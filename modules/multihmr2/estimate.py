"""
Multi-HMR 2 (Fiche et al. 2026, naver/multi-hmr2) single-image multi-person
detection + mesh recovery -> SMPL-X parameters.

Multi-HMR 2 is DETR-based and predicts the **Anny** body model (163 bones,
phenotype shape params) plus a scene camera — NOT SMPL-X. We bridge to SMPL-X in
two exact steps:
  1. Re-pose an Anny model built with ``topology="smplx"`` (naver's official
     Anny<->SMPL-X vertex correspondences) using the predicted bone poses + shape.
     Its vertices are 1:1 with SMPL-X's 10475, so this yields a *registered* target
     mesh (validated: reproduces the predicted mesh to ~1mm).
  2. Fit SMPL-X params (global_orient, body_pose, betas, transl, optionally hands)
     to that registered mesh with dense vertex + regressed-joint losses
     (~9mm mean vertex error on the demo image).

Source is loaded by path from vendor/multi-hmr2 (override with the MULTIHMR2_DIR
env var); install.py clones it. The ``anny`` package is a pip dependency
(``pip install anny warp-lang``).

Caveats (surfaced, not hidden):
  - The Anny "smplx" topology data is **non-commercial** (auto-downloaded by anny
    on first use), and the multihmr2 checkpoint follows Naver's release terms.
  - Anny covers all ages; extreme phenotypes (infants) sit outside SMPL-X's shape
    space, so the betas fit is best-effort there.
  - jaw/expression are not fitted -> graft Face: SMIRK, or pose in the editor.
  - Predicts in an OpenCV camera frame (Y-down, Z-forward); rotated 180° about X
    to Y-up, same as the other estimators.
"""

import os
import sys

import numpy as np
import torch

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# multi-hmr2 source checkout. install.py clones it into vendor/multi-hmr2; override
# with the MULTIHMR2_DIR env var.
MULTIHMR2_DIR = os.environ.get("MULTIHMR2_DIR", os.path.join(_PKG_ROOT, "vendor", "multi-hmr2"))

# OpenCV camera (Y-down, Z-forward) -> Y-up world (rotate 180° about X). Same as NLF.
_RFIX = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

_cache: dict = {}


def _import_multihmr2():
    src = os.path.join(MULTIHMR2_DIR, "src")
    if not os.path.isdir(os.path.join(src, "multihmr2")):
        raise RuntimeError(
            f"multi-hmr2 source not found at {src}. Clone "
            f"https://github.com/naver/multi-hmr2 into {MULTIHMR2_DIR} (install.py "
            f"does this for you), or set the MULTIHMR2_DIR env var to a checkout."
        )
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from multihmr2.api import InferenceSession
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"multi-hmr2 import failed: missing dependency '{getattr(e, 'name', e)}'. "
            f"Install with: pip install anny warp-lang torchmetrics termcolor scikit-learn"
        ) from e
    return InferenceSession


def load_multihmr2(ckpt_path: str, device: str):
    """Load (and cache) the Multi-HMR2 session + the Anny smplx-topology model.

    If ckpt_path does not exist, upstream InferenceSession downloads the official
    checkpoint from download.europe.naverlabs.com (via wget) into that path.
    First-ever run also builds the Anny blend-shape caches (~2 min, then cached
    in ~/.cache/anny) and fetches the non-commercial Anny<->SMPL-X data.
    """
    key = (os.path.abspath(ckpt_path), device)
    if key in _cache:
        return _cache[key]
    InferenceSession = _import_multihmr2()
    import anny

    session = InferenceSession(ckpt_path, device=device)
    # Anny re-posed on SMPL-X topology: vertices are 1:1 with SMPL-X's 10475.
    # Pose parameterization is irrelevant here (we override with "world" per call,
    # feeding the FK world matrices stored in PersonOutput.bone_poses).
    anny_smplx = anny.create_fullbody_model(
        local_changes=True, pose_parameterization="local-bone",
        remove_unattached_vertices=False, topology="smplx",
    ).to(dtype=torch.float32, device=device)
    anny_smplx.set_skinning_method("lbs")     # matches the decoder's body model
    bundle = {"session": session, "anny_smplx": anny_smplx}
    _cache[key] = bundle
    return bundle


def _pick_person(pred):
    """Largest person by 2D keypoint extent; returns a 1-slice PersonOutput."""
    p = pred.persons
    if p.num_person == 0:
        raise RuntimeError(
            "Multi-HMR2 detected no person in the image. Try lowering det_thresh, "
            "or a clearer photo."
        )
    j2d = p.j2d                                          # (N,J,2)
    ext = (j2d.max(dim=1).values - j2d.min(dim=1).values)  # (N,2)
    areas = (ext[:, 0] * ext[:, 1]).float()
    i = int(torch.argmax(areas))
    return p[i:i + 1]


def _repose_smplx_topology(anny_smplx, person):
    """Re-pose the Anny smplx-topology model with one person's prediction.

    Returns (10475,3) float32 numpy vertices in the OpenCV camera frame.
    PersonOutput.bone_poses are FK world matrices in root-relative space; feeding
    them back with pose_parameterization="world" reproduces the prediction exactly,
    and the offset re-adds the person's camera-space translation.
    """
    dev = anny_smplx.device
    labels = anny_smplx.phenotype_labels
    phen = {k: person.shape[:, l].to(dev, torch.float32) for l, k in enumerate(labels)}
    lc = {k: torch.zeros(1, dtype=torch.float32, device=dev)
          for k in anny_smplx.local_change_labels}
    out = anny_smplx(pose_parameters=person.bone_poses.to(dev, torch.float32),
                     phenotype_kwargs=phen, local_changes_kwargs=lc,
                     pose_parameterization="world")
    offset = (person.transl_pelvis[0] - person.bone_poses[0, 0, :3, 3]).to(dev)
    return (out["vertices"][0] + offset).detach().cpu().float().numpy()


def _kabsch(A, B):
    """Rotation matrix mapping point set A onto B (both (N,3) tensors)."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, _, Vt = torch.linalg.svd(Ac.T @ Bc)
    d = torch.sign(torch.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=A.device))
    return Vt.T @ D @ U.T


def _fit_smplx_to_registered(smplx_layer, target_verts, device, fit_hands=True):
    """Fit SMPL-X params to a registered (SMPL-X-topology) target mesh.

    target_verts: (10475,3) numpy, Y-up world. Dense vertex MSE + regressed-joint
    MSE, staged Adam (rigid+shape -> +body -> +hands). Returns dict of numpy
    params + mean vertex error (meters).
    """
    import roma

    tv = torch.as_tensor(np.asarray(target_verts, np.float32), device=device)
    J = smplx_layer.J_regressor.to(device=device, dtype=tv.dtype)
    tj = J @ tv                                          # (55,3) target joints

    with torch.no_grad():
        j0 = smplx_layer().joints[0, :55]
    R = _kabsch(j0[:22], tj[:22])

    params = {
        "global_orient": roma.rotmat_to_rotvec(R).reshape(1, 3).clone(),
        "transl": (tj[0] - j0[0]).reshape(1, 3).clone(),
        "betas": torch.zeros(1, 10, device=device),
        "body_pose": torch.zeros(1, 63, device=device),
        "left_hand_pose": torch.zeros(1, 45, device=device),
        "right_hand_pose": torch.zeros(1, 45, device=device),
    }
    for v in params.values():
        v.requires_grad_()

    def loss_fn():
        out = smplx_layer(**params)
        lv = (out.vertices[0] - tv).pow(2).sum(-1).mean()
        lj = (out.joints[0, :55] - tj).pow(2).sum(-1).mean()
        reg = (1e-4 * params["body_pose"].pow(2).mean()
               + 1e-3 * params["betas"].pow(2).mean()
               + 1e-3 * (params["left_hand_pose"].pow(2).mean()
                         + params["right_hand_pose"].pow(2).mean()))
        return lv + lj + reg

    stages = [(["global_orient", "transl", "betas"], 80, 0.05),
              (["global_orient", "transl", "betas", "body_pose"], 250, 0.02)]
    if fit_hands:
        stages.append((["global_orient", "transl", "betas", "body_pose",
                        "left_hand_pose", "right_hand_pose"], 120, 0.01))
    for keys, iters, lr in stages:
        opt = torch.optim.Adam([params[k] for k in keys], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()

    with torch.no_grad():
        err = float((smplx_layer(**params).vertices[0] - tv).norm(dim=-1).mean())
    out = {k: v.detach().cpu().numpy().reshape(-1).astype(np.float32)
           for k, v in params.items()}
    out["fit_err"] = err
    return out


def estimate_smplx_params(bundle, smplx_layer, image_rgb01, device,
                          conf_thresh=0.4, fit_hands=True, offload_after=True):
    """
    Run Multi-HMR2 on one image, bridge Anny -> SMPL-X, return SMPL-X params (Y-up).

    image_rgb01: (H,W,3) float RGB in [0,1] (a single ComfyUI IMAGE frame).
    smplx_layer: the smplx model from load_smplx (used as the fitting target model).
    Returns dict: global_orient(3), body_pose(63), betas(10), transl(3),
    left/right_hand_pose(45), fit_err, n_persons — numpy float32.
    """
    session, anny_smplx = bundle["session"], bundle["anny_smplx"]
    img_u8 = np.clip(np.asarray(image_rgb01, np.float32) * 255.0, 0, 255).astype(np.uint8)
    try:
        pred = session(img_u8, conf_thresh=conf_thresh)      # persons moved to CPU
        n_persons = int(pred.persons.num_person)
        person = _pick_person(pred)
        anny_smplx.to(device)
        verts_cam = _repose_smplx_topology(anny_smplx, person)
    finally:
        if offload_after:
            session.model.to("cpu")
            anny_smplx.to("cpu")
            if device != "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    target = verts_cam @ _RFIX.T                             # into Y-up world

    # ComfyUI executes under inference_mode; the fit needs autograd.
    with torch.inference_mode(False), torch.enable_grad():
        params = _fit_smplx_to_registered(smplx_layer, target, device,
                                          fit_hands=fit_hands)
    params["n_persons"] = n_persons
    return params
