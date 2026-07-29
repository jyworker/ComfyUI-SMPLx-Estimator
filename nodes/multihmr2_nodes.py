"""
Full Body: Multi-HMR2 — single-image multi-person body -> SMPL-X via Multi-HMR 2
(naver/multi-hmr2, 2026, Anny body model).

    Load Multi-HMR2 -> MULTIHMR2_MODEL ─► Full Body: Multi-HMR2 (model, image)
                                          -> SMPLX -> SMPL-X Editor

Multi-HMR2 predicts the Anny body model; we re-pose Anny on SMPL-X topology
(official correspondences) and fit SMPL-X to the registered mesh — so betas are
REAL (fitted) and hands are fitted too (fit_hands). Face stays flat -> graft
Face: SMIRK via `smplx_face`, or pose in the editor. Multiple people: the largest
person in view is used.
"""

import hashlib

import numpy as np

from ..modules.multihmr2.estimate import estimate_smplx_params
from ..modules.smplx_fit.model import load_smplx
from ..modules.smplx_fit.joint_maps import body_joint_names
from ..modules.smplx_fit.render import render_maps
from .smplx_nodes import _forward_mesh, _ground, _img


class MultiHMR2Estimator:
    """Multi-person body -> SMPL-X via Multi-HMR2 (fitted betas + hands)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MULTIHMR2_MODEL",),
                "image": ("IMAGE",),
                "det_thresh": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95,
                                         "step": 0.05,
                                         "tooltip": "Detection confidence threshold."}),
                "fit_hands": ("BOOLEAN", {"default": True,
                                          "tooltip": "Also fit finger pose from the "
                                                     "predicted mesh (slower)."}),
            },
            "optional": {
                "smplx_hands": ("SMPLX", {
                    "tooltip": "Optional SMPL-X (e.g. from Hand: WiLoR) to override the "
                               "fitted hand pose (wrist-relative — no melt)."}),
                "smplx_face": ("SMPLX", {
                    "tooltip": "Optional SMPL-X (e.g. from Face: SMIRK) to graft jaw + "
                               "expression. Takes precedence over smplx_hands' face."}),
            },
        }

    RETURN_TYPES = ("SMPLX", "IMAGE")
    RETURN_NAMES = ("smplx", "preview")
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "SMPLx Estimator"

    @classmethod
    def IS_CHANGED(cls, model, image, det_thresh, fit_hands,
                   smplx_hands=None, smplx_face=None):
        h = hashlib.sha256()
        h.update(np.asarray(image).tobytes())
        h.update(repr((model.get("smplx_parent"), model.get("gender"),
                       model.get("device"), det_thresh, fit_hands,
                       id(model.get("bundle")))).encode())
        for src in (smplx_hands, smplx_face):
            if src:
                for k in ("left_hand_pose", "right_hand_pose", "jaw_pose", "expression"):
                    if k in src:
                        h.update(np.asarray(src[k], np.float32).tobytes())
        return h.hexdigest()

    def estimate(self, model, image, det_thresh, fit_hands,
                 smplx_hands=None, smplx_face=None):
        b = model
        dev = b["device"]
        rgb01 = image[0].cpu().numpy().astype(np.float32)
        smplx_model = load_smplx(b["smplx_parent"], b["gender"], dev)
        params = estimate_smplx_params(b["bundle"], smplx_model, rgb01, dev,
                                       conf_thresh=det_thresh, fit_hands=fit_hands)
        z = lambda n: np.zeros(n, np.float32)  # noqa: E731
        smplx_dict = {
            "global_orient": params["global_orient"], "body_pose": params["body_pose"],
            "betas": params["betas"], "transl": params["transl"],
            "left_hand_pose": params["left_hand_pose"],
            "right_hand_pose": params["right_hand_pose"],
            "jaw_pose": z(3), "leye_pose": z(3), "reye_pose": z(3), "expression": z(10),
            "gender": b["gender"], "model_path": b["smplx_parent"],
            "joint_names": body_joint_names(), "joints_3d": np.zeros((55, 3), np.float32),
            "fit_loss": float(params["fit_err"]),
        }
        if smplx_hands:                                          # graft hands (+ any face)
            for k in ("left_hand_pose", "right_hand_pose", "jaw_pose", "expression"):
                if k in smplx_hands:
                    smplx_dict[k] = np.asarray(smplx_hands[k], np.float32).copy()
        if smplx_face:                                           # dedicated face wins jaw/expr
            for k in ("jaw_pose", "expression"):
                if k in smplx_face:
                    smplx_dict[k] = np.asarray(smplx_face[k], np.float32).copy()
        grafted = [s for s, on in (("hands", smplx_hands), ("face", smplx_face)) if on]
        print(f"[multihmr2] {params['n_persons']} person(s), largest -> SMPL-X on {dev} "
              f"(fit err {params['fit_err'] * 1000:.1f}mm)"
              + (f" + grafted {'/'.join(grafted)}" if grafted else ""))
        verts, faces, joints = _forward_mesh(smplx_dict, smplx_model, dev)
        smplx_dict["joints_3d"] = joints
        smplx_dict, verts = _ground(smplx_dict, verts)
        pose, _, _, _ = render_maps(verts, faces, dev, size=512, ground=False)
        return {"ui": {}, "result": (smplx_dict, _img(pose))}
