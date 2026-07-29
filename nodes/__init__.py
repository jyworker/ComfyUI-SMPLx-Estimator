from .model_loaders import (LoadSMPLX, LoadNLF, LoadMultiHMR, LoadMultiHMR2,
                            LoadWiLoR, LoadSMIRK)
from .nlf_nodes import NLFSMPLXEstimator
from .multihmr_nodes import MultiHMREstimator
from .multihmr2_nodes import MultiHMR2Estimator
from .wilor_nodes import WiLoRHandEstimator
from .smirk_nodes import SMIRKFaceEstimator
from .smplx_nodes import SMPLXEditor
from .export_nodes import ExportMesh

NODE_CLASSES = [
    LoadSMPLX, LoadNLF, LoadMultiHMR, LoadMultiHMR2, LoadWiLoR, LoadSMIRK,
    NLFSMPLXEstimator,
    MultiHMREstimator,
    MultiHMR2Estimator,
    WiLoRHandEstimator,
    SMIRKFaceEstimator,
    SMPLXEditor,
    ExportMesh,
]

NODE_CLASS_MAPPINGS = {
    "LoadSMPLX": LoadSMPLX,
    "LoadNLF": LoadNLF,
    "LoadMultiHMR": LoadMultiHMR,
    "LoadMultiHMR2": LoadMultiHMR2,
    "LoadWiLoR": LoadWiLoR,
    "LoadSMIRK": LoadSMIRK,
    "NLFSMPLXEstimator": NLFSMPLXEstimator,
    "MultiHMREstimator": MultiHMREstimator,
    "MultiHMR2Estimator": MultiHMR2Estimator,
    "WiLoRHandEstimator": WiLoRHandEstimator,
    "SMIRKFaceEstimator": SMIRKFaceEstimator,
    "SMPLXEditor": SMPLXEditor,
    "ExportMesh": ExportMesh,
}
