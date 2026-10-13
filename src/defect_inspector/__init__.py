"""Visual defect inspection for manufactured parts: scratch, discoloration, chip."""
from .config import InspectionConfig
from .detectors import CHIP, DEFECT_TYPES, DISCOLORATION, SCRATCH, Defect
from .inspector import InspectionResult, Inspector
from .preprocess import load_image
from .visualize import annotate

__all__ = ["InspectionConfig", "Inspector", "InspectionResult", "Defect", "annotate",
           "load_image", "SCRATCH", "DISCOLORATION", "CHIP", "DEFECT_TYPES"]
__version__ = "0.1.0"
