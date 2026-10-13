"""Inspection configuration.

All thresholds live here so a line engineer can tune the system for a new
part / camera / lighting setup by editing a JSON file instead of code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class InspectionConfig:
    # ---------------- Part segmentation ----------------
    # Smallest blob (as fraction of image area) accepted as "the part".
    min_part_area_ratio: float = 0.02
    # Pixels eroded from the part edge before looking for surface defects,
    # so the part outline itself is not mistaken for a scratch/stain.
    edge_margin_px: int = 8

    # ---------------- Scratch detection ----------------
    # Morphological kernel; features thinner than this are "line-like".
    scratch_kernel_px: int = 15
    # Minimum top-hat / black-hat response (grey levels).
    scratch_min_contrast: float = 18.0
    # Response must also be this many std-devs above the surface mean.
    scratch_sigma: float = 4.0
    scratch_min_length_px: float = 25.0
    # length / width of the fitted rectangle.
    scratch_min_elongation: float = 4.0

    # ---------------- Discoloration detection ----------------
    # Colour distance (CIE76 delta-E) from the part's median colour.
    discolor_delta_e: float = 12.0
    # Weight on lightness (L*) vs chroma (a*, b*). 1.0 catches dark/burn
    # stains best; lower it (e.g. 0.5) if uneven lighting causes false rejects.
    discolor_lightness_weight: float = 1.0
    discolor_blur_px: int = 9
    discolor_min_area_px: int = 150

    # ---------------- Chip detection ----------------
    # A concavity in the part outline deeper than this is an edge chip.
    chip_min_depth_px: float = 5.0
    # ...and also deeper than this fraction of the part's equivalent diameter.
    chip_min_depth_ratio: float = 0.012
    chip_min_area_px: int = 40

    # Which defect types are enabled.
    detect_scratch: bool = True
    detect_discoloration: bool = True
    detect_chip: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "InspectionConfig":
        data = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
