"""High-level inspection pipeline: image in, PASS/FAIL + defect list out."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import InspectionConfig
from .detectors import (DEFECT_TYPES, Defect, detect_chips, detect_discoloration,
                        detect_scratches)
from .preprocess import segment_part


@dataclass
class InspectionResult:
    verdict: str                      # "PASS" | "FAIL" | "NO_PART"
    defects: list[Defect] = field(default_factory=list)
    contour: np.ndarray | None = None
    elapsed_ms: float = 0.0
    source: str = ""

    @property
    def defect_types(self) -> set[str]:
        return {d.type for d in self.defects}

    def summary(self) -> dict:
        return {t: sum(d.type == t for d in self.defects) for t in DEFECT_TYPES}

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "verdict": self.verdict,
            "has_scratch": any(d.type == "scratch" for d in self.defects),
            "has_discoloration": any(d.type == "discoloration" for d in self.defects),
            "has_chip": any(d.type == "chip" for d in self.defects),
            "counts": self.summary(),
            "defects": [d.to_dict() for d in self.defects],
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class Inspector:
    def __init__(self, cfg: InspectionConfig | None = None):
        self.cfg = cfg or InspectionConfig()

    def inspect(self, img: np.ndarray, source: str = "") -> InspectionResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        mask, contour = segment_part(img, cfg)
        if mask is None:
            return InspectionResult("NO_PART", source=source,
                                    elapsed_ms=(time.perf_counter() - t0) * 1e3)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        defects: list[Defect] = []
        scratch_mask = None
        if cfg.detect_scratch:
            found, scratch_mask = detect_scratches(gray, mask, cfg)
            defects += found
        if cfg.detect_discoloration:
            defects += detect_discoloration(img, mask, cfg, exclude=scratch_mask)
        if cfg.detect_chip:
            defects += detect_chips(mask, contour, cfg)

        return InspectionResult("FAIL" if defects else "PASS", defects, contour,
                                (time.perf_counter() - t0) * 1e3, source)
