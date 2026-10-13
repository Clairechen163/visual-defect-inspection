"""The three defect detectors.

* Scratch        - thin, elongated line that is darker or brighter than the
                   surrounding surface (morphological top-hat / black-hat).
* Discoloration  - area whose colour deviates from the part's normal colour
                   (delta-E in CIE L*a*b*), e.g. rust, heat tint, stains.
* Chip           - material missing from the part edge (concavity in the
                   outline vs. its convex hull).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .config import InspectionConfig
from .preprocess import inner_mask

SCRATCH = "scratch"
DISCOLORATION = "discoloration"
CHIP = "chip"
DEFECT_TYPES = (SCRATCH, DISCOLORATION, CHIP)


@dataclass
class Defect:
    type: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    severity: float  # detector-specific strength, larger = worse
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = [int(v) for v in self.bbox]
        d["severity"] = round(float(self.severity), 2)
        d["details"] = {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v)
                        for k, v in self.details.items()}
        return d


# --------------------------------------------------------------------------
# Scratch
# --------------------------------------------------------------------------
def scratch_response(gray: np.ndarray, cfg: InspectionConfig) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.scratch_kernel_px,) * 2)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)  # dark thin lines
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)      # bright thin lines
    return np.maximum(blackhat, tophat)


def detect_scratches(gray: np.ndarray, mask: np.ndarray, cfg: InspectionConfig):
    """Returns (defects, scratch_pixel_mask)."""
    inner = inner_mask(mask, cfg.edge_margin_px)
    resp = scratch_response(cv2.GaussianBlur(gray, (3, 3), 0), cfg)
    vals = resp[inner > 0]
    if vals.size == 0:
        return [], np.zeros_like(gray)
    thr = max(cfg.scratch_min_contrast, float(vals.mean() + cfg.scratch_sigma * vals.std()))
    binary = ((resp > thr) & (inner > 0)).astype(np.uint8) * 255
    # Bridge small gaps along a scratch.
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    defects, scratch_mask = [], np.zeros_like(gray)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 10:
            continue
        pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
        length, width = max(w, h), max(min(w, h), 1.0)
        elong = length / width
        if length < cfg.scratch_min_length_px or elong < cfg.scratch_min_elongation:
            continue
        comp = labels == i
        scratch_mask[comp] = 255
        x, y, bw, bh = stats[i, :4]
        defects.append(Defect(SCRATCH, (x, y, bw, bh), int(stats[i, cv2.CC_STAT_AREA]),
                              severity=float(length),
                              details={"length_px": length, "width_px": width,
                                       "contrast": float(resp[comp].mean())}))
    return defects, scratch_mask


# --------------------------------------------------------------------------
# Discoloration
# --------------------------------------------------------------------------
def delta_e_map(img: np.ndarray, region: np.ndarray, cfg: InspectionConfig) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV 8-bit Lab -> real L*a*b*
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    ref = np.median(lab[region > 0], axis=0)
    diff = lab - ref
    diff[..., 0] *= cfg.discolor_lightness_weight
    de = np.sqrt((diff ** 2).sum(axis=2))
    b = cfg.discolor_blur_px | 1
    return cv2.GaussianBlur(de, (b, b), 0)


def detect_discoloration(img: np.ndarray, mask: np.ndarray, cfg: InspectionConfig,
                         exclude: np.ndarray | None = None):
    inner = inner_mask(mask, cfg.edge_margin_px)
    if not inner.any():
        return []
    de = delta_e_map(img, inner, cfg)
    binary = ((de > cfg.discolor_delta_e) & (inner > 0)).astype(np.uint8) * 255
    if exclude is not None and exclude.any():
        # Pixels already explained by a scratch should not double-count.
        grown = cv2.dilate(exclude, np.ones((7, 7), np.uint8))
        binary[grown > 0] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    defects = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.discolor_min_area_px:
            continue
        comp = labels == i
        x, y, w, h = stats[i, :4]
        defects.append(Defect(DISCOLORATION, (x, y, w, h), area,
                              severity=float(de[comp].max()),
                              details={"mean_delta_e": float(de[comp].mean()),
                                       "max_delta_e": float(de[comp].max())}))
    return defects


# --------------------------------------------------------------------------
# Chip
# --------------------------------------------------------------------------
def detect_chips(mask: np.ndarray, contour: np.ndarray, cfg: InspectionConfig):
    area = cv2.contourArea(contour)
    eq_diam = np.sqrt(4 * area / np.pi)
    min_depth = max(cfg.chip_min_depth_px, cfg.chip_min_depth_ratio * eq_diam)

    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        hull_idx = np.sort(hull_idx, axis=0)  # convexityDefects needs monotonic hull
        cdefs = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return []
    if cdefs is None:
        return []

    pts = contour[:, 0, :]
    n = len(pts)
    defects = []
    for s, e, f, d in np.asarray(cdefs).reshape(-1, 4):
        depth = d / 256.0
        if depth < min_depth:
            continue
        idx = np.arange(s, e + 1) if e >= s else np.r_[np.arange(s, n), np.arange(0, e + 1)]
        region = pts[idx]
        missing_area = cv2.contourArea(region.astype(np.int32))  # outline + hull chord
        if missing_area < cfg.chip_min_area_px:
            continue
        # The hull chord may span a whole straight side; keep only the
        # outline points that actually dip away from it (the chip itself).
        a, b = pts[s].astype(float), pts[e].astype(float)
        ab = b - a
        norm = np.linalg.norm(ab) or 1.0
        dist = np.abs(ab[0] * (region[:, 1] - a[1]) - ab[1] * (region[:, 0] - a[0])) / norm
        dip = region[dist > max(2.0, 0.15 * depth)]
        if len(dip) < 3:
            dip = region
        x, y, w, h = cv2.boundingRect(dip.astype(np.int32))
        width = float(max(w, h))
        defects.append(Defect(CHIP, (x, y, w, h), int(missing_area), severity=depth,
                              details={"depth_px": depth, "opening_px": width,
                                       "deepest_point": [int(v) for v in pts[f]]}))
    return defects
