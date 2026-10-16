"""Patch representations for the classifiers.

Two inputs are compared in the experiments:

1. ``patch_features`` - 18 interpretable statistics (colour deviation, line
   response, edge concavity). Good for linear models and for explaining *why*.
2. ``patch_pixels``   - the raw patch, colour-normalised and downsampled.
   A neural network has to learn its own features from these.
"""
from __future__ import annotations

import cv2
import numpy as np

PATCH = 64        # patch side in pixels (full resolution)
PIXEL_SIDE = 24   # side after downsampling for the pixel representation

FEATURE_NAMES = [
    "dL_mean", "dL_std", "da_abs_mean", "db_abs_mean",          # colour vs. part median
    "dE_mean", "dE_p95", "dE_max", "dE_frac_gt8", "dE_frac_gt15",
    "line_p99", "line_frac_gt18", "line_elongation", "line_length",  # scratch evidence
    "orient_coherence", "grad_mean",                            # texture / structure
    "part_fraction", "concavity_fraction", "concavity_depth",   # chip evidence
]


def lab_real(bgr: np.ndarray) -> np.ndarray:
    """OpenCV 8-bit Lab -> real-valued CIE L*a*b*."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def reference_lab(img: np.ndarray, mask: np.ndarray, margin: int = 8) -> np.ndarray:
    """Median colour of the part's interior = what 'normal' looks like for this part."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
    inner = cv2.erode(mask, k)
    region = inner if inner.any() else mask
    return np.median(lab_real(img)[region > 0], axis=0)


def crop(arr: np.ndarray, cx: int, cy: int, size: int = PATCH) -> np.ndarray:
    """Square crop centred on (cx, cy), edge-replicated past the image border."""
    h = size // 2
    pad = [(h, h), (h, h)] + [(0, 0)] * (arr.ndim - 2)
    padded = np.pad(arr, pad, mode="edge")
    return padded[cy:cy + size, cx:cx + size]


def _line_stats(gray, inner):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    resp = np.maximum(cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k),
                      cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)).astype(np.float32)
    vals = resp[inner > 0]
    if vals.size < 20:
        return 0.0, 0.0, 0.0, 0.0
    binary = ((resp > 18) & (inner > 0)).astype(np.uint8)
    elong = length = 0.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pts = np.column_stack(np.where(labels == i))[:, ::-1].astype(np.float32)
        if len(pts) >= 3:
            (_, _), (w, h), _ = cv2.minAreaRect(pts)
            length = max(w, h)
            elong = length / max(min(w, h), 1.0)
    return (float(np.percentile(vals, 99)), float((vals > 18).mean()),
            float(np.log1p(elong)), float(length))


def _structure(gray, inner):
    g = gray.astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    sel = inner > 0
    if sel.sum() < 20:
        return 0.0, 0.0
    jxx, jyy, jxy = (gx * gx)[sel].mean(), (gy * gy)[sel].mean(), (gx * gy)[sel].mean()
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return float(coh), float(np.sqrt(gx * gx + gy * gy)[sel].mean())


def _concavity(pmask):
    """How much of the part silhouette inside the patch is 'bitten out'.

    A convex part clipped by a square window is still convex, so any gap
    between the silhouette and its convex hull is evidence of a chip.
    """
    cnts, _ = cv2.findContours(pmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return 0.0, 0.0
    hull = cv2.convexHull(c)
    frac = max(cv2.contourArea(hull) - cv2.contourArea(c), 0.0) / pmask.size
    depth = 0.0
    try:
        idx = np.sort(cv2.convexHull(c, returnPoints=False), axis=0)
        d = cv2.convexityDefects(c, idx)
        if d is not None:
            depth = float(np.asarray(d).reshape(-1, 4)[:, 3].max()) / 256.0
    except cv2.error:
        pass
    return float(frac), depth


def patch_features(patch_bgr: np.ndarray, patch_mask: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    pm = (patch_mask > 0).astype(np.uint8) * 255
    inner = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    sel = inner > 0 if inner.sum() > 20 * 255 else pm > 0
    lab = lab_real(patch_bgr) - ref_lab
    if sel.sum() < 5:
        colour = [0.0] * 9
    else:
        d = lab[sel]
        de = np.sqrt((d ** 2).sum(axis=1))
        colour = [d[:, 0].mean(), d[:, 0].std(), np.abs(d[:, 1]).mean(), np.abs(d[:, 2]).mean(),
                  de.mean(), np.percentile(de, 95), de.max(), (de > 8).mean(), (de > 15).mean()]
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    line = _line_stats(gray, inner)
    struct = _structure(gray, inner)
    conc = _concavity(pm)
    return np.array(colour + list(line) + list(struct) + [pm.mean() / 255.0, *conc], np.float32)


def patch_pixels(patch_bgr: np.ndarray, patch_mask: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    """Lab difference from the part's normal colour + part mask, downsampled -> flat vector."""
    lab = (lab_real(patch_bgr) - ref_lab) / np.array([20.0, 10.0, 10.0], np.float32)
    lab = cv2.resize(lab, (PIXEL_SIDE, PIXEL_SIDE), interpolation=cv2.INTER_AREA)
    m = cv2.resize((patch_mask > 0).astype(np.float32), (PIXEL_SIDE, PIXEL_SIDE),
                   interpolation=cv2.INTER_AREA)
    lab = lab * m[..., None]  # background carries no colour information
    return np.concatenate([lab.reshape(-1), m.reshape(-1)]).astype(np.float32)
