"""Image loading and part segmentation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import InspectionConfig


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def segment_part(img: np.ndarray, cfg: InspectionConfig):
    """Separate the part from the background.

    Assumes a single part on a contrasting, roughly uniform background
    (backlight, dark conveyor, or inspection tray) - the standard setup
    for a fixed inspection station.

    Returns (mask, contour) where mask is uint8 {0,255} with interior holes
    filled, or (None, None) if no part is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The image border should be background; if it isn't, the part is darker
    # than the background, so invert.
    border = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]])
    if border.mean() > 127:
        th = cv2.bitwise_not(th)

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg.min_part_area_ratio * gray.size:
        return None, None

    # Fill the outer contour: a dark stain or deep scratch inside the part
    # must stay "part", otherwise it would look like a hole.
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask, contour


def inner_mask(mask: np.ndarray, margin_px: int) -> np.ndarray:
    if margin_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    return cv2.erode(mask, k)
