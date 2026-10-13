"""Draw inspection results onto the image for operators / audit trail."""
from __future__ import annotations

import cv2
import numpy as np

from .inspector import InspectionResult

COLORS = {  # BGR
    "scratch": (0, 0, 255),        # red
    "discoloration": (0, 200, 255),  # amber
    "chip": (255, 0, 255),         # magenta
}


def annotate(img: np.ndarray, result: InspectionResult) -> np.ndarray:
    out = img.copy()
    if result.contour is not None:
        cv2.drawContours(out, [result.contour], -1, (255, 200, 0), 1)
    for d in result.defects:
        x, y, w, h = d.bbox
        pad = 4
        c = COLORS.get(d.type, (255, 255, 255))
        cv2.rectangle(out, (x - pad, y - pad), (x + w + pad, y + h + pad), c, 2)
        label = d.type
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y - pad - 4, th + 2)
        cv2.rectangle(out, (x - pad, ty - th - 3), (x - pad + tw + 4, ty + 2), c, -1)
        cv2.putText(out, label, (x - pad + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)

    banner = {"PASS": (40, 160, 40), "FAIL": (30, 30, 200), "NO_PART": (90, 90, 90)}[result.verdict]
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), banner, -1)
    counts = ", ".join(f"{k}:{v}" for k, v in result.summary().items() if v)
    text = f"{result.verdict}  {counts}".strip()
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
