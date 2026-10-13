"""Measure the inspector against a labelled dataset.

Dataset layout (same as `defect-inspect generate` produces):

    dataset/
      images/part_0000.png ...
      labels.json   {"part_0000.png": [{"type": "chip", "bbox": [x,y,w,h]}, ...], ...}

An image with an empty list is a good part. Your own factory images can be
labelled in the same format (e.g. export from CVAT / Label Studio).
"""
from __future__ import annotations

import json
from pathlib import Path

from .detectors import DEFECT_TYPES
from .inspector import Inspector
from .preprocess import load_image


def _overlaps(a, b, pad=6):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad < bx or bx + bw + pad < ax or ay + ah + pad < by or by + bh + pad < ay)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "tp": tp, "fp": fp, "fn": fn}


def evaluate(dataset_dir: str | Path, inspector: Inspector | None = None) -> dict:
    root = Path(dataset_dir)
    labels = json.loads((root / "labels.json").read_text())
    inspector = inspector or Inspector()

    # Image-level: "does this part have defect type X?"
    img_counts = {t: [0, 0, 0] for t in DEFECT_TYPES}
    # Defect-level: is each labelled defect found (bbox overlap)?
    loc_counts = {t: [0, 0, 0] for t in DEFECT_TYPES}
    verdict = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    errors = []

    for name, gt in labels.items():
        res = inspector.inspect(load_image(root / "images" / name), source=name)
        gt_types = {g["type"] for g in gt}
        pred_types = res.defect_types

        for t in DEFECT_TYPES:
            g, p = t in gt_types, t in pred_types
            img_counts[t][0] += g and p
            img_counts[t][1] += p and not g
            img_counts[t][2] += g and not p

            gts = [g_["bbox"] for g_ in gt if g_["type"] == t]
            preds = [d.bbox for d in res.defects if d.type == t]
            loc_counts[t][0] += sum(any(_overlaps(gb, pb) for pb in preds) for gb in gts)
            loc_counts[t][2] += sum(not any(_overlaps(gb, pb) for pb in preds) for gb in gts)
            loc_counts[t][1] += sum(not any(_overlaps(gb, pb) for gb in gts) for pb in preds)

        bad, flagged = bool(gt), res.verdict == "FAIL"
        key = {(True, True): "tp", (False, True): "fp", (True, False): "fn", (False, False): "tn"}
        verdict[key[(bad, flagged)]] += 1
        if gt_types != pred_types:
            errors.append({"image": name, "expected": sorted(gt_types), "predicted": sorted(pred_types)})

    total = sum(verdict.values())
    return {
        "images": total,
        "pass_fail": {**verdict,
                      "accuracy": round((verdict["tp"] + verdict["tn"]) / total, 3) if total else 0,
                      "escape_rate": round(verdict["fn"] / max(verdict["tp"] + verdict["fn"], 1), 3),
                      "false_reject_rate": round(verdict["fp"] / max(verdict["fp"] + verdict["tn"], 1), 3)},
        "per_type_image_level": {t: _prf(*img_counts[t]) for t in DEFECT_TYPES},
        "per_type_localisation": {t: _prf(*loc_counts[t]) for t in DEFECT_TYPES},
        "mismatched_images": errors,
    }
