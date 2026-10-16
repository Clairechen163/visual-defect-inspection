"""Turn a labelled image dataset into a labelled *patch* dataset.

For every labelled defect we crop a 64x64 patch around it (with a little
jitter). "good" patches are sampled both from the part interior and from
its outline - otherwise a model could learn the shortcut "edge = chip".

Splitting is done by *image*, never by patch: patches from one image are
correlated, and mixing them across train/test would inflate the scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import InspectionConfig
from ..preprocess import load_image, segment_part
from .classifier import CLASSES
from .features import PATCH, crop, patch_features, patch_pixels, reference_lab


@dataclass
class PatchSet:
    features: np.ndarray   # (N, 18)
    pixels: np.ndarray     # (N, 2304)
    y: np.ndarray          # (N,) class index into CLASSES
    image: np.ndarray      # (N,) source image name (for grouped splits)

    def subset(self, sel):
        return PatchSet(self.features[sel], self.pixels[sel], self.y[sel], self.image[sel])

    def __len__(self):
        return len(self.y)


def _window(cx, cy, s=PATCH):
    return cx - s // 2, cy - s // 2, s, s


def _intersects(a, b, pad=0):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or ay + ah + pad <= by or by + bh + pad <= ay)


def extract_patches(dataset_dir: str | Path, n_good_per_image: int = 4, jitter: int = 6,
                    seed: int = 0) -> PatchSet:
    root = Path(dataset_dir)
    labels = json.loads((root / "labels.json").read_text())
    nfile = root / "nuisances.json"
    nuisances = json.loads(nfile.read_text()) if nfile.exists() else {}
    rng = np.random.default_rng(seed)
    cfg = InspectionConfig()
    feats, pix, ys, imgs = [], [], [], []

    def add(img, mask, ref, cx, cy, cls, name):
        p, m = crop(img, cx, cy), crop(mask, cx, cy)
        feats.append(patch_features(p, m, ref))
        pix.append(patch_pixels(p, m, ref))
        ys.append(CLASSES.index(cls))
        imgs.append(name)

    for name, gt in labels.items():
        img = load_image(root / "images" / name)
        mask, contour = segment_part(img, cfg)
        if mask is None:
            continue
        ref = reference_lab(img, mask)
        boxes = [(g["type"], g["bbox"]) for g in gt]

        for t, (x, y, w, h) in boxes:
            cx = int(x + w / 2 + rng.integers(-jitter, jitter + 1))
            cy = int(y + h / 2 + rng.integers(-jitter, jitter + 1))
            win = _window(cx, cy)
            # Skip ambiguous patches that also contain a *different* defect type.
            if any(t2 != t and _intersects(win, b2) for t2, b2 in boxes):
                continue
            add(img, mask, ref, cx, cy, t, name)

        # Hard negatives: harmless look-alikes are labelled "good".
        for nb in nuisances.get(name, []):
            x, y, w, h = nb["bbox"]
            cx = int(x + w / 2 + rng.integers(-jitter, jitter + 1))
            cy = int(y + h / 2 + rng.integers(-jitter, jitter + 1))
            if not any(_intersects(_window(cx, cy), b) for _, b in boxes):
                add(img, mask, ref, cx, cy, "good", name)

        # Good patches: half interior, half on the outline.
        ys_in, xs_in = np.where(mask > 0)
        outline = contour[:, 0, :]
        made, tries = 0, 0
        while made < n_good_per_image and tries < 50:
            tries += 1
            if made % 2 == 0:
                i = rng.integers(len(xs_in))
                cx, cy = int(xs_in[i]), int(ys_in[i])
            else:
                cx, cy = (int(v) for v in outline[rng.integers(len(outline))])
            if any(_intersects(_window(cx, cy), b, pad=8) for _, b in boxes):
                continue
            add(img, mask, ref, cx, cy, "good", name)
            made += 1

    return PatchSet(np.array(feats), np.array(pix), np.array(ys), np.array(imgs))


def group_split(ps: PatchSet, fractions=(0.6, 0.2, 0.2), seed: int = 0):
    """Train/val/test split by source image."""
    names = np.unique(ps.image)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_tr = int(round(fractions[0] * len(names)))
    n_va = int(round(fractions[1] * len(names)))
    groups = names[:n_tr], names[n_tr:n_tr + n_va], names[n_tr + n_va:]
    return tuple(ps.subset(np.isin(ps.image, g)) for g in groups)
