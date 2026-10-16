"""Synthetic part-image generator with ground-truth labels.

Real defect images are scarce on a new line, so this lets you develop,
test and tune the pipeline (and CI) before you have a labelled dataset.
It renders a machined metal part (disc or rounded rectangle) with brushed
texture and uneven lighting, then injects scratches, discoloration and
edge chips with known bounding boxes.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

H = W = 512


def _rounded_rect_mask(cx, cy, hw, hh, r):
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (cx - hw + r, cy - hh), (cx + hw - r, cy + hh), 255, -1)
    cv2.rectangle(m, (cx - hw, cy - hh + r), (cx + hw, cy + hh - r), 255, -1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cv2.circle(m, (cx + sx * (hw - r), cy + sy * (hh - r)), r, 255, -1)
    return m


def _make_part(rng):
    cx, cy = W // 2 + rng.integers(-20, 21), H // 2 + rng.integers(-20, 21)
    if rng.random() < 0.5:
        R = int(rng.integers(150, 195))
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, (cx, cy), R, 255, -1)
        shape = {"kind": "disc", "c": (cx, cy), "R": R}
    else:
        hw, hh = int(rng.integers(140, 200)), int(rng.integers(110, 170))
        r = int(rng.integers(15, 40))
        mask = _rounded_rect_mask(cx, cy, hw, hh, r)
        shape = {"kind": "rect", "c": (cx, cy), "hw": hw, "hh": hh, "r": r}
    return mask, shape


def _boundary_point(rng, shape):
    """Point on the part outline and the outward normal there."""
    cx, cy = shape["c"]
    if shape["kind"] == "disc":
        a = rng.uniform(0, 2 * np.pi)
        n = np.array([np.cos(a), np.sin(a)])
        return np.array([cx, cy]) + shape["R"] * n, n
    hw, hh, r = shape["hw"], shape["hh"], shape["r"]
    side = rng.integers(4)
    if side in (0, 1):  # top / bottom
        x = rng.uniform(cx - hw + r + 10, cx + hw - r - 10)
        y = cy - hh if side == 0 else cy + hh
        return np.array([x, y]), np.array([0, -1 if side == 0 else 1])
    y = rng.uniform(cy - hh + r + 10, cy + hh - r - 10)
    x = cx - hw if side == 2 else cx + hw
    return np.array([x, y]), np.array([-1 if side == 2 else 1, 0])


def _bbox(m):
    ys, xs = np.where(m > 0)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def generate_sample(rng: np.random.Generator, defects: list[str], intensity: float = 1.0,
                    n_nuisance: int = 0, return_nuisances: bool = False):
    """Render one image. Returns (bgr_image, labels) with labels as
    [{"type": ..., "bbox": [x, y, w, h]}, ...].

    `intensity` (0-1] scales how visible the defects are: 1.0 = clear,
    ~0.3 = faint scratches, pale stains and small chips (hard cases).

    `n_nuisance` adds harmless look-alikes that are NOT defects: dust specks,
    specular glare and short tool marks. They are the hard negatives a real
    line produces, and the reason an ML verifier is useful. With
    `return_nuisances=True` their boxes are returned as a third value."""
    part_mask, shape = _make_part(rng)
    labels = []

    # --- surface: brushed metal + tint + lighting gradient -----------------
    base = rng.uniform(150, 185)
    tint = rng.uniform(-6, 6, size=3)
    noise = rng.normal(0, 10, (H, W)).astype(np.float32)
    brushed = cv2.blur(noise, (31, 1)) * 1.6 + rng.normal(0, 2.5, (H, W))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ga = rng.uniform(0, 2 * np.pi)
    grad = ((xx - W / 2) * np.cos(ga) + (yy - H / 2) * np.sin(ga)) / (W / 2) * rng.uniform(4, 14)
    surf = base + brushed + grad
    img = np.stack([surf + tint[i] for i in range(3)], axis=2)

    # --- discoloration (heat tint / rust / dark stain) ----------------------
    inner = cv2.erode(part_mask, np.ones((61, 61), np.uint8))
    ys, xs = np.where(inner > 0)

    def random_inner_point():
        i = rng.integers(len(xs))
        return int(xs[i]), int(ys[i])

    for _ in range(defects.count("discoloration")):
        cx, cy = random_inner_point()
        ax, ay = int(rng.integers(18, 45)), int(rng.integers(14, 40))
        blob = np.zeros((H, W), np.uint8)
        cv2.ellipse(blob, (cx, cy), (ax, ay), float(rng.uniform(0, 180)), 0, 360, 255, -1)
        lbl_mask = blob.copy()
        soft = cv2.GaussianBlur(blob.astype(np.float32) / 255, (0, 0), 6)[..., None]
        palette = [(-45, -15, 30), (35, 5, -30), (-40, -40, -40), (-10, 25, 35)]  # BGR shifts
        shift = np.array(palette[rng.integers(len(palette))], np.float32) * rng.uniform(0.8, 1.2) * intensity
        img += soft * shift
        labels.append({"type": "discoloration", "bbox": _bbox(lbl_mask)})

    # --- scratches ----------------------------------------------------------
    for _ in range(defects.count("scratch")):
        p0 = np.array(random_inner_point(), float)
        ang = rng.uniform(0, np.pi)
        length = rng.uniform(50, 160)
        d = np.array([np.cos(ang), np.sin(ang)])
        bend = np.array([-d[1], d[0]]) * rng.uniform(-8, 8)
        pts = np.array([p0, p0 + d * length / 2 + bend, p0 + d * length]).astype(np.int32)
        line = np.zeros((H, W), np.uint8)
        cv2.polylines(line, [pts], False, 255, int(rng.integers(1, 4)), cv2.LINE_AA)
        line = cv2.bitwise_and(line, cv2.erode(part_mask, np.ones((25, 25), np.uint8)))
        if line.max() == 0:
            continue
        sign = -1 if rng.random() < 0.7 else 1  # most scratches read dark
        img += (line.astype(np.float32) / 255)[..., None] * sign * rng.uniform(45, 80) * intensity
        labels.append({"type": "scratch", "bbox": _bbox(line)})

    # --- nuisances (acceptable, not defects) --------------------------------
    nuisances = []
    for _ in range(n_nuisance):
        for _try in range(20):
            cx, cy = random_inner_point()
            if all(abs(cx - (b[0] + b[2] / 2)) > b[2] / 2 + 40 or abs(cy - (b[1] + b[3] / 2)) > b[3] / 2 + 40
                   for b in [l["bbox"] for l in labels] + [nb["bbox"] for nb in nuisances]):
                break
        else:
            continue
        kind = ("dust", "glare", "toolmark")[rng.integers(3)]
        layer = np.zeros((H, W), np.float32)
        if kind == "dust":            # a few tiny dark specks
            for _k in range(int(rng.integers(1, 4))):
                o = rng.normal(0, 4, 2)
                cv2.circle(layer, (int(cx + o[0]), int(cy + o[1])), int(rng.integers(1, 3)), 1.0, -1)
            img += layer[..., None] * -rng.uniform(25, 50)
        elif kind == "glare":         # soft, bright, colourless highlight
            cv2.ellipse(layer, (cx, cy), (int(rng.integers(10, 22)), int(rng.integers(5, 12))),
                        float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
            layer = cv2.GaussianBlur(layer, (0, 0), 5)
            img += layer[..., None] * rng.uniform(25, 45)
        else:                         # short, shallow machining mark
            a = rng.uniform(0, np.pi)
            L = rng.uniform(8, 16)
            p0 = (int(cx - L / 2 * np.cos(a)), int(cy - L / 2 * np.sin(a)))
            p1 = (int(cx + L / 2 * np.cos(a)), int(cy + L / 2 * np.sin(a)))
            cv2.line(layer, p0, p1, 1.0, 1, cv2.LINE_AA)
            img += layer[..., None] * -rng.uniform(20, 35)
        ys_, xs_ = np.where(layer > 0.05)
        nuisances.append({"type": kind, "bbox": [int(xs_.min()), int(ys_.min()),
                                                 int(np.ptp(xs_) + 1), int(np.ptp(ys_) + 1)]})

    # --- composite on background -------------------------------------------
    bg = rng.uniform(25, 50) + rng.normal(0, 3, (H, W, 1))
    pm = part_mask
    for _ in range(defects.count("chip")):
        p, n = _boundary_point(rng, shape)
        r = max(5, int(rng.integers(12, 28) * (0.4 + 0.6 * intensity)))
        c = p + n * rng.uniform(-0.3, 0.3) * r
        bite = np.zeros((H, W), np.uint8)
        # Irregular bite: a few overlapping circles = fractured look.
        for _k in range(3):
            o = rng.normal(0, r * 0.25, 2)
            cv2.circle(bite, tuple((c + o).astype(int)), int(r * rng.uniform(0.6, 1.0)), 255, -1)
        removed = cv2.bitwise_and(bite, pm)
        if removed.sum() == 0:
            continue
        pm = cv2.bitwise_and(pm, cv2.bitwise_not(bite))
        labels.append({"type": "chip", "bbox": _bbox(removed)})

    m = (cv2.GaussianBlur(pm, (3, 3), 0).astype(np.float32) / 255)[..., None]
    img = img * m + bg * (1 - m)
    img = cv2.GaussianBlur(img, (3, 3), 0) + rng.normal(0, 1.5, img.shape)
    out = np.clip(img, 0, 255).astype(np.uint8)
    return (out, labels, nuisances) if return_nuisances else (out, labels)


def generate_dataset(out_dir: str | Path, n: int = 100, seed: int = 0,
                     good_ratio: float = 0.3, min_intensity: float = 1.0,
                     nuisance_rate: float = 0.0) -> Path:
    """`nuisance_rate` = mean number of harmless look-alikes per image (Poisson).
    Their locations go to nuisances.json (they are *not* defects)."""
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_labels, all_nuis = {}, {}
    for i in range(n):
        defects: list[str] = []
        if rng.random() >= good_ratio:
            for t in ("scratch", "discoloration", "chip"):
                if rng.random() < 0.45:
                    defects += [t] * int(rng.integers(1, 3))
            if not defects:
                defects = [str(rng.choice(["scratch", "discoloration", "chip"]))]
        intensity = 1.0 if min_intensity >= 1.0 else float(rng.uniform(min_intensity, 1.0))
        n_nuis = int(rng.poisson(nuisance_rate)) if nuisance_rate > 0 else 0
        img, labels, nuis = generate_sample(rng, defects, intensity, n_nuis, return_nuisances=True)
        name = f"part_{i:04d}.png"
        cv2.imwrite(str(out / "images" / name), img)
        all_labels[name] = labels
        all_nuis[name] = nuis
    (out / "labels.json").write_text(json.dumps(all_labels, indent=1))
    if nuisance_rate > 0:
        (out / "nuisances.json").write_text(json.dumps(all_nuis, indent=1))
    return out
