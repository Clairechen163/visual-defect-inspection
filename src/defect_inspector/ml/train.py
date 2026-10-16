"""End-to-end ML experiment: `defect-inspect train-ml`.

1. Extract labelled patches and split them by image (train / val / test).
2. Train a 2 x 2 grid: {softmax regression, neural network} x {statistical
   features, raw pixels}.
3. Score on the held-out test patches with cluster-bootstrap 95% CIs,
   McNemar tests between models, and calibration error.
4. Plug the best feature model into the inspector as a verifier behind
   high-sensitivity rules, and compare PASS/FAIL performance on the
   held-out test *images*.
5. Save models, report.json and figures.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import InspectionConfig
from ..inspector import Inspector
from ..preprocess import load_image
from .classifier import CLASSES, PatchClassifier
from .dataset import extract_patches, group_split
from .features import FEATURE_NAMES
from .metrics import (accuracy, bootstrap_ci, confusion_matrix, expected_calibration_error,
                      macro_f1, mcnemar_exact, per_class)
from .mlp import MLP
from .softmax import SoftmaxRegression

# High-sensitivity rule thresholds: catch faint defects, accept false alarms
# that the ML verifier will remove. Also shipped as configs/high_sensitivity.json.
HIGH_SENSITIVITY = dict(scratch_min_contrast=12.0, scratch_sigma=3.0, scratch_min_length_px=18.0,
                        discolor_delta_e=7.0, discolor_min_area_px=80,
                        chip_min_depth_px=3.0, chip_min_depth_ratio=0.006)
K = len(CLASSES)


def build_models(n_features: int, n_pixels: int, seed: int = 0):
    # Hyper-parameters were chosen on the validation split only (never on test).
    return {
        ("softmax", "features"): SoftmaxRegression(K, l2=1e-3, lr=0.01, seed=seed),
        ("mlp", "features"): MLP([n_features, 64, 32, K], l2=1e-4, dropout=0.2, seed=seed),
        ("softmax", "pixels"): SoftmaxRegression(K, l2=1e-2, lr=3e-4, epochs=400, seed=seed),
        ("mlp", "pixels"): MLP([n_pixels, 128, 64, K], l2=1e-3, dropout=0.3, seed=seed),
    }


def _image_level(dataset_dir: Path, names, inspector: Inspector):
    labels = json.loads((dataset_dir / "labels.json").read_text())
    bad, flagged = [], []
    for n in names:
        res = inspector.inspect(load_image(dataset_dir / "images" / n), source=n)
        bad.append(bool(labels[n]))
        flagged.append(res.verdict == "FAIL")
    return np.array(bad), np.array(flagged)


def _pf_metrics(bad, flagged, seed=0):
    """PASS/FAIL accuracy, escape rate and false-reject rate with bootstrap CIs."""
    def esc(b, f):
        return float((b & ~f).sum() / max(b.sum(), 1))

    def frr(b, f):
        return float((~b & f).sum() / max((~b).sum(), 1))

    def acc(b, f):
        return float((b == f).mean())

    out = {}
    for name, fn in [("accuracy", acc), ("escape_rate", esc), ("false_reject_rate", frr)]:
        lo, hi = bootstrap_ci(bad, flagged, fn, seed=seed)
        out[name] = {"value": round(fn(bad, flagged), 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def run(dataset_dir: str | Path, out_dir: str | Path, seed: int = 0, verbose: bool = True) -> dict:
    t0 = time.time()
    root, out = Path(dataset_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    log("Extracting patches ...")
    ps = extract_patches(root, seed=seed)
    tr, va, te = group_split(ps, seed=seed)
    log(f"  {len(ps)} patches | train {len(tr)} / val {len(va)} / test {len(te)} "
        f"| classes {dict(zip(CLASSES, np.bincount(ps.y, minlength=K).tolist()))}")

    report = {"classes": CLASSES, "feature_names": FEATURE_NAMES,
              "split_images": {"train": int(len(np.unique(tr.image))), "val": int(len(np.unique(va.image))),
                               "test": int(len(np.unique(te.image)))},
              "split_patches": {"train": len(tr), "val": len(va), "test": len(te)},
              "models": {}, "mcnemar": {}}
    models, preds, probs = build_models(ps.features.shape[1], ps.pixels.shape[1], seed), {}, {}

    for (arch, kind), model in models.items():
        key = f"{arch}_{kind}"
        log(f"Training {key} ...")
        ts = time.time()
        model.fit(getattr(tr, kind), tr.y, getattr(va, kind), va.y)
        P = model.predict_proba(getattr(te, kind))
        yhat = P.argmax(1)
        preds[key], probs[key] = yhat, P
        cm = confusion_matrix(te.y, yhat, K)
        acc_ci = bootstrap_ci(te.y, yhat, accuracy, groups=te.image, seed=seed)
        f1_ci = bootstrap_ci(te.y, yhat, lambda a, b: macro_f1(a, b, K), groups=te.image, seed=seed)
        report["models"][key] = {
            "arch": arch, "input": kind,
            "n_params": int(sum(v.size for k, v in model.params.items())),
            "epochs_run": len(model.history["train_loss"]),
            "train_seconds": round(time.time() - ts, 1),
            "test_accuracy": round(accuracy(te.y, yhat), 4), "test_accuracy_ci95": [round(v, 4) for v in acc_ci],
            "test_macro_f1": round(macro_f1(te.y, yhat, K), 4), "test_macro_f1_ci95": [round(v, 4) for v in f1_ci],
            "ece": round(expected_calibration_error(P, te.y), 4),
            "confusion_matrix": cm.tolist(), "per_class": per_class(cm, CLASSES),
            "history": {k: [round(float(x), 5) for x in v] for k, v in model.history.items()},
        }
        PatchClassifier(model, kind, key).save(out / f"{key}.npz")
        log(f"  acc {report['models'][key]['test_accuracy']:.3f} "
            f"[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]  macro-F1 {report['models'][key]['test_macro_f1']:.3f}")

    for a, b in [("softmax_features", "mlp_features"), ("softmax_pixels", "mlp_pixels"),
                 ("mlp_features", "mlp_pixels")]:
        report["mcnemar"][f"{a}_vs_{b}"] = mcnemar_exact(te.y, preds[a], preds[b])

    # Interpretable softmax weights: standardised feature -> class log-odds.
    report["softmax_coefficients"] = models[("softmax", "features")].params["W"].round(3).tolist()

    # ----- the system-level question: does the model help the inspector? -----
    best = max(("softmax_features", "mlp_features"), key=lambda k: report["models"][k]["test_macro_f1"])
    clf = PatchClassifier.load(out / f"{best}.npz")
    test_images = sorted(set(te.image.tolist()))
    sens = InspectionConfig(**HIGH_SENSITIVITY)
    setups = {
        "Rules (default)": Inspector(InspectionConfig()),
        "Rules (high sensitivity)": Inspector(sens),
        "High sensitivity + softmax": Inspector(sens, PatchClassifier.load(out / "softmax_features.npz")),
        "High sensitivity + neural net": Inspector(sens, PatchClassifier.load(out / "mlp_features.npz")),
    }
    log(f"Image-level PASS/FAIL on {len(test_images)} held-out images ...")
    report["inspector_comparison"] = {}
    for name, ins in setups.items():
        bad, flagged = _image_level(root, test_images, ins)
        report["inspector_comparison"][name] = _pf_metrics(bad, flagged, seed=seed)
        m = report["inspector_comparison"][name]
        log(f"  {name:32s} acc {m['accuracy']['value']:.3f}  escape {m['escape_rate']['value']:.3f}  "
            f"false-reject {m['false_reject_rate']['value']:.3f}")
    report["deployed_model"] = best
    report["elapsed_seconds"] = round(time.time() - t0, 1)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    try:
        from .plots import make_all
        figs = make_all(report, out)
        log("Figures: " + ", ".join(str(f) for f in figs))
    except ImportError:
        log("matplotlib not installed - skipping figures (pip install -e '.[ml]')")
    log(f"Done in {report['elapsed_seconds']} s. Models + report in {out}")
    return report


def model_factories() -> dict:
    """name -> (input kind, make(n_inputs, seed)) with the same settings as build_models."""
    return {
        "softmax_features": ("features", lambda d, s: SoftmaxRegression(K, l2=1e-3, lr=0.01, seed=s)),
        "mlp_features": ("features", lambda d, s: MLP([d, 64, 32, K], l2=1e-4, dropout=0.2, seed=s)),
        "softmax_pixels": ("pixels", lambda d, s: SoftmaxRegression(K, l2=1e-2, lr=3e-4, epochs=400, seed=s)),
        "mlp_pixels": ("pixels", lambda d, s: MLP([d, 128, 64, K], l2=1e-3, dropout=0.3, seed=s)),
    }
