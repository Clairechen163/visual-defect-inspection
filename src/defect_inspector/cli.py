"""Command-line interface: `defect-inspect {inspect,generate,evaluate,config}`."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from .config import InspectionConfig
from .evaluate import evaluate
from .inspector import Inspector
from .preprocess import load_image
from .synth import generate_dataset
from .visualize import annotate

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_cfg(path):
    return InspectionConfig.from_json(path) if path else InspectionConfig()


def _iter_images(target: Path):
    if target.is_dir():
        yield from sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    else:
        yield target


def _load_model(path):
    if not path:
        return None
    from .ml.classifier import PatchClassifier
    return PatchClassifier.load(path)


def cmd_inspect(a) -> int:
    inspector = Inspector(_load_cfg(a.config), _load_model(a.model))
    out = Path(a.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    results, any_fail = [], False
    for path in _iter_images(Path(a.target)):
        res = inspector.inspect(load_image(path), source=str(path))
        results.append(res.to_dict())
        any_fail |= res.verdict != "PASS"
        cv2.imwrite(str(out / "annotated" / (path.stem + ".png")), annotate(load_image(path), res))
        flags = [t for t in ("scratch", "discoloration", "chip") if res.to_dict()[f"has_{t}"]]
        print(f"{res.verdict:7s} {path.name:30s} {', '.join(flags) or '-':35s} {res.elapsed_ms:6.1f} ms")

    (out / "report.json").write_text(json.dumps(results, indent=2))
    with open(out / "report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "verdict", "scratch", "discoloration", "chip", "n_defects"])
        for r in results:
            w.writerow([r["source"], r["verdict"], int(r["has_scratch"]),
                        int(r["has_discoloration"]), int(r["has_chip"]), len(r["defects"])])
    n_fail = sum(r["verdict"] != "PASS" for r in results)
    print(f"\n{len(results)} inspected, {n_fail} rejected. Report: {out / 'report.json'}")
    return 1 if (a.fail_exit and any_fail) else 0


def cmd_generate(a) -> int:
    out = generate_dataset(a.out, n=a.n, seed=a.seed, good_ratio=a.good_ratio,
                           min_intensity=a.min_intensity, nuisance_rate=a.nuisance_rate)
    print(f"Wrote {a.n} images + labels.json to {out}")
    return 0


def cmd_evaluate(a) -> int:
    report = evaluate(a.dataset, Inspector(_load_cfg(a.config), _load_model(a.model)))
    pf = report["pass_fail"]
    print(f"Images: {report['images']}   PASS/FAIL accuracy: {pf['accuracy']:.1%}   "
          f"escape rate: {pf['escape_rate']:.1%}   false reject rate: {pf['false_reject_rate']:.1%}\n")
    print(f"{'defect':15s} {'precision':>9s} {'recall':>7s} {'f1':>6s}   (image level)")
    for t, m in report["per_type_image_level"].items():
        print(f"{t:15s} {m['precision']:9.3f} {m['recall']:7.3f} {m['f1']:6.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nFull report: {a.out}")
    return 0


def cmd_train_ml(a) -> int:
    from .ml.train import run
    if a.generate:
        generate_dataset(a.dataset, n=a.generate, seed=a.seed + 11, min_intensity=0.25,
                         nuisance_rate=2.0)
        print(f"Generated {a.generate} hard synthetic images in {a.dataset}")
    run(a.dataset, a.out, seed=a.seed)
    return 0


def cmd_config(a) -> int:
    InspectionConfig().to_json(a.out)
    print(f"Default config written to {a.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="defect-inspect",
                                description="Visual defect inspection: scratch / discoloration / chip")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect", help="Inspect an image or a folder of images")
    s.add_argument("target")
    s.add_argument("--out", default="results")
    s.add_argument("--config")
    s.add_argument("--model", help="Trained .npz classifier used to verify rule-based candidates")
    s.add_argument("--fail-exit", action="store_true",
                   help="Exit with code 1 if any part fails (for PLC / line integration)")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("generate", help="Generate a synthetic labelled dataset")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--good-ratio", type=float, default=0.3)
    s.add_argument("--min-intensity", type=float, default=1.0,
                   help="<1 adds faint defects, e.g. 0.25 for a hard dataset")
    s.add_argument("--nuisance-rate", type=float, default=0.0,
                   help="mean harmless look-alikes (dust, glare, tool marks) per image")
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("evaluate", help="Score the inspector on a labelled dataset")
    s.add_argument("dataset")
    s.add_argument("--config")
    s.add_argument("--model", help="Trained .npz classifier used to verify rule-based candidates")
    s.add_argument("--out", help="Write full JSON report here")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("train-ml", help="Train & compare softmax regression and neural-network classifiers")
    s.add_argument("dataset", help="Labelled dataset folder (images/ + labels.json)")
    s.add_argument("--out", default="models/run")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--generate", type=int, metavar="N",
                   help="First generate N hard synthetic images (faint defects + nuisances) into DATASET")
    s.set_defaults(fn=cmd_train_ml)

    s = sub.add_parser("config", help="Write the default config to a JSON file")
    s.add_argument("--out", default="config.json")
    s.set_defaults(fn=cmd_config)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
