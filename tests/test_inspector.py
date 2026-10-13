import json

import cv2
import numpy as np
import pytest

from defect_inspector import InspectionConfig, Inspector, annotate
from defect_inspector.cli import main
from defect_inspector.evaluate import evaluate
from defect_inspector.synth import generate_dataset, generate_sample


@pytest.fixture(scope="module")
def inspector():
    return Inspector()


def _sample(seed, defects):
    return generate_sample(np.random.default_rng(seed), defects)


@pytest.mark.parametrize("seed", range(5))
def test_good_part_passes(inspector, seed):
    img, _ = _sample(seed, [])
    res = inspector.inspect(img)
    assert res.verdict == "PASS", res.to_dict()


@pytest.mark.parametrize("defect", ["scratch", "discoloration", "chip"])
@pytest.mark.parametrize("seed", range(3))
def test_each_defect_type_detected(inspector, defect, seed):
    img, labels = _sample(100 + seed, [defect])
    assert labels, "generator should have placed a defect"
    res = inspector.inspect(img)
    assert res.verdict == "FAIL"
    assert defect in res.defect_types, res.to_dict()


def test_combined_defects(inspector):
    img, _ = _sample(42, ["scratch", "discoloration", "chip"])
    res = inspector.inspect(img)
    assert res.defect_types == {"scratch", "discoloration", "chip"}


def test_empty_image_reports_no_part(inspector):
    img = np.full((256, 256, 3), 40, np.uint8)
    assert inspector.inspect(img).verdict == "NO_PART"


def test_dark_part_on_bright_background(inspector):
    img, _ = _sample(3, ["chip"])
    res = inspector.inspect(255 - img)  # inverted: dark part, bright background
    assert "chip" in res.defect_types


def test_disabling_a_detector(inspector):
    img, _ = _sample(101, ["chip"])
    res = Inspector(InspectionConfig(detect_chip=False)).inspect(img)
    assert "chip" not in res.defect_types


def test_report_and_annotation(inspector):
    img, _ = _sample(5, ["scratch"])
    res = inspector.inspect(img, source="x.png")
    d = res.to_dict()
    json.dumps(d)  # must be serialisable
    assert d["has_scratch"] and d["verdict"] == "FAIL"
    assert annotate(img, res).shape == img.shape


def test_config_roundtrip(tmp_path):
    p = tmp_path / "cfg.json"
    InspectionConfig(scratch_min_length_px=40).to_json(p)
    assert InspectionConfig.from_json(p).scratch_min_length_px == 40
    p.write_text('{"not_a_key": 1}')
    with pytest.raises(ValueError):
        InspectionConfig.from_json(p)


def test_accuracy_on_synthetic_set(tmp_path):
    ds = generate_dataset(tmp_path / "ds", n=40, seed=123)
    report = evaluate(ds)
    assert report["pass_fail"]["accuracy"] >= 0.9
    assert report["pass_fail"]["false_reject_rate"] <= 0.1


def test_cli_end_to_end(tmp_path):
    ds = tmp_path / "ds"
    assert main(["generate", "--n", "6", "--out", str(ds)]) == 0
    out = tmp_path / "out"
    assert main(["inspect", str(ds / "images"), "--out", str(out)]) == 0
    assert (out / "report.json").exists() and (out / "report.csv").exists()
    assert len(list((out / "annotated").glob("*.png"))) == 6
    assert main(["inspect", str(ds / "images"), "--out", str(out), "--fail-exit"]) in (0, 1)
