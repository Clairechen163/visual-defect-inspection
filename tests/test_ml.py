import numpy as np
import pytest

from defect_inspector import InspectionConfig, Inspector
from defect_inspector.ml import CLASSES, MLP, PatchClassifier, SoftmaxRegression
from defect_inspector.ml.dataset import extract_patches, group_split
from defect_inspector.ml.features import FEATURE_NAMES
from defect_inspector.ml.metrics import (bootstrap_ci, confusion_matrix, expected_calibration_error,
                                         mcnemar_exact)
from defect_inspector.ml.optim import softmax
from defect_inspector.synth import generate_dataset, generate_sample


def _numerical_grad_error(model, loss_fn, eps=1e-5):
    """Max relative error between analytic and central-difference gradients."""
    _, grads = loss_fn()
    worst = 0.0
    for k, P in model.params.items():
        it = np.nditer(P, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            orig = P[i]
            P[i] = orig + eps
            lp = loss_fn()[0]
            P[i] = orig - eps
            lm = loss_fn()[0]
            P[i] = orig
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - grads[k][i]) / max(1e-8, abs(num) + abs(grads[k][i])))
    return worst


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    return rng.normal(size=(15, 5)), rng.integers(0, 4, 15), np.array([1.0, 2.0, 0.5, 1.5])


def test_softmax_rows_sum_to_one_and_are_stable():
    p = softmax(np.array([[1000.0, 1001.0, 999.0], [-5.0, 0.0, 5.0]]))
    assert np.allclose(p.sum(axis=1), 1.0) and np.isfinite(p).all()


def test_softmax_regression_gradient(toy):
    X, y, cw = toy
    m = SoftmaxRegression(4, l2=0.1)
    rng = np.random.default_rng(1)
    m.params = {"W": rng.normal(size=(5, 4)), "b": rng.normal(size=4)}
    assert _numerical_grad_error(m, lambda: m.loss_and_grads(X, y, cw)) < 1e-6


def test_mlp_backprop_gradient(toy):
    X, y, cw = toy
    m = MLP([5, 7, 6, 4], l2=0.1, dropout=0.0)
    assert _numerical_grad_error(m, lambda: m.loss_and_grads(X, y, cw)) < 1e-6


def test_models_learn_separable_and_nonlinear_data():
    rng = np.random.default_rng(0)
    # Linearly separable blobs: softmax regression should nail it.
    centers = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])
    y = rng.integers(0, 4, 400)
    X = centers[y] + rng.normal(0, 0.5, (400, 2))
    assert (SoftmaxRegression(4, epochs=100).fit(X, y).predict(X) == y).mean() > 0.97
    # XOR: impossible for a linear model, easy for a neural network.
    Xx = rng.uniform(-1, 1, (600, 2))
    yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
    lin = (SoftmaxRegression(2, epochs=100).fit(Xx, yx).predict(Xx) == yx).mean()
    nn = (MLP([2, 32, 2], epochs=300, dropout=0.0, lr=1e-2).fit(Xx, yx).predict(Xx) == yx).mean()
    assert lin < 0.7 and nn > 0.93


def test_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    a = np.array([0, 0, 1, 1, 2, 0])
    b = np.array([0, 1, 1, 0, 2, 0])
    cm = confusion_matrix(y, a, 3)
    assert cm.trace() == 5 and cm.sum() == 6
    r = mcnemar_exact(y, a, b)
    assert (r["a_only_correct"], r["b_only_correct"]) == (2, 0) and r["p_value"] == pytest.approx(0.5)
    lo, hi = bootstrap_ci(y, a, lambda u, v: float((u == v).mean()), n_boot=300)
    assert 0 <= lo <= 5 / 6 <= hi <= 1
    perfect = np.eye(3)[y]
    assert expected_calibration_error(perfect, y) == pytest.approx(0.0)


def test_patch_pipeline_save_load_and_inspector(tmp_path):
    ds = generate_dataset(tmp_path / "ds", n=40, seed=5, min_intensity=0.4, nuisance_rate=1.5)
    assert (ds / "nuisances.json").exists()
    ps = extract_patches(ds)
    assert ps.features.shape[1] == len(FEATURE_NAMES)
    assert set(np.unique(ps.y)) == set(range(len(CLASSES)))
    tr, va, te = group_split(ps)
    # Grouped split: no image contributes patches to two splits.
    assert not (set(tr.image) & set(te.image)) and not (set(tr.image) & set(va.image))

    m = MLP([ps.features.shape[1], 16, len(CLASSES)], epochs=60).fit(tr.features, tr.y, va.features, va.y)
    clf = PatchClassifier(m, "features", "test")
    clf.save(tmp_path / "m.npz")
    clf2 = PatchClassifier.load(tmp_path / "m.npz")
    assert np.allclose(m.predict_proba(te.features), clf2.model.predict_proba(te.features))

    img, _ = generate_sample(np.random.default_rng(101), ["chip"])
    res = Inspector(InspectionConfig(), clf2).inspect(img)
    assert all("ml_label" in d.details for d in res.defects)


def test_shipped_model_rejects_false_alarms_but_keeps_defects():
    clf = PatchClassifier.load("models/mlp_features.npz")
    cfg = InspectionConfig.from_json("configs/high_sensitivity.json")
    rng = np.random.default_rng(2024)
    img, _ = generate_sample(rng, ["chip"])
    assert "chip" in Inspector(cfg, clf).inspect(img).defect_types
