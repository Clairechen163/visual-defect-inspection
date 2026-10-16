import numpy as np
import pytest

from defect_inspector import InspectionConfig, Inspector
from defect_inspector.ml import PatchClassifier
from defect_inspector.ml.dataset import PatchSet
from defect_inspector.ml.decision import (auc, average_precision, best_threshold, expected_cost,
                                          paired_bootstrap_saving, roc_curve)
from defect_inspector.ml.softmax import SoftmaxRegression
from defect_inspector.ml.validation import (corrected_resampled_ttest, grouped_kfold, learning_curve,
                                            repeated_cv, summarise_cv)
from defect_inspector.spc import p_chart, simulate_production


# ----------------------------------------------------------------- validation
def test_grouped_kfold_partitions_groups():
    groups = np.repeat(np.arange(20), 5)
    seen = []
    for tr, te in grouped_kfold(groups, k=5, seed=1):
        assert not set(groups[tr]) & set(groups[te])       # no group on both sides
        seen += list(np.unique(groups[te]))
    assert sorted(seen) == list(range(20))                 # every group tested once


def test_corrected_ttest_is_more_conservative_than_naive():
    rng = np.random.default_rng(0)
    a = 0.80 + rng.normal(0, 0.01, 10)
    b = a - 0.01 + rng.normal(0, 0.01, 10)
    r = corrected_resampled_ttest(a, b, n_train=800, n_test=200)
    assert r["p_value"] > r["naive_p_value"]
    assert r["df"] == 9


def _toy_patchset(n_img=30, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n_img * 6)
    X = np.eye(4)[y] * 3 + rng.normal(0, 1, (len(y), 4))
    return PatchSet(X, X, y, np.repeat([f"img{i}" for i in range(n_img)], 6))


def test_repeated_cv_and_learning_curve_run():
    ps = _toy_patchset()
    fac = {"softmax": ("features", lambda d, s: SoftmaxRegression(4, epochs=40, seed=s))}
    cv = repeated_cv(ps, fac, k=3, repeats=2)
    assert len(cv["scores"]["softmax"]["macro_f1"]) == 6
    assert summarise_cv(cv)["softmax"]["mean"] > 0.7
    lc = learning_curve(ps.subset(np.arange(120)), ps.subset(np.arange(120, 180)), fac,
                        n_images=[5, 15], repeats=2)
    assert lc["softmax"]["n_images"] == [5, 15]


# ----------------------------------------------------------------- decision
def test_roc_auc_and_average_precision():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(*roc_curve(y, perfect)[:2]) == pytest.approx(1.0)
    assert average_precision(y, perfect) == pytest.approx(1.0)
    worst = perfect[::-1]
    assert auc(*roc_curve(y, worst)[:2]) == pytest.approx(0.0)


def test_expected_cost_with_prevalence():
    y = np.array([1, 1, 0, 0], bool)
    flagged = np.array([1, 0, 1, 0], bool)       # FNR = 0.5, FPR = 0.5
    assert expected_cost(y, flagged, 100, 5) == pytest.approx((100 + 5) / 4)
    assert expected_cost(y, flagged, 100, 5, prevalence=0.1) == pytest.approx(100 * .1 * .5 + 5 * .9 * .5)


def test_best_threshold_and_bootstrap_saving():
    y = np.array([1, 1, 1, 0, 0, 0], bool)
    score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    t = best_threshold(y, score, 10, 1, prevalence=0.2)
    assert 0.3 < t <= 0.7
    never = np.zeros(6, bool)
    sav = paired_bootstrap_saving(y, never, score >= t, 10, 1, prevalence=0.2, n_boot=200)
    assert sav["saving_per_part"] == pytest.approx(10 * 0.2)   # all escapes avoided
    assert sav["ci95"][0] > 0


# ----------------------------------------------------------------- SPC
def test_p_chart_limits_and_signals():
    rejected = np.r_[np.full(15, 2), np.full(10, 9)]          # n = 25
    c = p_chart(rejected, n=25, phase1=15)
    assert c["center"] == pytest.approx(0.08)
    assert c["ucl"] == pytest.approx(0.08 + 3 * np.sqrt(0.08 * 0.92 / 25))
    assert c["first_signal"] == 15


def test_simulated_production_detects_shift():
    ins = Inspector(InspectionConfig.from_json("configs/high_sensitivity.json"),
                    PatchClassifier.load("models/mlp_features.npz"))
    run = simulate_production(ins, n_batches=6, batch_size=8, shift_at=3,
                              p_defect=0.0, p_defect_after_shift=1.0, seed=1)
    assert run.true_defective.tolist() == [0, 0, 0, 8, 8, 8]
    assert run.rejected[3:].sum() >= 20 and run.rejected[:3].sum() <= 2
