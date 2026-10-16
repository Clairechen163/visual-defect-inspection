"""Evaluation statistics for classifiers.

Beyond accuracy, these answer the questions a statistician would ask:
* How uncertain is the score?          -> bootstrap confidence intervals
* Is model A really better than B?     -> McNemar's exact test (paired)
* Can the probabilities be trusted?    -> expected calibration error
"""
from __future__ import annotations

from math import comb

import numpy as np


def confusion_matrix(y, yhat, k: int) -> np.ndarray:
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, yhat), 1)
    return cm  # rows = true, cols = predicted


def per_class(cm: np.ndarray, names) -> dict:
    out = {}
    for i, n in enumerate(names):
        tp = cm[i, i]
        p = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        r = tp / cm[i].sum() if cm[i].sum() else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[n] = {"precision": round(float(p), 3), "recall": round(float(r), 3),
                  "f1": round(float(f), 3), "support": int(cm[i].sum())}
    return out


def accuracy(y, yhat) -> float:
    return float((np.asarray(y) == np.asarray(yhat)).mean())


def macro_f1(y, yhat, k: int) -> float:
    cm = confusion_matrix(y, yhat, k)
    f1s = []
    for i in range(k):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
    return float(np.mean(f1s))


def bootstrap_ci(y, yhat, metric, n_boot: int = 2000, alpha: float = 0.05,
                 groups=None, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI. If `groups` (e.g. image ids) are given, whole
    groups are resampled (cluster bootstrap), respecting within-image correlation."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx_sets = [np.arange(len(y))[[i]] for i in range(len(y))]
    else:
        groups = np.asarray(groups)
        idx_sets = [np.where(groups == g)[0] for g in np.unique(groups)]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(len(idx_sets), size=len(idx_sets))
        idx = np.concatenate([idx_sets[i] for i in pick])
        stats.append(metric(y[idx], yhat[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_exact(y, pred_a, pred_b) -> dict:
    """Paired test of H0: models A and B have the same error rate.

    Only discordant pairs matter: b = A right & B wrong, c = A wrong & B right.
    Under H0, b ~ Binomial(b + c, 0.5); two-sided exact p-value.
    """
    y, a, b_ = map(np.asarray, (y, pred_a, pred_b))
    ra, rb = a == y, b_ == y
    b = int((ra & ~rb).sum())
    c = int((~ra & rb).sum())
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def expected_calibration_error(probs: np.ndarray, y, n_bins: int = 10) -> float:
    """Weighted mean |confidence - accuracy| over confidence bins."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(ece)
