"""From model scores to a business decision.

An inspection threshold is a business choice, not a modelling one: passing a
defective part (an *escape*) usually costs far more than scrapping a good
one (a *false reject*). This module

1. turns the inspector + verifier into a per-image risk score,
2. traces ROC and precision-recall curves (implemented from scratch),
3. picks the threshold that minimises **expected cost per part** on a
   validation set, re-weighted to the defect rate expected *in production*
   (a labelled dataset is usually enriched with defects, so its own
   defect mix would give the wrong answer), and
4. reports the cost on held-out test images, with a paired bootstrap CI
   for the saving against a baseline.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..inspector import Inspector
from ..preprocess import load_image


def image_scores(dataset_dir: str | Path, names, inspector: Inspector) -> tuple[np.ndarray, np.ndarray]:
    """Per-image risk score = max over rule candidates of P(defect) = 1 - P(good).

    The verifier is run with ml_reject_prob > 1 so nothing is dropped; the
    threshold is applied afterwards. Images with no candidate score 0.
    Returns (y_true_defective, score).
    """
    root = Path(dataset_dir)
    labels = json.loads((root / "labels.json").read_text())
    keep_all = Inspector(replace(inspector.cfg, ml_reject_prob=1.01), inspector.classifier)
    y, s = [], []
    for n in names:
        res = keep_all.inspect(load_image(root / "images" / n), source=n)
        p_defect = [1.0 - d.details["ml_p_good"] for d in res.defects if "ml_p_good" in d.details]
        y.append(bool(labels[n]))
        s.append(max(p_defect, default=0.0))
    return np.array(y), np.array(s)


def roc_curve(y, score):
    """(false-positive rate, true-positive rate, thresholds), thresholds descending."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    thr = np.r_[np.inf, np.unique(score)[::-1]]
    P, N = y.sum(), (~y).sum()
    tpr = np.array([(score[y] >= t).sum() / max(P, 1) for t in thr])
    fpr = np.array([(score[~y] >= t).sum() / max(N, 1) for t in thr])
    return fpr, tpr, thr


def pr_curve(y, score):
    """(recall, precision, thresholds)."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    thr = np.unique(score)[::-1]
    prec, rec = [], []
    for t in thr:
        flag = score >= t
        tp = (flag & y).sum()
        prec.append(tp / max(flag.sum(), 1))
        rec.append(tp / max(y.sum(), 1))
    return np.array(rec), np.array(prec), thr


def average_precision(y, score) -> float:
    """AP = sum_n (R_n - R_{n-1}) P_n over descending thresholds (no interpolation)."""
    rec, prec, _ = pr_curve(y, score)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def auc(x, y) -> float:
    """Trapezoidal area under a curve given by points sorted on x."""
    o = np.argsort(x, kind="stable")
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def expected_cost(y, flagged, cost_escape: float, cost_false_reject: float,
                  prevalence: float | None = None) -> float:
    """Expected cost per inspected part.

    With `prevalence` (production defect rate pi):
        cost = c_esc * pi * FNR + c_fr * (1 - pi) * FPR
    Without it, the dataset's own defect mix is used.
    """
    y, flagged = np.asarray(y, bool), np.asarray(flagged, bool)
    if prevalence is None:
        return float((cost_escape * (y & ~flagged).sum()
                      + cost_false_reject * (~y & flagged).sum()) / len(y))
    fnr = (y & ~flagged).sum() / max(y.sum(), 1)
    fpr = (~y & flagged).sum() / max((~y).sum(), 1)
    return float(cost_escape * prevalence * fnr + cost_false_reject * (1 - prevalence) * fpr)


def cost_curve(y, score, cost_escape, cost_false_reject, prevalence=None, thresholds=None):
    thresholds = np.linspace(0, 1, 201) if thresholds is None else np.asarray(thresholds)
    costs = np.array([expected_cost(y, score >= t, cost_escape, cost_false_reject, prevalence)
                      for t in thresholds])
    return thresholds, costs


def best_threshold(y, score, cost_escape, cost_false_reject, prevalence=None) -> float:
    """Threshold minimising validation cost (midpoint of the optimal plateau)."""
    t, c = cost_curve(y, score, cost_escape, cost_false_reject, prevalence)
    best = np.flatnonzero(np.isclose(c, c.min()))
    return float(t[best[len(best) // 2]])


def paired_bootstrap_saving(y, flag_base, flag_new, cost_escape, cost_false_reject,
                            prevalence=None, n_boot: int = 2000, seed: int = 0) -> dict:
    """Saving per part (baseline cost - new cost) with a paired percentile CI.

    Defective and good images are resampled separately (stratified), so each
    bootstrap sample keeps the class mix needed for the prevalence weighting.
    """
    y, a, b = (np.asarray(v, bool) for v in (y, flag_base, flag_new))
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(seed)

    def saving(idx):
        return (expected_cost(y[idx], a[idx], cost_escape, cost_false_reject, prevalence)
                - expected_cost(y[idx], b[idx], cost_escape, cost_false_reject, prevalence))

    point = saving(np.arange(len(y)))
    boots = [saving(np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"saving_per_part": float(point), "ci95": [float(lo), float(hi)],
            "p_saving_le_0": float(np.mean(np.array(boots) <= 0))}
