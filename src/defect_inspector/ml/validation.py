"""Model comparison that holds up statistically.

A single train/test split gives one number per model. Here we use
**repeated grouped k-fold cross-validation** (folds are made of whole images,
so correlated patches never straddle train and test), then compare models
with the **corrected resampled t-test** of Nadeau & Bengio (2003). The
correction accounts for the overlap between training sets across folds,
which makes the naive paired t-test badly over-confident.

Also: a **learning curve over the number of labelled images**, the
practical question "how much data do we need to label?".
"""
from __future__ import annotations

from math import sqrt

import numpy as np

from .classifier import CLASSES
from .dataset import PatchSet
from .metrics import accuracy, macro_f1

K = len(CLASSES)


def grouped_kfold(groups: np.ndarray, k: int, seed: int = 0):
    """Yield (train_idx, test_idx), each fold holding out ~1/k of the groups."""
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    for fold in np.array_split(names, k):
        test = np.isin(groups, fold)
        yield np.where(~test)[0], np.where(test)[0]


def _inner_val_split(groups: np.ndarray, frac: float, seed: int):
    """Hold out a fraction of the *training* images for early stopping."""
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    val = np.isin(groups, names[:max(1, int(round(frac * len(names))))])
    return np.where(~val)[0], np.where(val)[0]


def repeated_cv(ps: PatchSet, factories: dict, k: int = 5, repeats: int = 3, seed: int = 0,
                verbose: bool = False) -> dict:
    """factories: name -> (input_kind, callable(n_inputs, seed) -> unfitted model).

    Returns {"folds": [...], "scores": {name: {"macro_f1": [...], "accuracy": [...]}},
             "n_train": [...], "n_test": [...]}; fold scores are aligned across
    models, so they can be compared pairwise.
    """
    out = {"scores": {n: {"macro_f1": [], "accuracy": []} for n in factories},
           "n_train": [], "n_test": [], "folds": []}
    for r in range(repeats):
        for f, (tr_idx, te_idx) in enumerate(grouped_kfold(ps.image, k, seed=seed + r)):
            fit_idx, val_idx = _inner_val_split(ps.image[tr_idx], 0.2, seed=seed + 100 * r + f)
            fit_idx, val_idx = tr_idx[fit_idx], tr_idx[val_idx]
            out["n_train"].append(len(tr_idx))
            out["n_test"].append(len(te_idx))
            out["folds"].append((r, f))
            for name, (kind, make) in factories.items():
                X = getattr(ps, kind)
                model = make(X.shape[1], seed + r)
                model.fit(X[fit_idx], ps.y[fit_idx], X[val_idx], ps.y[val_idx])
                yhat = model.predict(X[te_idx])
                out["scores"][name]["macro_f1"].append(macro_f1(ps.y[te_idx], yhat, K))
                out["scores"][name]["accuracy"].append(accuracy(ps.y[te_idx], yhat))
            if verbose:
                print(f"  repeat {r + 1}/{repeats} fold {f + 1}/{k} done")
    return out


def _t_sf(t: float, df: int) -> float:
    """Survival function of Student's t (scipy if available, else normal approx.)."""
    try:
        from scipy import stats
        return float(stats.t.sf(t, df))
    except ImportError:  # pragma: no cover
        from math import erfc
        return 0.5 * erfc(t / sqrt(2))


def corrected_resampled_ttest(a, b, n_train: float, n_test: float) -> dict:
    """Nadeau & Bengio corrected resampled t-test on paired fold scores.

    t = mean(d) / sqrt( (1/J + n_test/n_train) * var(d) ),  df = J - 1
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return {"mean_diff": float(d.mean()), "t": float("inf") if d.mean() else 0.0,
                "df": J - 1, "p_value": 0.0 if d.mean() else 1.0}
    t = d.mean() / sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * _t_sf(abs(t), J - 1)
    naive_t = d.mean() / sqrt(var / J)
    return {"mean_diff": float(d.mean()), "t": float(t), "df": J - 1, "p_value": float(p),
            "naive_p_value": float(2 * _t_sf(abs(naive_t), J - 1))}


def summarise_cv(cv: dict, metric: str = "macro_f1") -> dict:
    return {n: {"mean": float(np.mean(s[metric])), "sd": float(np.std(s[metric], ddof=1)),
                "min": float(np.min(s[metric])), "max": float(np.max(s[metric]))}
            for n, s in cv["scores"].items()}


def learning_curve(train: PatchSet, test: PatchSet, factories: dict, n_images: list[int],
                   repeats: int = 3, seed: int = 0) -> dict:
    """Macro-F1 on a fixed test set when training on n randomly chosen images."""
    names = np.unique(train.image)
    res = {name: {"n_images": [], "mean": [], "sd": []} for name in factories}
    for n in n_images:
        scores = {name: [] for name in factories}
        for r in range(repeats):
            rng = np.random.default_rng(seed + 1000 * r + n)
            pick = rng.choice(names, size=min(n, len(names)), replace=False)
            sub = train.subset(np.isin(train.image, pick))
            fit_i, val_i = _inner_val_split(sub.image, 0.2, seed=seed + r)
            for name, (kind, make) in factories.items():
                X = getattr(sub, kind)
                m = make(X.shape[1], seed + r)
                m.fit(X[fit_i], sub.y[fit_i], X[val_i], sub.y[val_i])
                scores[name].append(macro_f1(test.y, m.predict(getattr(test, kind)), K))
        for name in factories:
            res[name]["n_images"].append(int(n))
            res[name]["mean"].append(float(np.mean(scores[name])))
            res[name]["sd"].append(float(np.std(scores[name], ddof=1)) if repeats > 1 else 0.0)
    return res
