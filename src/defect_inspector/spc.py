"""Statistical process control on the inspector's output.

An inspection system tells you about *each part*; quality engineering also
needs to know when the *process* changes (tool wear, a new material lot, a
drifting oven). The classic tool is the **p-chart**: the fraction of
rejected parts per production batch, plotted against 3-sigma binomial
control limits estimated from an in-control baseline (Phase I).

    p_bar = total rejects / total inspected            (Phase I batches)
    UCL/LCL = p_bar +/- 3 sqrt(p_bar (1 - p_bar) / n)

Signals: a point beyond the limits (Western Electric rule 1), or 8
consecutive points above the centre line (rule 4, a sustained shift).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .detectors import DEFECT_TYPES
from .inspector import Inspector
from .synth import generate_sample


@dataclass
class ProductionRun:
    batch_size: int
    true_defective: np.ndarray          # per batch: truly defective parts
    rejected: np.ndarray                # per batch: parts the inspector rejected
    defect_counts: Counter = field(default_factory=Counter)  # detected defects by type
    shift_at: int | None = None


def simulate_production(inspector: Inspector, n_batches: int = 30, batch_size: int = 25,
                        p_defect: float = 0.08, p_defect_after_shift: float = 0.22,
                        shift_at: int = 18, nuisance_rate: float = 1.0,
                        seed: int = 0) -> ProductionRun:
    """Generate and inspect parts batch by batch. From batch `shift_at` onward
    the process degrades (e.g. tool wear) and scratches become more common."""
    rng = np.random.default_rng(seed)
    true_def, rej, counts = [], [], Counter()
    for b in range(n_batches):
        shifted = b >= shift_at
        p = p_defect_after_shift if shifted else p_defect
        weights = np.array([0.6, 0.2, 0.2]) if shifted else np.array([1, 1, 1]) / 3
        n_def = n_rej = 0
        for _ in range(batch_size):
            defects = []
            if rng.random() < p:
                defects = [str(rng.choice(DEFECT_TYPES, p=weights))]
                n_def += 1
            img, _ = generate_sample(rng, defects, intensity=float(rng.uniform(0.4, 1.0)),
                                     n_nuisance=int(rng.poisson(nuisance_rate)))
            res = inspector.inspect(img)
            if res.verdict == "FAIL":
                n_rej += 1
                counts.update(d.type for d in res.defects)
        true_def.append(n_def)
        rej.append(n_rej)
    return ProductionRun(batch_size, np.array(true_def), np.array(rej), counts, shift_at)


def p_chart(rejected, n: int, phase1: int) -> dict:
    """Control limits from the first `phase1` batches, then signal detection."""
    rejected = np.asarray(rejected)
    p = rejected / n
    p_bar = rejected[:phase1].sum() / (phase1 * n)
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    ucl, lcl = p_bar + 3 * sigma, max(0.0, p_bar - 3 * sigma)
    beyond = [int(i) for i in np.flatnonzero(p > ucl)]
    run8, streak = [], 0
    for i, v in enumerate(p):
        streak = streak + 1 if v > p_bar else 0
        if streak >= 8:
            run8.append(i)
    signals = sorted(set(beyond) | set(run8))
    return {"p": p, "center": float(p_bar), "ucl": float(ucl), "lcl": float(lcl),
            "beyond_limits": beyond, "run_of_8": run8,
            "first_signal": signals[0] if signals else None}
