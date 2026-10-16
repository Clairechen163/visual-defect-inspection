"""Shared numerics: stable softmax, cross-entropy, Adam optimiser."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted by the row max for numerical stability."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray, class_weight: np.ndarray | None = None) -> float:
    """Mean negative log-likelihood of the true class (optionally class-weighted)."""
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    w = np.ones(len(y)) if class_weight is None else class_weight[y]
    return float(-(w * np.log(p)).sum() / w.sum())


def softmax_ce_grad(probs: np.ndarray, y: np.ndarray, class_weight: np.ndarray | None = None) -> np.ndarray:
    """d(cross-entropy)/d(logits) = (p - onehot(y)) * w / sum(w)."""
    g = probs.copy()
    g[np.arange(len(y)), y] -= 1.0
    w = np.ones(len(y)) if class_weight is None else class_weight[y]
    return g * (w / w.sum())[:, None]


def balanced_class_weight(y: np.ndarray, n_classes: int) -> np.ndarray:
    """w_k = N / (K * n_k): each class contributes equally to the loss."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return len(y) / (n_classes * np.maximum(counts, 1.0))


class Adam:
    """Adam (Kingma & Ba, 2015) over a dict of parameter arrays, updated in place."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def iterate_minibatches(n: int, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]
