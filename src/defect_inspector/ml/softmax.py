"""Softmax (multinomial logistic) regression, from scratch.

Model:   P(y = k | x) = exp(w_k . x + b_k) / sum_j exp(w_j . x + b_j)
Loss:    mean cross-entropy (negative log-likelihood) + (lambda/2) ||W||^2
Fit:     mini-batch Adam with early stopping on a validation set.

Because the model is linear in the (standardised) features, each weight
w_{k,f} is interpretable: +1 std of feature f multiplies the odds of class k
vs. the reference by exp(w_{k,f} - w_{ref,f}).
"""
from __future__ import annotations

import numpy as np

from .optim import (Adam, balanced_class_weight, cross_entropy, iterate_minibatches,
                    softmax, softmax_ce_grad)


class SoftmaxRegression:
    def __init__(self, n_classes: int, l2: float = 1e-3, lr: float = 0.05, epochs: int = 300,
                 batch_size: int = 64, patience: int = 30, balanced: bool = True, seed: int = 0):
        self.K, self.l2, self.lr, self.epochs = n_classes, l2, lr, epochs
        self.batch_size, self.patience, self.balanced, self.seed = batch_size, patience, balanced, seed
        self.params: dict[str, np.ndarray] = {}
        self.mu = self.sigma = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ---------------------------------------------------------------- core math
    def _standardise(self, X):
        return (X - self.mu) / self.sigma

    def logits(self, Xs: np.ndarray) -> np.ndarray:
        return Xs @ self.params["W"] + self.params["b"]

    def loss_and_grads(self, Xs: np.ndarray, y: np.ndarray, class_weight=None):
        """Loss and analytic gradients on already-standardised inputs."""
        W = self.params["W"]
        P = softmax(self.logits(Xs))
        loss = cross_entropy(P, y, class_weight) + 0.5 * self.l2 * float((W * W).sum())
        G = softmax_ce_grad(P, y, class_weight)          # dL/dlogits
        return loss, {"W": Xs.T @ G + self.l2 * W, "b": G.sum(axis=0)}

    # ---------------------------------------------------------------- training
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, np.float64)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardise(X)
        d = X.shape[1]
        self.params = {"W": rng.normal(0, 0.01, (d, self.K)), "b": np.zeros(self.K)}
        cw = balanced_class_weight(y, self.K) if self.balanced else None
        opt = Adam(self.params, lr=self.lr)

        best, best_params, bad = np.inf, None, 0
        for ep in range(self.epochs):
            for idx in iterate_minibatches(len(y), self.batch_size, rng):
                _, g = self.loss_and_grads(Xs[idx], y[idx], cw)
                opt.step(g)
            self.history["train_loss"].append(self.loss_and_grads(Xs, y, cw)[0])
            if X_val is not None:
                Pv = self.predict_proba(X_val)
                vl = cross_entropy(Pv, y_val)
                self.history["val_loss"].append(vl)
                self.history["val_acc"].append(float((Pv.argmax(1) == y_val).mean()))
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_params = {k: v.copy() for k, v in self.params.items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if verbose and ep % 25 == 0:
                print(f"  softmax ep {ep:3d} train {self.history['train_loss'][-1]:.4f}")
        if best_params is not None:
            self.params = best_params
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba(self, X) -> np.ndarray:
        return softmax(self.logits(self._standardise(np.asarray(X, np.float64))))

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ---------------------------------------------------------------- persistence
    def state(self) -> dict[str, np.ndarray]:
        return {"W": self.params["W"], "b": self.params["b"], "mu": self.mu, "sigma": self.sigma}

    @classmethod
    def from_state(cls, st: dict) -> "SoftmaxRegression":
        m = cls(n_classes=st["W"].shape[1])
        m.params = {"W": st["W"], "b": st["b"]}
        m.mu, m.sigma = st["mu"], st["sigma"]
        return m
