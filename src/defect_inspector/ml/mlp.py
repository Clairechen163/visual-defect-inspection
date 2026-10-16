"""Multi-layer perceptron (fully-connected neural network), from scratch.

Architecture:  x -> [Linear -> ReLU -> Dropout] x L -> Linear -> softmax
Init:          He-normal (suited to ReLU)
Loss:          cross-entropy + (lambda/2) sum ||W_l||^2
Training:      mini-batch Adam, inverted dropout, early stopping on val loss.

Backpropagation is written out explicitly:
    dZ_L   = (P - Y) / N                         (softmax + cross-entropy)
    dW_l   = A_{l-1}^T dZ_l + lambda W_l
    db_l   = sum_rows dZ_l
    dA_{l-1} = dZ_l W_l^T
    dZ_{l-1} = dA_{l-1} * 1[Z_{l-1} > 0] * dropout_mask / keep_prob
"""
from __future__ import annotations

import numpy as np

from .optim import (Adam, balanced_class_weight, cross_entropy, iterate_minibatches,
                    softmax, softmax_ce_grad)


class MLP:
    def __init__(self, layer_sizes: list[int], l2: float = 1e-4, lr: float = 1e-3,
                 epochs: int = 200, batch_size: int = 64, dropout: float = 0.2,
                 patience: int = 20, balanced: bool = True, seed: int = 0):
        assert len(layer_sizes) >= 2
        self.sizes, self.l2, self.lr, self.epochs = list(layer_sizes), l2, lr, epochs
        self.batch_size, self.dropout, self.patience = batch_size, dropout, patience
        self.balanced, self.seed = balanced, seed
        self.K = layer_sizes[-1]
        self.params: dict[str, np.ndarray] = {}
        self.mu = self.sigma = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self._init(np.random.default_rng(seed))

    @property
    def n_layers(self) -> int:
        return len(self.sizes) - 1

    def _init(self, rng):
        for i in range(self.n_layers):
            fan_in, fan_out = self.sizes[i], self.sizes[i + 1]
            self.params[f"W{i}"] = rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            self.params[f"b{i}"] = np.zeros(fan_out)

    # ---------------------------------------------------------------- forward / backward
    def forward(self, X, train: bool = False, rng=None):
        cache = {"A0": X}
        A = X
        for i in range(self.n_layers):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"Z{i}"] = Z
            if i < self.n_layers - 1:
                A = np.maximum(Z, 0.0)
                if train and self.dropout > 0:
                    keep = 1.0 - self.dropout
                    mask = (rng.random(A.shape) < keep) / keep
                    cache[f"D{i}"] = mask
                    A = A * mask
                cache[f"A{i + 1}"] = A
        return softmax(Z), cache

    def loss_and_grads(self, X, y, class_weight=None, train: bool = False, rng=None):
        P, cache = self.forward(X, train=train, rng=rng)
        reg = 0.5 * self.l2 * sum(float((self.params[f"W{i}"] ** 2).sum()) for i in range(self.n_layers))
        loss = cross_entropy(P, y, class_weight) + reg

        grads = {}
        dZ = softmax_ce_grad(P, y, class_weight)
        for i in reversed(range(self.n_layers)):
            A_prev = cache[f"A{i}"]
            grads[f"W{i}"] = A_prev.T @ dZ + self.l2 * self.params[f"W{i}"]
            grads[f"b{i}"] = dZ.sum(axis=0)
            if i > 0:
                dA = dZ @ self.params[f"W{i}"].T
                if f"D{i - 1}" in cache:
                    dA = dA * cache[f"D{i - 1}"]
                dZ = dA * (cache[f"Z{i - 1}"] > 0)
        return loss, grads

    # ---------------------------------------------------------------- training
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        rng = np.random.default_rng(self.seed + 1)
        X = np.asarray(X, np.float64)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sigma
        cw = balanced_class_weight(y, self.K) if self.balanced else None
        opt = Adam(self.params, lr=self.lr)

        best, best_params, bad = np.inf, None, 0
        for ep in range(self.epochs):
            ep_loss, n = 0.0, 0
            for idx in iterate_minibatches(len(y), self.batch_size, rng):
                loss, g = self.loss_and_grads(Xs[idx], y[idx], cw, train=True, rng=rng)
                opt.step(g)
                ep_loss += loss * len(idx)
                n += len(idx)
            self.history["train_loss"].append(ep_loss / n)
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
            if verbose and ep % 10 == 0:
                print(f"  mlp ep {ep:3d} train {self.history['train_loss'][-1]:.4f}"
                      + (f" val {self.history['val_loss'][-1]:.4f}" if X_val is not None else ""))
        if best_params is not None:
            self.params = best_params
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba(self, X) -> np.ndarray:
        Xs = (np.asarray(X, np.float64) - self.mu) / self.sigma
        return self.forward(Xs)[0]

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ---------------------------------------------------------------- persistence
    def state(self) -> dict[str, np.ndarray]:
        return {**self.params, "mu": self.mu, "sigma": self.sigma,
                "sizes": np.array(self.sizes)}

    @classmethod
    def from_state(cls, st: dict) -> "MLP":
        m = cls([int(s) for s in st["sizes"]], dropout=0.0)
        for i in range(m.n_layers):
            m.params[f"W{i}"], m.params[f"b{i}"] = st[f"W{i}"], st[f"b{i}"]
        m.mu, m.sigma = st["mu"], st["sigma"]
        return m
