"""A trained model bundled with its preprocessing, ready to plug into Inspector."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = ["good", "scratch", "discoloration", "chip"]


class PatchClassifier:
    """Classifies 64x64 patches around candidate defects.

    `kind` is "features" (18 statistics) or "pixels" (raw normalised patch);
    `model` is a SoftmaxRegression or MLP.
    """

    def __init__(self, model, kind: str, name: str = ""):
        assert kind in ("features", "pixels")
        self.model, self.kind, self.name = model, kind, name

    def _vectors(self, img, mask, centers):
        from .features import crop, patch_features, patch_pixels, reference_lab
        ref = reference_lab(img, mask)
        fn = patch_features if self.kind == "features" else patch_pixels
        return np.array([fn(crop(img, cx, cy), crop(mask, cx, cy), ref) for cx, cy in centers])

    def predict_proba_at(self, img: np.ndarray, mask: np.ndarray, centers) -> np.ndarray:
        if not len(centers):
            return np.zeros((0, len(CLASSES)))
        return self.model.predict_proba(self._vectors(img, mask, centers))

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        from .mlp import MLP
        arch = "mlp" if isinstance(self.model, MLP) else "softmax"
        meta = json.dumps({"arch": arch, "kind": self.kind, "name": self.name, "classes": CLASSES})
        np.savez_compressed(path, __meta__=np.array(meta), **self.model.state())

    @classmethod
    def load(cls, path: str | Path) -> "PatchClassifier":
        from .mlp import MLP
        from .softmax import SoftmaxRegression
        with np.load(path, allow_pickle=False) as z:
            st = {k: z[k] for k in z.files}
        meta = json.loads(str(st.pop("__meta__")))
        model = (MLP if meta["arch"] == "mlp" else SoftmaxRegression).from_state(st)
        return cls(model, meta["kind"], meta.get("name", ""))
