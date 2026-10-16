"""Machine-learning layer for defect classification, implemented from scratch in NumPy.

* ``softmax.SoftmaxRegression`` - multinomial logistic regression
* ``mlp.MLP``                   - fully-connected neural network (backprop by hand)
* ``features``                  - interpretable statistical features of an image patch
* ``dataset``                   - labelled patch extraction, split by image (no leakage)
* ``metrics``                   - confusion matrix, bootstrap CIs, McNemar test, calibration
* ``classifier.PatchClassifier``- trained model + preprocessing, pluggable into ``Inspector``
"""
from .classifier import CLASSES, PatchClassifier
from .mlp import MLP
from .softmax import SoftmaxRegression

__all__ = ["CLASSES", "PatchClassifier", "MLP", "SoftmaxRegression"]
