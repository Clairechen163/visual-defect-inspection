"""Figures for the ML report (matplotlib, optional dependency)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, RED, MID = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
TITLES = {"softmax_features": "Softmax regression · features", "mlp_features": "Neural network · features",
          "softmax_pixels": "Softmax regression · pixels", "mlp_pixels": "Neural network · pixels"}
ORDER = ["softmax_features", "mlp_features", "softmax_pixels", "mlp_pixels"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    })


def learning_curves(report, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, key in zip(axes.flat, ORDER):
        h = report["models"][key]["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=BLUE, lw=2, label="train")
        ax.plot(ep, h["val_loss"], color=ORANGE, lw=2, label="validation")
        best = int(np.argmin(h["val_loss"]))
        ax.plot(best + 1, h["val_loss"][best], "o", ms=7, color=ORANGE, mec=SURFACE, mew=2)
        ax.set_title(TITLES[key], loc="left")
        ax.set_yscale("log")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlabel("epoch")
        ax.set_xlim(0, ep[-1] * 1.03)
    axes[0, 0].set_ylabel("cross-entropy (log)")
    axes[1, 0].set_ylabel("cross-entropy (log)")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle("Learning curves (dot = early-stopping point, weights restored from there)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    p = out / "learning_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def confusion_matrices(report, out: Path) -> Path:
    classes = report["classes"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, key in zip(axes, ORDER):
        cm = np.array(report["models"][key]["confusion_matrix"], float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes if key == ORDER[0] else [])
        m = report["models"][key]
        ax.set_title(f"{TITLES[key]}\nacc {m['test_accuracy']:.1%}  ·  macro-F1 {m['test_macro_f1']:.2f}",
                     loc="left", fontsize=10)
        ax.set_xlabel("predicted")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("true")
    fig.suptitle("Confusion matrices on held-out test patches (rows = true class)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out / "confusion_matrices.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def softmax_coefficients(report, out: Path) -> Path:
    W = np.array(report["softmax_coefficients"])
    lim = np.abs(W).max()
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(W, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if abs(W[i, j]) > 0.35 * lim:
                ax.text(j, i, f"{W[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(W[i, j]) > 0.7 * lim else INK)
    ax.set_xticks(range(W.shape[1]), report["classes"])
    ax.set_yticks(range(W.shape[0]), report["feature_names"])
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Softmax regression weights\n(red raises, blue lowers the class log-odds per +1 SD)",
                 loc="left", fontsize=10, pad=28)
    fig.tight_layout()
    p = out / "softmax_coefficients.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def inspector_comparison(report, out: Path) -> Path:
    comp = report["inspector_comparison"]
    names = list(comp)
    metrics = [("accuracy", "PASS/FAIL accuracy  ↑"), ("escape_rate", "Defective parts passed  ↓"),
               ("false_reject_rate", "Good parts rejected  ↓")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (m, title) in zip(axes, metrics):
        vals = np.array([comp[n][m]["value"] for n in names])
        lo = vals - np.array([comp[n][m]["ci95"][0] for n in names])
        hi = np.array([comp[n][m]["ci95"][1] for n in names]) - vals
        ax.barh(y, vals, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for yi, v, h in zip(y, vals, hi):
            ax.text(v + h + 0.02, yi, f"{v:.0%}", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_title(title, loc="left")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, names)
    fig.suptitle(f"Inspector on {report['split_images']['test']} held-out images (bars = 95% bootstrap CI)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    p = out / "inspector_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def make_all(report, out: str | Path):
    out = Path(out)
    _style()
    return [learning_curves(report, out), confusion_matrices(report, out),
            softmax_coefficients(report, out), inspector_comparison(report, out)]
