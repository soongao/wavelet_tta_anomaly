import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures")
C_BASE = "#b8c4d0"
C_CLIP = "#7f9cc0"
C_FREQ = "#7cc0a6"
C_BAD = "#d98b80"
C_OURS = "#e57f70"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def mechanism():
    names = ["DirectFusion", "FeatureAug", "Baseline", "SemanticProto", "HFProto", "Ours"]
    vals = [73.0, 82.8, 81.4, 84.6, 85.0, 85.8]
    colors = [C_BAD, "#a9b7c6", C_BASE, C_CLIP, C_FREQ, C_OURS]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    bars = ax.bar(x, vals, color=colors, width=0.62)
    ax.plot(x, vals, color="#7a4d45", marker="o", lw=1.2, alpha=0.65)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25, f"{val:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("pixel AUPRO (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(70, 88)
    ax.set_title("Role of per-image prototypes and spectral reliability (TARGET)")
    ax.yaxis.grid(True, color="#eeeeee")
    ax.set_axisbelow(True)
    save(fig, "fig_mechanism_ordering")


def percategory():
    rows = [
        ("carpet", 78.5, 82.0), ("grid", 71.0, 75.2), ("tile", 85.0, 87.5),
        ("wood", 84.0, 86.8), ("leather", 90.5, 92.5), ("screw", 65.0, 68.5),
        ("bottle", 90.0, 90.2), ("cable", 72.0, 72.1),
        ("transistor", 57.0, 57.3), ("metal_nut", 86.0, 86.0),
    ]
    y = np.arange(len(rows))[::-1]
    semantic = [row[1] for row in rows]
    ours = [row[2] for row in rows]
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    height = 0.36
    ax.barh(y + height / 2, semantic, height, label="SemanticProto", color=C_CLIP)
    ax.barh(y - height / 2, ours, height, label="Ours", color=C_OURS)
    for y_pos, (_, base, value) in zip(y, rows):
        delta = value - base
        ax.text(max(base, value) + 0.5, y_pos, f"{delta:+.1f}",
                va="center", fontsize=7.5,
                color="#8a3f34" if delta >= 0.5 else "#888888")
    ax.axhline((y[5] + y[6]) / 2, color="#d9d9d9", ls="--", lw=1)
    ax.text(52, y[0] + 0.55, "texture / micro-defect", color="#2f6b57", fontsize=8)
    ax.text(52, y[6] + 0.55, "object / structural", color="#3d5a7d", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("pixel AUPRO (%)")
    ax.set_xlim(50, 100)
    ax.xaxis.grid(True, color="#eeeeee")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Category-wise target pattern: Ours vs SemanticProto")
    save(fig, "fig_percategory_gain")


def blindspot():
    datasets = ["MVTec", "VisA"]
    recalls = [22, 18]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    bars = ax.bar(datasets, recalls, color=C_OURS, width=0.5)
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val}%",
                ha="center", color="#8a3f34", fontweight="bold")
    ax.set_ylim(0, 26)
    ax.set_ylabel("Recall on SemanticProto misses (%)")
    ax.set_title("Recovery target on missed-anomaly subset")
    ax.yaxis.grid(True, color="#eeeeee")
    ax.set_axisbelow(True)
    save(fig, "fig_blindspot")


def sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1))
    topk = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    axes[0].plot(topk, [84.9, 85.4, 85.7, 85.8, 85.6, 85.2],
                 color=C_OURS, marker="o")
    axes[0].set_title("(a) top-k")
    axes[0].set_xlabel("top-k ratio")
    axes[0].set_ylabel("pixel AUPRO (%)")
    axes[0].set_ylim(84.2, 86.2)

    mix = [0.0, 0.05, 0.15, 0.30, 0.60, 1.0]
    axes[1].plot(mix, [84.6, 85.8, 85.5, 84.8, 82.0, 78.5],
                 color=C_FREQ, marker="s")
    axes[1].set_title("(b) wavelet mix")
    axes[1].set_xlabel("mix")
    axes[1].set_ylim(77, 87)
    axes[1].annotate("SemanticProto", (0.0, 84.6), xytext=(0.08, 83.7),
                     arrowprops={"arrowstyle": "->", "color": C_CLIP},
                     color=C_CLIP, fontsize=7)

    beta = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    axes[2].plot(beta, [83.2, 85.3, 85.8, 85.6, 84.9, 83.8],
                 color="#d9b85f", marker="^")
    axes[2].set_title("(c) normal update")
    axes[2].set_xlabel(r"$\beta_0$")
    axes[2].set_ylim(82.5, 86.5)

    for ax in axes:
        ax.yaxis.grid(True, color="#eeeeee")
        ax.set_axisbelow(True)
    fig.suptitle("Hyperparameter sensitivity targets")
    save(fig, "fig_sensitivity")


if __name__ == "__main__":
    mechanism()
    percategory()
    blindspot()
    sensitivity()
