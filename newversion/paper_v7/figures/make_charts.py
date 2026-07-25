#!/usr/bin/env python3
"""
面向论文的数据图（v7 叙事版）。
所有数值来自 EXPERIMENT_TARGETS.md / TABLES.md 的 EXPECTED 目标值，非实测。
做完实验后用真实 log 替换即可。
风格：干净扁平矢量、白底、柔和配色、导出 PDF。
"""
import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplcache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
OUT = os.path.dirname(__file__)

C_BASE   = "#b8c4d0"
C_CLIP   = "#7f9cc0"
C_FREQ   = "#7cc0a6"
C_GLOBAL = "#d9c07a"
C_BAD    = "#d98b80"
C_OURS   = "#e57f70"


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---- Fig 1: main results across datasets (grouped bar, pixel AUPRO) ----
def fig_main_results():
    datasets = ["MVTec", "VisA", "MPDD", "BTAD", "DTD-Synth"]
    base = [81.4, 86.7, 84.6, 70.5, 89.1]
    ours = [86.0, 91.3, 89.9, 78.2, 91.8]
    x = np.arange(len(datasets)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    b1 = ax.bar(x - w/2, base, w, label="AnomalyCLIP", color=C_BASE)
    b2 = ax.bar(x + w/2, ours, w, label="Ours", color=C_OURS)
    for r in list(b1) + list(b2):
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.4, f"{r.get_height():.1f}",
                ha="center", va="bottom", fontsize=8, color="#444")
    for xi, (bv, ov) in enumerate(zip(base, ours)):
        ax.text(xi, max(bv, ov)+3.2, f"+{ov-bv:.1f}", ha="center", fontsize=8.5,
                color="#8a3f34", fontweight="bold")
    ax.set_ylabel("pixel AUPRO (%)")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylim(65, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    ax.set_title("Localization gain across five datasets (EXPECTED)", fontsize=10.5, color="#333")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8); ax.set_axisbelow(True)
    savefig(fig, "fig_main_results")


# ---- Fig 2: mechanism ordering — where the tightening signal comes from ----
def fig_mechanism_ordering():
    names  = ["DirectHF\n(raw HF)", "Generic\nprompt", "GlobalRef\n(fixed)", "SelfRef\n(CLIP-only)", "Ours\n(per-image HF)"]
    vals   = [72.9, 81.4, 83.2, 84.6, 86.0]
    colors = [C_BAD, C_BASE, C_GLOBAL, C_CLIP, C_OURS]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    bars = ax.bar(x, vals, 0.62, color=colors)
    for r in bars:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.25, f"{r.get_height():.1f}",
                ha="center", va="bottom", fontsize=9.5, color="#444", fontweight="bold")
    ax.plot(x, vals, color="#8a3f34", lw=1.4, marker="o", ms=4, alpha=0.55, zorder=3)
    ax.annotate("", xy=(4, 86.0), xytext=(2, 83.2),
                arrowprops=dict(arrowstyle="->", color="#8a6d2e", lw=1.2))
    ax.text(3.0, 86.9, "per-image vs fixed  +2.8", fontsize=8.5, color="#8a6d2e", ha="center")
    ax.text(0, 71.2, "collapse", fontsize=8.5, color="#9c4b4b", ha="center", style="italic")
    ax.set_ylabel("pixel AUPRO (%)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(68, 90)
    ax.set_title("Source of the per-image tightening signal (MVTec, EXPECTED)", fontsize=10.5, color="#333")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8); ax.set_axisbelow(True)
    savefig(fig, "fig_mechanism_ordering")


# ---- Fig 3: per-category gain (Ours vs SelfRef) ----
def fig_percategory_gain():
    tex = [("carpet",78.5,82.0),("grid",71.0,75.2),("tile",85.0,87.5),
           ("wood",84.0,86.8),("leather",90.5,92.5),("screw",65.0,68.5)]
    obj = [("bottle",90.0,90.2),("cable",72.0,72.1),("transistor",57.0,57.3),("metal_nut",86.0,86.0)]
    rows = tex + obj
    labels=[r[0] for r in rows]; self_v=[r[1] for r in rows]; ours_v=[r[2] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.6)); h=0.36
    ax.barh(y+h/2, self_v, h, label="SelfRef (CLIP-only)", color=C_CLIP)
    ax.barh(y-h/2, ours_v, h, label="Ours", color=C_OURS)
    for yi,(l,s,o) in zip(y, rows):
        d=o-s; col="#8a3f34" if d>=0.5 else "#999"
        ax.text(max(o,s)+0.6, yi, f"+{d:.1f}" if d>=0 else f"{d:.1f}", va="center",
                fontsize=8.5, color=col, fontweight="bold" if d>=0.5 else "normal")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("pixel AUPRO (%)"); ax.set_xlim(50, 100)
    div=(y[5]+y[6])/2; ax.axhline(div, color="#dddddd", lw=1, ls="--")
    ax.text(52, y[0]+0.6, "texture / micro-defect", fontsize=8.5, color="#2f6b57", style="italic")
    ax.text(52, y[6]+0.6, "object / structural", fontsize=8.5, color="#3d5a7d", style="italic")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Gain concentrates on texture classes (MVTec, EXPECTED)", fontsize=10.5, color="#333")
    ax.xaxis.grid(True, color="#eeeeee", linewidth=0.8); ax.set_axisbelow(True)
    savefig(fig, "fig_percategory_gain")


# ---- Fig 4: sensitivity (3 subplots); wavelet-mix subplot doubles as mechanism curve ----
def fig_sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))

    topk = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    aupro_k = [84.9, 85.4, 85.7, 85.8, 85.6, 85.2]
    ax=axes[0]
    ax.plot(topk, aupro_k, color=C_OURS, lw=2, marker="o", ms=5)
    ax.axhline(84.6, color=C_CLIP, lw=1.2, ls="--")
    ax.text(0.055, 84.68, "SelfRef", fontsize=8, color=C_CLIP)
    ax.set_xlabel("top-k ratio (evidence)"); ax.set_ylabel("pixel AUPRO (%)")
    ax.set_ylim(84.2, 86.2); ax.set_title("(a) robust to top-k", fontsize=10)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)

    mix = [0.0, 0.05, 0.15, 0.30, 0.60, 1.0]
    aupro_m = [84.6, 86.0, 85.6, 84.8, 82.0, 78.5]
    ax=axes[1]
    ax.plot(mix, aupro_m, color=C_FREQ, lw=2, marker="s", ms=5)
    ax.scatter([0.05],[86.0], s=90, facecolors="none", edgecolors="#8a3f34", lw=1.6, zorder=5)
    ax.text(0.08, 86.05, "Ours", fontsize=8.5, color="#8a3f34")
    ax.text(0.0, 83.8, "=SelfRef", fontsize=7.5, color=C_CLIP)
    ax.text(1.0, 79.2, "raw HF\n(toward collapse)", fontsize=7.5, color="#9c4b4b", ha="right")
    ax.set_xlabel("wavelet mix"); ax.set_ylim(77, 87)
    ax.set_title("(b) mix: too much HF hurts", fontsize=10)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)

    beta = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    aupro_b = [83.2, 85.3, 86.0, 85.6, 84.9, 83.8]
    ax=axes[2]
    ax.plot(beta, aupro_b, color=C_GLOBAL, lw=2, marker="^", ms=6)
    ax.set_xlabel("normal update strength beta"); ax.set_ylim(82.5, 86.5)
    ax.set_title("(c) mild update is best", fontsize=10)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)

    fig.suptitle("Hyperparameter sensitivity (MVTec pixel AUPRO, EXPECTED); (b) is also a mechanism curve",
                 fontsize=10.5, color="#333", y=1.03)
    savefig(fig, "fig_sensitivity")


# ---- Fig 5: compact blind-spot recall ----
def fig_blindspot():
    datasets=["MVTec","VisA"]; ours=[22,18]
    x=np.arange(len(datasets))
    fig, ax=plt.subplots(figsize=(3.6, 3.4))
    bars=ax.bar(x, ours, 0.5, color=C_OURS)
    for r,v in zip(bars, ours):
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.5, f"{v}%", ha="center",
                fontsize=11, color="#8a3f34", fontweight="bold")
    ax.axhline(0, color=C_CLIP, lw=1.4)
    ax.text(1.4, 0.6, "SelfRef = 0%", fontsize=8, color=C_CLIP, ha="right")
    ax.set_ylabel("recall on CLIP blind-spot (%)")
    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylim(0, 26)
    ax.set_title("Recovering CLIP's misses\n(EXPECTED)", fontsize=10, color="#333")
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8); ax.set_axisbelow(True)
    savefig(fig, "fig_blindspot")


if __name__ == "__main__":
    fig_main_results()
    fig_mechanism_ordering()
    fig_percategory_gain()
    fig_sensitivity()
    fig_blindspot()
    print("done")
