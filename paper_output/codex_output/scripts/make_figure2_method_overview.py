#!/usr/bin/env python3
"""Generate a vector Figure 2 method overview for WPTA.

The figure is a pure vector diagram: boxes, arrows, text, and small
schematic glyphs are all drawn with Matplotlib primitives. It does not
contain experiment heatmaps or screenshots.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import fill

Path("outputs/.matplotlib_cache").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("outputs/figures")
PNG_OUT = OUT_DIR / "figure2_wpta_method_overview.png"
PDF_OUT = OUT_DIR / "figure2_wpta_method_overview.pdf"
SVG_OUT = OUT_DIR / "figure2_wpta_method_overview.svg"


COLORS = {
    "semantic": "#0072B2",
    "wavelet": "#009E73",
    "adapt": "#D55E00",
    "output": "#CC79A7",
    "neutral": "#4D4D4D",
    "light_semantic": "#E6F1F8",
    "light_wavelet": "#E6F5EF",
    "light_adapt": "#FBEDE6",
    "light_output": "#F7EAF3",
    "light_neutral": "#F4F4F4",
}


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
        }
    )


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str = "",
    face: str = "#FFFFFF",
    edge: str = "#333333",
    title_color: str = "#111111",
    lw: float = 1.25,
    fontsize_title: float = 9.5,
    fontsize_body: float = 8.3,
    wrap: int = 22,
    radius: float = 0.09,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.22,
        title,
        ha="center",
        va="top",
        color=title_color,
        fontsize=fontsize_title,
        fontweight="bold",
    )
    if body:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.10,
            fill(body, wrap),
            ha="center",
            va="center",
            color="#222222",
            fontsize=fontsize_body,
            linespacing=1.24,
        )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    color: str = "#555555",
    rad: float = 0.0,
    style: str = "-",
    label_offset: tuple[float, float] = (0.0, 0.0),
    fontsize: float = 8.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.15,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        lx = (start[0] + end[0]) / 2 + label_offset[0]
        ly = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9),
        )


def add_patch_grid(ax: plt.Axes, x: float, y: float, size: float = 0.56) -> None:
    cell = size / 4
    palette = ["#E8E8E8", "#D8ECF7", "#F8E5C4", "#DFF1E8"]
    for r in range(4):
        for c in range(4):
            ax.add_patch(
                Rectangle(
                    (x + c * cell, y + (3 - r) * cell),
                    cell * 0.9,
                    cell * 0.9,
                    facecolor=palette[(r + c) % len(palette)],
                    edgecolor="#777777",
                    linewidth=0.35,
                )
            )


def add_heatmap_glyph(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    colors = ["#F6E8C3", "#FDB863", "#E66101", "#B2182B"]
    for i, color in enumerate(colors):
        ax.add_patch(
            Rectangle(
                (x + i * w / len(colors), y),
                w / len(colors),
                h,
                facecolor=color,
                edgecolor="none",
            )
        )
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#555555", linewidth=0.6))


def add_anchor_points(ax: plt.Axes, x: float, y: float) -> None:
    ax.scatter([x, x + 0.20, x + 0.10], [y, y + 0.13, y + 0.28], s=28, c="#E69F00", edgecolors="#8C510A", linewidths=0.6)
    ax.scatter([x + 0.48, x + 0.65, x + 0.56], [y + 0.02, y + 0.18, y + 0.32], s=28, c="#56B4E9", edgecolors="#2166AC", linewidths=0.6)
    ax.text(x + 0.11, y - 0.12, "abnormal", ha="center", va="top", fontsize=8.0, color="#8C510A")
    ax.text(x + 0.56, y - 0.12, "normal", ha="center", va="top", fontsize=8.0, color="#2166AC")


def main() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.35, 3.15))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 4.55)
    ax.axis("off")

    ax.text(
        0.18,
        4.34,
        "Wavelet-Supervised Test-Time Prototype Adaptation (WPTA)",
        ha="left",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        11.85,
        4.34,
        "Frozen CLIP  |  no target training data  |  no backpropagation",
        ha="right",
        va="center",
        fontsize=8.2,
        color="#555555",
    )

    input_box = add_box(
        ax,
        (0.22, 1.58),
        (1.38, 1.26),
        "Input image",
        "x",
        face=COLORS["light_neutral"],
        edge="#666666",
        fontsize_title=9.2,
        fontsize_body=8.4,
    )
    add_patch_grid(ax, 0.55, 1.82, 0.54)

    clip_box = add_box(
        ax,
        (1.95, 1.44),
        (1.78, 1.54),
        "Frozen CLIP",
        "features F\nprototypes t_n, t_a",
        face="#FFFFFF",
        edge=COLORS["neutral"],
        fontsize_title=9.2,
        fontsize_body=8.0,
        wrap=20,
    )
    ax.add_patch(
        Rectangle(
            (2.22, 1.72),
            0.52,
            0.38,
            facecolor="#F2F2F2",
            edgecolor="#666666",
            linewidth=0.55,
        )
    )
    ax.text(2.48, 1.91, "E_img", ha="center", va="center", fontsize=8.0, color="#333333")
    ax.add_patch(
        Rectangle(
            (2.96, 1.72),
            0.45,
            0.38,
            facecolor="#F2F2F2",
            edgecolor="#666666",
            linewidth=0.55,
        )
    )
    ax.text(3.185, 1.91, "E_txt", ha="center", va="center", fontsize=8.0, color="#333333")

    semantic_box = add_box(
        ax,
        (4.15, 2.82),
        (1.88, 1.03),
        "Initial semantic prior",
        "S0 from CLIP similarities",
        face=COLORS["light_semantic"],
        edge=COLORS["semantic"],
        title_color=COLORS["semantic"],
        fontsize_title=9.0,
        fontsize_body=8.0,
        wrap=18,
    )
    add_heatmap_glyph(ax, 4.58, 3.02, 0.98, 0.12)

    wavelet_box = add_box(
        ax,
        (4.15, 0.62),
        (1.88, 1.24),
        "Haar wavelet reliability",
        "HF texture\nLF edge\nW = HF * (1 - LF_edge)",
        face=COLORS["light_wavelet"],
        edge=COLORS["wavelet"],
        title_color=COLORS["wavelet"],
        fontsize_title=9.0,
        fontsize_body=8.0,
        wrap=20,
    )

    evidence_box = add_box(
        ax,
        (6.55, 1.58),
        (1.76, 1.30),
        "Evidence selection",
        "select q_a and q_n reliable patches",
        face=COLORS["light_adapt"],
        edge=COLORS["adapt"],
        title_color=COLORS["adapt"],
        fontsize_title=9.0,
        fontsize_body=8.0,
        wrap=20,
    )
    add_anchor_points(ax, 7.02, 1.82)

    anchors_box = add_box(
        ax,
        (8.78, 1.58),
        (1.36, 1.30),
        "Visual anchors",
        "v_a and v_n",
        face="#FFF8E8",
        edge="#A6761D",
        title_color="#7A4E00",
        fontsize_title=9.0,
        fontsize_body=8.1,
        wrap=18,
    )
    ax.text(9.46, 1.95, "v_a", ha="center", va="center", fontsize=10.4, color="#8C510A", fontweight="bold")
    ax.text(9.46, 1.73, "v_n", ha="center", va="center", fontsize=10.4, color="#2166AC", fontweight="bold")

    proto_box = add_box(
        ax,
        (8.63, 3.12),
        (1.66, 0.72),
        "Conservative calibration",
        "t'_a, t'_n",
        face="#F7F7F7",
        edge="#666666",
        fontsize_title=8.7,
        fontsize_body=8.0,
        wrap=18,
    )

    output_box = add_box(
        ax,
        (10.62, 1.58),
        (1.14, 1.30),
        "Final scores",
        "anomaly map M\nimage score s_img",
        face=COLORS["light_output"],
        edge=COLORS["output"],
        title_color=COLORS["output"],
        fontsize_title=9.0,
        fontsize_body=8.0,
        wrap=18,
    )
    add_heatmap_glyph(ax, 10.87, 1.82, 0.62, 0.13)

    add_arrow(ax, (1.60, 2.21), (1.95, 2.21), "x", color="#555555", label_offset=(0, 0.16), fontsize=8.0)
    add_arrow(ax, (3.73, 2.22), (4.15, 3.33), "F, t_a, t_n", color=COLORS["semantic"], rad=0.08, label_offset=(0.04, 0.05), fontsize=8.0)
    add_arrow(ax, (3.73, 1.98), (4.15, 1.22), "patch grid F", color=COLORS["wavelet"], rad=-0.07, label_offset=(-0.05, -0.03), fontsize=8.0)
    add_arrow(ax, (6.03, 3.33), (6.55, 2.45), "S0", color=COLORS["semantic"], rad=-0.12, label_offset=(0.03, 0.06), fontsize=8.0)
    add_arrow(ax, (6.03, 1.22), (6.55, 2.05), "W", color=COLORS["wavelet"], rad=0.12, label_offset=(0.00, -0.07), fontsize=8.0)
    add_arrow(ax, (8.31, 2.23), (8.78, 2.23), "top-k evidence", color=COLORS["adapt"], label_offset=(0, 0.18), fontsize=8.0)
    add_arrow(ax, (9.46, 2.88), (9.46, 3.12), "gate", color="#666666", label_offset=(0.20, 0), fontsize=8.0)
    add_arrow(ax, (10.14, 2.23), (10.62, 2.23), "calib. prototypes", color="#666666", label_offset=(0.0, 0.18), fontsize=8.0)
    add_arrow(ax, (10.29, 3.47), (10.80, 2.88), "t'_a, t'_n", color="#666666", rad=-0.06, label_offset=(0.15, 0.02), fontsize=8.0)

    ax.plot([4.02, 6.22], [2.28, 2.28], linestyle=(0, (4, 3)), color="#AAAAAA", linewidth=0.8)
    ax.text(5.10, 2.42, "wavelet is reliability, not final score", ha="center", va="center", fontsize=8.0, color="#555555")

    ax.text(0.26, 0.16, "Inference-only: per-image prototype update with frozen CLIP.", fontsize=8.0, color="#444444")
    ax.text(11.75, 0.16, "Final scores use calibrated CLIP prototypes.", ha="right", fontsize=8.0, color="#444444")

    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SVG_OUT, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PNG_OUT, bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)

    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {SVG_OUT}")
    print(f"Wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
