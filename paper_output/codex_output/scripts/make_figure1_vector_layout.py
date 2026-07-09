#!/usr/bin/env python3
"""Build a publication-oriented Figure 1 layout from real model-output panels.

The seven input panels are raster model outputs. This script keeps their pixels
unchanged and redraws only paper-facing layout elements as vector graphics:
panel titles, borders, arrows, legend, and mechanism note.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib_cache").resolve()))

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
    }
)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


DEFAULT_PANELS_DIR = Path("outputs/figures/figure1_panels")
DEFAULT_OUT_DIR = Path("outputs/figures")

PANEL_FILES = [
    "panel_01_input.png",
    "panel_02_gt_target.png",
    "panel_03_fixed_prototype_map.png",
    "panel_04_direct_wavelet_cue.png",
    "panel_05_boundary-aware_reliability.png",
    "panel_06_selected_evidence.png",
    "panel_07_wpta_final_map.png",
]

PANEL_TITLES = [
    "Input image",
    "GT target",
    "Fixed prototype map",
    "Direct wavelet cue",
    "Boundary-aware reliability",
    "Selected evidence",
    "WPTA final map",
]

PANEL_LETTERS = list("abcdefg")


def load_panels(panels_dir: Path) -> list[Image.Image]:
    panels: list[Image.Image] = []
    for name in PANEL_FILES:
        path = panels_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing panel: {path}")
        panels.append(Image.open(path).convert("RGBA"))
    return panels


def add_panel(ax: plt.Axes, image: Image.Image, title: str, letter: str) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#505050")
    ax.set_title(title, fontsize=8.4, pad=4.2, color="#111111", fontweight="semibold")
    ax.text(
        0.02,
        0.98,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#111111",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#777777", lw=0.35, alpha=0.92),
    )


def axes_center(ax: plt.Axes) -> tuple[float, float]:
    box = ax.get_position()
    return (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2


def add_arrow(fig: plt.Figure, ax_from: plt.Axes, ax_to: plt.Axes, color: str = "#666666") -> None:
    x0, y0 = axes_center(ax_from)
    x1, y1 = axes_center(ax_to)
    start = (x0 + 0.055, y0)
    end = (x1 - 0.055, y1)
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=9,
        lw=0.75,
        color=color,
        shrinkA=2,
        shrinkB=2,
        connectionstyle="arc3,rad=0.0",
    )
    fig.patches.append(arrow)


def add_legend(fig: plt.Figure) -> None:
    legend_box = FancyBboxPatch(
        (0.055, 0.045),
        0.37,
        0.055,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        transform=fig.transFigure,
        fc="#fbfbfb",
        ec="#cfcfcf",
        lw=0.55,
    )
    fig.patches.append(legend_box)
    entries = [
        (0.074, "#d55e00", "abnormal evidence"),
        (0.218, "#0072b2", "normal evidence"),
    ]
    for x, color, label in entries:
        fig.patches.append(
            Rectangle((x, 0.064), 0.018, 0.018, transform=fig.transFigure, fc="none", ec=color, lw=1.25)
        )
        fig.text(x + 0.024, 0.063, label, ha="left", va="bottom", fontsize=8.0, color="#222222")


def add_note(fig: plt.Figure) -> None:
    note = (
        "WPTA uses boundary-aware wavelet reliability to select visual anchors.\n"
        "Final scoring still comes from calibrated CLIP prototypes."
    )
    note_box = FancyBboxPatch(
        (0.455, 0.045),
        0.49,
        0.055,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        transform=fig.transFigure,
        fc="#f7f7f7",
        ec="#cfcfcf",
        lw=0.55,
    )
    fig.patches.append(note_box)
    fig.text(0.475, 0.073, note, ha="left", va="center", fontsize=8.0, color="#222222", linespacing=1.15)


def build_figure(panels: list[Image.Image]) -> plt.Figure:
    fig = plt.figure(figsize=(7.08, 4.72), dpi=300)
    grid = fig.add_gridspec(
        2,
        4,
        left=0.045,
        right=0.965,
        top=0.875,
        bottom=0.145,
        wspace=0.08,
        hspace=0.26,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 3]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    empty_ax = fig.add_subplot(grid[1, 3])
    empty_ax.axis("off")

    for ax, image, title, letter in zip(axes, panels, PANEL_TITLES, PANEL_LETTERS):
        add_panel(ax, image, title, letter)

    fig.text(
        0.045,
        0.945,
        "Figure 1: Wavelet reliability guides prototype adaptation",
        ha="left",
        va="top",
        fontsize=10.6,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.045,
        0.912,
        "MVTec cable sample 000083. Raster panels are unchanged; annotations are vector.",
        ha="left",
        va="top",
        fontsize=8.0,
        color="#555555",
    )

    for idx in range(3):
        add_arrow(fig, axes[idx], axes[idx + 1])
    add_arrow(fig, axes[4], axes[5])
    add_arrow(fig, axes[5], axes[6])

    # A small connector indicating that the lower-row evidence path calibrates final scoring.
    start = axes_center(axes[3])
    end = axes_center(axes[6])
    fig.patches.append(
        FancyArrowPatch(
            (start[0], start[1] - 0.095),
            (end[0] + 0.02, end[1] + 0.105),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=0.7,
            color="#777777",
            linestyle=(0, (3, 2)),
            connectionstyle="arc3,rad=-0.18",
        )
    )

    add_legend(fig)
    add_note(fig)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels-dir", type=Path, default=DEFAULT_PANELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="figure1_motivated_example_mvtec_cable_vector_layout")
    args = parser.parse_args()

    panels = load_panels(args.panels_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(panels)

    pdf_path = args.out_dir / f"{args.stem}.pdf"
    svg_path = args.out_dir / f"{args.stem}.svg"
    png_path = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
