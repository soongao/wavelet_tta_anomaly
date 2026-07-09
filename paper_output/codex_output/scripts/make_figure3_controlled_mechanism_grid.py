#!/usr/bin/env python3
"""Build a controlled MVTec/VisA mechanism grid from real model-output strips.

The source images are seven-panel strips. This script crops the strips into
panels, selects six mechanism columns, and redraws only row/column labels as
vector text. It does not edit heatmaps, masks, or evidence overlays.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
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
from matplotlib.patches import Rectangle
from PIL import Image


SOURCE_DIR = Path("/Users/bytedance/code/AnomalyCLIP/cached_results/prototype_tuned/mechanism_viz")
OUT_DIR = Path("outputs/figures")


@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    category: str
    sample: str
    filename: str


CASES = [
    CaseSpec("MVTec", "bottle", "000000", "mvtec_bottle_000000_bottle.png"),
    CaseSpec("MVTec", "cable", "000083", "mvtec_cable_000083_cable.png"),
    CaseSpec("MVTec", "capsule", "000233", "mvtec_capsule_000233_capsule.png"),
    CaseSpec("VisA", "candle", "000100", "visa_candle_000100_candle.png"),
    CaseSpec("VisA", "capsules", "000260", "visa_capsules_000260_capsules.png"),
    CaseSpec("VisA", "cashew", "000410", "visa_cashew_000410_cashew.png"),
]

# Source strip columns:
# 0 Input, 1 GT target, 2 Fixed prototype map, 3 Direct wavelet cue,
# 4 Boundary-aware reliability, 5 Selected evidence, 6 WPTA final map.
SELECTED_COLUMNS = [
    (0, "Input"),
    (1, "GT target"),
    (2, "Fixed map"),
    (4, "Reliability W"),
    (5, "Selected evidence"),
    (6, "WPTA map"),
]


def crop_panels(image: Image.Image, n_panels: int = 7) -> list[Image.Image]:
    width, height = image.size
    panel_w = width // n_panels
    panels: list[Image.Image] = []
    for idx in range(n_panels):
        left = idx * panel_w
        right = width if idx == n_panels - 1 else (idx + 1) * panel_w
        panels.append(image.crop((left, 0, right, height)).convert("RGBA"))
    return panels


def load_case_panels(source_dir: Path, case: CaseSpec) -> list[Image.Image]:
    path = source_dir / case.filename
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGBA")
    if image.width % 7 != 0:
        raise ValueError(f"source strip width is not divisible by 7: {path} ({image.size})")
    panels = crop_panels(image)
    return [panels[col_idx] for col_idx, _ in SELECTED_COLUMNS]


def add_dataset_marker(ax: plt.Axes, dataset: str) -> None:
    color = "#0072B2" if dataset == "MVTec" else "#009E73"
    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            0.025,
            1.0,
            transform=ax.transAxes,
            fc=color,
            ec="none",
            alpha=0.95,
            zorder=5,
        )
    )


def build_figure(case_panels: list[list[Image.Image]]) -> plt.Figure:
    n_rows = len(CASES)
    n_cols = len(SELECTED_COLUMNS)
    fig = plt.figure(figsize=(7.08, 7.85), dpi=300)
    grid = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.125,
        right=0.985,
        top=0.915,
        bottom=0.075,
        wspace=0.035,
        hspace=0.105,
    )

    for row_idx, (case, panels) in enumerate(zip(CASES, case_panels)):
        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.42)
                spine.set_edgecolor("#5f5f5f")
            if col_idx == 0:
                add_dataset_marker(ax, case.dataset)
                row_label = f"{case.dataset}\n{case.category}\n#{case.sample}"
                ax.text(
                    -0.18,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.0,
                    color="#202020",
                    linespacing=1.08,
                )
            if row_idx == 0:
                _, title = SELECTED_COLUMNS[col_idx]
                ax.set_title(title, fontsize=8.4, pad=4.4, color="#111111", fontweight="semibold")

    fig.text(
        0.125,
        0.966,
        "Controlled MVTec/VisA mechanism visualization",
        ha="left",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.125,
        0.941,
        "All panels are cropped from real WPTA mechanism strips; heatmaps and evidence overlays are unchanged.",
        ha="left",
        va="top",
        fontsize=8.0,
        color="#4a4a4a",
    )
    fig.text(
        0.125,
        0.038,
        "Blue row marker: MVTec. Green row marker: VisA. This figure supports the controlled mechanism evidence only.",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#4a4a4a",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--stem", default="figure3_controlled_mechanism_mvtec_visa_grid")
    args = parser.parse_args()

    case_panels = [load_case_panels(args.source_dir, case) for case in CASES]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(case_panels)

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
