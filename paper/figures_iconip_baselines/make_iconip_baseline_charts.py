#!/usr/bin/env python3
"""Generate ICONIP-style baseline charts from the copied baseline tables.

The reference Ours column from /Users/bytedance/mypaper/paper_iconip/main.tex
is intentionally excluded.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "source_data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "CLIP": "#7a7a7a",
    "WinCLIP": "#4E79A7",
    "VAND": "#59A14F",
    "CoOp": "#9C755F",
    "AdaCLIP": "#B07AA1",
    "AnomalyCLIP": "#F28E2B",
    "AA-CLIP": "#E15759",
    "MVFA": "#76B7B2",
}

INDUSTRIAL_METHODS = [
    "CLIP",
    "WinCLIP",
    "VAND",
    "CoOp",
    "AdaCLIP",
    "AnomalyCLIP",
    "AA-CLIP",
]
MEDICAL_IMAGE_METHODS = ["CLIP", "WinCLIP", "VAND", "CoOp", "AnomalyCLIP"]
MEDICAL_PIXEL_METHODS = [
    "CLIP",
    "WinCLIP",
    "VAND",
    "CoOp",
    "MVFA",
    "AdaCLIP",
    "AnomalyCLIP",
    "AA-CLIP",
]
INDUSTRIAL_DATASETS = ["MVTec AD", "VisA", "MPDD", "BTAD", "DTD-Synthetic"]
MEDICAL_IMAGE_DATASETS = ["HeadCT", "BrainMRI", "Br35H"]
MEDICAL_PIXEL_DATASETS = ["ISIC", "ColonDB", "ClinicDB", "Kvasir"]


def build_industrial() -> pd.DataFrame:
    rows = {
        "MVTec AD": {
            "CLIP": (74.1, 87.6, 38.4, 11.3),
            "WinCLIP": (91.8, 96.5, 85.1, 64.6),
            "VAND": (86.1, 93.5, 87.6, 44.0),
            "CoOp": (88.8, 94.8, 33.3, 6.7),
            "AdaCLIP": (89.2, 96.4, 88.7, 37.8),
            "AnomalyCLIP": (91.5, 96.2, 91.1, 81.4),
            "AA-CLIP": (90.5, 94.9, 91.9, 84.6),
        },
        "VisA": {
            "CLIP": (66.4, 71.5, 46.6, 14.8),
            "WinCLIP": (78.1, 81.2, 79.6, 56.8),
            "VAND": (78.0, 81.4, 94.2, 86.8),
            "CoOp": (62.8, 68.1, 24.2, 3.8),
            "AdaCLIP": (85.8, 84.9, 95.5, 56.8),
            "AnomalyCLIP": (82.1, 85.4, 95.4, 87.0),
            "AA-CLIP": (84.6, 82.2, 95.5, 83.0),
        },
        "MPDD": {
            "CLIP": (54.3, 65.4, 62.1, 33.0),
            "WinCLIP": (63.6, 69.9, 76.4, 48.9),
            "VAND": (73.0, 80.2, 94.1, 83.2),
            "CoOp": (55.1, 64.2, 15.4, 2.3),
            "AdaCLIP": (76.0, 80.4, 96.1, 60.3),
            "AnomalyCLIP": (77.0, 82.0, 96.5, 88.7),
            "AA-CLIP": (75.1, 80.1, 96.7, 76.5),
        },
        "BTAD": {
            "CLIP": (34.5, 52.5, 30.6, 4.4),
            "WinCLIP": (68.2, 70.9, 72.7, 27.3),
            "VAND": (73.6, 68.6, 60.8, 25.0),
            "CoOp": (66.8, 77.4, 28.6, 3.8),
            "AdaCLIP": (88.6, 92.4, 92.1, 32.5),
            "AnomalyCLIP": (88.3, 87.3, 94.2, 74.8),
            "AA-CLIP": (94.8, 97.9, 97.0, 69.0),
        },
        "DTD-Synthetic": {
            "CLIP": (71.6, 85.7, 33.9, 12.5),
            "WinCLIP": (93.2, 92.6, 83.9, 57.8),
            "VAND": (86.4, 95.0, 95.3, 86.9),
            "CoOp": (np.nan, np.nan, np.nan, np.nan),
            "AdaCLIP": (95.5, 97.0, 97.7, 75.0),
            "AnomalyCLIP": (93.5, 97.0, 97.9, 92.3),
            "AA-CLIP": (93.3, 97.8, 96.4, 85.9),
        },
        "Average": {
            "CLIP": (60.2, 72.5, 42.3, 15.2),
            "WinCLIP": (79.0, 82.2, 79.5, 51.1),
            "VAND": (79.4, 83.7, 86.4, 65.2),
            "CoOp": (68.4, 76.1, 25.4, 4.2),
            "AdaCLIP": (87.0, 90.2, 94.0, 52.5),
            "AnomalyCLIP": (86.5, 89.6, 95.0, 84.8),
            "AA-CLIP": (87.7, 90.6, 95.5, 79.8),
        },
    }
    return to_long(rows, ["Image-AUROC", "Image-AP", "Pixel-AUROC", "Pixel-AUPRO"])


def build_medical_image() -> pd.DataFrame:
    rows = {
        "HeadCT": {
            "CLIP": (56.5, 58.4),
            "WinCLIP": (81.8, 80.2),
            "VAND": (89.1, 89.4),
            "CoOp": (78.4, 78.8),
            "AnomalyCLIP": (93.4, 91.6),
        },
        "BrainMRI": {
            "CLIP": (73.9, 81.7),
            "WinCLIP": (86.6, 91.5),
            "VAND": (89.3, 90.9),
            "CoOp": (61.3, 44.9),
            "AnomalyCLIP": (90.3, 92.2),
        },
        "Br35H": {
            "CLIP": (78.4, 78.8),
            "WinCLIP": (80.5, 82.2),
            "VAND": (93.1, 92.9),
            "CoOp": (86.0, 87.5),
            "AnomalyCLIP": (94.6, 94.7),
        },
        "Average": {
            "CLIP": (69.6, 73.0),
            "WinCLIP": (83.0, 84.6),
            "VAND": (90.5, 91.1),
            "CoOp": (75.2, 70.4),
            "AnomalyCLIP": (92.8, 92.8),
        },
    }
    return to_long(rows, ["Image-AUROC", "Image-AP"])


def build_medical_pixel() -> pd.DataFrame:
    rows = {
        "ISIC": {
            "CLIP": (33.1, 5.8),
            "WinCLIP": (83.3, 55.1),
            "VAND": (89.4, 77.2),
            "CoOp": (51.7, 15.9),
            "MVFA": (84.5, 69.1),
            "AdaCLIP": (89.3, 33.1),
            "AnomalyCLIP": (83.0, 63.8),
            "AA-CLIP": (93.9, 87.0),
        },
        "ColonDB": {
            "CLIP": (49.5, 15.8),
            "WinCLIP": (70.3, 32.5),
            "VAND": (78.4, 64.6),
            "CoOp": (40.5, 2.6),
            "MVFA": (80.8, 65.9),
            "AdaCLIP": (90.4, 56.8),
            "AnomalyCLIP": (np.nan, np.nan),
            "AA-CLIP": (84.0, 67.5),
        },
        "ClinicDB": {
            "CLIP": (47.5, 18.9),
            "WinCLIP": (51.2, 13.8),
            "VAND": (80.5, 60.7),
            "CoOp": (34.8, 2.4),
            "MVFA": (82.8, 66.3),
            "AdaCLIP": (84.4, 56.1),
            "AnomalyCLIP": (92.4, 82.9),
            "AA-CLIP": (89.9, 70.1),
        },
        "Kvasir": {
            "CLIP": (44.6, 17.7),
            "WinCLIP": (69.7, 24.5),
            "VAND": (75.0, 36.2),
            "CoOp": (44.1, 3.5),
            "MVFA": (81.9, 43.7),
            "AdaCLIP": (77.5, 41.3),
            "AnomalyCLIP": (92.5, 61.5),
            "AA-CLIP": (87.2, 52.6),
        },
        "Average": {
            "CLIP": (43.7, 14.6),
            "WinCLIP": (68.6, 31.5),
            "VAND": (80.8, 59.7),
            "CoOp": (42.8, 6.1),
            "MVFA": (82.5, 61.3),
            "AdaCLIP": (85.4, 46.8),
            "AnomalyCLIP": (89.3, 69.4),
            "AA-CLIP": (88.7, 69.3),
        },
    }
    return to_long(rows, ["Pixel-AUROC", "Pixel-AUPRO"])


def to_long(rows: dict, metrics: list[str]) -> pd.DataFrame:
    records = []
    for dataset, by_method in rows.items():
        for method, values in by_method.items():
            for metric, value in zip(metrics, values):
                records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
    return pd.DataFrame.from_records(records)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)


def plot_average_bars(df: pd.DataFrame, metrics: list[str], methods: list[str], stem: str) -> None:
    avg = df[df["dataset"] == "Average"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    y = np.arange(len(methods))
    for ax, metric in zip(axes, metrics):
        vals = []
        for method in methods:
            row = avg[(avg["method"] == method) & (avg["metric"] == metric)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
        ax.barh(y, vals, color=[COLORS[m] for m in methods], height=0.68)
        ax.set_title(metric)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(min(v + 1.2, 98.5), i, f"{v:.1f}", va="center", ha="left", fontsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Average score (%)")
    axes[0].set_yticks(y, methods)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.0)
    save_all(fig, stem)
    plt.close(fig)


def plot_dataset_lines(
    df: pd.DataFrame,
    metric: str,
    methods: list[str],
    dataset_order: list[str],
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(dataset_order))
    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    for idx, method in enumerate(methods):
        vals = []
        for dataset in dataset_order:
            row = df[(df["dataset"] == dataset) & (df["method"] == method) & (df["metric"] == metric)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
        ax.plot(
            x,
            vals,
            marker=markers[idx % len(markers)],
            linewidth=1.35,
            markersize=4.2,
            color=COLORS[method],
            label=method,
        )
    ax.set_xticks(x, dataset_order)
    ax.set_ylabel(f"{metric} (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    save_all(fig, stem)
    plt.close(fig)


def main() -> None:
    configure_style()
    industrial = build_industrial()
    medical_image = build_medical_image()
    medical_pixel = build_medical_pixel()

    industrial.to_csv(DATA_DIR / "industrial_baselines_long.csv", index=False)
    medical_image.to_csv(DATA_DIR / "medical_image_baselines_long.csv", index=False)
    medical_pixel.to_csv(DATA_DIR / "medical_pixel_baselines_long.csv", index=False)

    plot_average_bars(
        industrial,
        ["Image-AUROC", "Image-AP", "Pixel-AUROC", "Pixel-AUPRO"],
        INDUSTRIAL_METHODS,
        "industrial_average_baseline_bars",
    )
    plot_dataset_lines(
        industrial,
        "Image-AUROC",
        INDUSTRIAL_METHODS,
        INDUSTRIAL_DATASETS,
        "industrial_image_auroc_baseline_lines",
    )
    plot_dataset_lines(
        industrial,
        "Image-AP",
        INDUSTRIAL_METHODS,
        INDUSTRIAL_DATASETS,
        "industrial_image_ap_baseline_lines",
    )
    plot_dataset_lines(
        industrial,
        "Pixel-AUROC",
        INDUSTRIAL_METHODS,
        INDUSTRIAL_DATASETS,
        "industrial_pixel_auroc_baseline_lines",
    )
    plot_dataset_lines(
        industrial,
        "Pixel-AUPRO",
        INDUSTRIAL_METHODS,
        INDUSTRIAL_DATASETS,
        "industrial_pixel_aupro_baseline_lines",
    )
    plot_average_bars(
        medical_image,
        ["Image-AUROC", "Image-AP"],
        MEDICAL_IMAGE_METHODS,
        "medical_image_average_baseline_bars",
    )
    plot_dataset_lines(
        medical_image,
        "Image-AUROC",
        MEDICAL_IMAGE_METHODS,
        MEDICAL_IMAGE_DATASETS,
        "medical_image_auroc_baseline_lines",
    )
    plot_dataset_lines(
        medical_image,
        "Image-AP",
        MEDICAL_IMAGE_METHODS,
        MEDICAL_IMAGE_DATASETS,
        "medical_image_ap_baseline_lines",
    )
    plot_average_bars(
        medical_pixel,
        ["Pixel-AUROC", "Pixel-AUPRO"],
        MEDICAL_PIXEL_METHODS,
        "medical_pixel_average_baseline_bars",
    )
    plot_dataset_lines(
        medical_pixel,
        "Pixel-AUROC",
        MEDICAL_PIXEL_METHODS,
        MEDICAL_PIXEL_DATASETS,
        "medical_pixel_auroc_baseline_lines",
    )
    plot_dataset_lines(
        medical_pixel,
        "Pixel-AUPRO",
        MEDICAL_PIXEL_METHODS,
        MEDICAL_PIXEL_DATASETS,
        "medical_pixel_aupro_baseline_lines",
    )


if __name__ == "__main__":
    main()
