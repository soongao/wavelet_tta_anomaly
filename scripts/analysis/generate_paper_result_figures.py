#!/usr/bin/env python3
"""Generate publication-ready result figures from recorded experiment data.

The script reads the result workbook CSVs under ``paper/result_record`` and
exports paper figures under ``paper_output/generated_result_figures_20260717``.
It intentionally avoids error bars because the available records are aggregate
dataset-level summaries rather than repeated-run estimates.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


OUT_DIR = ROOT / "paper_output" / "generated_result_figures_20260717"
DATA_DIR = OUT_DIR / "source_data"
FIG_DIR = OUT_DIR / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"

RESULT_WORKBOOK = ROOT / "paper" / "result_record" / "result_table.csv"
MEDICAL_WORKBOOK = ROOT / "paper" / "result_record" / "medical_result_table.csv"

METRICS = [
    ("pixel_auroc", "Pixel AUROC"),
    ("pixel_aupro", "Pixel AUPRO"),
    ("image_auroc", "Image AUROC"),
    ("image_ap", "Image AP"),
]

KEY_METRICS = [
    ("pixel_aupro", "Pixel AUPRO"),
    ("image_auroc", "Image AUROC"),
]

CORE_METHODS = [
    "Baseline",
    "Direct wavelet fusion / no adaptation",
    "Semantic prototype adaptation",
    "Wavelet prototype adaptation no conservative",
    "Full wavelet prototype adaptation",
]

CORE_METHOD_LABELS = ["Base", "Direct\nfusion", "Semantic\nadapt", "WPTA\nno cons.", "Full\nWPTA"]

CORE_DELTA_METHODS = CORE_METHODS[1:]
CORE_DELTA_LABELS = ["Direct\nfusion", "Semantic\nadapt", "WPTA\nno cons.", "Full\nWPTA"]

WAVELET_METHODS = [
    "Semantic-only prototype adaptation",
    "Direct wavelet fusion",
    "HF-only W + prototype adaptation",
    "Boundary-aware W + prototype adaptation",
    "Full boundary-aware W + conservative",
]

WAVELET_DELTA_METHODS = WAVELET_METHODS[1:]
WAVELET_DELTA_LABELS = [
    "Direct fusion",
    "HF-only W",
    "Boundary-aware W",
    "Full boundary-aware",
]

COLORS = {
    "ink": "#1F2933",
    "muted": "#5F6B7A",
    "grid": "#D5DCE5",
    "baseline": "#667085",
    "full": "#D55E00",
    "mvtec": "#0072B2",
    "visa": "#009E73",
    "light": "#F6F8FA",
    "edge": "#D0D7DE",
}

CMAP_DELTA = LinearSegmentedColormap.from_list(
    "paper_delta",
    ["#3B6EA8", "#F7F7F7", "#D55E00"],
)


@dataclass(frozen=True)
class MetricRecord:
    dataset: str
    method: str
    values: dict[str, float]
    result_type: str = "current"
    notes: str = ""


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
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def parse_sections(path: Path) -> dict[str, list[dict[str, str]]]:
    sections: dict[str, list[dict[str, str]]] = {}
    current: str | None = None
    header: list[str] | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.reader(f):
            row = [cell.strip() for cell in raw]
            if not any(row):
                continue

            first = row[0]
            if first.endswith(": 表格 1"):
                current = first.split(":", 1)[0]
                sections[current] = []
                header = None
                continue

            if current is None:
                continue

            if header is None:
                header = row
                continue

            padded = row + [""] * max(0, len(header) - len(row))
            sections[current].append(dict(zip(header, padded)))

    return sections


def parse_float(value: str) -> float:
    value = value.strip()
    if value in {"", "-", "暂无完整对应结果"}:
        return math.nan
    return float(value)


def parse_slash_metrics(value: str) -> dict[str, float]:
    parts = [parse_float(part) for part in value.split("/")]
    if len(parts) != len(METRICS):
        return {key: math.nan for key, _ in METRICS}
    return {key: number for (key, _), number in zip(METRICS, parts)}


def main_results(sections: dict[str, list[dict[str, str]]]) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for row in sections["Main Result"]:
        if row.get("result_type") != "current":
            continue
        values = {key: parse_float(row[key]) for key, _ in METRICS}
        records.append(
            MetricRecord(
                dataset=row["dataset"],
                method=row["method"],
                values=values,
                result_type=row.get("result_type", "current"),
                notes=row.get("notes", ""),
            )
        )
    return records


def section_records(
    rows: Iterable[dict[str, str]],
    dataset_columns: list[tuple[str, str]],
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for row in rows:
        method = row.get("method") or row.get("wavelet_setting") or row.get("setting") or ""
        for dataset, col in dataset_columns:
            values = parse_slash_metrics(row.get(col, ""))
            if all(math.isnan(v) for v in values.values()):
                continue
            records.append(
                MetricRecord(
                    dataset=dataset,
                    method=method,
                    values=values,
                    notes=row.get("notes", ""),
                )
            )
    return records


def medical_pixel_records(sections: dict[str, list[dict[str, str]]]) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for row in sections["Medical Pixel Result"]:
        if row.get("result_type") != "current":
            continue
        values = {
            "pixel_auroc": parse_float(row.get("pixel_auroc", "")),
            "pixel_aupro": parse_float(row.get("pixel_aupro", "")),
        }
        records.append(
            MetricRecord(
                dataset=row["dataset"],
                method=row["method"],
                values=values,
                result_type=row.get("result_type", "current"),
                notes=row.get("notes", ""),
            )
        )
    return records


def records_to_csv(path: Path, records: list[MetricRecord], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", *metric_keys, "result_type", "notes"])
        for record in records:
            writer.writerow(
                [
                    record.dataset,
                    record.method,
                    *[record.values.get(metric_key, math.nan) for metric_key in metric_keys],
                    record.result_type,
                    record.notes,
                ]
            )


def write_delta_csv(
    path: Path,
    records: list[MetricRecord],
    baseline_method: str,
    compare_methods: list[str],
    metric_keys: list[str],
) -> None:
    by_key = pivot(records)
    datasets = sorted({record.dataset for record in records})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", *[f"delta_{metric_key}" for metric_key in metric_keys]])
        for dataset in datasets:
            for method in compare_methods:
                writer.writerow(
                    [
                        dataset,
                        method,
                        *[
                            by_key[(dataset, method)].values[metric_key]
                            - by_key[(dataset, baseline_method)].values[metric_key]
                            for metric_key in metric_keys
                        ],
                    ]
                )


def pivot(records: list[MetricRecord]) -> dict[tuple[str, str], MetricRecord]:
    return {(record.dataset, record.method): record for record in records}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


def save_figure(fig: plt.Figure, basename: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"{basename}.{ext}"
        fig.savefig(path, dpi=350 if ext == "png" else None, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    make_grayscale_preview(FIG_DIR / f"{basename}.png", FIG_DIR / f"{basename}_gray.png")
    outputs.append(FIG_DIR / f"{basename}_gray.png")
    return outputs


def make_grayscale_preview(source: Path, target: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    with Image.open(source) as image:
        image.convert("L").save(target)


def metric_label(metric_key: str) -> str:
    return dict(METRICS)[metric_key]


def draw_delta_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    vlim: float | None = None,
) -> None:
    finite = values[np.isfinite(values)]
    if vlim is None:
        vlim = max(0.5, float(np.nanmax(np.abs(finite))) if finite.size else 1.0)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    image = ax.imshow(values, cmap=CMAP_DELTA, norm=norm, aspect="auto")
    ax.set_title(title, pad=8, loc="left", fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(length=0)

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if not np.isfinite(value):
                text = "NA"
                color = COLORS["muted"]
            else:
                text = f"{value:+.1f}"
                color = "white" if abs(value) > vlim * 0.55 else COLORS["ink"]
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color=color)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["edge"])
        spine.set_linewidth(0.6)
    return image


def add_bar_labels(ax: plt.Axes, bars, *, horizontal: bool = False, fontsize: int = 6) -> None:
    for bar in bars:
        if horizontal:
            value = bar.get_width()
            x = value + (0.08 if value >= 0 else -0.08)
            ax.text(
                x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=fontsize,
                color=COLORS["ink"],
            )
        else:
            value = bar.get_height()
            y = value + (0.08 if value >= 0 else -0.08)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=fontsize,
                color=COLORS["ink"],
            )


def figure_main_results(records: list[MetricRecord]) -> None:
    by_key = pivot(records)
    datasets = ["MVTec", "VisA", "MPDD", "BTAD", "DTD-Synthetic"]
    methods = ("AnomalyCLIP baseline", "Full")

    fig = plt.figure(figsize=(7.16, 3.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.25], wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    y = np.arange(len(datasets))
    baseline = np.array([by_key[(dataset, methods[0])].values["pixel_aupro"] for dataset in datasets])
    full = np.array([by_key[(dataset, methods[1])].values["pixel_aupro"] for dataset in datasets])
    delta = full - baseline

    for i, (b_value, f_value) in enumerate(zip(baseline, full)):
        ax_a.plot([b_value, f_value], [i, i], color=COLORS["grid"], lw=2.0, solid_capstyle="round")
        ax_a.scatter(b_value, i, s=28, color=COLORS["baseline"], zorder=3, label="Baseline" if i == 0 else None)
        ax_a.scatter(f_value, i, s=34, color=COLORS["full"], zorder=4, label="Full" if i == 0 else None)
        ax_a.text(f_value + 0.45, i, f"{delta[i]:+.1f}", va="center", ha="left", fontsize=7, color=COLORS["full"])

    ax_a.set_yticks(y, datasets)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Pixel AUPRO (%)")
    ax_a.set_title("Localization improvement", loc="left", fontweight="bold", pad=8)
    ax_a.grid(axis="x", color=COLORS["grid"], alpha=0.8)
    ax_a.legend(frameon=False, loc="lower right", handletextpad=0.4)
    ax_a.set_xlim(min(baseline.min(), full.min()) - 2.0, max(baseline.max(), full.max()) + 5.0)
    add_panel_label(ax_a, "a")

    heat = np.zeros((len(datasets), len(METRICS)))
    for row_i, dataset in enumerate(datasets):
        for col_i, (metric_key, _) in enumerate(METRICS):
            heat[row_i, col_i] = (
                by_key[(dataset, methods[1])].values[metric_key]
                - by_key[(dataset, methods[0])].values[metric_key]
            )

    image = draw_delta_heatmap(
        ax_b,
        heat,
        datasets,
        [label.replace(" ", "\n") for _, label in METRICS],
        title="Full - baseline gain",
    )
    cbar = fig.colorbar(image, ax=ax_b, fraction=0.046, pad=0.02)
    cbar.set_label("points", rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=7, length=2)
    add_panel_label(ax_b, "b")

    save_figure(fig, "figure1_main_results")


def figure_core_ablation(records: list[MetricRecord]) -> None:
    datasets = ["MVTec", "VisA"]
    colors = {"MVTec": COLORS["mvtec"], "VisA": COLORS["visa"]}
    markers = {"MVTec": "o", "VisA": "s"}
    by_key = pivot(records)

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9), constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(len(CORE_METHODS))
    offsets = {"MVTec": -0.08, "VisA": 0.08}

    for ax, (metric_key, label), panel in zip(axes, METRICS, ["a", "b", "c", "d"]):
        metric_values = []
        for dataset in datasets:
            y = np.array([by_key[(dataset, method)].values[metric_key] for method in CORE_METHODS])
            metric_values.extend(y.tolist())
            ax.scatter(
                x + offsets[dataset],
                y,
                marker=markers[dataset],
                s=26,
                color=colors[dataset],
                label=dataset,
                zorder=4,
            )
            ax.axhline(
                y[0],
                color=colors[dataset],
                lw=0.75,
                linestyle=(0, (2, 2)),
                alpha=0.42,
                zorder=1,
            )

        y_min = min(metric_values)
        y_max = max(metric_values)
        pad = max(0.8, (y_max - y_min) * 0.16)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xticks(x, CORE_METHOD_LABELS)
        ax.set_ylabel(f"{label} (%)")
        ax.set_title(label, loc="left", fontweight="bold", pad=7)
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.85)
        add_panel_label(ax, panel)

    axes[0].legend(frameon=False, loc="lower right")
    save_figure(fig, "figure2_core_ablation")


def figure_core_ablation_bar(records: list[MetricRecord]) -> None:
    """Show component gains as bars relative to the fixed AnomalyCLIP baseline."""
    datasets = ["MVTec", "VisA"]
    colors = {"MVTec": COLORS["mvtec"], "VisA": COLORS["visa"]}
    by_key = pivot(records)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.95), constrained_layout=True)
    x = np.arange(len(CORE_DELTA_METHODS))
    width = 0.36
    offsets = {"MVTec": -width / 2, "VisA": width / 2}

    for ax, (metric_key, label), panel in zip(axes, KEY_METRICS, ["a", "b"]):
        all_values: list[float] = []
        for dataset in datasets:
            base = by_key[(dataset, CORE_METHODS[0])].values[metric_key]
            deltas = np.array([by_key[(dataset, method)].values[metric_key] - base for method in CORE_DELTA_METHODS])
            all_values.extend(deltas.tolist())
            bars = ax.bar(
                x + offsets[dataset],
                deltas,
                width=width,
                color=colors[dataset],
                alpha=0.9,
                label=dataset,
                edgecolor="white",
                linewidth=0.5,
            )
            add_bar_labels(ax, bars)

        y_min = min(0.0, min(all_values))
        y_max = max(0.0, max(all_values))
        pad = max(0.5, (y_max - y_min) * 0.18)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.axhline(0, color=COLORS["ink"], lw=0.7)
        ax.set_xticks(x, CORE_DELTA_LABELS)
        ax.set_ylabel(f"Gain in {label} (points)")
        ax.set_title(f"{label}: component gain", loc="left", fontweight="bold", pad=7)
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.8)
        add_panel_label(ax, panel)

    axes[0].legend(frameon=False, loc="lower left")
    save_figure(fig, "figure4_core_ablation_bar")


def figure_core_ablation_line(records: list[MetricRecord]) -> None:
    """Use a connected trend only for the ordered component-accumulation path."""
    datasets = ["MVTec", "VisA"]
    colors = {"MVTec": COLORS["mvtec"], "VisA": COLORS["visa"]}
    markers = {"MVTec": "o", "VisA": "s"}
    by_key = pivot(records)
    ordered_methods = [
        "Baseline",
        "Semantic prototype adaptation",
        "Wavelet prototype adaptation no conservative",
        "Full wavelet prototype adaptation",
    ]
    ordered_labels = ["Base", "Semantic\nadapt", "WPTA\nno cons.", "Full\nWPTA"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.95), constrained_layout=True)
    x = np.arange(len(ordered_methods))

    for ax, (metric_key, label), panel in zip(axes, KEY_METRICS, ["a", "b"]):
        metric_values: list[float] = []
        for dataset in datasets:
            values = np.array([by_key[(dataset, method)].values[metric_key] for method in ordered_methods])
            metric_values.extend(values.tolist())
            ax.plot(
                x,
                values,
                marker=markers[dataset],
                color=colors[dataset],
                label=dataset,
                lw=1.4,
                markersize=4.2,
            )
            for xi, value in zip(x, values):
                ax.text(xi, value + 0.16, f"{value:.1f}", ha="center", va="bottom", fontsize=6, color=colors[dataset])

        y_min = min(metric_values)
        y_max = max(metric_values)
        pad = max(0.8, (y_max - y_min) * 0.18)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xticks(x, ordered_labels)
        ax.set_ylabel(f"{label} (%)")
        ax.set_title(f"{label}: ordered accumulation", loc="left", fontweight="bold", pad=7)
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.8)
        add_panel_label(ax, panel)

    axes[0].legend(frameon=False, loc="lower right")
    save_figure(fig, "figure5_core_ablation_line")


def figure_wavelet_design(records: list[MetricRecord]) -> None:
    columns: list[tuple[str, str]] = []
    for dataset in ("MVTec", "VisA"):
        for metric_key, metric_name in KEY_METRICS:
            columns.append((dataset, metric_name))

    by_key = pivot(records)
    heat = np.zeros((len(WAVELET_DELTA_METHODS), len(columns)))
    for row_i, method in enumerate(WAVELET_DELTA_METHODS):
        for col_i, (dataset, metric_name) in enumerate(columns):
            metric_key = next(key for key, label in METRICS if label == metric_name)
            base = by_key[(dataset, WAVELET_METHODS[0])].values[metric_key]
            value = by_key[(dataset, method)].values[metric_key]
            heat[row_i, col_i] = value - base

    fig, ax = plt.subplots(figsize=(7.16, 2.75), constrained_layout=True)
    image = draw_delta_heatmap(
        ax,
        heat,
        WAVELET_DELTA_LABELS,
        [f"{dataset}\n{metric}" for dataset, metric in columns],
        title="Wavelet design gain over semantic-only adaptation",
        vlim=7.0,
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.015)
    cbar.set_label("points", rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=7, length=2)
    add_panel_label(ax, "a")
    save_figure(fig, "figure3_wavelet_design_ablation")


def figure_wavelet_design_bar(records: list[MetricRecord]) -> None:
    """Bar version of wavelet-design deltas for readers who prefer exact comparisons."""
    datasets = ["MVTec", "VisA"]
    colors = {"MVTec": COLORS["mvtec"], "VisA": COLORS["visa"]}
    by_key = pivot(records)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05), constrained_layout=True)
    x = np.arange(len(WAVELET_DELTA_METHODS))
    width = 0.36
    offsets = {"MVTec": -width / 2, "VisA": width / 2}

    for ax, (metric_key, label), panel in zip(axes, KEY_METRICS, ["a", "b"]):
        all_values: list[float] = []
        for dataset in datasets:
            base = by_key[(dataset, WAVELET_METHODS[0])].values[metric_key]
            deltas = np.array([by_key[(dataset, method)].values[metric_key] - base for method in WAVELET_DELTA_METHODS])
            all_values.extend(deltas.tolist())
            bars = ax.bar(
                x + offsets[dataset],
                deltas,
                width=width,
                color=colors[dataset],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
                label=dataset,
            )
            add_bar_labels(ax, bars)

        y_min = min(0.0, min(all_values))
        y_max = max(0.0, max(all_values))
        pad = max(0.45, (y_max - y_min) * 0.18)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.axhline(0, color=COLORS["ink"], lw=0.7)
        ax.set_xticks(x, ["Direct\nfusion", "HF-only", "Boundary\naware", "Full"])
        ax.set_ylabel(f"Gain in {label} (points)")
        ax.set_title(f"{label}: wavelet design", loc="left", fontweight="bold", pad=7)
        ax.grid(axis="y", color=COLORS["grid"], alpha=0.8)
        add_panel_label(ax, panel)

    axes[0].legend(frameon=False, loc="lower left")
    save_figure(fig, "figure6_wavelet_design_bar")


def figure_main_gain_bar(records: list[MetricRecord]) -> None:
    """Rank datasets by full-method gain over baseline for the two headline metrics."""
    by_key = pivot(records)
    datasets = ["MVTec", "VisA", "MPDD", "BTAD", "DTD-Synthetic"]
    metrics = [("pixel_aupro", "Pixel AUPRO"), ("image_auroc", "Image AUROC")]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0), constrained_layout=True)
    for ax, (metric_key, label), panel in zip(axes, metrics, ["a", "b"]):
        deltas = []
        for dataset in datasets:
            baseline = by_key[(dataset, "AnomalyCLIP baseline")].values[metric_key]
            full = by_key[(dataset, "Full")].values[metric_key]
            deltas.append((dataset, full - baseline))
        deltas.sort(key=lambda item: item[1])

        y = np.arange(len(deltas))
        values = np.array([value for _, value in deltas])
        labels = [dataset for dataset, _ in deltas]
        colors = [COLORS["full"] if value >= 0 else COLORS["baseline"] for value in values]
        bars = ax.barh(y, values, color=colors, alpha=0.9, edgecolor="white", linewidth=0.5)
        add_bar_labels(ax, bars, horizontal=True)
        ax.axvline(0, color=COLORS["ink"], lw=0.7)
        ax.set_yticks(y, labels)
        ax.set_xlabel(f"Full - baseline {label} (points)")
        ax.set_title(f"{label}: dataset gain", loc="left", fontweight="bold", pad=7)
        ax.grid(axis="x", color=COLORS["grid"], alpha=0.8)
        ax.set_xlim(min(0.0, values.min()) - 0.7, values.max() + 1.0)
        add_panel_label(ax, panel)

    save_figure(fig, "figure7_main_gain_bar")


def figure_medical(records: list[MetricRecord]) -> None:
    by_key = pivot(records)
    datasets = sorted({record.dataset for record in records})
    if not datasets:
        return
    dataset = datasets[0]
    baseline = by_key[(dataset, "AnomalyCLIP baseline")]
    full = by_key[(dataset, "Full")]

    metrics = [("pixel_auroc", "Pixel AUROC"), ("pixel_aupro", "Pixel AUPRO")]
    fig, ax = plt.subplots(figsize=(4.1, 2.3), constrained_layout=True)
    y = np.arange(len(metrics))
    b_values = np.array([baseline.values[key] for key, _ in metrics])
    f_values = np.array([full.values[key] for key, _ in metrics])

    for i, (b_value, f_value) in enumerate(zip(b_values, f_values)):
        ax.plot([b_value, f_value], [i, i], color=COLORS["grid"], lw=2.0, solid_capstyle="round")
        ax.scatter(b_value, i, s=30, color=COLORS["baseline"], label="Baseline" if i == 0 else None, zorder=3)
        ax.scatter(f_value, i, s=36, color=COLORS["full"], label="Full" if i == 0 else None, zorder=4)
        ax.text(f_value + 0.18, i, f"{f_value - b_value:+.1f}", va="center", ha="left", fontsize=7, color=COLORS["full"])

    ax.set_yticks(y, [label for _, label in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Score (%)")
    ax.set_title(f"Medical transfer: {dataset}", loc="left", fontweight="bold", pad=7)
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.85)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(min(b_values.min(), f_values.min()) - 1.0, max(b_values.max(), f_values.max()) + 2.0)
    add_panel_label(ax, "a")
    save_figure(fig, "supp_figure_medical_isbi")


def write_manifest(outputs: list[str]) -> None:
    manifest = OUT_DIR / "MANIFEST.md"
    manifest.write_text(
        "\n".join(
            [
                "# Generated Paper Result Figures",
                "",
                "Created by `scripts/analysis/generate_paper_result_figures.py`.",
                "",
                "## Data Sources",
                "",
                f"- Main, core ablation, and wavelet design data: `{RESULT_WORKBOOK.relative_to(ROOT)}`",
                f"- Medical pixel result data: `{MEDICAL_WORKBOOK.relative_to(ROOT)}`",
                "- Only `current` rows are plotted. `expected_pass` rows and `blocked` rows are excluded.",
                "- Error bars are intentionally omitted because the source records are aggregate dataset-level summaries, not repeated-run estimates.",
                "- `pubfig`/`pubtab` are not required for this package; figures are generated with Matplotlib and exported as PDF/SVG/PNG.",
                "",
                "## Figures",
                "",
                "- `figures/figure1_main_results.{pdf,svg,png}`: five-dataset main result summary. Panel a shows paired baseline/full Pixel AUPRO; panel b shows Full - baseline gains for all four metrics.",
                "- `figures/figure2_core_ablation.{pdf,svg,png}`: core component ablation on MVTec and VisA across all four metrics.",
                "- `figures/figure3_wavelet_design_ablation.{pdf,svg,png}`: wavelet design deltas over semantic-only prototype adaptation for the two key metrics.",
                "- `figures/figure4_core_ablation_bar.{pdf,svg,png}`: bar-chart version of component gains over the fixed baseline for Pixel AUPRO and Image AUROC.",
                "- `figures/figure5_core_ablation_line.{pdf,svg,png}`: ordered component-accumulation trend from baseline to the full method.",
                "- `figures/figure6_wavelet_design_bar.{pdf,svg,png}`: bar-chart version of wavelet design gains over semantic-only adaptation.",
                "- `figures/figure7_main_gain_bar.{pdf,svg,png}`: ranked dataset-level gains of the full method over the baseline.",
                "- `figures/supp_figure_medical_isbi.{pdf,svg,png}`: supplementary ISIC/ISBI pixel-level medical transfer result. Other medical rows are blocked in the source workbook and are not plotted.",
                "",
                "## Generated Source CSVs",
                "",
                "- `source_data/main_results_current.csv`",
                "- `source_data/core_ablation_current.csv`",
                "- `source_data/wavelet_design_current.csv`",
                "- `source_data/medical_pixel_current.csv`",
                "- `source_data/main_gain_vs_baseline.csv`",
                "- `source_data/core_ablation_gain_vs_baseline.csv`",
                "- `source_data/wavelet_design_gain_vs_semantic.csv`",
                "",
                "## LaTeX",
                "",
                "- `latex_figure_snippets.tex` contains ready-to-edit figure environments.",
                "- `FIGURE_GUIDE_ZH.md` explains every generated figure in Chinese.",
                "",
                "## Exported Files",
                "",
                *[f"- `{path}`" for path in outputs],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_latex_snippets() -> None:
    snippets = OUT_DIR / "latex_figure_snippets.tex"
    snippets.write_text(
        r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure1_main_results.pdf}
\caption{Main result summary across five anomaly detection datasets. (a) Pixel-level AUPRO comparison between the AnomalyCLIP baseline and the full method. Values on the right report absolute point gains. (b) Full-method gains over the baseline across pixel- and image-level metrics.}
\label{fig:main-results}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure2_core_ablation.pdf}
\caption{Core component ablation on MVTec and VisA. Each panel reports one evaluation metric for the fixed baseline, direct wavelet fusion without prototype adaptation, semantic-only prototype adaptation, wavelet prototype adaptation without conservative update, and the full method.}
\label{fig:core-ablation}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/figure3_wavelet_design_ablation.pdf}
\caption{Wavelet design ablation. Cells show point changes over semantic-only prototype adaptation for pixel AUPRO and image AUROC on MVTec and VisA.}
\label{fig:wavelet-design-ablation}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure4_core_ablation_bar.pdf}
\caption{Component-level ablation gains over the fixed AnomalyCLIP baseline. Bars report point changes in Pixel AUPRO and Image AUROC on MVTec and VisA.}
\label{fig:core-ablation-bar}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure5_core_ablation_line.pdf}
\caption{Ordered component accumulation from the fixed baseline to semantic prototype adaptation, wavelet-guided prototype adaptation, and the full conservative update. Lines are used only for this ordered build-up path.}
\label{fig:core-ablation-line}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure6_wavelet_design_bar.pdf}
\caption{Wavelet design ablation as grouped gain bars over semantic-only prototype adaptation.}
\label{fig:wavelet-design-bar}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/figure7_main_gain_bar.pdf}
\caption{Dataset-level gains of the full method over the AnomalyCLIP baseline on the two headline metrics.}
\label{fig:main-gain-bar}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=.72\linewidth]{figures/supp_figure_medical_isbi.pdf}
\caption{Supplementary medical transfer result on ISIC/ISBI. Only current completed rows are plotted; blocked medical datasets are excluded.}
\label{fig:supp-medical-isbi}
\end{figure}
""",
        encoding="utf-8",
    )


def write_main_gain_csv(path: Path, records: list[MetricRecord]) -> None:
    by_key = pivot(records)
    datasets = ["MVTec", "VisA", "MPDD", "BTAD", "DTD-Synthetic"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", *[f"delta_{metric_key}" for metric_key, _ in METRICS]])
        for dataset in datasets:
            baseline = by_key[(dataset, "AnomalyCLIP baseline")]
            full = by_key[(dataset, "Full")]
            writer.writerow(
                [
                    dataset,
                    *[full.values[metric_key] - baseline.values[metric_key] for metric_key, _ in METRICS],
                ]
            )


def write_chinese_figure_guide() -> None:
    guide = OUT_DIR / "FIGURE_GUIDE_ZH.md"
    guide.write_text(
        """# 论文图件中文说明

本文件说明 `paper_output/generated_result_figures_20260717/figures` 与 `paper/figures` 中每张图的含义、适合支撑的论文论点，以及写作时的使用建议。所有图件都由 `scripts/analysis/generate_paper_result_figures.py` 从 `paper/result_record/result_table.csv` 和 `paper/result_record/medical_result_table.csv` 自动生成。

## 总体原则

- PDF 适合直接放入 LaTeX 正文；SVG 适合后期在矢量编辑器里微调；PNG 适合快速预览或放入演示文稿。
- `_gray.png` 是灰度预览，用于检查打印或色弱读者场景下是否仍能区分图形元素。
- 当前数据是数据集级汇总结果，不是多次运行或多 seed 的重复实验。因此图中不画误差棒，也不标显著性星号。
- `expected_pass` 和 `blocked` 行没有进入图件；医学补充图只使用已经完成的 ISIC/ISBI current 结果。
- 柱状图主要展示相对增益，而不是截断后的绝对分数，目的是避免把 80-97% 的小差异视觉夸大。

## 图件清单

### Figure 1: `figure1_main_results`

**文件**：`figure1_main_results.pdf/svg/png`

**图中信息**：展示五个异常检测数据集上 baseline 与 full method 的主结果。左侧 panel 用配对点/线展示 Pixel AUPRO 的直接提升；右侧 panel 用热力图展示四个指标上的 full - baseline 增益。

**能支撑什么论点**：适合放在实验主结果部分，用来说明完整方法在多个数据集、多个指标上整体优于 AnomalyCLIP baseline，尤其突出定位指标 Pixel AUPRO 的稳定提升。

**适合怎么写**：可以配合主结果表格使用。图负责让读者快速看到“提升是否跨数据集一致”，表格负责给出所有精确数值。

**注意事项**：这张图不适合单独替代表格，因为审稿人通常仍需要完整的 benchmark 数值。

### Figure 2: `figure2_core_ablation`

**文件**：`figure2_core_ablation.pdf/svg/png`

**图中信息**：在 MVTec 和 VisA 上，对 baseline、直接小波融合、语义原型适配、无保守更新的小波原型适配、完整方法进行四个指标的消融对比。

**能支撑什么论点**：适合放在核心消融实验部分，用来证明提升不是来自某一个随意后处理，而是来自“原型适配 + 小波引导 + 保守更新”的组合。

**适合怎么写**：强调 direct wavelet fusion 作为负控，说明仅把小波图直接融合到 anomaly map 并不足够；真正有效的是让小波可靠性参与 patch evidence / prototype adaptation。

**注意事项**：这张图使用散点而不是连线，因为这些方法不是全部都构成严格连续路径。

### Figure 3: `figure3_wavelet_design_ablation`

**文件**：`figure3_wavelet_design_ablation.pdf/svg/png`

**图中信息**：以 semantic-only prototype adaptation 为参照，展示不同小波设计在 MVTec 和 VisA 的 Pixel AUPRO、Image AUROC 上带来的增益或下降。

**能支撑什么论点**：适合说明不是“任何小波特征都有效”，而是 boundary-aware wavelet reliability 与 conservative update 更符合异常定位需求。

**适合怎么写**：重点解释 direct fusion 为负控，HF-only 只能提供有限或不稳定收益，boundary-aware 版本才更接近最终方法。

**注意事项**：热力图适合快速比较正负方向和幅度，但如果正文需要更传统的展示方式，可以改用 Figure 6 的柱状图。

### Figure 4: `figure4_core_ablation_bar`

**文件**：`figure4_core_ablation_bar.pdf/svg/png`

**图中信息**：以 fixed AnomalyCLIP baseline 为零点，使用分组柱状图展示核心组件对 Pixel AUPRO 和 Image AUROC 的点数增益。

**能支撑什么论点**：适合在正文或补充材料中更直观地展示“每个组件相对 baseline 增益多少”。它比绝对分数柱状图更适合强调消融贡献。

**适合怎么写**：可以用于回答审稿人常问的“每个模块到底贡献多少”。在文中可描述：语义原型适配带来主要增益，小波引导和保守更新进一步补充提升。

**注意事项**：Direct fusion 出现负值或较弱提升时，不应回避；这正好支撑“简单小波后融合不是有效机制”的论点。

### Figure 5: `figure5_core_ablation_line`

**文件**：`figure5_core_ablation_line.pdf/svg/png`

**图中信息**：只沿着有明确构建顺序的路径连线：Baseline -> Semantic adaptation -> WPTA no conservative -> Full WPTA。

**能支撑什么论点**：适合展示方法逐步构建时性能如何变化，让读者看到完整方法不是孤立跳点，而是沿着设计路径逐步增强。

**适合怎么写**：可以用于方法消融段落的第二张图或补充图，解释“先适配原型，再引入小波可靠性，最后用保守更新稳定结果”的递进逻辑。

**注意事项**：折线只用于有顺序关系的组件累加路径。Direct fusion 没有放入这张折线图，因为它是负控分支，不是同一条递进路径。

### Figure 6: `figure6_wavelet_design_bar`

**文件**：`figure6_wavelet_design_bar.pdf/svg/png`

**图中信息**：以 semantic-only prototype adaptation 为零点，用分组柱状图展示 direct fusion、HF-only、boundary-aware、full boundary-aware 四种小波设计的增益。

**能支撑什么论点**：适合更传统地展示小波设计消融，尤其适合正文空间不足但需要让读者直接读出每个设计增益的情况。

**适合怎么写**：可用来说明 boundary-aware 设计相比 HF-only 更稳，full 版本在两个数据集上都能带来最终增益。

**注意事项**：如果已经在正文使用 Figure 3 热力图，Figure 6 更适合放补充材料，避免正文重复。

### Figure 7: `figure7_main_gain_bar`

**文件**：`figure7_main_gain_bar.pdf/svg/png`

**图中信息**：按数据集排序展示 full method 相比 baseline 在 Pixel AUPRO 和 Image AUROC 上的增益。

**能支撑什么论点**：适合补充说明方法的收益主要体现在哪些数据集和指标上。它能帮助读者快速看出 BTAD、MPDD、VisA 等数据集上的提升幅度差异。

**适合怎么写**：可放在主结果后，作为“增益分布”图；也可放在 rebuttal 或补充材料中回应“提升是否只来自单个数据集”的问题。

**注意事项**：这张图只展示两个 headline metrics，不展示所有指标；完整指标仍应参考 Figure 1 或结果表。

### Supplementary Figure: `supp_figure_medical_isbi`

**文件**：`supp_figure_medical_isbi.pdf/svg/png`

**图中信息**：展示 ISIC/ISBI 医学迁移实验中 baseline 与 full method 在 Pixel AUROC 和 Pixel AUPRO 上的对比。

**能支撑什么论点**：适合放在补充实验，说明方法在医学异常/病灶分割风格数据上也有一定迁移趋势。

**适合怎么写**：建议谨慎表述为 supplementary evidence 或 preliminary medical transfer result，不要写成大规模医学泛化结论。

**注意事项**：ColonDB、ClinicDB、Kvasir 等行在源表中是 blocked，没有完整 current 结果，因此没有绘制。

## 推荐放置顺序

正文主线建议：

1. `figure1_main_results`：主结果总览。
2. `figure2_core_ablation` 或 `figure4_core_ablation_bar`：核心消融。若正文偏结果解释，用 Figure 4；若正文需要完整四指标，用 Figure 2。
3. `figure3_wavelet_design_ablation` 或 `figure6_wavelet_design_bar`：小波设计消融。热力图更紧凑，柱状图更传统。

补充材料建议：

1. `figure5_core_ablation_line`：组件递进趋势。
2. `figure7_main_gain_bar`：主结果增益排序。
3. `supp_figure_medical_isbi`：医学迁移补充实验。

## 复现命令

```bash
python3 scripts/analysis/generate_paper_result_figures.py
```

运行后会更新：

- `paper_output/generated_result_figures_20260717/`
- `paper/figures/`

""",
        encoding="utf-8",
    )


def sync_figures_to_paper_dir() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for source in FIG_DIR.iterdir():
        if source.is_file() and source.suffix.lower() in {".pdf", ".svg", ".png"}:
            shutil.copy2(source, PAPER_FIG_DIR / source.name)


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sections = parse_sections(RESULT_WORKBOOK)
    medical_sections = parse_sections(MEDICAL_WORKBOOK)

    main = main_results(sections)
    core = section_records(
        sections["Core Ablation"],
        [("MVTec", "mvtec_current"), ("VisA", "visa_current")],
    )
    wavelet = section_records(
        sections["Wavelet Design"],
        [("MVTec", "mvtec_current"), ("VisA", "visa_current")],
    )
    medical = medical_pixel_records(medical_sections)

    records_to_csv(DATA_DIR / "main_results_current.csv", main, [key for key, _ in METRICS])
    records_to_csv(DATA_DIR / "core_ablation_current.csv", core, [key for key, _ in METRICS])
    records_to_csv(DATA_DIR / "wavelet_design_current.csv", wavelet, [key for key, _ in METRICS])
    records_to_csv(DATA_DIR / "medical_pixel_current.csv", medical, ["pixel_auroc", "pixel_aupro"])
    write_main_gain_csv(DATA_DIR / "main_gain_vs_baseline.csv", main)
    write_delta_csv(
        DATA_DIR / "core_ablation_gain_vs_baseline.csv",
        core,
        CORE_METHODS[0],
        CORE_DELTA_METHODS,
        [key for key, _ in METRICS],
    )
    write_delta_csv(
        DATA_DIR / "wavelet_design_gain_vs_semantic.csv",
        wavelet,
        WAVELET_METHODS[0],
        WAVELET_DELTA_METHODS,
        [key for key, _ in METRICS],
    )

    figure_main_results(main)
    figure_core_ablation(core)
    figure_wavelet_design(wavelet)
    figure_core_ablation_bar(core)
    figure_core_ablation_line(core)
    figure_wavelet_design_bar(wavelet)
    figure_main_gain_bar(main)
    figure_medical(medical)
    write_latex_snippets()
    write_chinese_figure_guide()
    sync_figures_to_paper_dir()

    output_paths = sorted(str(path.relative_to(OUT_DIR)) for path in OUT_DIR.rglob("*") if path.is_file())
    write_manifest(output_paths)
    print(f"Wrote {len(output_paths)} files under {OUT_DIR}")


if __name__ == "__main__":
    main()
