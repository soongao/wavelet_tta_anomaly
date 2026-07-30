#!/usr/bin/env python3
"""Generate paper-facing ablation tables and charts.

Source of truth:
- paper/tables/prototype_main_component_comparison.csv
- paper/tables/prototype_wavelet_effect_comparison.csv

The current paper-facing method name is Ours (unnamed). Old method names are
intentionally not used in generated tables or chart labels.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TABLE_SRC = REPO / "paper" / "tables"
TABLE_DIR = REPO / "paper" / "tables_ablation"
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "source_data"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("pixel_auroc", "Pixel AUROC"),
    ("pixel_aupro", "Pixel AUPRO"),
    ("image_auroc", "Image AUROC"),
    ("image_ap", "Image AP"),
]

DATASETS = ["MVTec", "VisA"]

CORE_SOURCE = TABLE_SRC / "prototype_main_component_comparison.csv"
WAVELET_SOURCE = TABLE_SRC / "prototype_wavelet_effect_comparison.csv"

CORE_METHODS = [
    "Baseline",
    "Direct wavelet fusion / no adaptation",
    "Semantic prototype adaptation",
    "Ours w/o conservative update",
    "Ours (unnamed)",
]
CORE_LABELS = {
    "Baseline": "Baseline",
    "Direct wavelet fusion / no adaptation": "Direct\nfusion",
    "Semantic prototype adaptation": "Semantic\nadapt.",
    "Ours w/o conservative update": "Ours\nw/o cons.",
    "Ours (unnamed)": "Ours",
}

CORE_PROGRESS_METHODS = [
    "Baseline",
    "Semantic prototype adaptation",
    "Ours w/o conservative update",
    "Ours (unnamed)",
]

WAVELET_METHODS = [
    "Direct wavelet fusion",
    "Semantic-only prototype adaptation",
    "HF-only reliability + prototype adaptation",
    "Boundary-aware reliability + prototype adaptation",
    "Ours (unnamed)",
]
WAVELET_LABELS = {
    "Direct wavelet fusion": "Direct\nfusion",
    "Semantic-only prototype adaptation": "Semantic\nonly",
    "HF-only reliability + prototype adaptation": "HF-only\nreliability",
    "Boundary-aware reliability + prototype adaptation": "Boundary-aware\nreliability",
    "Ours (unnamed)": "Ours",
}

WAVELET_PROGRESS_METHODS = [
    "Semantic-only prototype adaptation",
    "HF-only reliability + prototype adaptation",
    "Boundary-aware reliability + prototype adaptation",
    "Ours (unnamed)",
]

COLORS = {
    "Baseline": "#6B7280",
    "Direct wavelet fusion / no adaptation": "#C44E52",
    "Semantic prototype adaptation": "#4C78A8",
    "Ours w/o conservative update": "#F58518",
    "Ours (unnamed)": "#2A9D8F",
    "Direct wavelet fusion": "#C44E52",
    "Semantic-only prototype adaptation": "#4C78A8",
    "HF-only reliability + prototype adaptation": "#8172B2",
    "Boundary-aware reliability + prototype adaptation": "#F58518",
}

DATASET_COLORS = {"MVTec": "#4C78A8", "VisA": "#2A9D8F"}


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
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def parse_slash_metrics(value: str) -> dict[str, float]:
    values = [float(part.strip()) for part in value.split("/")]
    if len(values) != len(METRICS):
        raise ValueError(f"Expected {len(METRICS)} slash metrics, got {value!r}")
    return {metric: number for (metric, _), number in zip(METRICS, values)}


def load_ablation_csv(path: Path, method_column: str, method_order: list[str]) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            method = row[method_column].strip()
            for dataset, col in [("MVTec", "mvtec_current"), ("VisA", "visa_current")]:
                values = parse_slash_metrics(row[col])
                records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "notes": row.get("notes", "").strip(),
                        **values,
                    }
                )
    df = pd.DataFrame.from_records(records)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    return df.sort_values(["dataset", "method"]).reset_index(drop=True)


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt(
        id_vars=["dataset", "method", "notes"],
        value_vars=[key for key, _ in METRICS],
        var_name="metric",
        value_name="value",
    )


def write_source_data(core: pd.DataFrame, wavelet: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.to_csv(DATA_DIR / "controlled_core_ablation_wide.csv", index=False)
    wavelet.to_csv(DATA_DIR / "wavelet_design_ablation_wide.csv", index=False)
    to_long(core).to_csv(DATA_DIR / "controlled_core_ablation_long.csv", index=False)
    to_long(wavelet).to_csv(DATA_DIR / "wavelet_design_ablation_long.csv", index=False)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def format_value(value: float, best: float) -> str:
    rendered = f"{value:.1f}"
    if math.isclose(value, best, abs_tol=1e-9):
        return rf"\textbf{{{rendered}}}"
    return rendered


def write_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    best = {
        (dataset, metric): float(df[df["dataset"] == dataset][metric].max())
        for dataset in DATASETS
        for metric, _ in METRICS
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Variant & Pixel AUROC & Pixel AUPRO & Image AUROC & Image AP \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        if dataset_index:
            lines.append(r"\midrule")
        subset = df[df["dataset"] == dataset]
        for _, row in subset.iterrows():
            values = [
                format_value(float(row[metric]), best[(dataset, metric)])
                for metric, _ in METRICS
            ]
            lines.append(
                f"{latex_escape(str(row['dataset']))} & {latex_escape(str(row['method']))} "
                + "& "
                + " & ".join(values)
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_outputs(core: pd.DataFrame, wavelet: pd.DataFrame) -> None:
    write_latex_table(
        core,
        TABLE_DIR / "controlled_core_ablation.tex",
        (
            "Controlled core ablation on MVTec and VisA. "
            "All metrics are reported in percent; the best result within each dataset "
            "and metric is marked in bold."
        ),
        "tab:controlled-core-ablation",
    )
    write_latex_table(
        wavelet,
        TABLE_DIR / "wavelet_design_ablation.tex",
        (
            "Wavelet reliability design ablation on MVTec and VisA. "
            "All metrics are reported in percent; the best result within each dataset "
            "and metric is marked in bold."
        ),
        "tab:wavelet-design-ablation",
    )
    (TABLE_DIR / "all_ablation_tables.tex").write_text(
        "\n".join(
            [
                "% Current paper-facing ablation tables.",
                "% Generated from paper/figures_ablation/make_ablation_charts.py.",
                r"\input{paper/tables_ablation/controlled_core_ablation}",
                r"\input{paper/tables_ablation/wavelet_design_ablation}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)


def metric_value(df: pd.DataFrame, dataset: str, method: str, metric: str) -> float:
    row = df[(df["dataset"] == dataset) & (df["method"] == method)]
    if row.empty:
        return math.nan
    return float(row.iloc[0][metric])


def plot_absolute_bars(
    df: pd.DataFrame,
    methods: list[str],
    labels: dict[str, str],
    stem: str,
) -> None:
    fig, axes = plt.subplots(
        len(DATASETS),
        len(METRICS),
        figsize=(11.0, 5.2),
        sharex=True,
        sharey="row",
    )
    y = np.arange(len(methods))
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            vals = [metric_value(df, dataset, method, metric) for method in methods]
            ax.barh(y, vals, color=[COLORS[method] for method in methods], height=0.68)
            ax.set_xlim(75, 100)
            ax.grid(axis="x", color="#D5DCE5", alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_idx == 0:
                ax.set_title(metric_label)
            if col_idx == 0:
                ax.set_ylabel(dataset)
                ax.set_yticks(y, [labels[method].replace("\n", " ") for method in methods])
            else:
                ax.tick_params(axis="y", labelleft=False)
            for i, value in enumerate(vals):
                if math.isfinite(value):
                    ax.text(min(value + 0.35, 99.4), i, f"{value:.1f}", va="center", fontsize=6.3)
            if row_idx == len(DATASETS) - 1:
                ax.set_xlabel("Score (%)")
    axes[0, 0].invert_yaxis()
    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    save_all(fig, stem)
    plt.close(fig)


def plot_gain_bars(
    df: pd.DataFrame,
    reference_method: str,
    compare_methods: list[str],
    labels: dict[str, str],
    metrics: list[tuple[str, str]],
    stem: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(compare_methods))
    width = 0.34
    for ax, (metric, metric_label) in zip(axes, metrics):
        for dataset_idx, dataset in enumerate(DATASETS):
            ref = metric_value(df, dataset, reference_method, metric)
            deltas = [metric_value(df, dataset, method, metric) - ref for method in compare_methods]
            offset = (dataset_idx - 0.5) * width
            ax.bar(
                x + offset,
                deltas,
                width=width,
                color=DATASET_COLORS[dataset],
                label=dataset,
            )
            for xi, delta in zip(x + offset, deltas):
                va = "bottom" if delta >= 0 else "top"
                y_text = delta + (0.08 if delta >= 0 else -0.08)
                ax.text(xi, y_text, f"{delta:+.1f}", ha="center", va=va, fontsize=6.2)
        ax.axhline(0, color="#1F2933", linewidth=0.8)
        ax.set_title(metric_label)
        ax.set_xticks(x, [labels[method] for method in compare_methods])
        ax.grid(axis="y", color="#D5DCE5", alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Point change")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    save_all(fig, stem)
    plt.close(fig)


def plot_progression_lines(
    df: pd.DataFrame,
    methods: list[str],
    labels: dict[str, str],
    metrics: list[tuple[str, str]],
    stem: str,
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 3.7), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    x = np.arange(len(methods))
    for ax, (metric, metric_label) in zip(axes, metrics):
        for dataset in DATASETS:
            vals = [metric_value(df, dataset, method, metric) for method in methods]
            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=DATASET_COLORS[dataset],
                label=dataset,
            )
            for xi, value in zip(x, vals):
                ax.text(xi, value + 0.18, f"{value:.1f}", ha="center", va="bottom", fontsize=6.0)
        ax.set_title(metric_label)
        ax.set_xticks(x, [labels[method] for method in methods])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(78, 100)
        ax.grid(axis="y", color="#D5DCE5", alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    save_all(fig, stem)
    plt.close(fig)


def write_figure_snippets() -> None:
    snippets = [
        (
            "controlled_core_ablation_absolute_bars",
            "Controlled core ablation across all reported metrics.",
            "fig:controlled-core-ablation-bars",
            "figure*",
            r"\textwidth",
        ),
        (
            "controlled_core_ablation_gain_bars",
            "Controlled core ablation gains over the fixed baseline.",
            "fig:controlled-core-ablation-gains",
            "figure*",
            r"\textwidth",
        ),
        (
            "controlled_core_ablation_progression_lines",
            "Ordered controlled ablation path from baseline to Ours.",
            "fig:controlled-core-ablation-lines",
            "figure*",
            r"\textwidth",
        ),
        (
            "wavelet_design_ablation_absolute_bars",
            "Wavelet reliability design ablation across all reported metrics.",
            "fig:wavelet-design-ablation-bars",
            "figure*",
            r"\textwidth",
        ),
        (
            "wavelet_design_ablation_gain_bars",
            "Wavelet reliability design gains over semantic-only prototype adaptation.",
            "fig:wavelet-design-ablation-gains",
            "figure*",
            r"\textwidth",
        ),
        (
            "wavelet_design_ablation_progression_lines",
            "Wavelet reliability design progression over semantic-only prototype adaptation.",
            "fig:wavelet-design-ablation-lines",
            "figure*",
            r"\textwidth",
        ),
    ]
    lines = [
        "% Optional LaTeX snippets for current ablation figures.",
        "% Paths are relative to the repository root.",
        "",
    ]
    for stem, caption, label, env, width in snippets:
        lines.extend(
            [
                rf"\begin{{{env}}}[t]",
                r"\centering",
                rf"\includegraphics[width={width}]{{paper/figures_ablation/figures/{stem}.pdf}}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                rf"\end{{{env}}}",
                "",
            ]
        )
    (ROOT / "figure_snippets.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    core = load_ablation_csv(CORE_SOURCE, "method", CORE_METHODS)
    wavelet = load_ablation_csv(WAVELET_SOURCE, "wavelet_setting", WAVELET_METHODS)

    write_source_data(core, wavelet)
    write_latex_outputs(core, wavelet)

    plot_absolute_bars(
        core,
        CORE_METHODS,
        CORE_LABELS,
        "controlled_core_ablation_absolute_bars",
    )
    plot_gain_bars(
        core,
        "Baseline",
        CORE_METHODS[1:],
        CORE_LABELS,
        [("pixel_aupro", "Pixel AUPRO"), ("image_auroc", "Image AUROC")],
        "controlled_core_ablation_gain_bars",
    )
    plot_progression_lines(
        core,
        CORE_PROGRESS_METHODS,
        CORE_LABELS,
        [("pixel_aupro", "Pixel AUPRO"), ("image_auroc", "Image AUROC")],
        "controlled_core_ablation_progression_lines",
    )
    plot_absolute_bars(
        wavelet,
        WAVELET_METHODS,
        WAVELET_LABELS,
        "wavelet_design_ablation_absolute_bars",
    )
    plot_gain_bars(
        wavelet,
        "Semantic-only prototype adaptation",
        [
            "Direct wavelet fusion",
            "HF-only reliability + prototype adaptation",
            "Boundary-aware reliability + prototype adaptation",
            "Ours (unnamed)",
        ],
        WAVELET_LABELS,
        [("pixel_aupro", "Pixel AUPRO"), ("image_auroc", "Image AUROC")],
        "wavelet_design_ablation_gain_bars",
    )
    plot_progression_lines(
        wavelet,
        WAVELET_PROGRESS_METHODS,
        WAVELET_LABELS,
        [("pixel_aupro", "Pixel AUPRO"), ("image_auroc", "Image AUROC")],
        "wavelet_design_ablation_progression_lines",
    )
    write_figure_snippets()


if __name__ == "__main__":
    main()
