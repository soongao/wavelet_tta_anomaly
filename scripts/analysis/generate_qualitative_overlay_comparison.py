#!/usr/bin/env python3
"""Generate a compact two-column qualitative overlay figure.

The figure uses only:
1. original image with the ground-truth mask overlaid;
2. original image with the full-method anomaly heatmap overlaid.

Selected examples are taken from the evidence-ranked candidate pool produced by
the existing wavelet feature visualization tools.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "anomalyclip_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "anomalyclip_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools" / "wavelet_feature_viz"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from anomalyclip.visualization import apply_ad_scoremap  # noqa: E402
from visualize_experiment_effect import (  # noqa: E402
    _as_nhw,
    _build_baseline_args,
    _build_method_args,
    _build_multicrop_index,
    _evaluate_sample_light,
    _localization_visibility_metrics,
    _sample_cache_paths,
    _safe_ap,
)


@dataclass(frozen=True)
class SelectedCandidate:
    dataset: str
    class_name: str
    rank: int
    image: str
    visible_score: float
    delta_iou: float
    method_iou: float
    method_pixel_ap: float
    source_dir: str


@dataclass
class RenderedExample:
    candidate: SelectedCandidate
    sample_path: Path
    image_path: str
    gt_overlay: np.ndarray
    anomaly_overlay: np.ndarray
    match_error: float
    visible_score: float
    delta_iou: float
    method_iou: float
    method_pixel_ap: float


def _read_selected_candidates(path: Path, max_examples: int) -> list[SelectedCandidate]:
    rows = json.loads(path.read_text())
    selected: list[SelectedCandidate] = []
    seen_classes: set[tuple[str, str]] = set()
    for row in rows:
        dataset = str(row["dataset"])
        class_name = str(row["class_name"])
        key = (dataset, class_name)
        if key in seen_classes:
            continue
        selected.append(
            SelectedCandidate(
                dataset=dataset,
                class_name=class_name,
                rank=int(row["rank"]),
                image=str(row["image"]),
                visible_score=float(row["visible_score"]),
                delta_iou=float(row["delta_iou"]),
                method_iou=float(row["method_iou"]),
                method_pixel_ap=float(row["method_pixel_ap"]),
                source_dir=str(row["source_dir"]),
            )
        )
        seen_classes.add(key)
        if len(selected) >= max_examples:
            break
    if not selected:
        raise RuntimeError(f"no selected candidates found in {path}")
    return selected


def _cache_dir_for_dataset(dataset: str) -> Path:
    mapping = {
        "mvtec": Path("cache/mvtec_anomalyclip_features"),
        "visa": Path("cache/visa_anomalyclip_features"),
        "btad": Path("cache/btad_anomalyclip_features"),
        "mpdd": Path("cache/mpdd_anomalyclip_features"),
        "dtd_synthetic": Path("cache/dtd_synthetic_anomalyclip_features"),
        "dtd": Path("cache/dtd_synthetic_anomalyclip_features"),
    }
    key = dataset.lower()
    if key not in mapping:
        raise ValueError(f"unsupported dataset in candidate pool: {dataset}")
    return mapping[key]


def _multicrop_cache_dir_for_dataset(dataset: str) -> Path:
    key = dataset.lower()
    if key == "dtd_synthetic" or key == "dtd":
        return Path("cache/dtd_synthetic_multicrop_maps_grid2_ratio075_no_stratified_woven127")
    return Path(f"cache/{key}_multicrop_maps_grid2_ratio075")


def _load_metadata(cache_dir: Path) -> dict:
    metadata_path = cache_dir / "metadata.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    return torch.load(metadata_path, map_location="cpu")


def _candidate_sample_paths(cache_dir: Path, candidate: SelectedCandidate) -> list[Path]:
    sample_paths: list[Path] = []
    for sample_path in _sample_cache_paths(cache_dir):
        sample = torch.load(sample_path, map_location="cpu")
        if sample["cls_name"] != candidate.class_name:
            continue
        if Path(sample["img_path"]).name != candidate.image:
            continue
        mask = _as_nhw(sample["img_mask"].float())[0]
        if int(sample["anomaly"]) != 1 or float(mask.max()) <= 0.0:
            continue
        sample_paths.append(sample_path)
    if not sample_paths:
        raise RuntimeError(
            "no matching anomalous cached sample for "
            f"{candidate.dataset}/{candidate.class_name}/{candidate.image}"
        )
    return sample_paths


def _normalize_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    lo = float(np.nanmin(score))
    hi = float(np.nanmax(score))
    if hi <= lo:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0)


def _read_image(path: str, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _overlay_gt(image: np.ndarray, mask: np.ndarray, alpha: float = 0.46) -> np.ndarray:
    mask = _normalize_score(mask) >= 0.5
    overlay = image.astype(np.float32).copy()
    color = np.array([0, 188, 166], dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _overlay_anomaly(image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    return apply_ad_scoremap(image, _normalize_score(anomaly_map), alpha=0.50)


def _score_match(
    candidate: SelectedCandidate,
    visible_score: float,
    delta_iou: float,
    method_iou: float,
    method_pixel_ap: float,
) -> float:
    return (
        abs(visible_score - candidate.visible_score)
        + 4.0 * abs(delta_iou - candidate.delta_iou)
        + 4.0 * abs(method_iou - candidate.method_iou)
        + abs(method_pixel_ap - candidate.method_pixel_ap)
    )


def _build_eval_args(args: argparse.Namespace, dataset: str, image_size: int) -> tuple[argparse.Namespace, argparse.Namespace]:
    eval_context = argparse.Namespace(
        feature_map_layer=list(args.feature_map_layer),
        sigma=float(args.sigma),
        save_dir=args.output_dir / "_eval_context",
        method_preset=args.method_preset,
    )
    baseline_args = _build_baseline_args(eval_context)
    method_args = _build_method_args(eval_context)
    baseline_args.image_size = image_size
    method_args.image_size = image_size
    return baseline_args, method_args


def _render_candidate(args: argparse.Namespace, candidate: SelectedCandidate) -> RenderedExample:
    cache_dir = _cache_dir_for_dataset(candidate.dataset)
    metadata = _load_metadata(cache_dir)
    image_size = int(args.image_size or metadata["image_size"])
    text_features = metadata["text_features"].float()
    baseline_args, method_args = _build_eval_args(args, candidate.dataset, image_size)
    multicrop_index = _build_multicrop_index(_multicrop_cache_dir_for_dataset(candidate.dataset))

    best: Optional[RenderedExample] = None
    for sample_path in _candidate_sample_paths(cache_dir, candidate):
        sample = torch.load(sample_path, map_location="cpu")
        _, baseline_map = _evaluate_sample_light(
            sample,
            text_features,
            baseline_args,
            metadata,
            multicrop_index=None,
        )
        _, method_map = _evaluate_sample_light(
            sample,
            text_features,
            method_args,
            metadata,
            multicrop_index=multicrop_index,
        )
        mask = _as_nhw(sample["img_mask"].float())[0]
        baseline_np = _as_nhw(baseline_map)[0]
        method_np = _as_nhw(method_map)[0]
        visibility = _localization_visibility_metrics(mask, baseline_np, method_np)
        method_pixel_ap = _safe_ap(mask, method_np) or 0.0
        match_error = _score_match(
            candidate,
            visible_score=float(visibility["visible_score"]),
            delta_iou=float(visibility["delta_iou"]),
            method_iou=float(visibility["method_iou"]),
            method_pixel_ap=float(method_pixel_ap),
        )
        image = _read_image(sample["img_path"], image_size)
        rendered = RenderedExample(
            candidate=candidate,
            sample_path=sample_path,
            image_path=str(sample["img_path"]),
            gt_overlay=_overlay_gt(image, mask),
            anomaly_overlay=_overlay_anomaly(image, method_np),
            match_error=match_error,
            visible_score=float(visibility["visible_score"]),
            delta_iou=float(visibility["delta_iou"]),
            method_iou=float(visibility["method_iou"]),
            method_pixel_ap=float(method_pixel_ap),
        )
        if best is None or rendered.match_error < best.match_error:
            best = rendered

    if best is None:
        raise RuntimeError(f"failed to render candidate: {candidate}")
    return best


def _row_label(example: RenderedExample) -> str:
    dataset = "VisA" if example.candidate.dataset.lower() == "visa" else "MVTec"
    return f"{dataset}\n{example.candidate.class_name}"


def _plot_examples(examples: list[RenderedExample], output_stem: Path) -> None:
    rows = len(examples)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(5.8, 2.85 * rows),
        dpi=300,
        squeeze=False,
        constrained_layout=False,
    )
    column_titles = ["Image + GT", "Image + anomaly heatmap"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10, pad=7)

    for row, example in enumerate(examples):
        panels = [example.gt_overlay, example.anomaly_overlay]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#2f3437")
        axes[row, 0].set_ylabel(
            _row_label(example),
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=28,
        )

    fig.subplots_adjust(left=0.16, right=0.995, top=0.965, bottom=0.02, wspace=0.035, hspace=0.10)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=300)
    plt.close(fig)


def _write_source_table(examples: list[RenderedExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "class",
        "rank",
        "image",
        "sample_path",
        "image_path",
        "match_error",
        "visible_score",
        "delta_iou",
        "method_iou",
        "method_pixel_ap",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "dataset": example.candidate.dataset,
                    "class": example.candidate.class_name,
                    "rank": example.candidate.rank,
                    "image": example.candidate.image,
                    "sample_path": str(example.sample_path),
                    "image_path": example.image_path,
                    "match_error": f"{example.match_error:.6f}",
                    "visible_score": f"{example.visible_score:.6f}",
                    "delta_iou": f"{example.delta_iou:.6f}",
                    "method_iou": f"{example.method_iou:.6f}",
                    "method_pixel_ap": f"{example.method_pixel_ap:.6f}",
                }
            )


def _write_latex_snippet(path: Path, figure_name: str) -> None:
    snippet = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/{figure_name}.pdf}}
\caption{{Qualitative localization examples. Each row shows the input image with the ground-truth anomaly mask overlaid and the same image with the full-method anomaly heatmap overlaid.}}
\label{{fig:qualitative-overlay-comparison}}
\end{{figure}}
"""
    path.write_text(snippet)


def _write_manifest(path: Path, examples: list[RenderedExample], figure_name: str) -> None:
    lines = [
        "# Qualitative Overlay Comparison",
        "",
        f"- Figure: `figures/{figure_name}.{{pdf,svg,png}}`",
        "- Left column: original image with the ground-truth mask overlaid.",
        "- Right column: original image with the full-method anomaly heatmap overlaid.",
        "- Candidate examples are selected from the evidence-ranked localization pool.",
        "- Source table: `source_data/qualitative_overlay_examples.csv`",
        "- LaTeX snippet: `latex_qualitative_overlay_snippet.tex`",
        "",
        "## Rendered Examples",
    ]
    for example in examples:
        lines.append(
            "- "
            f"{example.candidate.dataset}/{example.candidate.class_name}/{example.candidate.image}; "
            f"sample={example.sample_path}; "
            f"method_iou={example.method_iou:.3f}; "
            f"method_pixel_ap={example.method_pixel_ap:.3f}"
        )
    path.write_text("\n".join(lines) + "\n")


def _sync_to_paper_figures(output_figure_dir: Path, paper_figure_dir: Path, figure_name: str) -> None:
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        src = output_figure_dir / f"{figure_name}{suffix}"
        shutil.copy2(src, paper_figure_dir / src.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Generate two-column qualitative overlay comparison")
    parser.add_argument(
        "--selected_candidates",
        type=Path,
        default=Path("outputs/wavelet_feature_viz/paper_visualization_selection/selected_examples.json"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("paper_output/qualitative_overlay_comparison_20260717"),
    )
    parser.add_argument("--paper_figure_dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--figure_name", default="figure8_qualitative_overlay_comparison")
    parser.add_argument("--max_examples", type=int, default=4)
    parser.add_argument("--method_preset", choices=["full_method"], default="full_method")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--image_size", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    figure_dir = args.output_dir / "figures"
    source_dir = args.output_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    candidates = _read_selected_candidates(args.selected_candidates, args.max_examples)
    examples = [_render_candidate(args, candidate) for candidate in candidates]
    output_stem = figure_dir / args.figure_name
    _plot_examples(examples, output_stem)
    _write_source_table(examples, source_dir / "qualitative_overlay_examples.csv")
    _write_latex_snippet(args.output_dir / "latex_qualitative_overlay_snippet.tex", args.figure_name)
    _write_manifest(args.output_dir / "MANIFEST.md", examples, args.figure_name)
    _sync_to_paper_figures(figure_dir, args.paper_figure_dir, args.figure_name)

    print(f"figure: {output_stem}.{{pdf,svg,png}}")
    print(f"paper_copy: {args.paper_figure_dir / (args.figure_name + '.pdf')}")
    print(f"examples: {len(examples)}")
    for example in examples:
        print(
            f"- {example.candidate.dataset}/{example.candidate.class_name}/{example.candidate.image} "
            f"sample={example.sample_path.name} match_error={example.match_error:.4f}"
        )


if __name__ == "__main__":
    main()
