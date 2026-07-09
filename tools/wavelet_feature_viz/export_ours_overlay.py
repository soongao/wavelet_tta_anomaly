#!/usr/bin/env python3
"""Export only our method's overlay heatmaps with dataset-native paths."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "anomalyclip_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "anomalyclip_cache"))

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

TOOLS_ROOT = REPO_ROOT / "tools" / "wavelet_feature_viz"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from anomalyclip.visualization import visualizer  # noqa: E402
from visualize_experiment_effect import (  # noqa: E402
    _as_nhw,
    _build_method_args,
    _build_multicrop_index,
    _evaluate_sample_light,
    _sample_cache_paths,
)


def _selected_classes(raw: Optional[Sequence[str]]) -> Optional[set[str]]:
    if raw is None:
        return None
    return {item for item in raw if item}


def _default_multicrop_cache(dataset: str) -> Path:
    return Path(f"cache/{dataset}_multicrop_maps_grid2_ratio075")


def _load_metadata(cache_dir: Path) -> dict:
    metadata_path = cache_dir / "metadata.pt"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    return torch.load(metadata_path, map_location="cpu")


def export_overlays(args: argparse.Namespace) -> None:
    metadata = _load_metadata(args.cache_dir)
    dataset = args.dataset or str(metadata.get("dataset", "unknown"))
    data_root = Path(args.data_root or metadata["data_path"]).expanduser()
    image_size = args.image_size or int(metadata["image_size"])

    method_args = _build_method_args(args)
    method_args.image_size = image_size
    text_features = metadata["text_features"].float()
    multicrop_index = (
        _build_multicrop_index(args.multicrop_cache_dir)
        if method_args.use_multicrop_fusion
        else None
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)
    class_filter = _selected_classes(args.classes)
    written_count = 0
    evaluated = 0

    for sample_path in _sample_cache_paths(args.cache_dir):
        sample = torch.load(sample_path, map_location="cpu")
        cls_name = sample["cls_name"]
        if class_filter is not None and cls_name not in class_filter:
            continue
        if args.anomalies_only and int(sample["anomaly"]) != 1:
            continue

        _, method_map = _evaluate_sample_light(
            sample,
            text_features,
            method_args,
            metadata,
            multicrop_index=multicrop_index,
        )
        map_np = _as_nhw(method_map)
        visualizer(
            [sample["img_path"]],
            map_np,
            image_size,
            str(args.save_dir),
            [cls_name],
            data_root=str(data_root),
            preserve_structure=True,
        )

        written_count += 1
        evaluated += 1
        if args.max_samples is not None and evaluated >= args.max_samples:
            break

    print(f"dataset: {dataset}")
    print(f"save_dir: {args.save_dir}")
    print(f"overlays: {written_count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Export our method overlay heatmaps")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--multicrop_cache_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--anomalies_only", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--method_preset", choices=["full_method", "wavelet_only"], default="full_method")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--image_size", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dataset is None:
        metadata = _load_metadata(args.cache_dir)
        args.dataset = str(metadata.get("dataset", "unknown"))
    if args.multicrop_cache_dir is None:
        args.multicrop_cache_dir = _default_multicrop_cache(args.dataset)
    export_overlays(args)


if __name__ == "__main__":
    main()
