from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from anomalyclip.cached_eval_utils import (
    build_anomaly_maps_from_patch_features,
    init_results,
    sample_cache_paths,
    smooth_anomaly_map,
)
from anomalyclip.dataset import generate_class_info
from anomalyclip.metrics import image_level_metrics, pixel_level_metrics
from anomalyclip.prototype_adaptation import apply_wavelet_prototype_adaptation
from scripts.evaluate.eval_cached_calibration import (
    _build_multicrop_index,
    _evaluate_sample,
    build_parser as build_eval_parser,
)


FINAL_PROTO_DEFAULTS = {
    "proto_alpha0": 0.0,
    "proto_beta0": 0.01,
    "proto_update_min_abnormal_confidence": 0.06,
    "proto_wavelet_mode": "boundary_aware",
    "proto_wavelet_mix": 0.05,
}


METHOD_NAMES = [
    "baseline",
    "clip_only_proto",
    "direct_wavelet_fusion",
    "hf_only_proto",
    "boundary_aware_proto",
    "full_no_conservative",
    "full_conservative",
]


def _eval_defaults():
    return build_eval_parser().parse_args([])


def _set_common_args(args, cli_args, save_path):
    args.cache_dir = cli_args.cache_dir
    args.dataset = cli_args.dataset
    args.save_path = save_path
    args.metrics = "image-pixel-level"
    args.aupro_steps = cli_args.aupro_steps
    args.feature_map_layer = cli_args.feature_map_layer
    args.sigma = cli_args.sigma
    args.image_size = cli_args.image_size
    args.classes = cli_args.classes
    return args


def _make_eval_args(cli_args, method_name, metadata):
    args = _eval_defaults()
    _set_common_args(
        args,
        cli_args,
        save_path=os.path.join(cli_args.output_dir, "_internal", method_name),
    )
    args.image_size = args.image_size or metadata["image_size"]
    args.feature_map_layer = args.feature_map_layer or metadata["feature_map_layer"]

    if method_name in {
        "clip_only_proto",
        "hf_only_proto",
        "boundary_aware_proto",
        "full_conservative",
        "full_no_conservative",
    }:
        args.use_wavelet_prototype_adaptation = True
        args.proto_alpha0 = cli_args.proto_alpha0
        args.proto_beta0 = cli_args.proto_beta0
        args.proto_update_min_abnormal_confidence = cli_args.proto_update_min_abnormal_confidence
        args.proto_wavelet_mode = cli_args.proto_wavelet_mode
        args.proto_wavelet_mix = cli_args.proto_wavelet_mix
        args.proto_topk_ratio = cli_args.proto_topk_ratio
        args.proto_eta = cli_args.proto_eta
        args.proto_gamma = cli_args.proto_gamma
        args.proto_tau_a = cli_args.proto_tau_a
        args.proto_temperature = cli_args.proto_temperature
        args.proto_conservative_update = method_name == "full_conservative"
        if method_name == "clip_only_proto":
            args.proto_wavelet_mode = "none"
        elif method_name == "hf_only_proto":
            args.proto_wavelet_mode = "hf_only"
        elif method_name == "boundary_aware_proto":
            args.proto_wavelet_mode = "boundary_aware"
        elif method_name == "full_no_conservative":
            args.proto_conservative_update = False

    if method_name == "direct_wavelet_fusion":
        args.use_direct_wavelet_fusion = True
        args.proto_wavelet_mode = cli_args.proto_wavelet_mode
        args.direct_wavelet_fusion_weight = cli_args.direct_wavelet_fusion_weight
        args.proto_temperature = cli_args.proto_temperature
        args.proto_anchor_layers = cli_args.proto_anchor_layers

    if cli_args.use_multicrop_fusion:
        args.use_multicrop_fusion = True
        args.multicrop_cache_dir = cli_args.multicrop_cache_dir
        args.multicrop_weight = cli_args.multicrop_weight

    if getattr(cli_args, "use_pixel_to_image_fusion", False):
        args.use_pixel_to_image_fusion = True
        args.pixel_to_image_weight = cli_args.pixel_to_image_weight
        args.pixel_to_image_topk_ratio = cli_args.pixel_to_image_topk_ratio
        args.pixel_to_image_normalize = cli_args.pixel_to_image_normalize

    return args


def _selected_sample_paths(cli_args):
    paths = sample_cache_paths(cli_args.cache_dir)
    if len(paths) == 0:
        raise FileNotFoundError(f"no sample cache files found under {cli_args.cache_dir}/samples")
    if cli_args.limit_samples > 0:
        paths = paths[: cli_args.limit_samples]
    return paths


def _class_filter(dataset, classes):
    obj_list, _ = generate_class_info(dataset)
    if not classes:
        return set(obj_list)
    requested = set(classes)
    return {obj for obj in obj_list if obj in requested}


def _sample_threshold_pixels(map_tensor, max_pixels_per_map):
    flat = map_tensor.detach().cpu().float().flatten()
    if flat.numel() <= max_pixels_per_map:
        return flat
    step = int(math.ceil(flat.numel() / max_pixels_per_map))
    return flat[::step]


def _limit_threshold_pixels(pixel_tensor, max_pixels):
    if max_pixels <= 0 or pixel_tensor.numel() <= max_pixels:
        return pixel_tensor
    step = int(math.ceil(pixel_tensor.numel() / max_pixels))
    return pixel_tensor[::step]


def _topk_mean(map_tensor, ratio):
    flat = map_tensor.detach().cpu().float().flatten()
    k = max(1, int(math.ceil(flat.numel() * ratio)))
    return float(torch.topk(flat, k=k).values.mean())


def _evaluate_normal_maps(cli_args, method_name, metadata, sample_paths, class_names):
    eval_args = _make_eval_args(cli_args, method_name, metadata)
    text_features = metadata["text_features"].float()
    multicrop_index = (
        _build_multicrop_index(eval_args.multicrop_cache_dir)
        if eval_args.use_multicrop_fusion
        else None
    )
    rows = []
    maps = []
    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu")
        if sample["cls_name"] not in class_names or int(sample["anomaly"]) != 0:
            continue
        _, anomaly_map = _evaluate_sample(
            sample,
            text_features,
            eval_args,
            metadata,
            multicrop_index=multicrop_index,
        )
        anomaly_map = anomaly_map.detach().cpu().float()
        maps.append(anomaly_map)
        rows.append(
            {
                "class": sample["cls_name"],
                "path": sample["img_path"],
                "mean_score": float(anomaly_map.mean()),
                "top1pct_score": _topk_mean(anomaly_map, ratio=0.01),
                "map": anomaly_map,
            }
        )
    return rows, maps


def run_normal_stability(cli_args):
    os.makedirs(cli_args.output_dir, exist_ok=True)
    metadata = torch.load(os.path.join(cli_args.cache_dir, "metadata.pt"), map_location="cpu")
    if cli_args.image_size is None:
        cli_args.image_size = metadata["image_size"]
    if cli_args.feature_map_layer is None:
        cli_args.feature_map_layer = metadata["feature_map_layer"]

    sample_paths = _selected_sample_paths(cli_args)
    class_names = _class_filter(cli_args.dataset, cli_args.classes)
    methods = ["baseline", "full_no_conservative", "full_conservative"]
    method_rows = {}
    method_maps = {}

    for method_name in methods:
        rows, maps = _evaluate_normal_maps(
            cli_args,
            method_name,
            metadata,
            sample_paths,
            class_names,
        )
        method_rows[method_name] = rows
        method_maps[method_name] = maps
        if len(rows) == 0:
            raise RuntimeError(f"no normal samples evaluated for {method_name}")

    baseline_pixels = []
    for anomaly_map in method_maps["baseline"]:
        baseline_pixels.append(
            _sample_threshold_pixels(
                anomaly_map,
                max_pixels_per_map=cli_args.max_threshold_pixels_per_map,
            )
        )
    baseline_pixels = torch.cat(baseline_pixels)
    baseline_pixels = _limit_threshold_pixels(
        baseline_pixels,
        max_pixels=cli_args.max_threshold_pixels_total,
    )
    threshold_p95 = float(torch.quantile(baseline_pixels, 0.95))
    threshold_p99 = float(torch.quantile(baseline_pixels, 0.99))

    summary_rows = []
    per_class = defaultdict(list)
    for method_name, rows in method_rows.items():
        fp95 = []
        fp99 = []
        mean_scores = []
        top_scores = []
        for row in rows:
            anomaly_map = row["map"]
            row_fp95 = float((anomaly_map > threshold_p95).float().mean())
            row_fp99 = float((anomaly_map > threshold_p99).float().mean())
            fp95.append(row_fp95)
            fp99.append(row_fp99)
            mean_scores.append(row["mean_score"])
            top_scores.append(row["top1pct_score"])
            per_class[(method_name, row["class"])].append((row_fp95, row_fp99, row["mean_score"], row["top1pct_score"]))
        summary_rows.append(
            {
                "method": method_name,
                "normal_images": len(rows),
                "fp_area_p95": float(np.mean(fp95)),
                "fp_area_p99": float(np.mean(fp99)),
                "mean_score": float(np.mean(mean_scores)),
                "top1pct_score": float(np.mean(top_scores)),
            }
        )

    csv_path = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_normal_stability.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "normal_images",
                "fp_area_p95",
                "fp_area_p99",
                "mean_score",
                "top1pct_score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    class_csv_path = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_normal_stability_per_class.csv")
    with open(class_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "class",
                "normal_images",
                "fp_area_p95",
                "fp_area_p99",
                "mean_score",
                "top1pct_score",
            ],
        )
        writer.writeheader()
        for (method_name, cls_name), values in sorted(per_class.items()):
            arr = np.asarray(values, dtype=np.float64)
            writer.writerow(
                {
                    "method": method_name,
                    "class": cls_name,
                    "normal_images": len(values),
                    "fp_area_p95": float(arr[:, 0].mean()),
                    "fp_area_p99": float(arr[:, 1].mean()),
                    "mean_score": float(arr[:, 2].mean()),
                    "top1pct_score": float(arr[:, 3].mean()),
                }
            )

    md_path = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_normal_stability.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {cli_args.dataset} normal image stability\n\n")
        handle.write("Normal labels are used only for evaluation-time grouping, not for inference.\n\n")
        handle.write(f"- baseline normal p95 threshold: `{threshold_p95:.6f}`\n")
        handle.write(f"- baseline normal p99 threshold: `{threshold_p99:.6f}`\n\n")
        handle.write("| method | normal images | FP area @ baseline p95 | FP area @ baseline p99 | mean score | top 1% score |\n")
        handle.write("|:--|--:|--:|--:|--:|--:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} | {row['normal_images']} | "
                f"{100.0 * row['fp_area_p95']:.3f}% | {100.0 * row['fp_area_p99']:.3f}% | "
                f"{row['mean_score']:.6f} | {row['top1pct_score']:.6f} |\n"
            )
        handle.write("\n")
        handle.write(f"Per-class CSV: `{class_csv_path}`\n")

    print(f"normal_stability: {md_path}")
    print(f"normal_stability_csv: {csv_path}")
    return md_path


def _normalize_for_display(x):
    x = np.asarray(x, dtype=np.float32)
    low, high = np.percentile(x, [1, 99])
    if high <= low:
        return np.zeros_like(x)
    return np.clip((x - low) / (high - low), 0.0, 1.0)


def _tensor_image_to_numpy(sample):
    img_path = sample.get("img_path")
    if img_path and os.path.exists(img_path):
        image = Image.open(img_path).convert("RGB")
        return np.asarray(image.resize((sample["img_mask"].shape[-1], sample["img_mask"].shape[-2])))
    return np.zeros((sample["img_mask"].shape[-2], sample["img_mask"].shape[-1], 3), dtype=np.uint8)


def _upsample_patch_map(x, image_size, mode="bilinear"):
    tensor = x.detach().float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    return F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    ).squeeze().detach().cpu().numpy()


def _select_visualization_samples(cli_args, sample_paths):
    class_names = list(_class_filter(cli_args.dataset, cli_args.classes))
    selected = []
    seen = set()
    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu")
        cls_name = sample["cls_name"]
        if cls_name not in class_names or cls_name in seen or int(sample["anomaly"]) == 0:
            continue
        selected.append(sample_path)
        seen.add(cls_name)
        if len(selected) >= cli_args.samples_per_dataset:
            break
    if len(selected) == 0:
        raise RuntimeError("no anomalous samples selected for visualization")
    return selected


def run_mechanism_visualization(cli_args):
    os.makedirs(cli_args.output_dir, exist_ok=True)
    metadata = torch.load(os.path.join(cli_args.cache_dir, "metadata.pt"), map_location="cpu")
    if cli_args.image_size is None:
        cli_args.image_size = metadata["image_size"]
    if cli_args.feature_map_layer is None:
        cli_args.feature_map_layer = metadata["feature_map_layer"]

    sample_paths = _selected_sample_paths(cli_args)
    selected_paths = _select_visualization_samples(cli_args, sample_paths)
    text_features = metadata["text_features"].float()
    written = []

    for sample_path in selected_paths:
        sample = torch.load(sample_path, map_location="cpu")
        patch_features = [patch_feature.float() for patch_feature in sample["patch_features"]]
        baseline_map, selected_patch_features = build_anomaly_maps_from_patch_features(
            patch_features,
            text_features,
            cli_args.feature_map_layer,
            cli_args.image_size,
            layer_weighting="sum",
        )
        baseline_map = smooth_anomaly_map(baseline_map, sigma=cli_args.sigma)[0].numpy()
        final_map, _, diagnostics = apply_wavelet_prototype_adaptation(
            selected_patch_features,
            text_features,
            image_size=cli_args.image_size,
            temperature=cli_args.proto_temperature,
            gamma=cli_args.proto_gamma,
            eta=cli_args.proto_eta,
            topk_ratio=cli_args.proto_topk_ratio,
            alpha0=cli_args.proto_alpha0,
            beta0=cli_args.proto_beta0,
            tau_a=cli_args.proto_tau_a,
            update_min_abnormal_confidence=cli_args.proto_update_min_abnormal_confidence,
            wavelet_mix=cli_args.proto_wavelet_mix,
            wavelet_mode=cli_args.proto_wavelet_mode,
            conservative_update=True,
        )
        final_map = smooth_anomaly_map(final_map, sigma=cli_args.sigma)[0].numpy()
        image_np = _tensor_image_to_numpy(sample)
        gt_mask = sample["img_mask"].detach().cpu().float().squeeze().numpy()
        w_map = _upsample_patch_map(diagnostics["w"][0], cli_args.image_size)
        abnormal_mask = _upsample_patch_map(
            diagnostics["abnormal_mask"][0].float(),
            cli_args.image_size,
            mode="nearest",
        )
        normal_mask = _upsample_patch_map(
            diagnostics["normal_mask"][0].float(),
            cli_args.image_size,
            mode="nearest",
        )

        panels = [
            ("image", image_np, None),
            ("gt mask", gt_mask, "gray"),
            ("baseline map", _normalize_for_display(baseline_map), "magma"),
            ("wavelet W", _normalize_for_display(w_map), "viridis"),
            ("abnormal patches", abnormal_mask, "Reds"),
            ("normal patches", normal_mask, "Blues"),
            ("final map", _normalize_for_display(final_map), "magma"),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
        for ax, (title, data, cmap) in zip(axes, panels):
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if cmap is None:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
        fig.suptitle(
            f"{cli_args.dataset}/{sample['cls_name']} - {Path(sample['img_path']).name}",
            fontsize=10,
        )
        fig.tight_layout()
        out_name = f"{cli_args.dataset}_{sample['cls_name']}_{Path(sample_path).stem}.png"
        out_path = os.path.join(cli_args.output_dir, out_name)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        written.append(out_path)

    index_path = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_mechanism_visualizations.md")
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {cli_args.dataset} mechanism visualizations\n\n")
        for path in written:
            handle.write(f"- `{path}`\n")
    print(f"mechanism_visualizations: {index_path}")
    return index_path


def _safe_percent_metric(results, obj_name, metric_name, aupro_steps):
    metric_results = results
    if metric_name.startswith("pixel-"):
        metric_results = {
            obj_name: {
                **results[obj_name],
                "imgs_masks": torch.cat(results[obj_name]["imgs_masks"]),
                "anomaly_maps": torch.cat(results[obj_name]["anomaly_maps"]).detach().cpu().numpy(),
            }
        }
    try:
        if metric_name.startswith("image-"):
            value = image_level_metrics(metric_results, obj_name, metric_name)
        else:
            value = pixel_level_metrics(metric_results, obj_name, metric_name, aupro_steps=aupro_steps)
    except ValueError:
        return float("nan")
    return 100.0 * float(value)


def _summarize_results(results, obj_list, aupro_steps):
    rows = []
    for obj_name in obj_list:
        if len(results[obj_name]["gt_sp"]) == 0:
            continue
        rows.append(
            {
                "class": obj_name,
                "pixel_auroc": _safe_percent_metric(results, obj_name, "pixel-auroc", aupro_steps),
                "pixel_aupro": _safe_percent_metric(results, obj_name, "pixel-aupro", aupro_steps),
                "image_auroc": _safe_percent_metric(results, obj_name, "image-auroc", aupro_steps),
                "image_ap": _safe_percent_metric(results, obj_name, "image-ap", aupro_steps),
            }
        )
    metric_names = ["pixel_auroc", "pixel_aupro", "image_auroc", "image_ap"]
    mean_row = {"class": "mean"}
    for metric_name in metric_names:
        values = np.asarray([row[metric_name] for row in rows], dtype=np.float64)
        mean_row[metric_name] = float(np.nanmean(values))
    return rows, mean_row


def run_metrics_runtime(cli_args):
    os.makedirs(cli_args.output_dir, exist_ok=True)
    metadata = torch.load(os.path.join(cli_args.cache_dir, "metadata.pt"), map_location="cpu")
    if cli_args.image_size is None:
        cli_args.image_size = metadata["image_size"]
    if cli_args.feature_map_layer is None:
        cli_args.feature_map_layer = metadata["feature_map_layer"]

    all_obj_list, _ = generate_class_info(cli_args.dataset)
    class_names = _class_filter(cli_args.dataset, cli_args.classes)
    obj_list = [obj_name for obj_name in all_obj_list if obj_name in class_names]
    sample_paths = _selected_sample_paths(cli_args)
    text_features = metadata["text_features"].float()
    summary_rows = []
    per_class_rows = []

    for method_name in cli_args.methods:
        eval_args = _make_eval_args(cli_args, method_name, metadata)
        multicrop_index = (
            _build_multicrop_index(eval_args.multicrop_cache_dir)
            if eval_args.use_multicrop_fusion
            else None
        )
        results = init_results(obj_list)
        sample_times = []
        evaluated = 0

        for sample_path in sample_paths:
            sample = torch.load(sample_path, map_location="cpu")
            cls_name = sample["cls_name"]
            if cls_name not in results:
                continue
            start = time.perf_counter()
            text_prob, anomaly_map = _evaluate_sample(
                sample,
                text_features,
                eval_args,
                metadata,
                multicrop_index=multicrop_index,
            )
            elapsed = time.perf_counter() - start
            if evaluated >= cli_args.runtime_warmup:
                sample_times.append(elapsed)
            results[cls_name]["imgs_masks"].append(sample["img_mask"].float())
            results[cls_name]["gt_sp"].append(int(sample["anomaly"]))
            results[cls_name]["pr_sp"].extend(text_prob.detach().cpu())
            results[cls_name]["anomaly_maps"].append(anomaly_map)
            evaluated += 1

        class_rows, mean_row = _summarize_results(results, obj_list, cli_args.aupro_steps)
        runtime_mean = float(np.mean(sample_times)) if len(sample_times) > 0 else float("nan")
        runtime_std = float(np.std(sample_times)) if len(sample_times) > 0 else float("nan")
        summary_row = {
            "dataset": cli_args.dataset,
            "method": method_name,
            "evaluated": evaluated,
            "runtime_mean_sec": runtime_mean,
            "runtime_std_sec": runtime_std,
            **mean_row,
        }
        summary_rows.append(summary_row)
        for row in class_rows:
            per_class_rows.append({"dataset": cli_args.dataset, "method": method_name, **row})

    summary_csv = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_metrics_runtime.csv")
    fieldnames = [
        "dataset",
        "method",
        "evaluated",
        "pixel_auroc",
        "pixel_aupro",
        "image_auroc",
        "image_ap",
        "runtime_mean_sec",
        "runtime_std_sec",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    per_class_csv = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_metrics_runtime_per_class.csv")
    class_fieldnames = [
        "dataset",
        "method",
        "class",
        "pixel_auroc",
        "pixel_aupro",
        "image_auroc",
        "image_ap",
    ]
    with open(per_class_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=class_fieldnames)
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow({name: row.get(name) for name in class_fieldnames})

    md_path = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_metrics_runtime.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {cli_args.dataset} metrics and runtime\n\n")
        handle.write("Metrics keep the existing evaluation protocol: pixel AUROC, pixel AUPRO(PRO), image AUROC, and image AP.\n\n")
        handle.write("| method | samples | pixel AUROC | pixel AUPRO | image AUROC | image AP | sec/image |\n")
        handle.write("|:--|--:|--:|--:|--:|--:|--:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['method']} | {row['evaluated']} | "
                f"{row['pixel_auroc']:.4f} | {row['pixel_aupro']:.4f} | "
                f"{row['image_auroc']:.4f} | {row['image_ap']:.4f} | "
                f"{row['runtime_mean_sec']:.6f} |\n"
            )
        handle.write("\n")
        handle.write(f"Summary CSV: `{summary_csv}`\n")
        handle.write(f"Per-class CSV: `{per_class_csv}`\n")

    print(f"metrics_runtime: {md_path}")
    print(f"metrics_runtime_csv: {summary_csv}")
    return md_path


def run_runtime_only(cli_args):
    os.makedirs(cli_args.output_dir, exist_ok=True)
    metadata = torch.load(os.path.join(cli_args.cache_dir, "metadata.pt"), map_location="cpu")
    if cli_args.image_size is None:
        cli_args.image_size = metadata["image_size"]
    if cli_args.feature_map_layer is None:
        cli_args.feature_map_layer = metadata["feature_map_layer"]

    sample_paths = _selected_sample_paths(cli_args)
    class_names = _class_filter(cli_args.dataset, cli_args.classes)
    text_features = metadata["text_features"].float()
    rows = []
    for method_name in cli_args.methods:
        eval_args = _make_eval_args(cli_args, method_name, metadata)
        multicrop_index = (
            _build_multicrop_index(eval_args.multicrop_cache_dir)
            if eval_args.use_multicrop_fusion
            else None
        )
        sample_times = []
        evaluated = 0
        for sample_path in sample_paths:
            sample = torch.load(sample_path, map_location="cpu")
            if sample["cls_name"] not in class_names:
                continue
            start = time.perf_counter()
            _evaluate_sample(
                sample,
                text_features,
                eval_args,
                metadata,
                multicrop_index=multicrop_index,
            )
            elapsed = time.perf_counter() - start
            if evaluated >= cli_args.runtime_warmup:
                sample_times.append(elapsed)
            evaluated += 1
        rows.append(
            {
                "dataset": cli_args.dataset,
                "method": method_name,
                "evaluated": evaluated,
                "runtime_mean_sec": float(np.mean(sample_times)) if sample_times else float("nan"),
                "runtime_std_sec": float(np.std(sample_times)) if sample_times else float("nan"),
            }
        )

    csv_path = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_runtime.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "method", "evaluated", "runtime_mean_sec", "runtime_std_sec"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(cli_args.output_dir, f"{cli_args.dataset}_runtime.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {cli_args.dataset} runtime\n\n")
        handle.write("Runtime is measured around the cached inference path per image. It excludes model feature extraction because both baseline and prototype runs reuse the same cached AnomalyCLIP features.\n\n")
        handle.write("| method | samples | sec/image | std sec |\n")
        handle.write("|:--|--:|--:|--:|\n")
        for row in rows:
            handle.write(
                f"| {row['method']} | {row['evaluated']} | "
                f"{row['runtime_mean_sec']:.6f} | {row['runtime_std_sec']:.6f} |\n"
            )
        handle.write("\nThe extra cost of the full method comes from one Haar DWT on the patch grid, patch-evidence prototype construction, and one recalculation of patch logits with calibrated prototypes.\n")
        handle.write(f"\nCSV: `{csv_path}`\n")
    print(f"runtime: {md_path}")
    print(f"runtime_csv: {csv_path}")
    return md_path


def build_parser():
    parser = argparse.ArgumentParser("Prototype adaptation validation helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser):
        subparser.add_argument("--cache_dir", required=True)
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--output_dir", required=True)
        subparser.add_argument("--classes", nargs="+", default=None)
        subparser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
        subparser.add_argument("--image_size", type=int, default=None)
        subparser.add_argument("--sigma", type=float, default=5.0)
        subparser.add_argument("--aupro_steps", type=int, default=200)
        subparser.add_argument("--limit_samples", type=int, default=0)
        subparser.add_argument("--proto_temperature", type=float, default=0.07)
        subparser.add_argument("--proto_gamma", type=float, default=1.0)
        subparser.add_argument("--proto_eta", type=float, default=1.0)
        subparser.add_argument("--proto_topk_ratio", type=float, default=0.20)
        subparser.add_argument("--proto_alpha0", type=float, default=FINAL_PROTO_DEFAULTS["proto_alpha0"])
        subparser.add_argument("--proto_beta0", type=float, default=FINAL_PROTO_DEFAULTS["proto_beta0"])
        subparser.add_argument("--proto_tau_a", type=float, default=0.15)
        subparser.add_argument(
            "--proto_update_min_abnormal_confidence",
            type=float,
            default=FINAL_PROTO_DEFAULTS["proto_update_min_abnormal_confidence"],
        )
        subparser.add_argument(
            "--proto_wavelet_mode",
            default=FINAL_PROTO_DEFAULTS["proto_wavelet_mode"],
            choices=["none", "hf_only", "boundary_aware"],
        )
        subparser.add_argument("--proto_wavelet_mix", type=float, default=FINAL_PROTO_DEFAULTS["proto_wavelet_mix"])
        subparser.add_argument("--proto_anchor_layers", type=str, default="last", choices=["last", "mean"])
        subparser.add_argument("--proto_layer_fusion", type=str, default="sum", choices=["sum", "mean"])
        subparser.add_argument("--proto_percentile_low", type=float, default=1.0)
        subparser.add_argument("--proto_percentile_high", type=float, default=99.0)
        subparser.add_argument("--direct_wavelet_fusion_weight", type=float, default=0.5)

    normal_parser = subparsers.add_parser("normal-stability")
    add_common(normal_parser)
    normal_parser.add_argument("--max_threshold_pixels_per_map", type=int, default=25000)
    normal_parser.add_argument("--max_threshold_pixels_total", type=int, default=5000000)
    normal_parser.add_argument("--use_multicrop_fusion", action="store_true")
    normal_parser.add_argument("--multicrop_cache_dir", default=None)
    normal_parser.add_argument("--multicrop_weight", type=float, default=0.5)

    visual_parser = subparsers.add_parser("mechanism-viz")
    add_common(visual_parser)
    visual_parser.add_argument("--samples_per_dataset", type=int, default=3)

    metrics_parser = subparsers.add_parser("metrics-runtime")
    add_common(metrics_parser)
    metrics_parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES, default=["baseline", "full_conservative"])
    metrics_parser.add_argument("--runtime_warmup", type=int, default=5)
    metrics_parser.add_argument("--use_multicrop_fusion", action="store_true")
    metrics_parser.add_argument("--multicrop_cache_dir", default=None)
    metrics_parser.add_argument("--multicrop_weight", type=float, default=0.5)
    metrics_parser.add_argument("--use_pixel_to_image_fusion", action="store_true")
    metrics_parser.add_argument("--pixel_to_image_weight", type=float, default=0.1)
    metrics_parser.add_argument("--pixel_to_image_topk_ratio", type=float, default=0.01)
    metrics_parser.add_argument("--pixel_to_image_normalize", action="store_true")

    runtime_parser = subparsers.add_parser("runtime")
    add_common(runtime_parser)
    runtime_parser.add_argument("--methods", nargs="+", choices=METHOD_NAMES, default=["baseline", "full_conservative"])
    runtime_parser.add_argument("--runtime_warmup", type=int, default=5)
    runtime_parser.add_argument("--use_multicrop_fusion", action="store_true")
    runtime_parser.add_argument("--multicrop_cache_dir", default=None)
    runtime_parser.add_argument("--multicrop_weight", type=float, default=0.5)
    runtime_parser.add_argument("--use_pixel_to_image_fusion", action="store_true")
    runtime_parser.add_argument("--pixel_to_image_weight", type=float, default=0.1)
    runtime_parser.add_argument("--pixel_to_image_topk_ratio", type=float, default=0.01)
    runtime_parser.add_argument("--pixel_to_image_normalize", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "normal-stability":
        run_normal_stability(args)
    elif args.command == "mechanism-viz":
        run_mechanism_visualization(args)
    elif args.command == "metrics-runtime":
        run_metrics_runtime(args)
    elif args.command == "runtime":
        run_runtime_only(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
