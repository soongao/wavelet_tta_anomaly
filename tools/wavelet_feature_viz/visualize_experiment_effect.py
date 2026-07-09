#!/usr/bin/env python3
"""Visualize the class where an experiment improves pixel localization most.

This tool uses the experiment logs to select a class, then recomputes baseline
and method anomaly maps with the same cached-evaluation settings. It is meant
for evidence-driven visualization: pick the class from measured AUPRO deltas,
then show score distributions and qualitative maps for that class.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "anomalyclip_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "anomalyclip_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import find_objects, gaussian_filter, label as connected_label
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anomalyclip.visualization import apply_ad_scoremap  # noqa: E402
from anomalyclip.test_time_rectification import rectify_text_features_with_multi_layer_anchors  # noqa: E402
from anomalyclip.wavelet_calibration import (  # noqa: E402
    apply_structure_texture_calibration,
    compute_texture_reliability,
    fuse_image_score_with_pixel_score,
    global_image_confidence_gate,
    topk_pixel_score,
)


METRIC_NAMES = ("pixel_auroc", "pixel_aupro", "image_auroc", "image_ap")


def _cal_pro_score(masks: np.ndarray, amaps: np.ndarray, max_step: int = 200, expect_fpr: float = 0.3) -> float:
    masks = np.asarray(masks).astype(bool)
    amaps = np.asarray(amaps)
    min_th, max_th = amaps.min(), amaps.max()
    if max_th <= min_th:
        return float("nan")

    delta = (max_th - min_th) / max_step
    thresholds = np.arange(min_th, max_th, delta)
    region_scores = []
    for mask, amap in zip(masks, amaps):
        label_img, num_regions = connected_label(mask)
        if num_regions <= 0:
            continue
        for region_slice in find_objects(label_img):
            if region_slice is None:
                continue
            region_mask = label_img[region_slice] > 0
            region_scores.append(amap[region_slice][region_mask])
    if not region_scores:
        return float("nan")

    background_scores = amaps[~masks]
    if background_scores.size == 0:
        return float("nan")

    pros, fprs = [], []
    for threshold in thresholds:
        pros.append(np.mean([(scores > threshold).sum() / scores.size for scores in region_scores]))
        fprs.append((background_scores > threshold).sum() / background_scores.size)
    pros = np.asarray(pros)
    fprs = np.asarray(fprs)
    keep = fprs < expect_fpr
    if keep.sum() < 2:
        return float("nan")
    fprs = fprs[keep]
    if fprs.max() <= fprs.min():
        return float("nan")
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    kept_pros = pros[keep]
    order = np.argsort(fprs)
    fprs = fprs[order]
    kept_pros = kept_pros[order]
    return float(np.sum((fprs[1:] - fprs[:-1]) * (kept_pros[1:] + kept_pros[:-1]) * 0.5))


def _sample_cache_paths(cache_dir: Path) -> List[Path]:
    sample_dir = cache_dir / "samples"
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample directory not found: {sample_dir}")
    return sorted(sample_dir.glob("*.pt"))


def _build_multicrop_index(multicrop_cache_dir: Path) -> Dict[str, torch.Tensor]:
    index = {}
    for sample_path in _sample_cache_paths(multicrop_cache_dir):
        sample = torch.load(sample_path, map_location="cpu")
        index[sample["img_path"]] = sample["stitched_crop_map"].float()
    if not index:
        raise FileNotFoundError(f"no multi-crop samples found under {multicrop_cache_dir}")
    return index


def _fuse_multicrop_map(anomaly_map: torch.Tensor, crop_map: torch.Tensor, weight: float) -> torch.Tensor:
    anomaly_map = anomaly_map.float()
    crop_map = crop_map.float()
    if crop_map.dim() == 2:
        crop_map = crop_map.unsqueeze(0)
    if crop_map.shape[-2:] != anomaly_map.shape[-2:]:
        crop_map = F.interpolate(
            crop_map.unsqueeze(1),
            size=anomaly_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    weight = min(float(weight), 1.0)
    return ((1.0 - weight) * anomaly_map + weight * crop_map).clamp_min(0.0)


def _compute_image_text_prob(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)
    text_probs = image_features @ text_features.unsqueeze(0).permute(0, 2, 1)
    return (text_probs / temperature).softmax(-1)[:, 0, 1]


def _similarity_map_from_patch_features(
    patch_feature: torch.Tensor,
    text_features: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    patch_feature = F.normalize(patch_feature.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    batch, num_tokens, channels = patch_feature.shape
    spatial = patch_feature[:, 1:, :]
    similarity = (
        spatial.reshape(batch, spatial.size(1), 1, channels)
        * text_features.reshape(1, 1, text_features.size(0), channels)
    ).sum(-1)
    similarity = (similarity / 0.07).softmax(-1)
    side = int(spatial.size(1) ** 0.5)
    similarity = similarity.reshape(batch, side, side, -1).permute(0, 3, 1, 2)
    similarity = F.interpolate(similarity, image_size, mode="bilinear", align_corners=False)
    similarity = similarity.permute(0, 2, 3, 1)
    return (similarity[..., 1] + 1 - similarity[..., 0]) / 2.0


def _build_anomaly_map_from_patch_features(
    patch_features: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    feature_map_layer: Sequence[int],
    image_size: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    maps = []
    selected_patch_features = []
    first_selected_layer = feature_map_layer[0] if feature_map_layer else 0
    for idx, patch_feature in enumerate(patch_features):
        if idx >= first_selected_layer:
            maps.append(_similarity_map_from_patch_features(patch_feature, text_features, image_size))
            selected_patch_features.append(patch_feature.float())
    if not maps:
        raise ValueError("No patch feature layer was selected.")
    return torch.stack(maps).sum(dim=0), selected_patch_features


def _smooth_anomaly_map(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
    anomaly_map = anomaly_map.detach().cpu().float()
    if sigma <= 0:
        return anomaly_map
    return torch.stack(
        [torch.from_numpy(gaussian_filter(image.numpy(), sigma=sigma)) for image in anomaly_map],
        dim=0,
    )


def _evaluate_sample_light(
    sample: dict,
    text_features: torch.Tensor,
    args: argparse.Namespace,
    metadata: dict,
    multicrop_index: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_features = sample["image_features"].float()
    text_features_for_map = text_features
    patch_features = [patch_feature.float() for patch_feature in sample["patch_features"]]

    need_recompute = args.use_tta_rectification or list(args.feature_map_layer) != list(
        metadata["feature_map_layer"]
    )
    if need_recompute:
        anomaly_map, selected_patch_features = _build_anomaly_map_from_patch_features(
            patch_features,
            text_features_for_map,
            args.feature_map_layer,
            args.image_size,
        )
        texture_gate = sample["texture_gate"].float()
    else:
        anomaly_map = sample["base_anomaly_map"].float()
        selected_patch_features = None
        texture_gate = sample["texture_gate"].float()

    reliability = compute_texture_reliability(
        anomaly_map,
        texture_gate,
        topk_ratio=args.wavelet_reliability_topk_ratio,
    )

    if args.use_tta_rectification:
        rectified_text_features, _ = rectify_text_features_with_multi_layer_anchors(
            text_features_for_map,
            selected_patch_features,
            anomaly_map,
            wavelet_gate=texture_gate,
            reliability=reliability if not args.disable_adaptive_wavelet else None,
            mode=args.tta_mode,
            alpha=args.tta_alpha,
            topk_ratio=args.tta_topk_ratio,
            update_abnormal=args.tta_update_abnormal,
            min_confidence=args.tta_min_confidence,
            min_confidence_margin=args.tta_min_confidence_margin,
            repulsion_weight=args.tta_repulsion_weight,
            abnormal_alpha_scale=args.tta_abnormal_alpha_scale,
            fusion=args.tta_anchor_layers,
        )
        text_features_for_map = rectified_text_features[0]
        anomaly_map, selected_patch_features = _build_anomaly_map_from_patch_features(
            patch_features,
            text_features_for_map,
            args.feature_map_layer,
            args.image_size,
        )
        text_prob = _compute_image_text_prob(image_features, text_features_for_map)
    else:
        text_prob = sample.get("text_prob")
        if text_prob is None:
            text_prob = _compute_image_text_prob(image_features, text_features_for_map)
        else:
            text_prob = text_prob.float()

    if args.use_wavelet:
        image_confidence_gate = None
        if args.use_image_to_pixel_gate:
            image_confidence_gate = global_image_confidence_gate(
                text_prob,
                power=args.image_to_pixel_power,
                min_gate=args.image_to_pixel_min_gate,
                max_gate=args.image_to_pixel_max_gate,
            )
        anomaly_map, _ = apply_structure_texture_calibration(
            anomaly_map,
            texture_gate,
            beta=args.wavelet_beta,
            condition_power=args.wavelet_condition_power,
            suppress_beta=args.wavelet_suppress_beta,
            texture_max_delta_ratio=args.texture_max_delta_ratio,
            texture_suppression_weight=args.texture_suppression_weight,
            texture_local_contrast_kernel=args.texture_local_contrast_kernel,
            texture_local_contrast_weight=args.texture_local_contrast_weight,
            rank_preserve_topk_ratio=args.rank_preserve_topk_ratio,
            rank_gate_mode=args.rank_gate_mode,
            rank_gate_temperature=args.rank_gate_temperature,
            use_wavelet_confidence=args.use_wavelet_confidence,
            wavelet_confidence_power=args.wavelet_confidence_power,
            adaptive=not args.disable_adaptive_wavelet,
            reliability=reliability,
            reliability_power=args.wavelet_reliability_power,
            reliability_topk_ratio=args.wavelet_reliability_topk_ratio,
            min_reliability=args.wavelet_min_reliability,
            texture_delta_reliability_power=args.texture_delta_reliability_power,
            image_confidence_gate=image_confidence_gate,
            image_confidence_weight=args.image_to_pixel_weight,
        )

    if args.use_multicrop_fusion:
        if multicrop_index is None:
            raise ValueError("Multi-crop fusion needs a multicrop index.")
        crop_map = multicrop_index.get(sample["img_path"])
        if crop_map is None:
            if args.multicrop_missing_policy == "error":
                raise KeyError(f"missing multi-crop map for {sample['img_path']}")
        else:
            anomaly_map = _fuse_multicrop_map(anomaly_map, crop_map, weight=args.multicrop_weight)

    anomaly_map = _smooth_anomaly_map(anomaly_map, sigma=args.sigma)
    if args.use_pixel_to_image_fusion:
        pixel_score = topk_pixel_score(
            anomaly_map,
            topk_ratio=args.pixel_to_image_topk_ratio,
            normalize=args.pixel_to_image_normalize,
        )
        text_prob = fuse_image_score_with_pixel_score(
            text_prob,
            pixel_score,
            weight=args.pixel_to_image_weight,
        )

    return text_prob.detach().cpu(), anomaly_map


def _parse_metric_table(path: Path) -> Dict[str, Tuple[float, float, float, float]]:
    metrics: Dict[str, Tuple[float, float, float, float]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|") or line.startswith("|:") or "objects" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] == "mean":
            continue
        try:
            metrics[cells[0]] = tuple(float(value) for value in cells[1:])  # type: ignore[assignment]
        except ValueError:
            continue
    if not metrics:
        raise ValueError(f"no metric table rows found in {path}")
    return metrics


def _rank_classes(
    baseline_metrics: Dict[str, Tuple[float, float, float, float]],
    method_metrics: Dict[str, Tuple[float, float, float, float]],
) -> List[dict]:
    rows = []
    for cls_name, base_values in baseline_metrics.items():
        if cls_name not in method_metrics:
            continue
        method_values = method_metrics[cls_name]
        delta = tuple(method_values[i] - base_values[i] for i in range(4))
        rows.append(
            {
                "class": cls_name,
                "baseline": dict(zip(METRIC_NAMES, base_values)),
                "method": dict(zip(METRIC_NAMES, method_values)),
                "delta": dict(zip(METRIC_NAMES, delta)),
            }
        )
    rows.sort(
        key=lambda row: (
            row["delta"]["pixel_aupro"],
            row["delta"]["pixel_auroc"],
            row["delta"]["image_auroc"],
        ),
        reverse=True,
    )
    if not rows:
        raise ValueError("baseline and method logs have no shared class rows")
    return rows


def _write_ranking(rows: Sequence[dict], save_path: Path) -> None:
    lines = [
        "class\t"
        "delta_pixel_auroc\tdelta_pixel_aupro\tdelta_image_auroc\tdelta_image_ap\t"
        "baseline_pixel_aupro\tmethod_pixel_aupro"
    ]
    for row in rows:
        lines.append(
            "{class_name}\t{d_auc:+.1f}\t{d_aup:+.1f}\t{d_img:+.1f}\t{d_ap:+.1f}\t"
            "{base_aup:.1f}\t{method_aup:.1f}".format(
                class_name=row["class"],
                d_auc=row["delta"]["pixel_auroc"],
                d_aup=row["delta"]["pixel_aupro"],
                d_img=row["delta"]["image_auroc"],
                d_ap=row["delta"]["image_ap"],
                base_aup=row["baseline"]["pixel_aupro"],
                method_aup=row["method"]["pixel_aupro"],
            )
        )
    save_path.write_text("\n".join(lines) + "\n")


def _build_baseline_args(args: argparse.Namespace) -> argparse.Namespace:
    eval_args = argparse.Namespace()
    eval_args.feature_map_layer = list(args.feature_map_layer)
    eval_args.sigma = args.sigma
    eval_args.image_size = None
    eval_args.use_wavelet = False
    eval_args.use_wavelet_confidence = False
    eval_args.use_tta_rectification = False
    eval_args.use_multicrop_fusion = False
    eval_args.use_pixel_to_image_fusion = False
    eval_args.use_image_to_pixel_gate = False
    eval_args.disable_adaptive_wavelet = False
    eval_args.wavelet_reliability_topk_ratio = 0.05
    eval_args.wavelet_reliability_power = 1.0
    eval_args.wavelet_min_reliability = 0.0
    eval_args.wavelet_beta = 0.5
    eval_args.wavelet_condition_power = 1.0
    eval_args.wavelet_suppress_beta = 0.3
    eval_args.texture_max_delta_ratio = 0.05
    eval_args.texture_suppression_weight = 0.0
    eval_args.texture_local_contrast_kernel = 0
    eval_args.texture_local_contrast_weight = 0.0
    eval_args.texture_delta_reliability_power = 0.0
    eval_args.rank_preserve_topk_ratio = 0.0
    eval_args.rank_gate_mode = "hard"
    eval_args.rank_gate_temperature = 0.05
    eval_args.wavelet_confidence_power = 1.0
    eval_args.tta_mode = "legacy"
    eval_args.tta_alpha = 0.2
    eval_args.tta_topk_ratio = 0.05
    eval_args.tta_update_abnormal = False
    eval_args.tta_min_confidence = 0.0
    eval_args.tta_min_confidence_margin = 0.0
    eval_args.tta_repulsion_weight = 0.25
    eval_args.tta_abnormal_alpha_scale = 1.0
    eval_args.tta_anchor_layers = "last"
    eval_args.multicrop_weight = 0.25
    eval_args.multicrop_missing_policy = "error"
    eval_args.image_to_pixel_power = 1.0
    eval_args.image_to_pixel_min_gate = 0.0
    eval_args.image_to_pixel_max_gate = 1.0
    eval_args.image_to_pixel_weight = 0.0
    eval_args.pixel_to_image_weight = 0.0
    eval_args.pixel_to_image_topk_ratio = 0.01
    eval_args.pixel_to_image_normalize = False
    return eval_args


def _build_method_args(args: argparse.Namespace) -> argparse.Namespace:
    eval_args = _build_baseline_args(args)
    eval_args.save_path = str(args.save_dir / "_method_eval_context")

    if args.method_preset == "wavelet_only":
        eval_args.use_wavelet = True
        eval_args.wavelet_beta = 0.2
        eval_args.wavelet_condition_power = 2.0
        eval_args.wavelet_suppress_beta = 0.0
        eval_args.texture_max_delta_ratio = 0.05
        eval_args.texture_suppression_weight = 0.0
        eval_args.texture_local_contrast_kernel = 17
        eval_args.texture_local_contrast_weight = 0.5
        eval_args.rank_preserve_topk_ratio = 0.35
        eval_args.rank_gate_mode = "hard"
        eval_args.rank_gate_temperature = 0.05
        eval_args.use_wavelet_confidence = True
        eval_args.wavelet_confidence_power = 1.0
        return eval_args

    if args.method_preset != "full_method":
        raise ValueError(f"unsupported method preset: {args.method_preset}")

    eval_args.use_wavelet = True
    eval_args.wavelet_beta = 0.2
    eval_args.wavelet_condition_power = 2.0
    eval_args.wavelet_suppress_beta = 0.0
    eval_args.texture_max_delta_ratio = 0.05
    eval_args.texture_suppression_weight = 0.0
    eval_args.texture_local_contrast_kernel = 17
    eval_args.texture_local_contrast_weight = 0.5
    eval_args.rank_preserve_topk_ratio = 0.35
    eval_args.rank_gate_mode = "hard"
    eval_args.rank_gate_temperature = 0.05
    eval_args.use_wavelet_confidence = True
    eval_args.wavelet_confidence_power = 1.0

    eval_args.use_tta_rectification = True
    eval_args.tta_mode = "wavelet_guided"
    eval_args.tta_alpha = 0.01
    eval_args.tta_topk_ratio = 0.02
    eval_args.tta_min_confidence = 0.2
    eval_args.tta_anchor_layers = "mean"
    eval_args.tta_repulsion_weight = 0.1
    eval_args.tta_abnormal_alpha_scale = 0.75

    eval_args.use_multicrop_fusion = True
    eval_args.multicrop_weight = 0.5
    eval_args.multicrop_missing_policy = "error"

    eval_args.use_pixel_to_image_fusion = True
    eval_args.pixel_to_image_weight = 0.1
    eval_args.pixel_to_image_topk_ratio = 0.01
    return eval_args


def _safe_roc_auc(gt: np.ndarray, score: np.ndarray) -> Optional[float]:
    if np.unique(gt).size < 2:
        return None
    return float(roc_auc_score(gt.ravel(), score.ravel()))


def _safe_ap(gt: np.ndarray, score: np.ndarray) -> Optional[float]:
    if np.unique(gt).size < 2:
        return None
    return float(average_precision_score(gt.ravel(), score.ravel()))


def _top_area_prediction(score: np.ndarray, mask: np.ndarray, area_scale: float = 1.0) -> np.ndarray:
    mask_bool = mask >= 0.5
    positive_area = int(mask_bool.sum())
    if positive_area <= 0:
        positive_area = max(1, int(round(score.size * 0.01)))
    k = max(1, min(score.size, int(round(positive_area * area_scale))))
    flat = score.reshape(-1)
    selected = np.argpartition(flat, -k)[-k:]
    pred = np.zeros_like(flat, dtype=bool)
    pred[selected] = True
    return pred.reshape(score.shape)


def _binary_overlap(mask: np.ndarray, pred: np.ndarray) -> dict:
    mask_bool = mask >= 0.5
    pred_bool = pred.astype(bool)
    tp = int((mask_bool & pred_bool).sum())
    fp = int((~mask_bool & pred_bool).sum())
    fn = int((mask_bool & ~pred_bool).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def _prediction_error_rgb(mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mask_bool = mask >= 0.5
    pred_bool = pred.astype(bool)
    rgb = np.full((*mask.shape, 3), 245, dtype=np.uint8)
    rgb[mask_bool & pred_bool] = np.array([46, 160, 67], dtype=np.uint8)  # TP
    rgb[~mask_bool & pred_bool] = np.array([218, 54, 51], dtype=np.uint8)  # FP
    rgb[mask_bool & ~pred_bool] = np.array([49, 104, 176], dtype=np.uint8)  # FN
    return rgb


def _localization_visibility_metrics(mask: np.ndarray, baseline_map: np.ndarray, method_map: np.ndarray) -> dict:
    mask_bool = mask >= 0.5
    normal = ~mask_bool
    baseline_pred = _top_area_prediction(baseline_map, mask)
    method_pred = _top_area_prediction(method_map, mask)
    baseline_overlap = _binary_overlap(mask, baseline_pred)
    method_overlap = _binary_overlap(mask, method_pred)
    baseline_norm = _normalize(baseline_map)
    method_norm = _normalize(method_map)
    baseline_contrast = float(baseline_norm[mask_bool].mean() - baseline_norm[normal].mean())
    method_contrast = float(method_norm[mask_bool].mean() - method_norm[normal].mean())
    background_drop = float(baseline_norm[normal].mean() - method_norm[normal].mean())
    anomaly_gain = float(method_norm[mask_bool].mean() - baseline_norm[mask_bool].mean())
    delta_iou = method_overlap["iou"] - baseline_overlap["iou"]
    delta_f1 = method_overlap["f1"] - baseline_overlap["f1"]
    visible_score = (
        3.0 * delta_iou
        + 1.5 * delta_f1
        + 0.5 * (method_contrast - baseline_contrast)
        + 0.25 * background_drop
        + 0.25 * anomaly_gain
    )
    return {
        "baseline_pred": baseline_pred,
        "method_pred": method_pred,
        "baseline_iou": baseline_overlap["iou"],
        "method_iou": method_overlap["iou"],
        "delta_iou": float(delta_iou),
        "baseline_f1": baseline_overlap["f1"],
        "method_f1": method_overlap["f1"],
        "delta_f1": float(delta_f1),
        "baseline_precision": baseline_overlap["precision"],
        "method_precision": method_overlap["precision"],
        "baseline_recall": baseline_overlap["recall"],
        "method_recall": method_overlap["recall"],
        "baseline_contrast": baseline_contrast,
        "method_contrast": method_contrast,
        "delta_contrast": float(method_contrast - baseline_contrast),
        "background_drop": background_drop,
        "anomaly_gain": anomaly_gain,
        "visible_score": float(visible_score),
    }


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


def _normalize_with_range(arr: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    arr = arr.astype(np.float32)
    denom = max(float(max_value) - float(min_value), 1e-6)
    return np.clip((arr - float(min_value)) / denom, 0.0, 1.0)


def _overlay_scoremap(image: np.ndarray, scoremap: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    mask = _normalize_with_range(scoremap, min_value, max_value)
    return apply_ad_scoremap(image, mask)


def _as_nhw(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float().numpy()
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None]
    return arr


def _evaluate_selected_class(
    args: argparse.Namespace,
    class_name: str,
    baseline_args: argparse.Namespace,
    method_args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    metadata = torch.load(args.cache_dir / "metadata.pt", map_location="cpu")
    baseline_args.image_size = baseline_args.image_size or metadata["image_size"]
    method_args.image_size = method_args.image_size or metadata["image_size"]
    text_features = metadata["text_features"].float()
    multicrop_index = (
        _build_multicrop_index(args.multicrop_cache_dir)
        if method_args.use_multicrop_fusion
        else None
    )

    records = []
    for sample_path in _sample_cache_paths(args.cache_dir):
        sample = torch.load(sample_path, map_location="cpu")
        if sample["cls_name"] != class_name:
            continue

        baseline_prob, baseline_map = _evaluate_sample_light(
            sample,
            text_features,
            baseline_args,
            metadata,
            multicrop_index=None,
        )
        method_prob, method_map = _evaluate_sample_light(
            sample,
            text_features,
            method_args,
            metadata,
            multicrop_index=multicrop_index,
        )

        mask = sample["img_mask"].float()
        mask_np = _as_nhw(mask)[0]
        base_np = _as_nhw(baseline_map)[0]
        method_np = _as_nhw(method_map)[0]
        if int(sample["anomaly"]) == 1 and mask_np.max() > 0:
            visibility = _localization_visibility_metrics(mask_np, base_np, method_np)
            baseline_pred = visibility["baseline_pred"]
            method_pred = visibility["method_pred"]
        else:
            visibility = {
                "baseline_iou": 0.0,
                "method_iou": 0.0,
                "delta_iou": 0.0,
                "baseline_f1": 0.0,
                "method_f1": 0.0,
                "delta_f1": 0.0,
                "baseline_precision": 0.0,
                "method_precision": 0.0,
                "baseline_recall": 0.0,
                "method_recall": 0.0,
                "baseline_contrast": 0.0,
                "method_contrast": 0.0,
                "delta_contrast": 0.0,
                "background_drop": 0.0,
                "anomaly_gain": 0.0,
                "visible_score": 0.0,
            }
            baseline_pred = np.zeros_like(mask_np, dtype=bool)
            method_pred = np.zeros_like(mask_np, dtype=bool)
        records.append(
            {
                "sample_path": sample_path,
                "img_path": sample["img_path"],
                "anomaly": int(sample["anomaly"]),
                "mask": mask_np,
                "baseline_map": base_np,
                "method_map": method_np,
                "baseline_prob": float(baseline_prob.view(-1)[0].item()),
                "method_prob": float(method_prob.view(-1)[0].item()),
                "baseline_pixel_auroc": _safe_roc_auc(mask_np, base_np),
                "method_pixel_auroc": _safe_roc_auc(mask_np, method_np),
                "baseline_pixel_ap": _safe_ap(mask_np, base_np),
                "method_pixel_ap": _safe_ap(mask_np, method_np),
                "visibility": {
                    key: value
                    for key, value in visibility.items()
                    if key not in {"baseline_pred", "method_pred"}
                },
                "baseline_pred": baseline_pred,
                "method_pred": method_pred,
            }
        )
        if args.max_samples is not None and len(records) >= args.max_samples:
            break

    if not records:
        raise RuntimeError(f"no cached samples found for class {class_name}")

    masks = np.stack([record["mask"] for record in records], axis=0)
    baseline_maps = np.stack([record["baseline_map"] for record in records], axis=0)
    method_maps = np.stack([record["method_map"] for record in records], axis=0)
    image_gt = np.asarray([record["anomaly"] for record in records], dtype=np.int64)
    baseline_prob = np.asarray([record["baseline_prob"] for record in records], dtype=np.float32)
    method_prob = np.asarray([record["method_prob"] for record in records], dtype=np.float32)

    flat_gt = masks.reshape(-1)
    metrics = {
        "num_samples": len(records),
        "num_anomaly_samples": int(image_gt.sum()),
        "pixel_positive_ratio": float(flat_gt.mean()),
        "baseline_pixel_auroc": _safe_roc_auc(flat_gt, baseline_maps.reshape(-1)),
        "method_pixel_auroc": _safe_roc_auc(flat_gt, method_maps.reshape(-1)),
        "baseline_pixel_ap": _safe_ap(flat_gt, baseline_maps.reshape(-1)),
        "method_pixel_ap": _safe_ap(flat_gt, method_maps.reshape(-1)),
        "baseline_pixel_aupro": float(_cal_pro_score(masks, baseline_maps, max_step=args.aupro_steps)),
        "method_pixel_aupro": float(_cal_pro_score(masks, method_maps, max_step=args.aupro_steps)),
        "baseline_image_auroc": _safe_roc_auc(image_gt, baseline_prob),
        "method_image_auroc": _safe_roc_auc(image_gt, method_prob),
        "baseline_image_ap": _safe_ap(image_gt, baseline_prob),
        "method_image_ap": _safe_ap(image_gt, method_prob),
    }
    for metric in (
        "pixel_auroc",
        "pixel_ap",
        "pixel_aupro",
        "image_auroc",
        "image_ap",
    ):
        base_key = f"baseline_{metric}"
        method_key = f"method_{metric}"
        if metrics[base_key] is not None and metrics[method_key] is not None:
            metrics[f"delta_{metric}"] = float(metrics[method_key] - metrics[base_key])
        else:
            metrics[f"delta_{metric}"] = None

    return records, metrics


def _plot_metric_comparison(selected_row: dict, recomputed: dict, save_path: Path) -> None:
    labels = ["pixel AUROC", "pixel AUPRO", "image AUROC", "image AP"]
    baseline_log = [
        selected_row["baseline"]["pixel_auroc"],
        selected_row["baseline"]["pixel_aupro"],
        selected_row["baseline"]["image_auroc"],
        selected_row["baseline"]["image_ap"],
    ]
    method_log = [
        selected_row["method"]["pixel_auroc"],
        selected_row["method"]["pixel_aupro"],
        selected_row["method"]["image_auroc"],
        selected_row["method"]["image_ap"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    width = 0.34
    ax.bar(x - width / 2, baseline_log, width, color="#2f6fbb", label="baseline log")
    ax.bar(x + width / 2, method_log, width, color="#d64b3c", label="method log")
    for idx, (base_value, method_value) in enumerate(zip(baseline_log, method_log)):
        ax.text(
            idx,
            max(base_value, method_value) + 1.0,
            f"{method_value - base_value:+.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(f"Experiment metrics: {selected_row['class']}")
    ax.set_ylabel("score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")

    text = (
        f"recomputed pixel AUPRO: "
        f"{100 * recomputed['baseline_pixel_aupro']:.1f} -> "
        f"{100 * recomputed['method_pixel_aupro']:.1f}"
    )
    fig.text(0.02, 0.02, text, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(save_path)
    plt.close(fig)


def _plot_pixel_distribution(records: Sequence[dict], save_path: Path, seed: int, max_pixels: int) -> None:
    rng = np.random.default_rng(seed)
    masks = np.concatenate([record["mask"].reshape(-1) for record in records])
    baseline = np.concatenate([record["baseline_map"].reshape(-1) for record in records])
    method = np.concatenate([record["method_map"].reshape(-1) for record in records])
    if masks.size > max_pixels:
        idx = rng.choice(masks.size, size=max_pixels, replace=False)
        masks = masks[idx]
        baseline = baseline[idx]
        method = method[idx]

    normal = masks < 0.5
    anomaly = masks >= 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160, sharey=True)
    for ax, title, scores in [
        (axes[0], "Baseline pixel scores", baseline),
        (axes[1], "Method pixel scores", method),
    ]:
        bins = np.linspace(float(scores.min()), float(scores.max()), 80)
        if normal.any():
            ax.hist(scores[normal], bins=bins, density=True, alpha=0.6, color="#2f6fbb", label="normal")
        if anomaly.any():
            ax.hist(scores[anomaly], bins=bins, density=True, alpha=0.6, color="#d64b3c", label="anomaly")
        ax.set_title(title)
        ax.set_xlabel("pixel anomaly score")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _read_image(path: str, image_size: int) -> Optional[np.ndarray]:
    try:
        image = Image.open(path).convert("RGB").resize((image_size, image_size))
    except Exception:
        return None
    return np.asarray(image)


def _plot_examples(records: Sequence[dict], save_dir: Path, image_size: int, top_k: int) -> List[str]:
    out_dir = save_dir / "visible_localization_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    anomaly_records = [record for record in records if record["anomaly"] == 1 and record["mask"].max() > 0]
    anomaly_records.sort(
        key=lambda record: (
            record["visibility"]["visible_score"],
            record["visibility"]["delta_iou"],
            record["visibility"]["delta_contrast"],
            record["method_pixel_ap"] or 0.0,
        ),
        reverse=True,
    )
    (save_dir / "visible_examples.tsv").write_text(
        "\n".join(
            [
                "rank\timage\tvisible_score\tdelta_iou\tbaseline_iou\tmethod_iou\t"
                "delta_f1\tbaseline_f1\tmethod_f1\tdelta_contrast\tbackground_drop\tanomaly_gain\t"
                "baseline_pixel_ap\tmethod_pixel_ap"
            ]
            + [
                (
                    f"{idx}\t{Path(record['img_path']).name}\t"
                    f"{record['visibility']['visible_score']:.6f}\t"
                    f"{record['visibility']['delta_iou']:.6f}\t"
                    f"{record['visibility']['baseline_iou']:.6f}\t"
                    f"{record['visibility']['method_iou']:.6f}\t"
                    f"{record['visibility']['delta_f1']:.6f}\t"
                    f"{record['visibility']['baseline_f1']:.6f}\t"
                    f"{record['visibility']['method_f1']:.6f}\t"
                    f"{record['visibility']['delta_contrast']:.6f}\t"
                    f"{record['visibility']['background_drop']:.6f}\t"
                    f"{record['visibility']['anomaly_gain']:.6f}\t"
                    f"{record['baseline_pixel_ap'] or 0.0:.6f}\t"
                    f"{record['method_pixel_ap'] or 0.0:.6f}"
                )
                for idx, record in enumerate(anomaly_records)
            ]
        )
        + "\n"
    )
    written: List[str] = []
    for idx, record in enumerate(anomaly_records[:top_k]):
        image = _read_image(record["img_path"], image_size)
        mask = record["mask"]
        baseline_map = record["baseline_map"]
        method_map = record["method_map"]
        score_min = float(min(baseline_map.min(), method_map.min()))
        score_max = float(max(baseline_map.max(), method_map.max()))

        panels = []
        if image is not None:
            panels.append(("Image", image, {}))
            panels.append(("Baseline heatmap", _overlay_scoremap(image, baseline_map, score_min, score_max), {}))
            panels.append(("Ours heatmap", _overlay_scoremap(image, method_map, score_min, score_max), {}))
        else:
            heatmap_kwargs = {"cmap": "magma", "vmin": score_min, "vmax": score_max}
            panels.append(("Baseline heatmap", baseline_map, heatmap_kwargs))
            panels.append(("Ours heatmap", method_map, heatmap_kwargs))
        panels.insert(1 if image is not None else 0, ("GT", mask, {"cmap": "gray"}))

        cols = len(panels)
        fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.2), dpi=160)
        for ax, (title_text, arr, kwargs) in zip(axes, panels):
            ax.imshow(arr, **kwargs)
            ax.set_title(title_text)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        base_ap = record["baseline_pixel_ap"]
        method_ap = record["method_pixel_ap"]
        title = Path(record["img_path"]).name
        if base_ap is not None and method_ap is not None:
            title += f" | pixel AP {100 * base_ap:.1f}->{100 * method_ap:.1f}"
        title += (
            f" | IoU {record['visibility']['baseline_iou']:.2f}->"
            f"{record['visibility']['method_iou']:.2f}"
        )
        fig.suptitle(title)
        fig.tight_layout()
        out_path = out_dir / f"{idx:02d}_{Path(record['img_path']).stem}.png"
        fig.savefig(out_path)
        plt.close(fig)
        written.append(str(out_path))
    return written


def _write_summary(
    args: argparse.Namespace,
    selected_row: dict,
    ranking: Sequence[dict],
    recomputed_metrics: dict,
    example_paths: Sequence[str],
) -> None:
    payload = {
        "selected_class": selected_row["class"],
        "selection_rule": "largest positive pixel AUPRO delta in method log vs baseline log",
        "method_preset": args.method_preset,
        "baseline_log": str(args.baseline_log),
        "method_log": str(args.method_log),
        "selected_log_metrics": selected_row,
        "top_ranked_classes": list(ranking[:10]),
        "recomputed_metrics": recomputed_metrics,
        "example_paths": list(example_paths),
    }
    (args.save_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Visualize experiment-backed wavelet effects")
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/mvtec_anomalyclip_features"))
    parser.add_argument(
        "--multicrop_cache_dir",
        type=Path,
        default=Path("cache/mvtec_multicrop_maps_grid2_ratio075"),
    )
    parser.add_argument(
        "--baseline_log",
        type=Path,
        default=Path("ablation_results/20260622_094146_component/mvtec/01_cached_baseline_l123/log.txt"),
    )
    parser.add_argument(
        "--method_log",
        type=Path,
        default=Path("ablation_results/20260622_094146_component/mvtec/07_full_method/log.txt"),
    )
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/wavelet_feature_viz/mvtec_experiment_effect"))
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--class_name", default=None)
    parser.add_argument("--method_preset", choices=["full_method", "wavelet_only"], default="full_method")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--aupro_steps", type=int, default=200)
    parser.add_argument("--top_k_examples", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--max_pixels", type=int, default=500_000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=111)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = _parse_metric_table(args.baseline_log)
    method_metrics = _parse_metric_table(args.method_log)
    ranking = _rank_classes(baseline_metrics, method_metrics)
    _write_ranking(ranking, args.save_dir / "class_ranking.tsv")

    if args.class_name:
        selected = next((row for row in ranking if row["class"] == args.class_name), None)
        if selected is None:
            raise ValueError(f"class {args.class_name} not found in both logs")
    else:
        selected = ranking[0]

    baseline_args = _build_baseline_args(args)
    method_args = _build_method_args(args)
    records, recomputed_metrics = _evaluate_selected_class(
        args,
        selected["class"],
        baseline_args,
        method_args,
    )

    _plot_metric_comparison(selected, recomputed_metrics, args.save_dir / "experiment_metric_comparison.png")
    _plot_pixel_distribution(records, args.save_dir / "pixel_score_distribution.png", args.seed, args.max_pixels)
    example_paths = _plot_examples(records, args.save_dir, args.image_size, args.top_k_examples)
    _write_summary(args, selected, ranking, recomputed_metrics, example_paths)

    print(f"save_dir: {args.save_dir}")
    print(f"selected_class: {selected['class']}")
    print(
        "log pixel AUPRO: "
        f"{selected['baseline']['pixel_aupro']:.1f} -> {selected['method']['pixel_aupro']:.1f} "
        f"({selected['delta']['pixel_aupro']:+.1f})"
    )
    print(
        "recomputed pixel AUPRO: "
        f"{100 * recomputed_metrics['baseline_pixel_aupro']:.1f} -> "
        f"{100 * recomputed_metrics['method_pixel_aupro']:.1f} "
        f"({100 * recomputed_metrics['delta_pixel_aupro']:+.1f})"
    )
    print(f"examples: {len(example_paths)}")


if __name__ == "__main__":
    main()
