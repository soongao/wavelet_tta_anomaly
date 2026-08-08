from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import AnomalyCLIP_lib

from anomalyclip.cached_eval_utils import (
    build_anomaly_maps_from_patch_features,
    build_structure_texture_gate,
    build_wavelet_gate,
    compute_image_text_prob,
    format_metrics_table,
    init_results,
    load_model_and_text_features,
    sample_cache_paths,
    selected_classes,
    smooth_anomaly_map,
)
from anomalyclip.config_utils import parse_args_with_config
from anomalyclip.dataset import generate_class_info
from anomalyclip.logger import get_logger, log_run_context
from anomalyclip.test_time_rectification import rectify_text_features_with_multi_layer_anchors
from anomalyclip.normal_change_manifold import (
    apply_ncma_adaptation,
    compute_image_text_prob_with_ncma,
    load_normal_change_manifold,
)
from anomalyclip.wavelet_calibration import (
    apply_layer_consistency_calibration,
    apply_low_rank_residual_calibration,
    apply_structure_texture_calibration,
    apply_wavelet_residual_map_calibration,
    apply_wavelet_calibration,
    compute_texture_reliability,
    compute_wavelet_reliability,
    fuse_image_score_with_pixel_score,
    global_image_confidence_gate,
    topk_pixel_score,
)
from anomalyclip.prototype_adaptation import (
    apply_direct_wavelet_fusion,
    apply_wavelet_prototype_adaptation,
    compute_image_text_prob_with_adapted_prototypes,
)


def _load_patch_features(sample):
    if "patch_features" not in sample:
        raise ValueError(
            "This cache was created with --maps_only, so TTA or map recomputation "
            "cannot be evaluated. Rebuild the cache without --maps_only."
        )
    return [patch_feature.float() for patch_feature in sample["patch_features"]]


def _transport_kwargs(args):
    return {
        "ot_mode": args.ot_mode,
        "ot_cost": args.ot_cost,
        "ot_anchor_mode": args.ot_anchor_mode,
        "ot_epsilon": args.ot_epsilon,
        "ot_tau_patch": args.ot_tau_patch,
        "ot_tau_anchor": args.ot_tau_anchor,
        "ot_partial_mass": args.ot_partial_mass,
        "ot_iterations": args.ot_iterations,
        "ot_score": args.ot_score,
        "ot_alpha": args.ot_alpha,
        "ot_beta": args.ot_beta,
        "curvature": args.hyperbolic_curvature,
        "temperature": args.hyperbolic_temperature,
        "radius_scale": args.hyperbolic_radius_scale,
        "cone_aperture": args.cone_aperture,
        "margin": args.entailment_margin,
    }


def _build_transport_maps_from_patch_features(
    patch_features,
    text_features_raw,
    args,
    return_layer_maps=False,
):
    anomaly_map_list = []
    selected_patch_features = []
    selected_layers = set(int(layer) for layer in args.feature_map_layer)
    for idx, patch_feature in enumerate(patch_features):
        if idx not in selected_layers:
            continue

        similarity, energy, _ = AnomalyCLIP_lib.compute_transport_anomaly(
            patch_feature.float(),
            text_features_raw.float(),
            **_transport_kwargs(args),
        )
        if args.patch_score_space == "energy":
            anomaly_map = AnomalyCLIP_lib.get_similarity_map(
                energy[:, 1:].unsqueeze(-1),
                args.image_size,
            )[..., 0]
        else:
            similarity_map = AnomalyCLIP_lib.get_similarity_map(
                similarity[:, 1:, :],
                args.image_size,
            )
            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0

        anomaly_map_list.append(anomaly_map)
        selected_patch_features.append(patch_feature.float())

    if len(anomaly_map_list) == 0:
        raise ValueError("No patch feature layer was selected. Check --feature_map_layer.")

    anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
    if return_layer_maps:
        return anomaly_map, selected_patch_features, anomaly_map_list
    return anomaly_map, selected_patch_features


def _build_multicrop_index(multicrop_cache_dir):
    if not multicrop_cache_dir:
        return None
    sample_paths = sample_cache_paths(multicrop_cache_dir)
    if len(sample_paths) == 0:
        raise FileNotFoundError(f"no multi-crop sample cache files found under {multicrop_cache_dir}/samples")
    index = {}
    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu")
        index[sample["img_path"]] = sample["stitched_crop_map"].float()
    return index


def _fuse_multicrop_map(anomaly_map, crop_map, weight):
    if crop_map is None or weight <= 0:
        return anomaly_map
    anomaly_map = anomaly_map.float()
    crop_map = crop_map.float()
    if crop_map.dim() == 2:
        crop_map = crop_map.unsqueeze(0)
    if crop_map.shape[-2:] != anomaly_map.shape[-2:]:
        crop_map = torch.nn.functional.interpolate(
            crop_map.unsqueeze(1),
            size=anomaly_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    weight = min(float(weight), 1.0)
    return ((1.0 - weight) * anomaly_map + weight * crop_map).clamp_min(0.0)


MVTec_OBJECT_CLASSES = {
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
}


def _should_apply_foreground_gate(sample, args):
    if not args.use_foreground_gate:
        return False
    cls_name = sample["cls_name"]
    if args.foreground_classes is None:
        return cls_name in MVTec_OBJECT_CLASSES
    requested = set(args.foreground_classes)
    return "all" in requested or cls_name in requested


def _foreground_gate_from_image(image_path, image_size, args):
    output_size = image_size if isinstance(image_size, int) else image_size[0]
    image = Image.open(image_path).convert("RGB").resize((output_size, output_size), Image.BILINEAR)
    image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)

    height, width, _ = image_tensor.shape
    border = max(2, min(height, width) // 32)
    border_pixels = torch.cat(
        [
            image_tensor[:border].reshape(-1, 3),
            image_tensor[-border:].reshape(-1, 3),
            image_tensor[:, :border].reshape(-1, 3),
            image_tensor[:, -border:].reshape(-1, 3),
        ],
        dim=0,
    )
    background = border_pixels.median(dim=0).values.view(1, 1, 3)
    distance = (image_tensor - background).square().mean(dim=-1).sqrt()

    contrast = distance.max() - distance.min()
    if contrast < float(args.foreground_min_contrast):
        return torch.ones(1, height, width)

    distance = (distance - distance.min()) / contrast.clamp_min(1e-6)
    kernel = int(args.foreground_smooth_kernel)
    if kernel > 1:
        if kernel % 2 == 0:
            kernel += 1
        distance = F.avg_pool2d(
            distance.view(1, 1, height, width),
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        ).view(height, width)
        distance = (distance - distance.min()) / (distance.max() - distance.min()).clamp_min(1e-6)

    quantile = min(max(float(args.foreground_quantile), 0.0), 0.95)
    threshold = torch.quantile(distance.flatten(), quantile)
    foreground = ((distance - threshold) / (1.0 - threshold).clamp_min(1e-6)).clamp(0.0, 1.0)
    if args.foreground_power > 0:
        foreground = foreground.pow(float(args.foreground_power))

    outside_weight = min(max(float(args.foreground_outside_weight), 0.0), 1.0)
    gate = outside_weight + (1.0 - outside_weight) * foreground
    return gate.unsqueeze(0)


def _apply_foreground_gate(sample, anomaly_map, args):
    if not _should_apply_foreground_gate(sample, args):
        return anomaly_map
    foreground_gate = _foreground_gate_from_image(sample["img_path"], anomaly_map.shape[-1], args)
    if foreground_gate.shape[-2:] != anomaly_map.shape[-2:]:
        foreground_gate = F.interpolate(
            foreground_gate.unsqueeze(1),
            size=anomaly_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return anomaly_map.float() * foreground_gate.to(dtype=anomaly_map.dtype)


def _normalize_map_per_image(anomaly_map, args):
    mode = args.map_normalization
    if mode == "none":
        return anomaly_map

    anomaly_map = anomaly_map.float()
    batch = anomaly_map.shape[0]
    flat = anomaly_map.flatten(1)
    eps = 1e-6

    if mode == "minmax":
        lower = flat.min(dim=1).values
        upper = flat.max(dim=1).values
    elif mode == "percentile":
        q_low = min(max(float(args.map_norm_percentile_low) / 100.0, 0.0), 1.0)
        q_high = min(max(float(args.map_norm_percentile_high) / 100.0, 0.0), 1.0)
        if q_high <= q_low:
            raise ValueError("--map_norm_percentile_high must be greater than --map_norm_percentile_low")
        lower = torch.quantile(flat, q_low, dim=1)
        upper = torch.quantile(flat, q_high, dim=1)
    else:
        raise ValueError(f"unsupported map normalization mode: {mode}")

    view_shape = (batch,) + (1,) * (anomaly_map.dim() - 1)
    normalized = ((anomaly_map - lower.view(view_shape)) / (upper - lower).view(view_shape).clamp_min(eps)).clamp(0.0, 1.0)
    if args.map_norm_power > 0 and args.map_norm_power != 1.0:
        normalized = normalized.pow(float(args.map_norm_power))
    return normalized


def _build_maps_and_gate(sample, text_features, args, metadata, text_features_raw=None):
    feature_layers_changed = list(args.feature_map_layer) != list(metadata["feature_map_layer"])
    wavelet_fusion_changed = args.wavelet_fusion != metadata.get("wavelet_fusion", args.wavelet_fusion)
    can_recompute_from_patch = "patch_features" in sample
    needs_texture_gate_recompute = (
        (args.wavelet_mode == "dual_route" or args.use_wavelet_residual)
        and "texture_gate" not in sample
        and can_recompute_from_patch
    )
    need_patch_features = (
        args.score_mode != "cosine"
        or args.reload_text_features_from_checkpoint
        or args.use_tta_rectification
        or args.use_ncma_adaptation
        or args.use_wavelet_prototype_adaptation
        or args.use_direct_wavelet_fusion
        or args.recompute_maps
        or feature_layers_changed
        or needs_texture_gate_recompute
        or args.use_layer_consistency
        or args.use_low_rank_residual
        or args.use_wavelet_residual
        or (args.use_wavelet and wavelet_fusion_changed and can_recompute_from_patch)
    )
    if need_patch_features:
        patch_features = _load_patch_features(sample)
        if args.score_mode == "hyperbolic_uot":
            if text_features_raw is None:
                raise ValueError("--score_mode hyperbolic_uot needs raw text features")
            map_outputs = _build_transport_maps_from_patch_features(
                patch_features,
                text_features_raw,
                args,
                return_layer_maps=args.use_layer_consistency,
            )
        else:
            map_outputs = build_anomaly_maps_from_patch_features(
                patch_features,
                text_features,
                args.feature_map_layer,
                args.image_size,
                layer_weighting=args.layer_weighting,
                layer_weight_temperature=args.layer_weight_temperature,
                layer_weights=args.layer_weights,
                return_layer_maps=args.use_layer_consistency,
            )
        if args.use_layer_consistency:
            anomaly_map, selected_patch_features, layer_maps = map_outputs
        else:
            anomaly_map, selected_patch_features = map_outputs
            layer_maps = None
        wavelet_gate = build_wavelet_gate(
            selected_patch_features,
            image_size=args.image_size,
            fusion=args.wavelet_fusion,
            wavelet_levels=args.wavelet_levels,
            wavelet_level_fusion=args.wavelet_level_fusion,
        )
        texture_gate = build_structure_texture_gate(
            selected_patch_features,
            image_size=args.image_size,
            fusion=args.wavelet_fusion,
            edge_power=args.texture_edge_power,
            wavelet_levels=args.wavelet_levels,
            wavelet_level_fusion=args.wavelet_level_fusion,
        )
        return anomaly_map, wavelet_gate, texture_gate, patch_features, selected_patch_features, layer_maps

    anomaly_map = sample["base_anomaly_map"].float()
    wavelet_gate = sample.get("wavelet_gate")
    if wavelet_gate is not None:
        wavelet_gate = wavelet_gate.float()
    texture_gate = sample.get("texture_gate")
    if texture_gate is not None:
        texture_gate = texture_gate.float()
    return anomaly_map, wavelet_gate, texture_gate, None, None, None


def _evaluate_sample(
    sample,
    text_features,
    args,
    metadata,
    multicrop_index=None,
    normal_change_manifold=None,
    text_features_raw=None,
):
    text_features_for_map = text_features
    image_features = sample["image_features"].float()
    text_prob = None

    anomaly_map, wavelet_gate, texture_gate, patch_features, selected_patch_features, layer_maps = _build_maps_and_gate(
        sample,
        text_features_for_map,
        args,
        metadata,
        text_features_raw=text_features_raw,
    )
    wavelet_reliability = None
    if args.wavelet_mode == "dual_route":
        if texture_gate is None:
            raise ValueError(
                "dual_route mode needs texture_gate. Rebuild cache or use a patch-feature cache."
            )
        wavelet_reliability = compute_texture_reliability(
            anomaly_map,
            texture_gate,
            topk_ratio=args.wavelet_reliability_topk_ratio,
        )
    elif wavelet_gate is not None:
        wavelet_reliability = compute_wavelet_reliability(
            anomaly_map,
            wavelet_gate,
            topk_ratio=args.wavelet_reliability_topk_ratio,
        )

    if args.use_wavelet_prototype_adaptation and args.use_direct_wavelet_fusion:
        raise ValueError(
            "--use_wavelet_prototype_adaptation and --use_direct_wavelet_fusion "
            "are mutually exclusive"
        )
    if args.use_ncma_adaptation and (
        args.use_wavelet_prototype_adaptation or args.use_direct_wavelet_fusion
    ):
        raise ValueError(
            "--use_ncma_adaptation is mutually exclusive with prototype adaptation "
            "and direct wavelet fusion"
        )
    strict_proto_path = args.use_wavelet_prototype_adaptation or args.use_direct_wavelet_fusion
    legacy_mixins = (
        args.use_tta_rectification
        or args.use_wavelet
        or args.use_layer_consistency
        or args.use_wavelet_residual
        or args.use_low_rank_residual
    )
    if strict_proto_path and legacy_mixins:
        raise ValueError(
            "strict prototype adaptation/direct fusion ablations must be run "
            "without legacy TTA or legacy map calibration"
        )
    if args.use_ncma_adaptation and legacy_mixins:
        raise ValueError(
            "NCMA must be evaluated without legacy wavelet/TTA/low-rank calibration "
            "so that on-manifold projection and off-manifold residuals are isolated"
        )
    if args.use_ncma_adaptation:
        anomaly_map, adapted_text_features, _ = apply_ncma_adaptation(
            selected_patch_features,
            text_features_for_map,
            base_anomaly_map=anomaly_map,
            image_size=args.image_size,
            manifold=normal_change_manifold,
            fit_from_sample=args.ncma_fit_from_sample,
            anchor_layers=args.ncma_anchor_layers,
            num_anchors=args.ncma_num_anchors,
            tangent_rank=args.ncma_tangent_rank,
            pca_neighbors=args.ncma_pca_neighbors,
            fit_topk_ratio=args.ncma_fit_topk_ratio,
            max_fit_tokens=args.ncma_max_fit_tokens,
            kmeans_iters=args.ncma_kmeans_iters,
            residual_temperature=args.ncma_residual_temperature,
            update_topk_ratio=args.ncma_update_topk_ratio,
            update_alpha=args.ncma_update_alpha,
            normal_gamma=args.ncma_normal_gamma,
            residual_weight=args.ncma_residual_weight,
            temperature=args.ncma_temperature,
            layer_fusion=args.ncma_layer_fusion,
            use_projection_update=args.ncma_use_projection_update,
            use_residual_score=args.ncma_use_residual_score,
            update_abnormal=args.ncma_update_abnormal,
            abnormal_alpha=args.ncma_abnormal_alpha,
            seed=args.ncma_seed,
        )
        text_features_for_map = adapted_text_features
        text_prob = compute_image_text_prob_with_ncma(
            image_features,
            adapted_text_features,
            temperature=args.ncma_temperature,
        )
    elif args.use_wavelet_prototype_adaptation:
        anomaly_map, adapted_text_features, _ = apply_wavelet_prototype_adaptation(
            selected_patch_features,
            text_features_for_map,
            image_size=args.image_size,
            temperature=args.proto_temperature,
            gamma=args.proto_gamma,
            eta=args.proto_eta,
            topk_ratio=args.proto_topk_ratio,
            alpha0=args.proto_alpha0,
            beta0=args.proto_beta0,
            tau_a=args.proto_tau_a,
            update_min_abnormal_confidence=args.proto_update_min_abnormal_confidence,
            wavelet_mix=args.proto_wavelet_mix,
            wavelet_mode=args.proto_wavelet_mode,
            conservative_update=args.proto_conservative_update,
            anchor_layers=args.proto_anchor_layers,
            layer_fusion=args.proto_layer_fusion,
            clip_percentile_low=args.proto_percentile_low,
            clip_percentile_high=args.proto_percentile_high,
        )
        text_features_for_map = adapted_text_features
        text_prob = compute_image_text_prob_with_adapted_prototypes(
            image_features,
            adapted_text_features,
            temperature=args.proto_temperature,
        )
    elif args.use_direct_wavelet_fusion:
        anomaly_map, _ = apply_direct_wavelet_fusion(
            selected_patch_features,
            text_features_for_map,
            image_size=args.image_size,
            temperature=args.proto_temperature,
            weight=args.direct_wavelet_fusion_weight,
            wavelet_mode=args.proto_wavelet_mode,
            anchor_layers=args.proto_anchor_layers,
            clip_percentile_low=args.proto_percentile_low,
            clip_percentile_high=args.proto_percentile_high,
        )

    if args.use_tta_rectification:
        tta_gate = texture_gate if args.wavelet_mode == "dual_route" else wavelet_gate
        rectified_text_features, _ = rectify_text_features_with_multi_layer_anchors(
            text_features_for_map,
            selected_patch_features,
            anomaly_map,
            wavelet_gate=tta_gate,
            reliability=wavelet_reliability if not args.disable_adaptive_wavelet else None,
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
        map_outputs = build_anomaly_maps_from_patch_features(
            patch_features,
            text_features_for_map,
            args.feature_map_layer,
            args.image_size,
            layer_weighting=args.layer_weighting,
            layer_weight_temperature=args.layer_weight_temperature,
            layer_weights=args.layer_weights,
            return_layer_maps=args.use_layer_consistency,
        )
        if args.use_layer_consistency:
            anomaly_map, selected_patch_features, layer_maps = map_outputs
        else:
            anomaly_map, selected_patch_features = map_outputs
            layer_maps = None
        text_prob = compute_image_text_prob(image_features, text_features_for_map)
    elif (
        text_prob is None
        and (
            args.score_mode != "hyperbolic_uot"
            or args.use_cached_image_score_with_alt_map
        )
    ):
        text_prob = None if args.reload_text_features_from_checkpoint else sample.get("text_prob")
        if text_prob is None:
            text_prob = compute_image_text_prob(image_features, text_features_for_map)
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
        if args.wavelet_mode == "dual_route":
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
                reliability=wavelet_reliability,
                reliability_power=args.wavelet_reliability_power,
                reliability_topk_ratio=args.wavelet_reliability_topk_ratio,
                min_reliability=args.wavelet_min_reliability,
                texture_delta_reliability_power=args.texture_delta_reliability_power,
                image_confidence_gate=image_confidence_gate,
                image_confidence_weight=args.image_to_pixel_weight,
            )
        else:
            if wavelet_gate is None:
                raise ValueError("Wavelet calibration needs a cached or recomputed wavelet gate.")
            anomaly_map, _ = apply_wavelet_calibration(
                anomaly_map,
                wavelet_gate,
                beta=args.wavelet_beta,
                condition_power=args.wavelet_condition_power,
                suppress_beta=args.wavelet_suppress_beta,
                adaptive=not args.disable_adaptive_wavelet,
                reliability=wavelet_reliability,
                reliability_power=args.wavelet_reliability_power,
                reliability_topk_ratio=args.wavelet_reliability_topk_ratio,
                )

    if args.use_layer_consistency:
        if layer_maps is None:
            raise ValueError("Layer consistency needs patch-feature maps. Rebuild or use patch-feature cache.")
        anomaly_map, _ = apply_layer_consistency_calibration(
            anomaly_map,
            layer_maps,
            topk_ratio=args.layer_consistency_topk_ratio,
            min_layers=args.layer_consistency_min_layers,
            boost_weight=args.layer_consistency_boost,
            suppress_weight=args.layer_consistency_suppress,
            condition_power=args.layer_consistency_condition_power,
        )

    if args.use_wavelet_residual:
        residual_gate = texture_gate if args.wavelet_mode == "dual_route" else wavelet_gate
        if residual_gate is None:
            raise ValueError("Wavelet residual calibration needs a wavelet or texture gate.")
        anomaly_map, _ = apply_wavelet_residual_map_calibration(
            anomaly_map,
            residual_gate,
            weight=args.wavelet_residual_weight,
            topk_ratio=args.wavelet_residual_topk_ratio,
            condition_power=args.wavelet_residual_condition_power,
            max_delta_ratio=args.wavelet_residual_max_delta_ratio,
            local_contrast_kernel=args.wavelet_residual_local_contrast_kernel,
            local_contrast_weight=args.wavelet_residual_local_contrast_weight,
            rank_gate_mode=args.wavelet_residual_rank_gate_mode,
            rank_gate_temperature=args.wavelet_residual_rank_gate_temperature,
            confidence_power=args.wavelet_residual_confidence_power,
        )

    if args.use_low_rank_residual:
        if selected_patch_features is None:
            raise ValueError("Low-rank residual calibration needs patch-feature cache.")
        anomaly_map, _ = apply_low_rank_residual_calibration(
            anomaly_map,
            selected_patch_features,
            output_size=args.image_size,
            weight=args.low_rank_weight,
            rank_ratio=args.low_rank_rank_ratio,
            max_rank=args.low_rank_max_rank,
            sample_patches=args.low_rank_sample_patches,
            fusion=args.low_rank_fusion,
            condition_power=args.low_rank_condition_power,
            max_delta_ratio=args.low_rank_max_delta_ratio,
            clip_topk_ratio=args.low_rank_clip_topk_ratio,
            clip_gate_mode=args.low_rank_clip_gate_mode,
            clip_gate_temperature=args.low_rank_clip_gate_temperature,
            center=not args.low_rank_no_center,
            layer_selection=args.low_rank_layer_selection,
        )

    if args.use_multicrop_fusion:
        if multicrop_index is None:
            raise ValueError("Multi-crop fusion needs --multicrop_cache_dir.")
        crop_map = multicrop_index.get(sample["img_path"])
        if crop_map is None:
            if args.multicrop_missing_policy == "error":
                raise KeyError(f"missing multi-crop map for {sample['img_path']}")
        else:
            anomaly_map = _fuse_multicrop_map(
                anomaly_map,
                crop_map,
                weight=args.multicrop_weight,
            )

    if text_prob is None:
        text_prob = anomaly_map.flatten(1).max(dim=1).values

    if args.use_image_score_map_gate:
        gate = text_prob.float().view(-1, 1, 1).clamp(
            float(args.image_score_map_min_gate),
            float(args.image_score_map_max_gate),
        )
        if args.image_score_map_gate_power > 0:
            gate = gate.pow(float(args.image_score_map_gate_power))
        anomaly_map = anomaly_map.float() * gate

    anomaly_map = smooth_anomaly_map(anomaly_map, sigma=args.sigma)
    anomaly_map = _apply_foreground_gate(sample, anomaly_map, args)
    anomaly_map = _normalize_map_per_image(anomaly_map, args)
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


def evaluate_cache(args) -> None:
    metadata_path = os.path.join(args.cache_dir, "metadata.pt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata not found: {metadata_path}")

    metadata = torch.load(metadata_path, map_location="cpu")
    args.image_size = args.image_size or metadata["image_size"]
    args.feature_map_layer = args.feature_map_layer or metadata["feature_map_layer"]
    args.checkpoint_path = args.checkpoint_path or metadata.get("checkpoint_path")
    args.depth = args.depth or metadata.get("depth", 9)
    args.n_ctx = args.n_ctx or metadata.get("n_ctx", 12)
    args.t_n_ctx = args.t_n_ctx or metadata.get("t_n_ctx", 4)
    args.dpam_layer = args.dpam_layer or metadata.get("dpam_layer", 20)
    sample_paths = sample_cache_paths(args.cache_dir)
    if len(sample_paths) == 0:
        raise FileNotFoundError(f"no sample cache files found under {args.cache_dir}/samples")

    if metadata.get("cache_mode") == "maps_only":
        if args.score_mode != "cosine":
            raise ValueError(
                "This cache was created with --maps_only. Rebuild without --maps_only "
                "to evaluate alternate patch scorers."
            )
        if list(args.feature_map_layer) != list(metadata["feature_map_layer"]):
            raise ValueError(
                "This cache was created with --maps_only, so --feature_map_layer "
                "cannot be changed. Rebuild the cache without --maps_only."
            )
        if args.recompute_maps or args.use_tta_rectification:
            raise ValueError(
                "This cache was created with --maps_only. Rebuild without --maps_only "
                "to recompute maps or run TTA rectification."
            )
        if args.use_wavelet_prototype_adaptation or args.use_direct_wavelet_fusion:
            raise ValueError(
                "This cache was created with --maps_only. Rebuild without --maps_only "
                "to run prototype adaptation or direct wavelet fusion."
            )
        if args.use_ncma_adaptation:
            raise ValueError(
                "This cache was created with --maps_only. Rebuild without --maps_only "
                "to run NCMA projection and residual scoring."
            )
        if (
            args.wavelet_mode == "dual_route"
            and "texture_gate" not in torch.load(sample_paths[0], map_location="cpu")
        ):
            raise ValueError(
                "This maps_only cache does not contain texture_gate. Rebuild the cache for dual_route mode."
            )
        if args.wavelet_fusion != metadata.get("wavelet_fusion", args.wavelet_fusion):
            print(
                "warning: maps_only cache stores one precomputed wavelet gate; "
                f"using cached fusion={metadata.get('wavelet_fusion')} instead of {args.wavelet_fusion}"
            )

    obj_list, _ = generate_class_info(metadata.get("dataset", args.dataset))
    obj_list = selected_classes(obj_list, args.classes)
    results = init_results(obj_list)
    logger = get_logger(args.save_path)
    text_features = metadata["text_features"].float()
    text_features_raw = None
    if args.reload_text_features_from_checkpoint or args.score_mode == "hyperbolic_uot":
        _, loaded_text_features, loaded_text_features_raw = load_model_and_text_features(
            args,
            device="cpu",
            return_raw=True,
        )
        if args.reload_text_features_from_checkpoint:
            text_features = loaded_text_features.float()
        if args.score_mode == "hyperbolic_uot":
            text_features_raw = loaded_text_features_raw.float()
    normal_change_manifold = None
    if args.use_ncma_adaptation:
        if args.ncma_manifold_path:
            normal_change_manifold = load_normal_change_manifold(
                args.ncma_manifold_path,
                map_location="cpu",
            )
        elif not args.ncma_fit_from_sample:
            raise ValueError(
                "NCMA strict zero-shot evaluation needs --ncma_fit_from_sample "
                "when no optional --ncma_manifold_path is provided."
            )
    multicrop_index = _build_multicrop_index(args.multicrop_cache_dir) if args.use_multicrop_fusion else None
    run_title = (
        "NCMA cached calibration evaluation"
        if args.use_ncma_adaptation
        else "AnomalyCLIP cached calibration evaluation"
    )
    log_run_context(
        logger,
        args,
        title=run_title,
        extra_info={
            "method": "NCMA" if args.use_ncma_adaptation else "AnomalyCLIP cached calibration",
            "cache_dir": args.cache_dir,
            "cache_mode": metadata.get("cache_mode", "unknown"),
            "cache_num_samples": metadata.get("num_samples", "unknown"),
            "cache_dataset": metadata.get("dataset", "unknown"),
            "cache_data_path": metadata.get("data_path", "unknown"),
            "cache_checkpoint_path": metadata.get("checkpoint_path", "unknown"),
            "cache_features_list": metadata.get("features_list", "unknown"),
            "cache_feature_map_layer": metadata.get("feature_map_layer", "unknown"),
            "ncma_manifold_path": args.ncma_manifold_path if args.use_ncma_adaptation else "disabled",
            "ncma_fit_from_sample": args.ncma_fit_from_sample if args.use_ncma_adaptation else "disabled",
            "sample_files": len(sample_paths),
            "multicrop_cache_dir": args.multicrop_cache_dir if args.use_multicrop_fusion else "disabled",
            "multicrop_samples": len(multicrop_index) if multicrop_index is not None else 0,
        },
    )

    evaluated = 0
    for sample_path in sample_paths:
        sample = torch.load(sample_path, map_location="cpu")
        cls_name = sample["cls_name"]
        if cls_name not in results:
            continue
        text_prob, anomaly_map = _evaluate_sample(
            sample,
            text_features,
            args,
            metadata,
            multicrop_index=multicrop_index,
            normal_change_manifold=normal_change_manifold,
            text_features_raw=text_features_raw,
        )
        results[cls_name]["imgs_masks"].append(sample["img_mask"].float())
        results[cls_name]["gt_sp"].append(int(sample["anomaly"]))
        results[cls_name]["pr_sp"].extend(text_prob.detach().cpu())
        results[cls_name]["anomaly_maps"].append(anomaly_map)
        evaluated += 1

    table = format_metrics_table(
        results,
        obj_list,
        args.metrics,
        aupro_steps=args.aupro_steps,
    )
    logger.info("\n%s", table)
    print(f"cache_dir: {args.cache_dir}")
    print(f"samples evaluated: {evaluated}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate cached AnomalyCLIP calibration", add_help=True)
    parser.add_argument("--cache_dir", type=str, default="./cache/mvtec_anomalyclip_features")
    parser.add_argument("--save_path", type=str, default="./cached_results_mvtec")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument(
        "--metrics",
        type=str,
        default="image-pixel-level",
        choices=["image-level", "pixel-level", "image-pixel-level", "all"],
    )
    parser.add_argument("--aupro_steps", type=int, default=200)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--reload_text_features_from_checkpoint", action="store_true")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--n_ctx", type=int, default=None)
    parser.add_argument("--t_n_ctx", type=int, default=None)
    parser.add_argument("--dpam_layer", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=None)
    parser.add_argument("--score_mode", type=str, default="cosine", choices=["cosine", "hyperbolic_uot"])
    parser.add_argument("--patch_score_space", type=str, default="prob", choices=["prob", "energy"])
    parser.add_argument("--use_cached_image_score_with_alt_map", action="store_true")
    parser.add_argument("--hyperbolic_curvature", type=float, default=1.0)
    parser.add_argument("--hyperbolic_temperature", type=float, default=1.0)
    parser.add_argument("--hyperbolic_radius_scale", type=float, default=0.1)
    parser.add_argument("--cone_aperture", type=float, default=0.1)
    parser.add_argument("--entailment_margin", type=float, default=0.2)
    parser.add_argument("--ot_mode", type=str, default="unbalanced", choices=["balanced", "partial", "unbalanced"])
    parser.add_argument("--ot_cost", type=str, default="hyperbolic_cone", choices=["cosine", "euclidean", "hyperbolic_distance", "hyperbolic_cone"])
    parser.add_argument("--ot_anchor_mode", type=str, default="normal", choices=["normal", "anomaly", "both", "normal_anomaly"])
    parser.add_argument("--ot_epsilon", type=float, default=0.05)
    parser.add_argument("--ot_tau_patch", type=float, default=0.5)
    parser.add_argument("--ot_tau_anchor", type=float, default=0.5)
    parser.add_argument("--ot_partial_mass", type=float, default=0.9)
    parser.add_argument("--ot_iterations", type=int, default=50)
    parser.add_argument("--ot_score", type=str, default="combined", choices=["unmatched", "cost", "combined"])
    parser.add_argument("--ot_alpha", type=float, default=1.0)
    parser.add_argument("--ot_beta", type=float, default=1.0)
    parser.add_argument("--layer_weighting", type=str, default="sum", choices=["sum", "dynamic_wavelet"])
    parser.add_argument("--layer_weight_temperature", type=float, default=1.0)
    parser.add_argument("--layer_weights", type=float, nargs="+", default=None)
    parser.add_argument("--sigma", type=float, default=4)
    parser.add_argument("--classes", type=str, nargs="+", default=None)
    parser.add_argument("--recompute_maps", action="store_true", help="rebuild base maps from cached patch features")
    parser.add_argument("--use_wavelet", action="store_true")
    parser.add_argument("--wavelet_beta", type=float, default=0.5)
    parser.add_argument("--wavelet_condition_power", type=float, default=1.0)
    parser.add_argument("--wavelet_suppress_beta", type=float, default=0.3)
    parser.add_argument("--wavelet_fusion", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--wavelet_levels", type=int, default=1)
    parser.add_argument("--wavelet_level_fusion", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--wavelet_mode", type=str, default="dual_route", choices=["dual_route", "adaptive"])
    parser.add_argument("--texture_edge_power", type=float, default=1.0)
    parser.add_argument("--texture_max_delta_ratio", type=float, default=0.05)
    parser.add_argument("--texture_suppression_weight", type=float, default=0.0)
    parser.add_argument("--texture_local_contrast_kernel", type=int, default=0)
    parser.add_argument("--texture_local_contrast_weight", type=float, default=0.0)
    parser.add_argument("--rank_preserve_topk_ratio", type=float, default=0.0)
    parser.add_argument("--rank_gate_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--rank_gate_temperature", type=float, default=0.05)
    parser.add_argument("--use_wavelet_confidence", action="store_true")
    parser.add_argument("--wavelet_confidence_power", type=float, default=1.0)
    parser.add_argument("--disable_adaptive_wavelet", action="store_true")
    parser.add_argument("--wavelet_reliability_power", type=float, default=1.0)
    parser.add_argument("--wavelet_reliability_topk_ratio", type=float, default=0.05)
    parser.add_argument("--wavelet_min_reliability", type=float, default=0.0)
    parser.add_argument("--texture_delta_reliability_power", type=float, default=0.0)
    parser.add_argument("--use_layer_consistency", action="store_true")
    parser.add_argument("--layer_consistency_topk_ratio", type=float, default=0.10)
    parser.add_argument("--layer_consistency_min_layers", type=int, default=2)
    parser.add_argument("--layer_consistency_boost", type=float, default=0.03)
    parser.add_argument("--layer_consistency_suppress", type=float, default=0.0)
    parser.add_argument("--layer_consistency_condition_power", type=float, default=1.0)
    parser.add_argument("--use_wavelet_residual", action="store_true")
    parser.add_argument("--wavelet_residual_weight", type=float, default=0.03)
    parser.add_argument("--wavelet_residual_topk_ratio", type=float, default=0.35)
    parser.add_argument("--wavelet_residual_condition_power", type=float, default=2.0)
    parser.add_argument("--wavelet_residual_max_delta_ratio", type=float, default=0.03)
    parser.add_argument("--wavelet_residual_local_contrast_kernel", type=int, default=17)
    parser.add_argument("--wavelet_residual_local_contrast_weight", type=float, default=0.5)
    parser.add_argument("--wavelet_residual_rank_gate_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--wavelet_residual_rank_gate_temperature", type=float, default=0.05)
    parser.add_argument("--wavelet_residual_confidence_power", type=float, default=1.0)
    parser.add_argument("--use_low_rank_residual", action="store_true")
    parser.add_argument("--low_rank_weight", type=float, default=0.03)
    parser.add_argument("--low_rank_rank_ratio", type=float, default=0.15)
    parser.add_argument("--low_rank_max_rank", type=int, default=32)
    parser.add_argument("--low_rank_sample_patches", type=int, default=128)
    parser.add_argument("--low_rank_fusion", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--low_rank_condition_power", type=float, default=2.0)
    parser.add_argument("--low_rank_max_delta_ratio", type=float, default=0.03)
    parser.add_argument("--low_rank_clip_topk_ratio", type=float, default=0.35)
    parser.add_argument("--low_rank_clip_gate_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--low_rank_clip_gate_temperature", type=float, default=0.05)
    parser.add_argument("--low_rank_no_center", action="store_true")
    parser.add_argument("--low_rank_layer_selection", type=str, default="all", choices=["all", "last"])
    parser.add_argument("--use_multicrop_fusion", action="store_true")
    parser.add_argument("--multicrop_cache_dir", type=str, default=None)
    parser.add_argument("--multicrop_weight", type=float, default=0.25)
    parser.add_argument("--multicrop_missing_policy", type=str, default="error", choices=["error", "base"])
    parser.add_argument("--use_foreground_gate", action="store_true")
    parser.add_argument("--foreground_classes", type=str, nargs="+", default=None)
    parser.add_argument("--foreground_outside_weight", type=float, default=0.2)
    parser.add_argument("--foreground_power", type=float, default=1.0)
    parser.add_argument("--foreground_quantile", type=float, default=0.2)
    parser.add_argument("--foreground_smooth_kernel", type=int, default=31)
    parser.add_argument("--foreground_min_contrast", type=float, default=0.03)
    parser.add_argument("--map_normalization", type=str, default="none", choices=["none", "minmax", "percentile"])
    parser.add_argument("--map_norm_percentile_low", type=float, default=1.0)
    parser.add_argument("--map_norm_percentile_high", type=float, default=99.0)
    parser.add_argument("--map_norm_power", type=float, default=1.0)
    parser.add_argument("--use_image_to_pixel_gate", action="store_true")
    parser.add_argument("--image_to_pixel_weight", type=float, default=0.0)
    parser.add_argument("--image_to_pixel_power", type=float, default=1.0)
    parser.add_argument("--image_to_pixel_min_gate", type=float, default=0.0)
    parser.add_argument("--image_to_pixel_max_gate", type=float, default=1.0)
    parser.add_argument("--use_pixel_to_image_fusion", action="store_true")
    parser.add_argument("--pixel_to_image_weight", type=float, default=0.0)
    parser.add_argument("--pixel_to_image_topk_ratio", type=float, default=0.01)
    parser.add_argument("--pixel_to_image_normalize", action="store_true")
    parser.add_argument("--use_image_score_map_gate", action="store_true")
    parser.add_argument("--image_score_map_gate_power", type=float, default=1.0)
    parser.add_argument("--image_score_map_min_gate", type=float, default=0.0)
    parser.add_argument("--image_score_map_max_gate", type=float, default=1.0)
    parser.add_argument("--use_tta_rectification", action="store_true")
    parser.add_argument("--tta_mode", type=str, default="legacy", choices=["legacy", "wavelet_guided"])
    parser.add_argument("--tta_alpha", type=float, default=0.2)
    parser.add_argument("--tta_topk_ratio", type=float, default=0.05)
    parser.add_argument("--tta_min_confidence", type=float, default=0.0)
    parser.add_argument("--tta_min_confidence_margin", type=float, default=0.0)
    parser.add_argument("--tta_anchor_layers", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--tta_update_abnormal", action="store_true")
    parser.add_argument("--tta_repulsion_weight", type=float, default=0.25)
    parser.add_argument("--tta_abnormal_alpha_scale", type=float, default=1.0)
    parser.add_argument("--use_wavelet_prototype_adaptation", action="store_true")
    parser.add_argument("--use_direct_wavelet_fusion", action="store_true")
    parser.add_argument("--direct_wavelet_fusion_weight", type=float, default=0.5)
    parser.add_argument("--proto_temperature", type=float, default=0.07)
    parser.add_argument("--proto_gamma", type=float, default=1.0)
    parser.add_argument("--proto_eta", type=float, default=1.0)
    parser.add_argument("--proto_topk_ratio", type=float, default=0.20)
    parser.add_argument("--proto_alpha0", type=float, default=0.0)
    parser.add_argument("--proto_beta0", type=float, default=0.01)
    parser.add_argument("--proto_tau_a", type=float, default=0.15)
    parser.add_argument("--proto_update_min_abnormal_confidence", type=float, default=0.06)
    parser.add_argument("--proto_wavelet_mix", type=float, default=0.05)
    parser.add_argument("--proto_wavelet_mode", type=str, default="boundary_aware", choices=["none", "hf_only", "boundary_aware"])
    parser.add_argument("--proto_conservative_update", action="store_true", default=True)
    parser.add_argument("--no_proto_conservative_update", dest="proto_conservative_update", action="store_false")
    parser.add_argument("--proto_anchor_layers", type=str, default="last", choices=["last", "mean"])
    parser.add_argument("--proto_layer_fusion", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--proto_percentile_low", type=float, default=1.0)
    parser.add_argument("--proto_percentile_high", type=float, default=99.0)
    parser.add_argument("--use_ncma_adaptation", action="store_true")
    parser.add_argument("--ncma_manifold_path", type=str, default=None, help="optional precomputed manifold for source-calibration runs")
    parser.add_argument("--ncma_fit_from_sample", action="store_true", help="fit a temporary manifold from the current unlabeled test sample")
    parser.add_argument("--ncma_anchor_layers", type=str, default="mean", choices=["last", "mean"])
    parser.add_argument("--ncma_num_anchors", type=int, default=64)
    parser.add_argument("--ncma_tangent_rank", type=int, default=8)
    parser.add_argument("--ncma_pca_neighbors", type=int, default=128)
    parser.add_argument("--ncma_fit_topk_ratio", type=float, default=0.35)
    parser.add_argument("--ncma_max_fit_tokens", type=int, default=8192)
    parser.add_argument("--ncma_kmeans_iters", type=int, default=5)
    parser.add_argument("--ncma_residual_temperature", type=float, default=1.0)
    parser.add_argument("--ncma_update_topk_ratio", type=float, default=0.20)
    parser.add_argument("--ncma_update_alpha", type=float, default=0.10)
    parser.add_argument("--ncma_normal_gamma", type=float, default=1.0)
    parser.add_argument("--ncma_residual_weight", type=float, default=0.35)
    parser.add_argument("--ncma_temperature", type=float, default=0.07)
    parser.add_argument("--ncma_layer_fusion", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--ncma_use_projection_update", action="store_true", default=True)
    parser.add_argument("--ncma_use_residual_score", action="store_true", default=True)
    parser.add_argument("--ncma_update_abnormal", action="store_true")
    parser.add_argument("--ncma_abnormal_alpha", type=float, default=0.05)
    parser.add_argument("--ncma_seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    args, _ = parse_args_with_config(build_parser())
    evaluate_cache(args)
