import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _extract_spatial_tokens(patch_features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Drop CLS token when present and return [B, H*W, C] spatial tokens."""
    if patch_features.dim() != 3:
        raise ValueError(
            f"patch_features must be [B, N, C], got {tuple(patch_features.shape)}"
        )

    _, num_tokens, _ = patch_features.shape
    side_with_cls = int(math.sqrt(num_tokens - 1))
    if side_with_cls * side_with_cls == num_tokens - 1:
        return patch_features[:, 1:, :], side_with_cls, side_with_cls

    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return patch_features, side, side

    raise ValueError(
        "Cannot infer a square patch grid from patch_features with "
        f"{num_tokens} tokens."
    )


def _as_b1hw(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) == 1:
        return x
    raise ValueError(f"{name} must be [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")


def _minmax_norm_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = _as_b1hw(x, "x")
    batch = x.size(0)
    flat = x.flatten(1)
    min_val = flat.min(dim=1)[0].view(batch, 1, 1, 1)
    max_val = flat.max(dim=1)[0].view(batch, 1, 1, 1)
    return ((x - min_val) / (max_val - min_val + eps)).clamp(0.0, 1.0)


def _resize_scores_to_tokens(score_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
    score_map = _as_b1hw(score_map, "score_map")
    if score_map.shape[-2:] != (height, width):
        score_map = F.interpolate(
            score_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return score_map.flatten(1)


def _weighted_topk_anchor(
    tokens: torch.Tensor,
    confidence: torch.Tensor,
    topk_ratio: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one visual anchor per image from the top-k confidence tokens."""
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    batch, num_tokens, channels = tokens.shape
    k = max(1, int(math.ceil(num_tokens * topk_ratio)))
    topk_conf, topk_idx = torch.topk(confidence, k=k, dim=1)

    gather_idx = topk_idx.unsqueeze(-1).expand(batch, k, channels)
    topk_tokens = torch.gather(tokens, dim=1, index=gather_idx)

    weights = topk_conf / topk_conf.sum(dim=1, keepdim=True).clamp_min(eps)
    anchor = (topk_tokens * weights.unsqueeze(-1)).sum(dim=1)
    anchor = F.normalize(anchor, dim=-1, eps=eps)
    return anchor, topk_conf.mean(dim=1)


def rectify_text_features_with_visual_anchors(
    text_features: torch.Tensor,
    patch_features: torch.Tensor,
    anomaly_map: torch.Tensor,
    wavelet_gate: Optional[torch.Tensor] = None,
    reliability: Optional[torch.Tensor] = None,
    mode: str = "legacy",
    alpha: float = 0.2,
    topk_ratio: float = 0.05,
    update_abnormal: bool = False,
    min_confidence: float = 0.0,
    min_confidence_margin: float = 0.0,
    repulsion_weight: float = 0.25,
    abnormal_alpha_scale: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Rectify normal/anomaly text features using reliable visual anchors.

    Reliable normal patches have low CLIP anomaly response and low wavelet
    response. Reliable abnormal patches, when enabled, have high values in
    both maps. No labels or gradient updates are used.

    Args:
        text_features: [2, C] normalized text features ordered as
            [normal, anomaly].
        patch_features: [B, N, C] patch features from one selected layer.
        anomaly_map: [B, H, W] initial AnomalyCLIP anomaly map.
        wavelet_gate: optional [B, H, W] feature-wavelet reliability gate.
        reliability: optional [B] or [B, 1, 1, 1] CLIP-wavelet agreement.
            When provided, it gates the effective text-feature update strength.
        mode: ``legacy`` keeps the original normal-anchor update. 
            ``wavelet_guided`` uses both normal and abnormal anchors: the
            normal text feature is pushed away from wavelet-confirmed abnormal
            evidence, and the abnormal text feature is pulled toward it.
        alpha: interpolation strength for text-feature rectification.
        topk_ratio: fraction of patches used to form each visual anchor.
        update_abnormal: whether to also rectify the anomaly text feature.
        min_confidence: skip an anchor update when its mean top-k confidence
            is below this value.
        min_confidence_margin: skip an update unless the selected anchor is
            more confident than the opposite anchor by this margin.
        repulsion_weight: strength for pushing the normal text feature away
            from the abnormal anchor in ``wavelet_guided`` mode.
        abnormal_alpha_scale: relative update strength for the abnormal text
            feature in ``wavelet_guided`` mode.

    Returns:
        ``(rectified_text_features, diagnostics)``. The rectified features are
        [B, 2, C] because each image gets its own test-time text features.
    """
    if text_features.dim() != 2 or text_features.size(0) != 2:
        raise ValueError(
            f"text_features must be [2, C] ordered as [normal, anomaly], got {tuple(text_features.shape)}"
        )
    if mode not in {"legacy", "wavelet_guided"}:
        raise ValueError(f"unsupported TTA rectification mode: {mode}")

    spatial_tokens, height, width = _extract_spatial_tokens(patch_features)
    spatial_tokens = F.normalize(spatial_tokens.float(), dim=-1, eps=eps)

    anomaly_prior = _minmax_norm_per_image(anomaly_map, eps=eps)
    anomaly_scores = _resize_scores_to_tokens(anomaly_prior, height, width)

    if wavelet_gate is None:
        wavelet_scores = torch.ones_like(anomaly_scores)
    else:
        wavelet_prior = _minmax_norm_per_image(wavelet_gate, eps=eps)
        wavelet_scores = _resize_scores_to_tokens(wavelet_prior, height, width)

    normal_conf = (1.0 - anomaly_scores) * (1.0 - wavelet_scores)
    abnormal_conf = anomaly_scores * wavelet_scores

    normal_anchor, normal_anchor_conf = _weighted_topk_anchor(
        spatial_tokens,
        normal_conf,
        topk_ratio=topk_ratio,
        eps=eps,
    )
    abnormal_anchor, abnormal_anchor_conf = _weighted_topk_anchor(
        spatial_tokens,
        abnormal_conf,
        topk_ratio=topk_ratio,
        eps=eps,
    )

    batch = spatial_tokens.size(0)
    base_text = F.normalize(text_features.float(), dim=-1, eps=eps)
    rectified = base_text.unsqueeze(0).expand(batch, -1, -1).clone()
    if reliability is None:
        effective_alpha = torch.full(
            (batch, 1),
            float(alpha),
            dtype=rectified.dtype,
            device=rectified.device,
        )
    else:
        reliability = reliability.to(device=rectified.device, dtype=rectified.dtype)
        reliability = reliability.view(batch, -1).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        effective_alpha = float(alpha) * reliability

    normal_mask = normal_anchor_conf >= min_confidence
    if min_confidence_margin > 0:
        normal_mask = normal_mask & (
            normal_anchor_conf >= abnormal_anchor_conf + float(min_confidence_margin)
        )
    if normal_mask.any():
        if mode == "wavelet_guided" and repulsion_weight > 0:
            abnormal_mask_for_repulsion = (
                abnormal_anchor_conf >= min_confidence
            ) & (
                abnormal_anchor_conf >= normal_anchor_conf + float(min_confidence_margin)
            )
            repelled_normal_anchor = F.normalize(
                normal_anchor - float(repulsion_weight) * abnormal_anchor,
                dim=-1,
                eps=eps,
            )
            normal_target = torch.where(
                abnormal_mask_for_repulsion.unsqueeze(1),
                repelled_normal_anchor,
                normal_anchor,
            )
        else:
            normal_target = normal_anchor

        rectified_normal = F.normalize(
            (1.0 - effective_alpha) * rectified[:, 0, :] + effective_alpha * normal_target,
            dim=-1,
            eps=eps,
        )
        rectified[normal_mask, 0, :] = rectified_normal[normal_mask]

    should_update_abnormal = update_abnormal or mode == "wavelet_guided"
    if should_update_abnormal:
        abnormal_mask = abnormal_anchor_conf >= min_confidence
        if min_confidence_margin > 0:
            abnormal_mask = abnormal_mask & (
                abnormal_anchor_conf >= normal_anchor_conf + float(min_confidence_margin)
            )
        if abnormal_mask.any():
            abnormal_alpha = (effective_alpha * float(abnormal_alpha_scale)).clamp(0.0, 1.0)
            rectified_abnormal = F.normalize(
                (1.0 - abnormal_alpha) * rectified[:, 1, :] + abnormal_alpha * abnormal_anchor,
                dim=-1,
                eps=eps,
            )
            rectified[abnormal_mask, 1, :] = rectified_abnormal[abnormal_mask]

    diagnostics = {
        "normal_anchor_conf": normal_anchor_conf.detach(),
        "abnormal_anchor_conf": abnormal_anchor_conf.detach(),
        "effective_alpha": effective_alpha.squeeze(1).detach(),
        "mode": mode,
        "min_confidence_margin": torch.tensor(float(min_confidence_margin)),
        "repulsion_weight": torch.tensor(float(repulsion_weight)),
        "abnormal_alpha_scale": torch.tensor(float(abnormal_alpha_scale)),
        "normal_conf_map": normal_conf.view(batch, height, width).detach(),
        "abnormal_conf_map": abnormal_conf.view(batch, height, width).detach(),
    }
    return rectified, diagnostics


def rectify_text_features_with_multi_layer_anchors(
    text_features: torch.Tensor,
    patch_features_list: Sequence[torch.Tensor],
    anomaly_map: torch.Tensor,
    wavelet_gate: Optional[torch.Tensor] = None,
    reliability: Optional[torch.Tensor] = None,
    mode: str = "legacy",
    alpha: float = 0.2,
    topk_ratio: float = 0.05,
    update_abnormal: bool = False,
    min_confidence: float = 0.0,
    min_confidence_margin: float = 0.0,
    repulsion_weight: float = 0.25,
    abnormal_alpha_scale: float = 1.0,
    fusion: str = "mean",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Rectify text features by fusing visual anchors from multiple layers."""
    if len(patch_features_list) == 0:
        raise ValueError("patch_features_list must contain at least one layer")
    if fusion == "last":
        return rectify_text_features_with_visual_anchors(
            text_features,
            patch_features_list[-1],
            anomaly_map,
            wavelet_gate=wavelet_gate,
            reliability=reliability,
            mode=mode,
            alpha=alpha,
            topk_ratio=topk_ratio,
            update_abnormal=update_abnormal,
            min_confidence=min_confidence,
            min_confidence_margin=min_confidence_margin,
            repulsion_weight=repulsion_weight,
            abnormal_alpha_scale=abnormal_alpha_scale,
            eps=eps,
        )
    if fusion != "mean":
        raise ValueError(f"unsupported multi-layer anchor fusion: {fusion}")

    rectified_list = []
    diagnostics_list = []
    for patch_features in patch_features_list:
        rectified, diagnostics = rectify_text_features_with_visual_anchors(
            text_features,
            patch_features,
            anomaly_map,
            wavelet_gate=wavelet_gate,
            reliability=reliability,
            mode=mode,
            alpha=alpha,
            topk_ratio=topk_ratio,
            update_abnormal=update_abnormal,
            min_confidence=min_confidence,
            min_confidence_margin=min_confidence_margin,
            repulsion_weight=repulsion_weight,
            abnormal_alpha_scale=abnormal_alpha_scale,
            eps=eps,
        )
        rectified_list.append(rectified)
        diagnostics_list.append(diagnostics)

    rectified = F.normalize(torch.stack(rectified_list, dim=0).mean(dim=0), dim=-1, eps=eps)
    diagnostics = {
        "normal_anchor_conf": torch.stack(
            [item["normal_anchor_conf"] for item in diagnostics_list],
            dim=0,
        ).mean(dim=0),
        "abnormal_anchor_conf": torch.stack(
            [item["abnormal_anchor_conf"] for item in diagnostics_list],
            dim=0,
        ).mean(dim=0),
        "effective_alpha": diagnostics_list[-1]["effective_alpha"],
        "mode": mode,
        "anchor_fusion": fusion,
    }
    return rectified, diagnostics
