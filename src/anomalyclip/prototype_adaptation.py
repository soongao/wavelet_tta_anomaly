import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_hw(size) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"image_size must be an int or a pair, got {size}")
    return int(size[0]), int(size[1])


def _extract_spatial_tokens(patch_features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
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


def _normalize_percentile(
    x: torch.Tensor,
    low: float = 1.0,
    high: float = 99.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"x must be [B, H, W], got {tuple(x.shape)}")
    low = float(low)
    high = float(high)
    if not (0.0 <= low < high <= 100.0):
        raise ValueError(f"invalid percentile clip range: low={low}, high={high}")

    batch = x.size(0)
    flat = x.flatten(1)
    q_low = torch.quantile(flat.float(), low / 100.0, dim=1).view(batch, 1, 1)
    q_high = torch.quantile(flat.float(), high / 100.0, dim=1).view(batch, 1, 1)
    x = x.clamp(q_low.to(x.dtype), q_high.to(x.dtype))
    return ((x - q_low) / (q_high - q_low + eps)).clamp(0.0, 1.0)


def _haar_patch_reliability(
    tokens: torch.Tensor,
    height: int,
    width: int,
    wavelet_mode: str,
    clip_percentile_low: float = 1.0,
    clip_percentile_high: float = 99.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if wavelet_mode not in {"none", "hf_only", "boundary_aware"}:
        raise ValueError(f"unsupported proto_wavelet_mode: {wavelet_mode}")
    if wavelet_mode == "none":
        batch = tokens.size(0)
        zero = torch.zeros(batch, height, width, dtype=tokens.dtype, device=tokens.device)
        return zero.flatten(1), {"hf": zero, "lf_edge": zero, "w": zero}

    batch, _, channels = tokens.shape
    grid = tokens.transpose(1, 2).reshape(batch, channels, height, width)
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        grid = F.pad(grid, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = grid[:, :, 0::2, 0::2]
    x01 = grid[:, :, 0::2, 1::2]
    x10 = grid[:, :, 1::2, 0::2]
    x11 = grid[:, :, 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)

    hf = (lh.abs() + hl.abs() + hh.abs()).mean(dim=1)

    ll_h, ll_w = ll.shape[-2:]
    grad_y = torch.zeros(batch, 1, ll_h, ll_w, dtype=ll.dtype, device=ll.device)
    grad_x = torch.zeros_like(grad_y)
    if ll_h > 1:
        dy = ll[:, :, 1:, :] - ll[:, :, :-1, :]
        grad_y[:, :, 1:, :] = torch.sqrt(dy.square().mean(dim=1, keepdim=True) + eps)
    if ll_w > 1:
        dx = ll[:, :, :, 1:] - ll[:, :, :, :-1]
        grad_x[:, :, :, 1:] = torch.sqrt(dx.square().mean(dim=1, keepdim=True) + eps)
    lf_edge = torch.sqrt(grad_x.square() + grad_y.square() + eps).squeeze(1)

    hf = F.interpolate(
        hf.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    lf_edge = F.interpolate(
        lf_edge.unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    hf = _normalize_percentile(
        hf,
        low=clip_percentile_low,
        high=clip_percentile_high,
        eps=eps,
    )
    lf_edge = _normalize_percentile(
        lf_edge,
        low=clip_percentile_low,
        high=clip_percentile_high,
        eps=eps,
    )
    if wavelet_mode == "hf_only":
        w = hf
    else:
        w = hf * (1.0 - lf_edge).clamp(0.0, 1.0)
        w = _normalize_percentile(
            w,
            low=clip_percentile_low,
            high=clip_percentile_high,
            eps=eps,
        )
    return w.flatten(1), {"hf": hf, "lf_edge": lf_edge, "w": w}


def _prepare_text_features(
    text_features: torch.Tensor,
    batch: int,
    per_image: bool = False,
    eps: float = 1e-6,
):
    text_features = F.normalize(text_features.float(), dim=-1, eps=eps)
    if text_features.dim() == 2:
        if text_features.size(0) != 2:
            raise ValueError(
                f"text_features must be [2, C] or [K, 2, C], got {tuple(text_features.shape)}"
            )
        return text_features.unsqueeze(0).expand(batch, -1, -1), False
    if text_features.dim() == 3:
        if text_features.size(1) != 2:
            raise ValueError(
                f"text_features must be [2, C] or [K, 2, C], got {tuple(text_features.shape)}"
            )
        if per_image:
            if text_features.size(0) != batch:
                raise ValueError(
                    f"per-image text_features must be [B, 2, C], got {tuple(text_features.shape)} "
                    f"for batch={batch}"
                )
            return text_features, False
        return text_features.unsqueeze(0).expand(batch, -1, -1, -1), True
    if text_features.dim() == 4:
        if text_features.size(0) != batch or text_features.size(2) != 2:
            raise ValueError(
                "per-image ensemble text_features must be [B, K, 2, C], "
                f"got {tuple(text_features.shape)} for batch={batch}"
            )
        return text_features, True
    raise ValueError(
        "text_features must be [2, C], [K, 2, C], [B, 2, C], or [B, K, 2, C], "
        f"got {tuple(text_features.shape)}"
    )


def _prototype_logits(
    tokens: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float,
    per_image_text: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    tokens = F.normalize(tokens.float(), dim=-1, eps=eps)
    batch = tokens.size(0)
    prepared, ensemble = _prepare_text_features(
        text_features,
        batch=batch,
        per_image=per_image_text,
        eps=eps,
    )
    if ensemble:
        logits = torch.einsum("bnc,bkdc->bnkd", tokens, prepared).mean(dim=2)
    else:
        logits = torch.einsum("bnc,bdc->bnd", tokens, prepared)
    return logits / max(float(temperature), eps)


def _select_anchor_tokens(
    selected_patch_features: Sequence[torch.Tensor],
    anchor_layers: str,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, int, int]:
    if len(selected_patch_features) == 0:
        raise ValueError("selected_patch_features must contain at least one layer")
    if anchor_layers == "last":
        tokens, height, width = _extract_spatial_tokens(selected_patch_features[-1])
        return F.normalize(tokens.float(), dim=-1, eps=eps), height, width
    if anchor_layers != "mean":
        raise ValueError(f"unsupported proto_anchor_layers: {anchor_layers}")

    token_list = []
    height = width = None
    for patch_features in selected_patch_features:
        tokens, h, w = _extract_spatial_tokens(patch_features)
        if height is None:
            height, width = h, w
        elif (h, w) != (height, width):
            raise ValueError("all selected layers must share the same patch grid for mean anchors")
        token_list.append(F.normalize(tokens.float(), dim=-1, eps=eps))
    tokens = F.normalize(torch.stack(token_list, dim=0).mean(dim=0), dim=-1, eps=eps)
    return tokens, height, width


def _weighted_visual_prototype(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    topk_ratio: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0.0 < float(topk_ratio) <= 1.0):
        raise ValueError(f"proto_topk_ratio must be in (0, 1], got {topk_ratio}")
    batch, num_tokens, channels = tokens.shape
    k = max(1, int(math.ceil(num_tokens * float(topk_ratio))))
    top_weights, top_idx = torch.topk(weights, k=k, dim=1)
    gather_idx = top_idx.unsqueeze(-1).expand(batch, k, channels)
    top_tokens = torch.gather(tokens, dim=1, index=gather_idx)
    norm_weights = top_weights / top_weights.sum(dim=1, keepdim=True).clamp_min(eps)
    prototype = (top_tokens * norm_weights.unsqueeze(-1)).sum(dim=1)
    prototype = F.normalize(prototype, dim=-1, eps=eps)
    mask = torch.zeros_like(weights, dtype=torch.bool)
    mask.scatter_(1, top_idx, True)
    return prototype, top_weights.mean(dim=1), mask


def _calibrate_text_features(
    text_features: torch.Tensor,
    visual_normal: torch.Tensor,
    visual_abnormal: torch.Tensor,
    normal_confidence: torch.Tensor,
    abnormal_confidence: torch.Tensor,
    alpha0: float,
    beta0: float,
    tau_a: float,
    update_min_abnormal_confidence: float,
    conservative_update: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch = visual_normal.size(0)
    prepared, ensemble = _prepare_text_features(
        text_features,
        batch=batch,
        per_image=False,
        eps=eps,
    )
    if conservative_update:
        image_update_mask = abnormal_confidence >= float(update_min_abnormal_confidence)
        beta = float(beta0) * normal_confidence * image_update_mask.to(normal_confidence.dtype)
        alpha = float(alpha0) * abnormal_confidence * image_update_mask.to(abnormal_confidence.dtype)
    else:
        beta = torch.full_like(normal_confidence, float(beta0))
        alpha = torch.full_like(abnormal_confidence, float(alpha0))
    beta = beta.clamp(0.0, 1.0).view(batch, *([1, 1] if ensemble else [1]))
    alpha = alpha.clamp(0.0, 1.0).view(batch, *([1, 1] if ensemble else [1]))

    normal_target = visual_normal.view(batch, *([1, -1] if ensemble else [-1]))
    abnormal_target = visual_abnormal.view(batch, *([1, -1] if ensemble else [-1]))
    normal = F.normalize((1.0 - beta) * prepared[..., 0, :] + beta * normal_target, dim=-1, eps=eps)
    abnormal_candidate = F.normalize(
        (1.0 - alpha) * prepared[..., 1, :] + alpha * abnormal_target,
        dim=-1,
        eps=eps,
    )
    if conservative_update:
        update_mask = (abnormal_confidence >= max(float(tau_a), float(update_min_abnormal_confidence))).view(
            batch,
            *([1] if ensemble else []),
            1,
        )
        abnormal = torch.where(update_mask, abnormal_candidate, prepared[..., 1, :])
    else:
        abnormal = abnormal_candidate
    return torch.stack([normal, abnormal], dim=-2)


def _final_map_from_layers(
    selected_patch_features: Sequence[torch.Tensor],
    adapted_text_features: torch.Tensor,
    image_size,
    temperature: float,
    layer_fusion: str,
    eps: float = 1e-6,
) -> torch.Tensor:
    output_hw = _as_hw(image_size)
    maps = []
    for patch_features in selected_patch_features:
        tokens, height, width = _extract_spatial_tokens(patch_features)
        logits = _prototype_logits(
            tokens,
            adapted_text_features,
            temperature=temperature,
            per_image_text=True,
            eps=eps,
        )
        scores = logits.softmax(dim=-1)[..., 1].view(tokens.size(0), height, width)
        score_map = F.interpolate(
            scores.unsqueeze(1),
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        maps.append(score_map)
    stacked = torch.stack(maps, dim=0)
    if layer_fusion == "sum":
        return stacked.sum(dim=0)
    if layer_fusion == "mean":
        return stacked.mean(dim=0)
    raise ValueError(f"unsupported proto_layer_fusion: {layer_fusion}")


def apply_wavelet_prototype_adaptation(
    selected_patch_features: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    image_size,
    temperature: float = 0.07,
    gamma: float = 1.0,
    eta: float = 1.0,
    topk_ratio: float = 0.20,
    alpha0: float = 0.0,
    beta0: float = 0.01,
    tau_a: float = 0.15,
    update_min_abnormal_confidence: float = 0.06,
    wavelet_mix: float = 0.05,
    wavelet_mode: str = "boundary_aware",
    conservative_update: bool = True,
    anchor_layers: str = "last",
    layer_fusion: str = "sum",
    clip_percentile_low: float = 1.0,
    clip_percentile_high: float = 99.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply training-free wavelet-supervised text prototype adaptation.

    The returned map is recomputed from calibrated normal/abnormal prototypes.
    No gradients or model parameter updates are used.
    """
    tokens, height, width = _select_anchor_tokens(
        selected_patch_features,
        anchor_layers=anchor_layers,
        eps=eps,
    )
    logits = _prototype_logits(tokens, text_features, temperature=temperature, eps=eps)
    s0 = logits.softmax(dim=-1)[..., 1].clamp(0.0, 1.0)
    w, wavelet_diagnostics = _haar_patch_reliability(
        tokens,
        height=height,
        width=width,
        wavelet_mode=wavelet_mode,
        clip_percentile_low=clip_percentile_low,
        clip_percentile_high=clip_percentile_high,
        eps=eps,
    )

    semantic_abnormal = s0.clamp_min(eps).pow(float(gamma))
    semantic_normal = (1.0 - s0).clamp_min(eps).pow(float(gamma))
    wavelet_mix = max(0.0, min(float(wavelet_mix), 1.0))
    if wavelet_mode == "none" or wavelet_mix <= 0.0:
        abnormal_weights = semantic_abnormal
        normal_weights = semantic_normal
    else:
        abnormal_wavelet = ((1.0 - wavelet_mix) + wavelet_mix * w).clamp_min(eps)
        normal_wavelet = ((1.0 - wavelet_mix) + wavelet_mix * (1.0 - w)).clamp_min(eps)
        abnormal_weights = semantic_abnormal * abnormal_wavelet.pow(float(eta))
        normal_weights = semantic_normal * normal_wavelet.pow(float(eta))

    visual_abnormal, abnormal_confidence, abnormal_mask = _weighted_visual_prototype(
        tokens,
        abnormal_weights,
        topk_ratio=topk_ratio,
        eps=eps,
    )
    visual_normal, normal_confidence, normal_mask = _weighted_visual_prototype(
        tokens,
        normal_weights,
        topk_ratio=topk_ratio,
        eps=eps,
    )
    adapted_text_features = _calibrate_text_features(
        text_features,
        visual_normal=visual_normal,
        visual_abnormal=visual_abnormal,
        normal_confidence=normal_confidence,
        abnormal_confidence=abnormal_confidence,
        alpha0=alpha0,
        beta0=beta0,
        tau_a=tau_a,
        update_min_abnormal_confidence=update_min_abnormal_confidence,
        conservative_update=conservative_update,
        eps=eps,
    )
    final_map = _final_map_from_layers(
        selected_patch_features,
        adapted_text_features,
        image_size=image_size,
        temperature=temperature,
        layer_fusion=layer_fusion,
        eps=eps,
    )
    diagnostics = {
        "s0": s0.view(tokens.size(0), height, width).detach(),
        "w": wavelet_diagnostics["w"].detach(),
        "hf": wavelet_diagnostics["hf"].detach(),
        "lf_edge": wavelet_diagnostics["lf_edge"].detach(),
        "abnormal_weights": abnormal_weights.view(tokens.size(0), height, width).detach(),
        "normal_weights": normal_weights.view(tokens.size(0), height, width).detach(),
        "abnormal_mask": abnormal_mask.view(tokens.size(0), height, width).detach(),
        "normal_mask": normal_mask.view(tokens.size(0), height, width).detach(),
        "abnormal_confidence": abnormal_confidence.detach(),
        "normal_confidence": normal_confidence.detach(),
    }
    return final_map, adapted_text_features, diagnostics


def apply_direct_wavelet_fusion(
    selected_patch_features: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    image_size,
    temperature: float = 0.07,
    weight: float = 0.5,
    wavelet_mode: str = "boundary_aware",
    anchor_layers: str = "last",
    clip_percentile_low: float = 1.0,
    clip_percentile_high: float = 99.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Direct S0/W map fusion used only as an ablation baseline."""
    if wavelet_mode == "none":
        raise ValueError("direct wavelet fusion needs hf_only or boundary_aware wavelet mode")
    tokens, height, width = _select_anchor_tokens(
        selected_patch_features,
        anchor_layers=anchor_layers,
        eps=eps,
    )
    logits = _prototype_logits(tokens, text_features, temperature=temperature, eps=eps)
    s0 = logits.softmax(dim=-1)[..., 1].clamp(0.0, 1.0)
    w, wavelet_diagnostics = _haar_patch_reliability(
        tokens,
        height=height,
        width=width,
        wavelet_mode=wavelet_mode,
        clip_percentile_low=clip_percentile_low,
        clip_percentile_high=clip_percentile_high,
        eps=eps,
    )
    weight = max(0.0, min(float(weight), 1.0))
    fused = ((1.0 - weight) * s0 + weight * w).view(tokens.size(0), height, width)
    fused = F.interpolate(
        fused.unsqueeze(1),
        size=_as_hw(image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    diagnostics = {
        "s0": s0.view(tokens.size(0), height, width).detach(),
        "w": wavelet_diagnostics["w"].detach(),
        "hf": wavelet_diagnostics["hf"].detach(),
        "lf_edge": wavelet_diagnostics["lf_edge"].detach(),
    }
    return fused, diagnostics


def compute_image_text_prob_with_adapted_prototypes(
    image_features: torch.Tensor,
    adapted_text_features: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-6,
) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1, eps=eps)
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)
    batch = image_features.size(0)
    prepared, ensemble = _prepare_text_features(
        adapted_text_features,
        batch=batch,
        per_image=True,
        eps=eps,
    )
    if ensemble:
        logits = torch.einsum("bc,bkdc->bkd", image_features, prepared).mean(dim=1)
    else:
        logits = torch.einsum("bc,bdc->bd", image_features, prepared)
    return (logits / max(float(temperature), eps)).softmax(dim=-1)[:, 1]
