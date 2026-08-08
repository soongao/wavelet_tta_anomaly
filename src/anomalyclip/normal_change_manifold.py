import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NormalChangeManifold:
    """Local atlas of normal patch variation."""

    anchors: torch.Tensor
    bases: torch.Tensor
    scales: torch.Tensor
    tangent_rank: int
    metadata: Dict[str, object]

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "NormalChangeManifold":
        target_dtype = dtype or self.anchors.dtype
        return NormalChangeManifold(
            anchors=self.anchors.to(device=device, dtype=target_dtype),
            bases=self.bases.to(device=device, dtype=target_dtype),
            scales=self.scales.to(device=device, dtype=target_dtype),
            tangent_rank=self.tangent_rank,
            metadata=dict(self.metadata),
        )


def _as_hw(size) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"image_size must be an int or a pair, got {size}")
    return int(size[0]), int(size[1])


def extract_spatial_tokens(patch_features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
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


def select_manifold_tokens(
    selected_patch_features: Sequence[torch.Tensor],
    anchor_layers: str = "mean",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, int, int]:
    if len(selected_patch_features) == 0:
        raise ValueError("selected_patch_features must contain at least one layer")
    if anchor_layers == "last":
        tokens, height, width = extract_spatial_tokens(selected_patch_features[-1])
        return F.normalize(tokens.float(), dim=-1, eps=eps), height, width
    if anchor_layers != "mean":
        raise ValueError(f"unsupported NCMA anchor_layers: {anchor_layers}")

    token_list = []
    height = width = None
    for patch_features in selected_patch_features:
        tokens, h, w = extract_spatial_tokens(patch_features)
        if height is None:
            height, width = h, w
        elif (h, w) != (height, width):
            raise ValueError("all selected layers must share the same patch grid")
        token_list.append(F.normalize(tokens.float(), dim=-1, eps=eps))
    tokens = F.normalize(torch.stack(token_list, dim=0).mean(dim=0), dim=-1, eps=eps)
    return tokens, height, width


def _resize_scores_to_tokens(score_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if score_map.dim() == 3:
        score_map = score_map.unsqueeze(1)
    if score_map.dim() != 4 or score_map.size(1) != 1:
        raise ValueError(
            f"score_map must be [B, H, W] or [B, 1, H, W], got {tuple(score_map.shape)}"
        )
    if score_map.shape[-2:] != (height, width):
        score_map = F.interpolate(
            score_map.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return score_map.flatten(1)


def _normalize_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"x must be [B, H, W], got {tuple(x.shape)}")
    batch = x.size(0)
    flat = x.flatten(1)
    min_val = flat.min(dim=1)[0].view(batch, 1, 1)
    max_val = flat.max(dim=1)[0].view(batch, 1, 1)
    return ((x - min_val) / (max_val - min_val + eps)).clamp(0.0, 1.0)


def _sample_rows(tokens: torch.Tensor, max_tokens: int, seed: int = 0) -> torch.Tensor:
    if max_tokens <= 0 or tokens.size(0) <= max_tokens:
        return tokens
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed))
    idx = torch.randperm(tokens.size(0), generator=generator, device=tokens.device)[:max_tokens]
    return tokens.index_select(0, idx)


def _fit_anchors(
    tokens: torch.Tensor,
    num_anchors: int,
    kmeans_iters: int,
    seed: int,
    eps: float,
) -> torch.Tensor:
    tokens = F.normalize(tokens.float(), dim=-1, eps=eps)
    num_anchors = min(max(1, int(num_anchors)), tokens.size(0))
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed))
    init_idx = torch.randperm(tokens.size(0), generator=generator, device=tokens.device)[:num_anchors]
    anchors = tokens.index_select(0, init_idx).clone()

    for _ in range(max(0, int(kmeans_iters))):
        assignment = (tokens @ anchors.t()).argmax(dim=1)
        updated = []
        for anchor_idx in range(num_anchors):
            members = tokens[assignment == anchor_idx]
            if members.numel() == 0:
                updated.append(anchors[anchor_idx])
            else:
                updated.append(members.mean(dim=0))
        anchors = F.normalize(torch.stack(updated, dim=0), dim=-1, eps=eps)
    return anchors


def fit_normal_change_manifold_from_tokens(
    tokens: torch.Tensor,
    num_anchors: int = 64,
    tangent_rank: int = 8,
    pca_neighbors: int = 128,
    max_fit_tokens: int = 8192,
    kmeans_iters: int = 5,
    seed: int = 0,
    eps: float = 1e-6,
    metadata: Optional[Dict[str, object]] = None,
) -> NormalChangeManifold:
    """Fit a local normal-change manifold atlas from selected normal patch tokens."""
    if tokens.dim() != 2:
        raise ValueError(f"tokens must be [M, C], got {tuple(tokens.shape)}")
    if tokens.size(0) == 0:
        raise ValueError("cannot fit a normal-change manifold from zero tokens")

    tokens = _sample_rows(tokens, max_tokens=max_fit_tokens, seed=seed)
    tokens = F.normalize(tokens.float(), dim=-1, eps=eps)
    anchors = _fit_anchors(
        tokens,
        num_anchors=num_anchors,
        kmeans_iters=kmeans_iters,
        seed=seed,
        eps=eps,
    )

    rank = max(0, min(int(tangent_rank), tokens.size(1)))
    neighbor_count = min(max(rank + 2, int(pca_neighbors)), tokens.size(0))
    basis_list = []
    scale_list = []
    similarities = tokens @ anchors.t()
    distances = (2.0 - 2.0 * similarities).clamp_min(0.0).sqrt()
    for anchor_idx in range(anchors.size(0)):
        nearest = torch.topk(
            similarities[:, anchor_idx],
            k=neighbor_count,
            largest=True,
        ).indices
        neighbors = tokens.index_select(0, nearest)
        centered = neighbors - anchors[anchor_idx].unsqueeze(0)
        if rank > 0 and centered.size(0) > 1:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            basis = F.normalize(vh[:rank], dim=-1, eps=eps)
            if basis.size(0) < rank:
                pad = torch.zeros(
                    rank - basis.size(0),
                    tokens.size(1),
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                basis = torch.cat([basis, pad], dim=0)
            coeff = centered @ basis.t()
            tangent = coeff @ basis
            residual = (centered - tangent).norm(dim=-1)
        else:
            basis = torch.zeros(rank, tokens.size(1), dtype=tokens.dtype, device=tokens.device)
            residual = centered.norm(dim=-1)
        local_scale = torch.quantile(residual.float(), 0.75).clamp_min(eps)
        local_scale = torch.maximum(local_scale, distances[:, anchor_idx].median().clamp_min(eps))
        basis_list.append(basis)
        scale_list.append(local_scale)

    payload_metadata = dict(metadata or {})
    payload_metadata.update(
        {
            "num_anchors": int(anchors.size(0)),
            "tangent_rank": int(rank),
            "pca_neighbors": int(neighbor_count),
            "max_fit_tokens": int(max_fit_tokens),
        }
    )
    return NormalChangeManifold(
        anchors=anchors.detach(),
        bases=torch.stack(basis_list, dim=0).detach(),
        scales=torch.stack(scale_list, dim=0).detach(),
        tangent_rank=rank,
        metadata=payload_metadata,
    )


def fit_normal_change_manifold_from_patch_features(
    selected_patch_features: Sequence[torch.Tensor],
    anomaly_map: Optional[torch.Tensor] = None,
    normal_topk_ratio: float = 0.35,
    anchor_layers: str = "mean",
    num_anchors: int = 64,
    tangent_rank: int = 8,
    pca_neighbors: int = 128,
    max_fit_tokens: int = 8192,
    kmeans_iters: int = 5,
    seed: int = 0,
    eps: float = 1e-6,
    metadata: Optional[Dict[str, object]] = None,
) -> NormalChangeManifold:
    tokens, height, width = select_manifold_tokens(
        selected_patch_features,
        anchor_layers=anchor_layers,
        eps=eps,
    )
    batch, num_tokens, channels = tokens.shape
    if anomaly_map is None:
        normal_tokens = tokens.reshape(batch * num_tokens, channels)
    else:
        anomaly_scores = _resize_scores_to_tokens(anomaly_map, height, width)
        ratio = min(max(float(normal_topk_ratio), 0.0), 1.0)
        k = max(1, int(math.ceil(num_tokens * ratio)))
        normal_idx = torch.topk(1.0 - anomaly_scores, k=k, dim=1, largest=True).indices
        gather_idx = normal_idx.unsqueeze(-1).expand(batch, k, channels)
        normal_tokens = torch.gather(tokens, dim=1, index=gather_idx).reshape(batch * k, channels)

    fit_metadata = dict(metadata or {})
    fit_metadata.update(
        {
            "anchor_layers": anchor_layers,
            "normal_topk_ratio": float(normal_topk_ratio),
            "fit_grid": [int(height), int(width)],
        }
    )
    return fit_normal_change_manifold_from_tokens(
        normal_tokens,
        num_anchors=num_anchors,
        tangent_rank=tangent_rank,
        pca_neighbors=pca_neighbors,
        max_fit_tokens=max_fit_tokens,
        kmeans_iters=kmeans_iters,
        seed=seed,
        eps=eps,
        metadata=fit_metadata,
    )


def save_normal_change_manifold(manifold: NormalChangeManifold, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "anchors": manifold.anchors.detach().cpu(),
            "bases": manifold.bases.detach().cpu(),
            "scales": manifold.scales.detach().cpu(),
            "tangent_rank": int(manifold.tangent_rank),
            "metadata": dict(manifold.metadata),
        },
        path,
    )


def load_normal_change_manifold(path: str, map_location="cpu") -> NormalChangeManifold:
    payload = torch.load(path, map_location=map_location)
    return NormalChangeManifold(
        anchors=payload["anchors"],
        bases=payload["bases"],
        scales=payload["scales"],
        tangent_rank=int(payload.get("tangent_rank", payload["bases"].size(1))),
        metadata=dict(payload.get("metadata", {})),
    )


def project_tokens_to_manifold(
    tokens: torch.Tensor,
    manifold: NormalChangeManifold,
    residual_temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be [B, N, C], got {tuple(tokens.shape)}")
    tokens = F.normalize(tokens.float(), dim=-1, eps=eps)
    atlas = manifold.to(tokens.device, dtype=tokens.dtype)
    anchors = F.normalize(atlas.anchors, dim=-1, eps=eps)
    assignment = torch.einsum("bnc,kc->bnk", tokens, anchors).argmax(dim=-1)
    assigned_anchors = anchors.index_select(0, assignment.reshape(-1)).view_as(tokens)
    assigned_scales = atlas.scales.index_select(0, assignment.reshape(-1)).view(tokens.size(0), tokens.size(1))
    assigned_bases = atlas.bases.index_select(0, assignment.reshape(-1)).view(
        tokens.size(0),
        tokens.size(1),
        atlas.bases.size(1),
        tokens.size(2),
    )

    delta = tokens - assigned_anchors
    if assigned_bases.size(2) > 0:
        coeff = torch.einsum("bnc,bnrc->bnr", delta, assigned_bases)
        tangent = torch.einsum("bnr,bnrc->bnc", coeff, assigned_bases)
    else:
        tangent = torch.zeros_like(delta)
    projected = F.normalize(assigned_anchors + tangent, dim=-1, eps=eps)
    residual = (delta - tangent).norm(dim=-1) / assigned_scales.clamp_min(eps)
    compatibility = torch.exp(-residual / max(float(residual_temperature), eps)).clamp(0.0, 1.0)
    return projected, residual, compatibility, assignment


def _prepare_text_features(
    text_features: torch.Tensor,
    batch: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, bool]:
    text_features = F.normalize(text_features.float(), dim=-1, eps=eps)
    if text_features.dim() == 2:
        if text_features.size(0) != 2:
            raise ValueError(f"text_features must be [2, C] or [K, 2, C], got {tuple(text_features.shape)}")
        return text_features.unsqueeze(0).expand(batch, -1, -1), False
    if text_features.dim() == 3:
        if text_features.size(1) != 2:
            raise ValueError(f"text_features must be [B, 2, C] or [K, 2, C], got {tuple(text_features.shape)}")
        if text_features.size(0) == batch:
            return text_features, False
        return text_features.unsqueeze(0).expand(batch, -1, -1, -1), True
    if text_features.dim() == 4:
        if text_features.size(0) != batch or text_features.size(2) != 2:
            raise ValueError(f"per-image ensemble text_features must be [B, K, 2, C], got {tuple(text_features.shape)}")
        return text_features, True
    raise ValueError(f"unsupported text_features shape: {tuple(text_features.shape)}")


def _prototype_logits(
    tokens: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float,
    per_image_text: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    tokens = F.normalize(tokens.float(), dim=-1, eps=eps)
    batch = tokens.size(0)
    prepared, ensemble = _prepare_text_features(text_features, batch=batch, eps=eps)
    if text_features.dim() == 3 and text_features.size(0) == batch:
        per_image_text = True
    if ensemble:
        logits = torch.einsum("bnc,bkdc->bnkd", tokens, prepared).mean(dim=2)
    elif per_image_text:
        logits = torch.einsum("bnc,bdc->bnd", tokens, prepared)
    else:
        logits = torch.einsum("bnc,bdc->bnd", tokens, prepared)
    return logits / max(float(temperature), eps)


def _weighted_prototype(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    topk_ratio: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0.0 < float(topk_ratio) <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
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
        tokens, height, width = extract_spatial_tokens(patch_features)
        logits = _prototype_logits(
            tokens,
            adapted_text_features,
            temperature=temperature,
            per_image_text=True,
            eps=eps,
        )
        scores = logits.softmax(dim=-1)[..., 1].view(tokens.size(0), height, width)
        maps.append(
            F.interpolate(
                scores.unsqueeze(1),
                size=output_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        )
    stacked = torch.stack(maps, dim=0)
    if layer_fusion == "sum":
        return stacked.sum(dim=0)
    if layer_fusion == "mean":
        return stacked.mean(dim=0)
    raise ValueError(f"unsupported NCMA layer_fusion: {layer_fusion}")


def apply_ncma_adaptation(
    selected_patch_features: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    base_anomaly_map: torch.Tensor,
    image_size,
    manifold: Optional[NormalChangeManifold] = None,
    fit_from_sample: bool = False,
    anchor_layers: str = "mean",
    num_anchors: int = 64,
    tangent_rank: int = 8,
    pca_neighbors: int = 128,
    fit_topk_ratio: float = 0.35,
    max_fit_tokens: int = 8192,
    kmeans_iters: int = 5,
    residual_temperature: float = 1.0,
    update_topk_ratio: float = 0.20,
    update_alpha: float = 0.10,
    normal_gamma: float = 1.0,
    residual_weight: float = 0.35,
    temperature: float = 0.07,
    layer_fusion: str = "sum",
    use_projection_update: bool = True,
    use_residual_score: bool = True,
    update_abnormal: bool = False,
    abnormal_alpha: float = 0.05,
    seed: int = 0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply NCMA: update from on-manifold projections, score off-manifold residuals."""
    tokens, height, width = select_manifold_tokens(
        selected_patch_features,
        anchor_layers=anchor_layers,
        eps=eps,
    )
    if manifold is None:
        if not fit_from_sample:
            raise ValueError(
                "NCMA strict zero-shot evaluation needs --ncma_fit_from_sample "
                "when no optional manifold path is provided."
            )
        manifold = fit_normal_change_manifold_from_patch_features(
            selected_patch_features,
            anomaly_map=base_anomaly_map,
            normal_topk_ratio=fit_topk_ratio,
            anchor_layers=anchor_layers,
            num_anchors=num_anchors,
            tangent_rank=tangent_rank,
            pca_neighbors=pca_neighbors,
            max_fit_tokens=max_fit_tokens,
            kmeans_iters=kmeans_iters,
            seed=seed,
            eps=eps,
            metadata={"source": "test_sample_bootstrap"},
        )

    projected_tokens, residual, compatibility, assignment = project_tokens_to_manifold(
        tokens,
        manifold,
        residual_temperature=residual_temperature,
        eps=eps,
    )
    logits = _prototype_logits(tokens, text_features, temperature=temperature, eps=eps)
    semantic_abnormal = logits.softmax(dim=-1)[..., 1].clamp(0.0, 1.0)
    normal_weights = compatibility * (1.0 - semantic_abnormal).clamp_min(eps).pow(float(normal_gamma))
    normal_target, normal_confidence, normal_mask = _weighted_prototype(
        projected_tokens,
        normal_weights,
        topk_ratio=update_topk_ratio,
        eps=eps,
    )

    batch = tokens.size(0)
    prepared, ensemble = _prepare_text_features(text_features, batch=batch, eps=eps)
    if ensemble:
        target = normal_target.view(batch, 1, -1)
        alpha = (float(update_alpha) * normal_confidence).clamp(0.0, 1.0).view(batch, 1, 1)
        updated_normal = F.normalize((1.0 - alpha) * prepared[..., 0, :] + alpha * target, dim=-1, eps=eps)
        adapted = prepared.clone()
        if use_projection_update:
            adapted[..., 0, :] = updated_normal
    else:
        alpha = (float(update_alpha) * normal_confidence).clamp(0.0, 1.0).view(batch, 1)
        updated_normal = F.normalize((1.0 - alpha) * prepared[:, 0, :] + alpha * normal_target, dim=-1, eps=eps)
        adapted = prepared.clone()
        if use_projection_update:
            adapted[:, 0, :] = updated_normal

    if update_abnormal:
        residual_weights = residual * semantic_abnormal.clamp_min(eps)
        abnormal_target, abnormal_confidence, abnormal_mask = _weighted_prototype(
            tokens.detach(),
            residual_weights,
            topk_ratio=update_topk_ratio,
            eps=eps,
        )
        if ensemble:
            a_alpha = (float(abnormal_alpha) * abnormal_confidence).clamp(0.0, 1.0).view(batch, 1, 1)
            target = abnormal_target.view(batch, 1, -1)
            adapted[..., 1, :] = F.normalize(
                (1.0 - a_alpha) * adapted[..., 1, :] + a_alpha * target,
                dim=-1,
                eps=eps,
            )
        else:
            a_alpha = (float(abnormal_alpha) * abnormal_confidence).clamp(0.0, 1.0).view(batch, 1)
            adapted[:, 1, :] = F.normalize(
                (1.0 - a_alpha) * adapted[:, 1, :] + a_alpha * abnormal_target,
                dim=-1,
                eps=eps,
            )
    else:
        abnormal_confidence = torch.zeros(batch, dtype=tokens.dtype, device=tokens.device)
        abnormal_mask = torch.zeros_like(normal_mask)

    semantic_map = _final_map_from_layers(
        selected_patch_features,
        adapted,
        image_size=image_size,
        temperature=temperature,
        layer_fusion=layer_fusion,
        eps=eps,
    )
    residual_map = _normalize_per_image(residual.view(batch, height, width), eps=eps)
    residual_map = F.interpolate(
        residual_map.unsqueeze(1),
        size=_as_hw(image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    if use_residual_score:
        weight = min(max(float(residual_weight), 0.0), 1.0)
        final_map = ((1.0 - weight) * semantic_map + weight * residual_map).clamp_min(0.0)
    else:
        final_map = semantic_map

    diagnostics = {
        "semantic_abnormal": semantic_abnormal.view(batch, height, width).detach(),
        "on_manifold_compatibility": compatibility.view(batch, height, width).detach(),
        "off_manifold_residual": residual.view(batch, height, width).detach(),
        "off_manifold_residual_map": residual_map.detach(),
        "normal_update_mask": normal_mask.view(batch, height, width).detach(),
        "abnormal_update_mask": abnormal_mask.view(batch, height, width).detach(),
        "normal_confidence": normal_confidence.detach(),
        "abnormal_confidence": abnormal_confidence.detach(),
        "assignment": assignment.detach(),
    }
    return final_map, adapted.detach(), diagnostics


def compute_image_text_prob_with_ncma(
    image_features: torch.Tensor,
    adapted_text_features: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-6,
) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1, eps=eps)
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)
    batch = image_features.size(0)
    prepared, ensemble = _prepare_text_features(adapted_text_features, batch=batch, eps=eps)
    if ensemble:
        logits = torch.einsum("bc,bkdc->bkd", image_features, prepared).mean(dim=1)
    else:
        logits = torch.einsum("bc,bdc->bd", image_features, prepared)
    return (logits / max(float(temperature), eps)).softmax(dim=-1)[:, 1]
