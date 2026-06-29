import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F


SizeLike = Union[int, Tuple[int, int]]


def _as_hw(size: SizeLike) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"output size must be an int or a pair, got {size}")
    return int(size[0]), int(size[1])


def _as_b1hw(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) == 1:
        return x
    raise ValueError(f"{name} must be [B, H, W] or [B, 1, H, W], got {tuple(x.shape)}")


def _minmax_norm_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each image independently to [0, 1]."""
    x = _as_b1hw(x, "x")
    b = x.size(0)
    flat = x.flatten(1)
    x_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
    x_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return ((x - x_min) / (x_max - x_min + eps)).clamp(0.0, 1.0)


def _local_positive_residual(
    x: torch.Tensor,
    kernel_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a normalized positive residual against a local average."""
    x = _as_b1hw(x, "x")
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    local_mean = F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    residual = (x - local_mean).clamp_min(0.0)
    return _minmax_norm_per_image(residual, eps=eps)


def _clip_topk_mask(
    clip_prior: torch.Tensor,
    topk_ratio: float,
) -> torch.Tensor:
    """Return a hard top-k mask from a normalized CLIP anomaly prior."""
    clip_prior = _as_b1hw(clip_prior, "clip_prior")
    if topk_ratio <= 0.0 or topk_ratio >= 1.0:
        return torch.ones_like(clip_prior)

    batch, _, height, width = clip_prior.shape
    flat = clip_prior.flatten(1)
    k = max(1, int(flat.size(1) * topk_ratio))
    threshold = torch.topk(flat, k=k, dim=1).values[:, -1].view(batch, 1, 1, 1)
    return (clip_prior >= threshold).to(dtype=clip_prior.dtype).view(batch, 1, height, width)


def _clip_rank_gate(
    clip_prior: torch.Tensor,
    topk_ratio: float,
    mode: str = "hard",
    temperature: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gate texture evidence by the CLIP anomaly ranking.

    ``hard`` is the previous top-k mask. ``soft`` keeps the same top-k
    threshold but uses a sigmoid transition, which preserves weak but
    CLIP-consistent texture evidence near the boundary.
    """
    clip_prior = _as_b1hw(clip_prior, "clip_prior")
    if mode == "hard":
        return _clip_topk_mask(clip_prior, topk_ratio)
    if mode != "soft":
        raise ValueError(f"unsupported rank gate mode: {mode}")
    if topk_ratio <= 0.0 or topk_ratio >= 1.0:
        return torch.ones_like(clip_prior)

    batch, _, height, width = clip_prior.shape
    flat = clip_prior.flatten(1)
    k = max(1, int(flat.size(1) * topk_ratio))
    threshold = torch.topk(flat, k=k, dim=1).values[:, -1].view(batch, 1, 1, 1)
    scale = max(float(temperature), eps)
    gate = torch.sigmoid((clip_prior - threshold) / scale)
    gate = gate / gate.flatten(1).max(dim=1)[0].view(batch, 1, 1, 1).clamp_min(eps)
    return gate.view(batch, 1, height, width)


def _extract_spatial_tokens(patch_features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Return spatial patch tokens and their square grid size.

    AnomalyCLIP patch features normally include a leading CLS token:
    [B, 1 + H * W, C]. This function drops it when present.
    """
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


def _haar_low_high_components(
    patch_features: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return feature-space Haar low-frequency tokens and high-frequency energy."""
    spatial_tokens, height, width = _extract_spatial_tokens(patch_features)
    spatial_tokens = F.normalize(spatial_tokens.float(), dim=-1, eps=eps)

    b, _, channels = spatial_tokens.shape
    feature_grid = spatial_tokens.transpose(1, 2).reshape(b, channels, height, width)

    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        feature_grid = F.pad(feature_grid, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = feature_grid[:, :, 0::2, 0::2]
    x01 = feature_grid[:, :, 0::2, 1::2]
    x10 = feature_grid[:, :, 1::2, 0::2]
    x11 = feature_grid[:, :, 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)

    high_energy = torch.sqrt((lh.square() + hl.square() + hh.square()).mean(dim=1, keepdim=True) + eps)
    return ll, high_energy


def _haar_grid_low_high_components(
    feature_grid: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one Haar split to an already spatial feature grid."""
    if feature_grid.dim() != 4:
        raise ValueError(f"feature_grid must be [B, C, H, W], got {tuple(feature_grid.shape)}")

    height, width = feature_grid.shape[-2:]
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        feature_grid = F.pad(feature_grid, (0, pad_w, 0, pad_h), mode="replicate")

    x00 = feature_grid[:, :, 0::2, 0::2]
    x01 = feature_grid[:, :, 0::2, 1::2]
    x10 = feature_grid[:, :, 1::2, 0::2]
    x11 = feature_grid[:, :, 1::2, 1::2]

    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    high_energy = torch.sqrt((lh.square() + hl.square() + hh.square()).mean(dim=1, keepdim=True) + eps)
    return ll, high_energy


def _feature_grid_from_patch_features(
    patch_features: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    spatial_tokens, height, width = _extract_spatial_tokens(patch_features)
    spatial_tokens = F.normalize(spatial_tokens.float(), dim=-1, eps=eps)
    batch, _, channels = spatial_tokens.shape
    return spatial_tokens.transpose(1, 2).reshape(batch, channels, height, width)


def _multi_scale_haar_high_frequency_energies(
    patch_features: torch.Tensor,
    levels: int = 1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, ...]:
    """Return Haar high-frequency energy maps from repeated LL decomposition."""
    levels = max(1, int(levels))
    feature_grid = _feature_grid_from_patch_features(patch_features, eps=eps)
    energies = []
    current = feature_grid
    for _ in range(levels):
        current, high_energy = _haar_grid_low_high_components(current, eps=eps)
        energies.append(high_energy)
        if current.shape[-2] <= 1 and current.shape[-1] <= 1:
            break
    return tuple(energies)


def _haar_high_frequency_energy(patch_features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a feature-space Haar high-frequency energy map.

    The returned tensor is [B, 1, ceil(H / 2), ceil(W / 2)]. Odd patch grids
    are padded by one row/column before the 2D Haar split.
    """
    _, high_energy = _haar_low_high_components(patch_features, eps=eps)
    return high_energy


def _low_frequency_structure_edge(low_frequency: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Estimate semantic structure edges from low-frequency feature changes."""
    b, _, height, width = low_frequency.shape
    grad_y = torch.zeros(b, 1, height, width, dtype=low_frequency.dtype, device=low_frequency.device)
    grad_x = torch.zeros_like(grad_y)

    if height > 1:
        dy = low_frequency[:, :, 1:, :] - low_frequency[:, :, :-1, :]
        grad_y[:, :, 1:, :] = torch.sqrt(dy.square().mean(dim=1, keepdim=True) + eps)
    if width > 1:
        dx = low_frequency[:, :, :, 1:] - low_frequency[:, :, :, :-1]
        grad_x[:, :, :, 1:] = torch.sqrt(dx.square().mean(dim=1, keepdim=True) + eps)

    return torch.sqrt(grad_x.square() + grad_y.square() + eps)


def wavelet_gate_from_patch_features(
    patch_features: torch.Tensor,
    output_size: SizeLike,
    levels: int = 1,
    level_fusion: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a normalized wavelet gate from one layer of patch features.

    Returns:
        A [B, H, W] gate resized to ``output_size``. Higher values indicate
        stronger local high-frequency changes in CLIP feature space.
    """
    output_hw = _as_hw(output_size)
    energies = _multi_scale_haar_high_frequency_energies(
        patch_features,
        levels=levels,
        eps=eps,
    )
    gates = [
        F.interpolate(
            _minmax_norm_per_image(high_energy, eps=eps),
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )
        for high_energy in energies
    ]
    stacked = torch.stack(gates, dim=0)
    if level_fusion == "mean":
        gate = stacked.mean(dim=0)
    elif level_fusion == "sum":
        gate = stacked.sum(dim=0)
    else:
        raise ValueError(f"unsupported wavelet level fusion mode: {level_fusion}")
    gate = _minmax_norm_per_image(gate, eps=eps)
    return gate.squeeze(1)


def structure_texture_gates_from_patch_features(
    patch_features: torch.Tensor,
    output_size: SizeLike,
    edge_power: float = 1.0,
    levels: int = 1,
    level_fusion: str = "mean",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build structure and texture gates from one patch-feature layer.

    The texture route uses high-frequency Haar energy, but suppresses responses
    that coincide with low-frequency semantic structure edges. This separates
    local texture residuals from object boundaries and part edges.

    Returns:
        ``(texture_gate, structure_edge, high_gate)``, all shaped [B, H, W].
    """
    output_hw = _as_hw(output_size)
    levels = max(1, int(levels))
    current = _feature_grid_from_patch_features(patch_features, eps=eps)
    high_gates = []
    texture_gates = []
    structure_edges = []
    for _ in range(levels):
        current, high_energy = _haar_grid_low_high_components(current, eps=eps)
        structure_edge = _low_frequency_structure_edge(current, eps=eps)

        high_gate = _minmax_norm_per_image(high_energy, eps=eps)
        structure_edge = _minmax_norm_per_image(structure_edge, eps=eps)
        if edge_power > 0:
            texture_gate = high_gate * (1.0 - structure_edge).clamp(0.0, 1.0).pow(edge_power)
        else:
            texture_gate = high_gate
        texture_gate = _minmax_norm_per_image(texture_gate, eps=eps)

        high_gates.append(
            F.interpolate(high_gate, size=output_hw, mode="bilinear", align_corners=False)
        )
        structure_edges.append(
            F.interpolate(structure_edge, size=output_hw, mode="bilinear", align_corners=False)
        )
        texture_gates.append(
            F.interpolate(texture_gate, size=output_hw, mode="bilinear", align_corners=False)
        )
        if current.shape[-2] <= 1 and current.shape[-1] <= 1:
            break

    high_stack = torch.stack(high_gates, dim=0)
    structure_stack = torch.stack(structure_edges, dim=0)
    texture_stack = torch.stack(texture_gates, dim=0)
    if level_fusion == "mean":
        high_gate = high_stack.mean(dim=0)
        structure_edge = structure_stack.mean(dim=0)
        texture_gate = texture_stack.mean(dim=0)
    elif level_fusion == "sum":
        high_gate = high_stack.sum(dim=0)
        structure_edge = structure_stack.sum(dim=0)
        texture_gate = texture_stack.sum(dim=0)
    else:
        raise ValueError(f"unsupported wavelet level fusion mode: {level_fusion}")

    high_gate = _minmax_norm_per_image(high_gate, eps=eps)
    structure_edge = _minmax_norm_per_image(structure_edge, eps=eps)
    texture_gate = _minmax_norm_per_image(texture_gate, eps=eps)
    return texture_gate.squeeze(1), structure_edge.squeeze(1), high_gate.squeeze(1)


def fuse_wavelet_gates(gates: Iterable[torch.Tensor], mode: str = "mean", eps: float = 1e-6) -> torch.Tensor:
    """Fuse per-layer wavelet gates into one normalized [B, H, W] gate."""
    gates = list(gates)
    if len(gates) == 0:
        raise ValueError("at least one wavelet gate is required")

    stacked = torch.stack(gates, dim=0)
    if mode == "mean":
        fused = stacked.mean(dim=0)
    elif mode == "sum":
        fused = stacked.sum(dim=0)
    else:
        raise ValueError(f"unsupported wavelet fusion mode: {mode}")

    return _minmax_norm_per_image(fused, eps=eps).squeeze(1)


def compute_wavelet_reliability(
    anomaly_map: torch.Tensor,
    wavelet_gate: torch.Tensor,
    topk_ratio: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate per-image CLIP-wavelet agreement without labels.

    The reliability is high only when the CLIP anomaly map and the wavelet gate
    have both positive linear agreement and overlapping top-response regions.
    It is shaped [B, 1, 1, 1], so it can gate each image independently.
    """
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    anomaly_map_4d = _as_b1hw(anomaly_map, "anomaly_map")
    wavelet_gate_4d = _as_b1hw(wavelet_gate, "wavelet_gate")
    if wavelet_gate_4d.shape[-2:] != anomaly_map_4d.shape[-2:]:
        wavelet_gate_4d = F.interpolate(
            wavelet_gate_4d,
            size=anomaly_map_4d.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    anomaly_prior = _minmax_norm_per_image(anomaly_map_4d, eps=eps)
    wavelet_prior = _minmax_norm_per_image(wavelet_gate_4d, eps=eps)
    anomaly_flat = anomaly_prior.flatten(1)
    wavelet_flat = wavelet_prior.flatten(1)

    anomaly_centered = anomaly_flat - anomaly_flat.mean(dim=1, keepdim=True)
    wavelet_centered = wavelet_flat - wavelet_flat.mean(dim=1, keepdim=True)
    corr = (anomaly_centered * wavelet_centered).sum(dim=1)
    corr = corr / (
        anomaly_centered.square().sum(dim=1).sqrt()
        * wavelet_centered.square().sum(dim=1).sqrt()
        + eps
    )
    corr = corr.clamp(min=0.0, max=1.0)

    batch, num_pixels = anomaly_flat.shape
    k = max(1, int(num_pixels * topk_ratio))
    anomaly_idx = torch.topk(anomaly_flat, k=k, dim=1).indices
    wavelet_idx = torch.topk(wavelet_flat, k=k, dim=1).indices
    anomaly_mask = torch.zeros(batch, num_pixels, dtype=torch.bool, device=anomaly_flat.device)
    wavelet_mask = torch.zeros_like(anomaly_mask)
    anomaly_mask.scatter_(1, anomaly_idx, True)
    wavelet_mask.scatter_(1, wavelet_idx, True)
    topk_overlap = (anomaly_mask & wavelet_mask).sum(dim=1).float() / float(k)

    # Random top-k overlap is roughly topk_ratio, so remove that baseline.
    overlap = ((topk_overlap - topk_ratio) / (1.0 - topk_ratio + eps)).clamp(0.0, 1.0)
    reliability = torch.sqrt((corr * overlap).clamp_min(0.0) + eps)
    return reliability.view(batch, 1, 1, 1).clamp(0.0, 1.0)


def compute_texture_reliability(
    anomaly_map: torch.Tensor,
    texture_gate: torch.Tensor,
    topk_ratio: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate whether a texture residual is concentrated and CLIP-consistent."""
    texture_gate_4d = _as_b1hw(texture_gate, "texture_gate")
    texture_prior = _minmax_norm_per_image(texture_gate_4d, eps=eps)
    flat = texture_prior.flatten(1)
    k = max(1, int(flat.size(1) * topk_ratio))
    topk_mean = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
    global_mean = flat.mean(dim=1)
    concentration = ((topk_mean - global_mean) / (1.0 - global_mean + eps)).clamp(0.0, 1.0)

    agreement = compute_wavelet_reliability(
        anomaly_map,
        texture_gate,
        topk_ratio=topk_ratio,
        eps=eps,
    ).view(-1)
    reliability = torch.sqrt((concentration * agreement).clamp_min(0.0) + eps)
    return reliability.view(-1, 1, 1, 1).clamp(0.0, 1.0)


def global_image_confidence_gate(
    image_score: torch.Tensor,
    power: float = 1.0,
    min_gate: float = 0.0,
    max_gate: float = 1.0,
) -> torch.Tensor:
    """Convert image-level anomaly confidence to a [B, 1, 1, 1] gate."""
    gate = image_score.float().view(-1, 1, 1, 1).clamp(0.0, 1.0)
    if power > 0:
        gate = gate.pow(power)
    return gate.clamp(float(min_gate), float(max_gate))


def topk_pixel_score(
    anomaly_map: torch.Tensor,
    topk_ratio: float = 0.01,
    normalize: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Summarize a pixel anomaly map with its normalized top-k response."""
    anomaly_map_4d = _as_b1hw(anomaly_map, "anomaly_map")
    if normalize:
        anomaly_map_4d = _minmax_norm_per_image(anomaly_map_4d, eps=eps)
    flat = anomaly_map_4d.float().flatten(1)
    if topk_ratio <= 0.0:
        k = 1
    else:
        k = max(1, int(flat.size(1) * min(float(topk_ratio), 1.0)))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def fuse_image_score_with_pixel_score(
    image_score: torch.Tensor,
    pixel_score: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor:
    """Fuse global CLIP image score with local top-k pixel evidence."""
    image_score = image_score.float().view(-1).clamp(0.0, 1.0)
    pixel_score = pixel_score.float().view(-1).clamp(0.0, 1.0)
    weight = float(weight)
    if weight <= 0.0:
        return image_score
    weight = min(weight, 1.0)
    return ((1.0 - weight) * image_score + weight * pixel_score).clamp(0.0, 1.0)


def low_rank_residual_gate_from_patch_features(
    patch_features: torch.Tensor,
    output_size: SizeLike,
    rank_ratio: float = 0.15,
    max_rank: int = 32,
    sample_patches: int = 128,
    center: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build a normalized low-rank reconstruction residual map.

    The low-rank basis is estimated from a deterministic subset of patches for
    CPU efficiency, then every patch is projected onto that basis. Patches with
    large feature-space reconstruction error are treated as image-local
    outliers.
    """
    output_hw = _as_hw(output_size)
    spatial_tokens, height, width = _extract_spatial_tokens(patch_features)
    spatial_tokens = F.normalize(spatial_tokens.float(), dim=-1, eps=eps)

    residual_maps = []
    for tokens in spatial_tokens:
        x = tokens
        if center:
            x = x - x.mean(dim=0, keepdim=True)

        if 0 < sample_patches < x.size(0):
            sample_idx = torch.linspace(
                0,
                x.size(0) - 1,
                steps=int(sample_patches),
                device=x.device,
            ).round().long().unique()
            fit_tokens = x.index_select(0, sample_idx)
        else:
            fit_tokens = x

        min_dim = min(fit_tokens.size(0), fit_tokens.size(1))
        rank = int(math.ceil(float(rank_ratio) * float(min_dim)))
        rank = max(1, min(rank, int(max_rank), min_dim))

        try:
            _, _, vh = torch.linalg.svd(fit_tokens, full_matrices=False)
            basis = vh[:rank].transpose(0, 1).contiguous()
        except RuntimeError:
            # SVD can fail on degenerate inputs; QR still gives a stable
            # low-dimensional sampled subspace for residual estimation.
            q, _ = torch.linalg.qr(fit_tokens.transpose(0, 1), mode="reduced")
            basis = q[:, :rank].contiguous()

        projection = (x @ basis) @ basis.transpose(0, 1)
        residual = torch.sqrt((x - projection).square().mean(dim=-1) + eps)
        residual_maps.append(residual.view(1, height, width))

    residual_map = torch.stack(residual_maps, dim=0)
    residual_map = _minmax_norm_per_image(residual_map, eps=eps)
    residual_map = F.interpolate(
        residual_map,
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    )
    return _minmax_norm_per_image(residual_map, eps=eps).squeeze(1)


def apply_low_rank_residual_calibration(
    anomaly_map: torch.Tensor,
    selected_patch_features: Iterable[torch.Tensor],
    output_size: SizeLike,
    weight: float = 0.03,
    rank_ratio: float = 0.15,
    max_rank: int = 32,
    sample_patches: int = 128,
    fusion: str = "mean",
    condition_power: float = 2.0,
    max_delta_ratio: float = 0.03,
    clip_topk_ratio: float = 0.35,
    clip_gate_mode: str = "hard",
    clip_gate_temperature: float = 0.05,
    center: bool = True,
    layer_selection: str = "all",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse low-rank feature residual evidence into a pixel anomaly map.

    The residual is gated by the CLIP anomaly prior, so feature-space outliers
    only promote regions that the semantic branch already considers plausible.
    """
    patch_features = list(selected_patch_features)
    if len(patch_features) == 0:
        raise ValueError("at least one patch feature layer is required for low-rank residual")
    if layer_selection == "last":
        patch_features = [patch_features[-1]]
    elif layer_selection != "all":
        raise ValueError(f"unsupported low-rank layer selection: {layer_selection}")

    residual_gates = [
        low_rank_residual_gate_from_patch_features(
            patch_feature,
            output_size=output_size,
            rank_ratio=rank_ratio,
            max_rank=max_rank,
            sample_patches=sample_patches,
            center=center,
            eps=eps,
        )
        for patch_feature in patch_features
    ]
    residual_gate = fuse_wavelet_gates(residual_gates, mode=fusion, eps=eps)

    anomaly_map_4d = _as_b1hw(anomaly_map, "anomaly_map")
    residual_gate_4d = _as_b1hw(residual_gate, "residual_gate")
    if residual_gate_4d.shape[-2:] != anomaly_map_4d.shape[-2:]:
        residual_gate_4d = F.interpolate(
            residual_gate_4d,
            size=anomaly_map_4d.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    clip_prior = _minmax_norm_per_image(anomaly_map_4d, eps=eps)
    residual_prior = _minmax_norm_per_image(residual_gate_4d, eps=eps)
    clip_gate = _clip_rank_gate(
        clip_prior,
        clip_topk_ratio,
        mode=clip_gate_mode,
        temperature=clip_gate_temperature,
        eps=eps,
    )
    if condition_power > 0:
        residual = residual_prior * clip_prior.pow(condition_power) * clip_gate
    else:
        residual = residual_prior * clip_gate

    flat_map = anomaly_map_4d.flatten(1)
    score_scale = (flat_map.max(dim=1)[0] - flat_map.min(dim=1)[0]).view(
        anomaly_map_4d.size(0),
        1,
        1,
        1,
    ).clamp_min(eps)
    bonus = float(weight) * residual * score_scale
    if max_delta_ratio > 0:
        bonus = torch.minimum(bonus, float(max_delta_ratio) * score_scale)
    calibrated = anomaly_map_4d + bonus
    return calibrated.clamp_min(0.0).squeeze(1), residual.squeeze(1)


def apply_layer_consistency_calibration(
    anomaly_map: torch.Tensor,
    layer_maps: Union[Iterable[torch.Tensor], torch.Tensor],
    topk_ratio: float = 0.10,
    min_layers: int = 2,
    boost_weight: float = 0.03,
    suppress_weight: float = 0.0,
    condition_power: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use cross-layer agreement to refine a pixel anomaly map.

    A position is reliable when several selected CLIP layers rank it among
    their top anomaly responses. The gate is label-free and image-local, so it
    can be used in zero-shot evaluation without category-specific tuning.
    """
    anomaly_map_4d = _as_b1hw(anomaly_map, "anomaly_map")
    if isinstance(layer_maps, torch.Tensor):
        maps = layer_maps
        if maps.dim() == 4:
            maps = maps.unsqueeze(2)
        elif maps.dim() != 5:
            raise ValueError(
                "layer_maps tensor must be [L, B, H, W] or [L, B, 1, H, W], "
                f"got {tuple(layer_maps.shape)}"
            )
        layer_stack = maps.float()
    else:
        layer_list = [_as_b1hw(layer_map, "layer_map").float() for layer_map in layer_maps]
        if len(layer_list) == 0:
            raise ValueError("at least one layer map is required for consistency calibration")
        layer_stack = torch.stack(layer_list, dim=0)

    if layer_stack.shape[-2:] != anomaly_map_4d.shape[-2:]:
        layer_stack = F.interpolate(
            layer_stack.flatten(0, 1),
            size=anomaly_map_4d.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).view(layer_stack.size(0), anomaly_map_4d.size(0), 1, *anomaly_map_4d.shape[-2:])

    normalized_layers = torch.stack(
        [_minmax_norm_per_image(layer_stack[i], eps=eps) for i in range(layer_stack.size(0))],
        dim=0,
    )
    num_layers, batch, _, height, width = normalized_layers.shape
    min_layers = max(1, min(int(min_layers), num_layers))

    flat = normalized_layers.flatten(2)
    if topk_ratio <= 0.0 or topk_ratio >= 1.0:
        topk_mask = torch.ones_like(normalized_layers)
    else:
        k = max(1, int(flat.size(-1) * float(topk_ratio)))
        threshold = torch.topk(flat, k=k, dim=2).values[:, :, -1].view(
            num_layers,
            batch,
            1,
            1,
            1,
        )
        topk_mask = (normalized_layers >= threshold).to(dtype=normalized_layers.dtype)

    agreement = topk_mask.mean(dim=0)
    required_fraction = float(min_layers) / float(num_layers)
    consistency_gate = torch.where(
        agreement >= required_fraction,
        agreement,
        torch.zeros_like(agreement),
    )

    clip_prior = _minmax_norm_per_image(anomaly_map_4d, eps=eps)
    if condition_power > 0:
        conditioned_gate = consistency_gate * clip_prior.pow(condition_power)
        inconsistent_prior = clip_prior.pow(condition_power) * (1.0 - agreement)
    else:
        conditioned_gate = consistency_gate
        inconsistent_prior = 1.0 - agreement

    flat_map = anomaly_map_4d.flatten(1)
    score_scale = (flat_map.max(dim=1)[0] - flat_map.min(dim=1)[0]).view(
        batch,
        1,
        1,
        1,
    ).clamp_min(eps)
    calibrated = anomaly_map_4d
    if boost_weight > 0:
        calibrated = calibrated + float(boost_weight) * conditioned_gate * score_scale
    if suppress_weight > 0:
        calibrated = calibrated - float(suppress_weight) * inconsistent_prior * score_scale

    return calibrated.clamp_min(0.0).squeeze(1), consistency_gate.squeeze(1)


def apply_wavelet_residual_map_calibration(
    anomaly_map: torch.Tensor,
    wavelet_gate: torch.Tensor,
    weight: float = 0.03,
    topk_ratio: float = 0.35,
    condition_power: float = 2.0,
    max_delta_ratio: float = 0.03,
    local_contrast_kernel: int = 0,
    local_contrast_weight: float = 0.0,
    rank_gate_mode: str = "hard",
    rank_gate_temperature: float = 0.05,
    confidence_power: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add a bounded wavelet residual directly to the pixel anomaly map.

    This is intentionally conservative: the wavelet evidence is only promoted
    where CLIP already gives a compatible anomaly prior, which reduces the risk
    of treating normal texture edges as defects.
    """
    anomaly_map_4d = _as_b1hw(anomaly_map, "anomaly_map")
    wavelet_gate_4d = _as_b1hw(wavelet_gate, "wavelet_gate")
    if wavelet_gate_4d.shape[-2:] != anomaly_map_4d.shape[-2:]:
        wavelet_gate_4d = F.interpolate(
            wavelet_gate_4d,
            size=anomaly_map_4d.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    clip_prior = _minmax_norm_per_image(anomaly_map_4d, eps=eps)
    residual_gate = _minmax_norm_per_image(wavelet_gate_4d, eps=eps)
    if local_contrast_weight > 0.0 and local_contrast_kernel > 1:
        local_residual = _local_positive_residual(
            residual_gate,
            kernel_size=local_contrast_kernel,
            eps=eps,
        )
        residual_gate = (
            (1.0 - float(local_contrast_weight)) * residual_gate
            + float(local_contrast_weight) * local_residual
        )
        residual_gate = _minmax_norm_per_image(residual_gate, eps=eps)

    rank_gate = _clip_rank_gate(
        clip_prior,
        topk_ratio,
        mode=rank_gate_mode,
        temperature=rank_gate_temperature,
        eps=eps,
    )
    if condition_power > 0:
        residual = residual_gate * clip_prior.pow(condition_power) * rank_gate
    else:
        residual = residual_gate * rank_gate

    if confidence_power > 0:
        agreement = (1.0 - (clip_prior - residual_gate).abs()).clamp(0.0, 1.0)
        residual = residual * agreement.pow(confidence_power)

    flat_map = anomaly_map_4d.flatten(1)
    score_scale = (flat_map.max(dim=1)[0] - flat_map.min(dim=1)[0]).view(
        anomaly_map_4d.size(0),
        1,
        1,
        1,
    ).clamp_min(eps)
    bonus = float(weight) * residual * score_scale
    if max_delta_ratio > 0:
        bonus = torch.minimum(bonus, float(max_delta_ratio) * score_scale)
    calibrated = anomaly_map_4d + bonus
    return calibrated.clamp_min(0.0).squeeze(1), residual.squeeze(1)


def apply_structure_texture_calibration(
    anomaly_map: torch.Tensor,
    texture_gate: torch.Tensor,
    beta: float = 0.5,
    condition_power: float = 1.0,
    suppress_beta: float = 0.3,
    texture_max_delta_ratio: float = 0.05,
    texture_suppression_weight: float = 0.0,
    texture_local_contrast_kernel: int = 0,
    texture_local_contrast_weight: float = 0.0,
    rank_preserve_topk_ratio: float = 0.0,
    rank_gate_mode: str = "hard",
    rank_gate_temperature: float = 0.05,
    use_wavelet_confidence: bool = False,
    wavelet_confidence_power: float = 1.0,
    adaptive: bool = True,
    reliability: Optional[torch.Tensor] = None,
    reliability_power: float = 1.0,
    reliability_topk_ratio: float = 0.05,
    min_reliability: float = 0.0,
    texture_delta_reliability_power: float = 0.0,
    image_confidence_gate: Optional[torch.Tensor] = None,
    image_confidence_weight: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse a CLIP structure route with a non-structure texture route.

    The structure route is the original CLIP anomaly map. The texture route is
    a high-frequency residual after suppressing low-frequency structure edges.
    The texture route is used as a bounded additive residual by default, so it
    can refine high-confidence texture regions without aggressively changing
    CLIP's global ranking on structure-heavy images.
    """
    anomaly_map_4d = _as_b1hw(anomaly_map, "anomaly_map")
    texture_gate_4d = _as_b1hw(texture_gate, "texture_gate")
    if texture_gate_4d.shape[-2:] != anomaly_map_4d.shape[-2:]:
        texture_gate_4d = F.interpolate(
            texture_gate_4d,
            size=anomaly_map_4d.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    clip_prior = _minmax_norm_per_image(anomaly_map_4d, eps=eps)
    texture_gate_4d = _minmax_norm_per_image(texture_gate_4d, eps=eps)
    if texture_local_contrast_weight > 0.0 and texture_local_contrast_kernel > 1:
        local_residual = _local_positive_residual(
            texture_gate_4d,
            kernel_size=texture_local_contrast_kernel,
            eps=eps,
        )
        texture_gate_4d = (
            (1.0 - float(texture_local_contrast_weight)) * texture_gate_4d
            + float(texture_local_contrast_weight) * local_residual
        )
        texture_gate_4d = _minmax_norm_per_image(texture_gate_4d, eps=eps)

    rank_mask = _clip_rank_gate(
        clip_prior,
        rank_preserve_topk_ratio,
        mode=rank_gate_mode,
        temperature=rank_gate_temperature,
        eps=eps,
    )
    if condition_power > 0:
        semantic_texture = texture_gate_4d * clip_prior.pow(condition_power) * rank_mask
        nonsemantic_texture = texture_gate_4d * (1.0 - clip_prior).pow(condition_power)
    else:
        semantic_texture = texture_gate_4d * rank_mask
        nonsemantic_texture = torch.zeros_like(texture_gate_4d)

    if reliability is None:
        if adaptive:
            reliability = compute_texture_reliability(
                anomaly_map_4d,
                texture_gate_4d,
                topk_ratio=reliability_topk_ratio,
                eps=eps,
            )
        else:
            reliability = torch.ones(
                anomaly_map_4d.size(0),
                1,
                1,
                1,
                dtype=anomaly_map_4d.dtype,
                device=anomaly_map_4d.device,
            )
    elif reliability.dim() == 1:
        reliability = reliability.view(-1, 1, 1, 1)
    elif reliability.dim() == 3:
        reliability = reliability.mean(dim=(1, 2), keepdim=True).unsqueeze(1)

    reliability = reliability.to(device=anomaly_map_4d.device, dtype=anomaly_map_4d.dtype)
    reliability = reliability.clamp(0.0, 1.0)
    if min_reliability > 0.0:
        reliability = torch.where(
            reliability >= float(min_reliability),
            reliability,
            torch.zeros_like(reliability),
        )
    if reliability_power > 0:
        reliability = reliability.pow(reliability_power)

    flat = anomaly_map_4d.flatten(1)
    score_scale = (flat.max(dim=1)[0] - flat.min(dim=1)[0]).view(-1, 1, 1, 1).clamp_min(eps)
    confidence_gate = reliability
    if use_wavelet_confidence:
        local_agreement = (1.0 - (clip_prior - texture_gate_4d).abs()).clamp(0.0, 1.0)
        if wavelet_confidence_power > 0:
            local_agreement = local_agreement.pow(wavelet_confidence_power)
        confidence_gate = confidence_gate * local_agreement
    if image_confidence_gate is not None and image_confidence_weight > 0:
        image_confidence_gate = image_confidence_gate.to(
            device=anomaly_map_4d.device,
            dtype=anomaly_map_4d.dtype,
        )
        image_confidence_gate = image_confidence_gate.view(anomaly_map_4d.size(0), 1, 1, 1)
        image_confidence_gate = image_confidence_gate.clamp(0.0, 1.0)
        gate_mix = min(float(image_confidence_weight), 1.0)
        confidence_gate = confidence_gate * (
            (1.0 - gate_mix) + gate_mix * image_confidence_gate
        )

    texture_bonus = beta * confidence_gate * semantic_texture * score_scale
    if texture_max_delta_ratio > 0:
        max_bonus = texture_max_delta_ratio * score_scale
        if texture_delta_reliability_power > 0:
            max_bonus = max_bonus * reliability.pow(texture_delta_reliability_power)
        texture_bonus = torch.minimum(texture_bonus, max_bonus)

    if texture_suppression_weight > 0:
        texture_suppress = (
            1.0
            + texture_suppression_weight
            * suppress_beta
            * (1.0 - confidence_gate)
            * nonsemantic_texture
        )
        calibrated = (anomaly_map_4d + texture_bonus) / texture_suppress.clamp_min(eps)
        diagnostic_gate = confidence_gate * semantic_texture - (
            texture_suppression_weight * (1.0 - confidence_gate) * nonsemantic_texture
        )
    else:
        calibrated = anomaly_map_4d + texture_bonus
        diagnostic_gate = confidence_gate * semantic_texture
    return calibrated.squeeze(1), diagnostic_gate.squeeze(1)


def apply_wavelet_calibration(
    anomaly_map: torch.Tensor,
    wavelet_gate: torch.Tensor,
    beta: float = 0.5,
    condition_power: float = 1.0,
    suppress_beta: float = 0.3,
    adaptive: bool = True,
    reliability: Optional[torch.Tensor] = None,
    reliability_power: float = 1.0,
    reliability_topk_ratio: float = 0.05,
    image_confidence_gate: Optional[torch.Tensor] = None,
    image_confidence_weight: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply adaptive CLIP-conditioned wavelet calibration.

    ``wavelet_gate`` is not used as an anomaly map by itself. It is split by
    the CLIP anomaly prior into two parts:

        confirm gate: W * norm(A_clip) ** condition_power
        texture gate: W * (1 - norm(A_clip)) ** condition_power

    The image-level reliability decides whether wavelet evidence is helpful for
    the current image. When CLIP and wavelet agree, high-frequency details are
    used as positive evidence. When they disagree, high-frequency responses are
    treated as likely texture/background noise and are used mostly for
    suppression:

        A_final = A_clip
                  * (1 + beta * R * confirm)
                  / (1 + suppress_beta * (1 - R) * texture)

    Returns:
        ``(calibrated_map, diagnostic_gate)``, both shaped [B, H, W].
        ``diagnostic_gate`` is positive for enhanced regions and negative for
        suppressed high-frequency texture regions.
    """
    anomaly_map_4d = _as_b1hw(anomaly_map, "anomaly_map")
    wavelet_gate_4d = _as_b1hw(wavelet_gate, "wavelet_gate")

    if wavelet_gate_4d.shape[-2:] != anomaly_map_4d.shape[-2:]:
        wavelet_gate_4d = F.interpolate(
            wavelet_gate_4d,
            size=anomaly_map_4d.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    clip_prior = _minmax_norm_per_image(anomaly_map_4d, eps=eps)
    if condition_power > 0:
        anomaly_confidence = clip_prior.pow(condition_power)
        normal_confidence = (1.0 - clip_prior).pow(condition_power)
    else:
        anomaly_confidence = torch.ones_like(clip_prior)
        normal_confidence = torch.zeros_like(clip_prior)

    wavelet_gate_4d = wavelet_gate_4d.clamp(0.0, 1.0)
    confirm_gate = wavelet_gate_4d * anomaly_confidence
    texture_gate = wavelet_gate_4d * normal_confidence

    if reliability is None:
        if adaptive:
            reliability = compute_wavelet_reliability(
                anomaly_map_4d,
                wavelet_gate_4d,
                topk_ratio=reliability_topk_ratio,
                eps=eps,
            )
        else:
            reliability = torch.ones(
                anomaly_map_4d.size(0),
                1,
                1,
                1,
                dtype=anomaly_map_4d.dtype,
                device=anomaly_map_4d.device,
            )
    elif reliability.dim() == 1:
        reliability = reliability.view(-1, 1, 1, 1)
    elif reliability.dim() == 3:
        reliability = reliability.mean(dim=(1, 2), keepdim=True).unsqueeze(1)

    reliability = reliability.to(device=anomaly_map_4d.device, dtype=anomaly_map_4d.dtype)
    reliability = reliability.clamp(0.0, 1.0)
    if reliability_power > 0:
        reliability = reliability.pow(reliability_power)

    enhance_weight = reliability
    if image_confidence_gate is not None and image_confidence_weight > 0:
        image_confidence_gate = image_confidence_gate.to(
            device=anomaly_map_4d.device,
            dtype=anomaly_map_4d.dtype,
        )
        image_confidence_gate = image_confidence_gate.view(anomaly_map_4d.size(0), 1, 1, 1)
        image_confidence_gate = image_confidence_gate.clamp(0.0, 1.0)
        gate_mix = min(float(image_confidence_weight), 1.0)
        enhance_weight = enhance_weight * (
            (1.0 - gate_mix) + gate_mix * image_confidence_gate
        )
    suppress_weight = 1.0 - reliability
    enhance_scale = 1.0 + beta * enhance_weight * confirm_gate
    suppress_scale = 1.0 + suppress_beta * suppress_weight * texture_gate
    calibrated = anomaly_map_4d * enhance_scale / suppress_scale.clamp_min(eps)

    diagnostic_gate = enhance_weight * confirm_gate - suppress_weight * texture_gate
    return calibrated.squeeze(1), diagnostic_gate.squeeze(1)
