#!/usr/bin/env python3
"""Visualize AnomalyCLIP patch features before and after wavelet calibration.

The script reads the feature cache produced by ``cache_mvtec_features.py``.
It does not run the model. Each sampled patch is labeled by downsampling the
cached pixel mask to the ViT patch grid, then the tool compares:

1. baseline patch anomaly scores from ``base_anomaly_map``;
2. wavelet-calibrated patch anomaly scores;
3. raw patch token embeddings and score/gate context after calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "anomalyclip_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "anomalyclip_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anomalyclip.wavelet_calibration import (  # noqa: E402
    apply_structure_texture_calibration,
    apply_wavelet_calibration,
    compute_texture_reliability,
    compute_wavelet_reliability,
)


@dataclass
class PatchBatch:
    raw_features: np.ndarray
    labels: np.ndarray
    baseline_scores: np.ndarray
    wavelet_scores: np.ndarray
    gate_values: np.ndarray
    classes: List[str]


def _sample_cache_paths(cache_dir: Path, classes: Optional[Sequence[str]]) -> List[Path]:
    sample_dir = cache_dir / "samples"
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample directory not found: {sample_dir}")
    paths = sorted(sample_dir.glob("*.pt"))
    if classes:
        wanted = set(classes)
        filtered = []
        for path in paths:
            suffix = path.stem.split("_", 1)[1] if "_" in path.stem else ""
            if suffix in wanted:
                filtered.append(path)
        paths = filtered
    if not paths:
        raise FileNotFoundError(f"no sample cache files found under {sample_dir}")
    return paths


def _load_metadata(cache_dir: Path) -> dict:
    metadata_path = cache_dir / "metadata.pt"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    return torch.load(metadata_path, map_location="cpu")


def _spatial_tokens(patch_feature: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    if patch_feature.dim() != 3:
        raise ValueError(f"patch_feature must be [B, N, C], got {tuple(patch_feature.shape)}")
    num_tokens = patch_feature.size(1)
    side_with_cls = int(math.sqrt(num_tokens - 1))
    if side_with_cls * side_with_cls == num_tokens - 1:
        return patch_feature[:, 1:, :].float(), side_with_cls, side_with_cls
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return patch_feature.float(), side, side
    raise ValueError(f"cannot infer square patch grid from {num_tokens} tokens")


def _as_b1hw(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.size(1) == 1:
        return x
    raise ValueError(f"expected [H,W], [B,H,W], or [B,1,H,W], got {tuple(x.shape)}")


def _resize_to_grid(x: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    if mode in {"nearest", "area"}:
        resized = F.interpolate(_as_b1hw(x), size=(height, width), mode=mode)
    else:
        resized = F.interpolate(_as_b1hw(x), size=(height, width), mode=mode, align_corners=False)
    return resized.squeeze(1)


def _normalize_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x4 = _as_b1hw(x)
    flat = x4.flatten(1)
    x_min = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    x_max = flat.max(dim=1)[0].view(-1, 1, 1, 1)
    return ((x4 - x_min) / (x_max - x_min + eps)).clamp(0.0, 1.0).squeeze(1)


def _patch_labels(mask: torch.Tensor, height: int, width: int, threshold: float) -> torch.Tensor:
    resized = _resize_to_grid(mask.float(), height, width, mode="area")
    return (resized >= float(threshold)).view(-1).to(torch.int64)


def _patch_scores(score_map: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return _resize_to_grid(score_map.float(), height, width, mode="bilinear").view(-1)


def _choose_layer(patch_features: Sequence[torch.Tensor], layer_index: int) -> torch.Tensor:
    if layer_index < 0:
        layer_index = len(patch_features) + layer_index
    if layer_index < 0 or layer_index >= len(patch_features):
        raise IndexError(
            f"feature layer {layer_index} is out of range for {len(patch_features)} cached layers"
        )
    return patch_features[layer_index]


def _calibrate_sample(sample: dict, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    baseline_map = sample["base_anomaly_map"].float()
    if args.wavelet_mode == "dual_route":
        gate = sample.get("texture_gate")
        if gate is None:
            gate = sample.get("wavelet_gate")
        if gate is None:
            raise ValueError("sample has no texture_gate or wavelet_gate")
        reliability = compute_texture_reliability(
            baseline_map,
            gate.float(),
            topk_ratio=args.wavelet_reliability_topk_ratio,
        )
        calibrated, _ = apply_structure_texture_calibration(
            baseline_map,
            gate.float(),
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
        )
        return calibrated.float(), gate.float()

    gate = sample.get("wavelet_gate")
    if gate is None:
        raise ValueError("adaptive wavelet mode needs wavelet_gate")
    reliability = compute_wavelet_reliability(
        baseline_map,
        gate.float(),
        topk_ratio=args.wavelet_reliability_topk_ratio,
    )
    calibrated, _ = apply_wavelet_calibration(
        baseline_map,
        gate.float(),
        beta=args.wavelet_beta,
        condition_power=args.wavelet_condition_power,
        suppress_beta=args.wavelet_suppress_beta,
        adaptive=not args.disable_adaptive_wavelet,
        reliability=reliability,
        reliability_power=args.wavelet_reliability_power,
        reliability_topk_ratio=args.wavelet_reliability_topk_ratio,
    )
    return calibrated.float(), gate.float()


def _stratified_patch_indices(
    labels: torch.Tensor,
    scores: torch.Tensor,
    gate: torch.Tensor,
    max_per_sample: int,
    abnormal_fraction: float,
    rng: random.Random,
) -> torch.Tensor:
    labels_np = labels.cpu().numpy()
    anomaly_idx = np.flatnonzero(labels_np == 1)
    normal_idx = np.flatnonzero(labels_np == 0)
    if anomaly_idx.size == 0:
        score_np = scores.detach().cpu().numpy()
        gate_np = gate.detach().cpu().numpy()
        k = min(max_per_sample // 2, score_np.size)
        top_score = np.argpartition(score_np, -k)[-k:] if k > 0 else np.array([], dtype=np.int64)
        top_gate = np.argpartition(gate_np, -k)[-k:] if k > 0 else np.array([], dtype=np.int64)
        selected = np.unique(np.concatenate([top_score, top_gate]))
        remaining = max(0, max_per_sample - selected.size)
        if remaining > 0:
            pool = np.setdiff1d(normal_idx, selected, assume_unique=False)
            if pool.size > 0:
                extra = rng.sample(pool.tolist(), k=min(remaining, pool.size))
                selected = np.concatenate([selected, np.array(extra, dtype=np.int64)])
        return torch.from_numpy(selected[:max_per_sample]).long()

    anomaly_quota = max(1, int(max_per_sample * abnormal_fraction))
    normal_quota = max_per_sample - anomaly_quota
    chosen_anomaly = rng.sample(anomaly_idx.tolist(), k=min(anomaly_quota, anomaly_idx.size))
    chosen_normal = rng.sample(normal_idx.tolist(), k=min(normal_quota, normal_idx.size))
    selected = chosen_anomaly + chosen_normal
    if len(selected) < max_per_sample:
        pool = np.setdiff1d(np.arange(labels_np.size), np.array(selected, dtype=np.int64))
        if pool.size > 0:
            selected.extend(rng.sample(pool.tolist(), k=min(max_per_sample - len(selected), pool.size)))
    rng.shuffle(selected)
    return torch.tensor(selected, dtype=torch.long)


def _collect_patch_batch(
    sample_paths: Sequence[Path],
    args: argparse.Namespace,
) -> Tuple[PatchBatch, List[dict]]:
    rng = random.Random(args.seed)
    patch_records = []
    qualitative_records = []

    paths = list(sample_paths)
    if args.shuffle:
        rng.shuffle(paths)
    if args.max_samples is not None:
        paths = paths[: args.max_samples]

    for path in paths:
        sample = torch.load(path, map_location="cpu")
        if "patch_features" not in sample:
            raise ValueError(f"{path} was built with maps_only; rebuild cache with patch_features")

        patch_feature = _choose_layer(sample["patch_features"], args.feature_layer)
        tokens, grid_h, grid_w = _spatial_tokens(patch_feature)
        tokens = F.normalize(tokens.squeeze(0).float(), dim=-1)

        labels = _patch_labels(sample["img_mask"], grid_h, grid_w, args.mask_threshold)
        baseline_grid = _patch_scores(sample["base_anomaly_map"], grid_h, grid_w)
        calibrated_map, gate_map = _calibrate_sample(sample, args)
        if args.sigma > 0:
            smoothed = [
                torch.from_numpy(gaussian_filter(img.numpy(), sigma=args.sigma))
                for img in calibrated_map.detach().cpu().float()
            ]
            calibrated_map = torch.stack(smoothed, dim=0)
        wavelet_grid = _patch_scores(calibrated_map, grid_h, grid_w)
        gate_grid = _patch_scores(gate_map, grid_h, grid_w).clamp(0.0, 1.0)
        indices = _stratified_patch_indices(
            labels,
            wavelet_grid,
            gate_grid,
            args.max_patches_per_sample,
            args.abnormal_patch_fraction,
            rng,
        )
        if indices.numel() == 0:
            continue

        for idx in indices.tolist():
            patch_records.append(
                (
                    tokens[idx].numpy(),
                    int(labels[idx].item()),
                    float(baseline_grid[idx].item()),
                    float(wavelet_grid[idx].item()),
                    float(gate_grid[idx].item()),
                    str(sample["cls_name"]),
                )
            )

        if len(qualitative_records) < args.max_qualitative and int(sample["anomaly"]) == 1:
            qualitative_records.append(
                {
                    "sample_path": str(path),
                    "sample": sample,
                    "calibrated_map": calibrated_map.detach().cpu(),
                    "gate_map": gate_map.detach().cpu(),
                }
            )

    if not patch_records:
        raise RuntimeError("no patch records collected")

    if len(patch_records) > args.max_total_patches:
        patch_records = rng.sample(patch_records, k=args.max_total_patches)

    raw_features, labels, base_scores, wave_scores, gates, classes = zip(*patch_records)
    return (
        PatchBatch(
            raw_features=np.asarray(raw_features, dtype=np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            baseline_scores=np.asarray(base_scores, dtype=np.float32),
            wavelet_scores=np.asarray(wave_scores, dtype=np.float32),
            gate_values=np.asarray(gates, dtype=np.float32),
            classes=list(classes),
        ),
        qualitative_records,
    )


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> Optional[float]:
    if len(np.unique(labels)) < 2:
        return None
    if features.shape[0] < 3:
        return None
    try:
        return float(silhouette_score(features, labels))
    except Exception:
        return None


def _score_context_features(batch: PatchBatch) -> np.ndarray:
    delta = batch.wavelet_scores - batch.baseline_scores
    features = np.stack(
        [
            batch.baseline_scores,
            batch.wavelet_scores,
            batch.gate_values,
            delta,
            np.abs(delta),
        ],
        axis=1,
    ).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + 1e-6)


def _score_summary(batch: PatchBatch) -> dict:
    labels = batch.labels
    normal = labels == 0
    anomaly = labels == 1
    delta = batch.wavelet_scores - batch.baseline_scores
    summary = {
        "num_patches": int(labels.size),
        "num_normal_patches": int(normal.sum()),
        "num_anomaly_patches": int(anomaly.sum()),
    }
    for prefix, scores in [
        ("baseline", batch.baseline_scores),
        ("wavelet", batch.wavelet_scores),
    ]:
        if normal.any():
            summary[f"{prefix}_normal_mean"] = float(scores[normal].mean())
        if anomaly.any():
            summary[f"{prefix}_anomaly_mean"] = float(scores[anomaly].mean())
        if normal.any() and anomaly.any():
            summary[f"{prefix}_score_gap"] = float(scores[anomaly].mean() - scores[normal].mean())
            summary[f"{prefix}_normal_std"] = float(scores[normal].std())
            summary[f"{prefix}_anomaly_std"] = float(scores[anomaly].std())
    if normal.any() and anomaly.any():
        summary["score_gap_delta"] = float(
            (batch.wavelet_scores[anomaly].mean() - batch.wavelet_scores[normal].mean())
            - (batch.baseline_scores[anomaly].mean() - batch.baseline_scores[normal].mean())
        )
        summary["gate_normal_mean"] = float(batch.gate_values[normal].mean())
        summary["gate_anomaly_mean"] = float(batch.gate_values[anomaly].mean())
        summary["gate_anomaly_minus_normal"] = float(
            batch.gate_values[anomaly].mean() - batch.gate_values[normal].mean()
        )
        summary["delta_normal_mean"] = float(delta[normal].mean())
        summary["delta_anomaly_mean"] = float(delta[anomaly].mean())
        summary["delta_anomaly_minus_normal"] = float(delta[anomaly].mean() - delta[normal].mean())
        summary["absolute_delta_normal_mean"] = float(np.abs(delta[normal]).mean())
        summary["absolute_delta_anomaly_mean"] = float(np.abs(delta[anomaly]).mean())
    summary["absolute_delta_mean"] = float(np.abs(delta).mean())
    if labels.size > 1:
        summary["baseline_wavelet_score_corr"] = float(
            np.corrcoef(batch.baseline_scores, batch.wavelet_scores)[0, 1]
        )

    pca_raw = PCA(n_components=min(20, batch.raw_features.shape[0], batch.raw_features.shape[1]), random_state=0).fit_transform(
        batch.raw_features
    )
    summary["raw_feature_silhouette_pca"] = _safe_silhouette(pca_raw, labels)
    summary["score_context_silhouette"] = _safe_silhouette(_score_context_features(batch), labels)
    return summary


def _plot_score_distribution(batch: PatchBatch, save_path: Path) -> None:
    labels = batch.labels
    normal = labels == 0
    anomaly = labels == 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160, sharey=True)
    for ax, title, scores in [
        (axes[0], "Baseline patch anomaly score", batch.baseline_scores),
        (axes[1], "Wavelet-calibrated patch anomaly score", batch.wavelet_scores),
    ]:
        bins = np.linspace(float(scores.min()), float(scores.max()), 60)
        if normal.any():
            ax.hist(scores[normal], bins=bins, density=True, alpha=0.62, color="#2f6fbb", label="normal")
        if anomaly.any():
            ax.hist(scores[anomaly], bins=bins, density=True, alpha=0.62, color="#d64b3c", label="anomaly")
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _embedding(features: np.ndarray, method: str, seed: int, max_points: int) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(features)

    pre_dim = min(50, features.shape[1], features.shape[0] - 1)
    reduced = PCA(n_components=pre_dim, random_state=seed).fit_transform(features)
    perplexity = min(30.0, max(5.0, (features.shape[0] - 1) / 3.0))
    tsne_kwargs = {
        "n_components": 2,
        "init": "pca",
        "learning_rate": "auto",
        "perplexity": perplexity,
        "random_state": seed,
    }
    iter_arg = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    tsne_kwargs[iter_arg] = 1000 if max_points <= 4000 else 750
    return TSNE(**tsne_kwargs).fit_transform(reduced)


def _plot_feature_embedding(batch: PatchBatch, method: str, seed: int, save_path: Path) -> None:
    raw_xy = _embedding(batch.raw_features, method=method, seed=seed, max_points=batch.raw_features.shape[0])
    score_context = _score_context_features(batch)
    context_method = method if score_context.shape[0] >= 20 else "pca"
    context_xy = _embedding(score_context, method=context_method, seed=seed, max_points=score_context.shape[0])
    normal = batch.labels == 0
    anomaly = batch.labels == 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    for ax, title, xy in [
        (axes[0], "Raw CLIP patch features", raw_xy),
        (axes[1], "Score/gate context after wavelet", context_xy),
    ]:
        if normal.any():
            ax.scatter(
                xy[normal, 0],
                xy[normal, 1],
                s=6,
                c="#2f6fbb",
                alpha=0.35,
                linewidths=0,
                label="normal",
            )
        if anomaly.any():
            ax.scatter(
                xy[anomaly, 0],
                xy[anomaly, 1],
                s=10,
                c="#d64b3c",
                alpha=0.72,
                linewidths=0,
                label="anomaly",
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.12)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(f"{method.upper()} projection")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_calibration_effect(batch: PatchBatch, save_path: Path) -> None:
    labels = batch.labels
    normal = labels == 0
    anomaly = labels == 1
    delta = batch.wavelet_scores - batch.baseline_scores

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=160)

    for mask, color, label, alpha in [
        (normal, "#2f6fbb", "normal", 0.35),
        (anomaly, "#d64b3c", "anomaly", 0.7),
    ]:
        if mask.any():
            axes[0].scatter(
                batch.baseline_scores[mask],
                batch.wavelet_scores[mask],
                s=7,
                c=color,
                alpha=alpha,
                linewidths=0,
                label=label,
            )
    min_score = float(min(batch.baseline_scores.min(), batch.wavelet_scores.min()))
    max_score = float(max(batch.baseline_scores.max(), batch.wavelet_scores.max()))
    axes[0].plot([min_score, max_score], [min_score, max_score], "--", color="#555555", linewidth=1)
    axes[0].set_title("Baseline vs wavelet score")
    axes[0].set_xlabel("baseline score")
    axes[0].set_ylabel("wavelet score")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    bins = np.linspace(float(delta.min()), float(delta.max()), 60)
    if normal.any():
        axes[1].hist(delta[normal], bins=bins, density=True, alpha=0.62, color="#2f6fbb", label="normal")
    if anomaly.any():
        axes[1].hist(delta[anomaly], bins=bins, density=True, alpha=0.62, color="#d64b3c", label="anomaly")
    axes[1].axvline(0.0, color="#333333", linewidth=1)
    axes[1].set_title("Wavelet score delta")
    axes[1].set_xlabel("wavelet - baseline")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.2)

    for mask, color, label, alpha in [
        (normal, "#2f6fbb", "normal", 0.35),
        (anomaly, "#d64b3c", "anomaly", 0.7),
    ]:
        if mask.any():
            axes[2].scatter(
                batch.gate_values[mask],
                delta[mask],
                s=7,
                c=color,
                alpha=alpha,
                linewidths=0,
                label=label,
            )
    axes[2].axhline(0.0, color="#333333", linewidth=1)
    axes[2].set_title("Gate vs score delta")
    axes[2].set_xlabel("wavelet/texture gate")
    axes[2].set_ylabel("wavelet - baseline")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _read_image(path: str, image_size: int) -> Optional[np.ndarray]:
    try:
        image = Image.open(path).convert("RGB").resize((image_size, image_size))
    except Exception:
        return None
    return np.asarray(image)


def _map_to_numpy(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().float()
    if arr.dim() == 3:
        arr = arr[0]
    if arr.dim() == 4:
        arr = arr[0, 0]
    arr_np = arr.numpy()
    vmin = float(arr_np.min())
    vmax = float(arr_np.max())
    return (arr_np - vmin) / (vmax - vmin + 1e-6)


def _plot_qualitative(records: Sequence[dict], save_dir: Path, image_size: int) -> List[str]:
    out_dir = save_dir / "qualitative_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, record in enumerate(records):
        sample = record["sample"]
        cls_name = str(sample["cls_name"])
        image = _read_image(sample["img_path"], image_size)
        gt = _map_to_numpy(sample["img_mask"])
        baseline = _map_to_numpy(sample["base_anomaly_map"])
        gate = _map_to_numpy(record["gate_map"])
        calibrated = _map_to_numpy(record["calibrated_map"])
        baseline_raw = _as_b1hw(sample["base_anomaly_map"]).squeeze().detach().cpu().float().numpy()
        calibrated_raw = _as_b1hw(record["calibrated_map"]).squeeze().detach().cpu().float().numpy()
        delta = calibrated_raw - baseline_raw
        delta_abs = float(np.abs(delta).max())
        delta_limit = max(delta_abs, 1e-6)

        cols = 6 if image is not None else 5
        fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2), dpi=160)
        ax_idx = 0
        if image is not None:
            axes[ax_idx].imshow(image)
            axes[ax_idx].set_title("image")
            ax_idx += 1
        for ax, title, arr, cmap in [
            (axes[ax_idx], "GT mask", gt, "gray"),
            (axes[ax_idx + 1], "baseline map", baseline, "magma"),
            (axes[ax_idx + 2], "wavelet/texture gate", gate, "viridis"),
            (axes[ax_idx + 3], "wavelet map", calibrated, "magma"),
        ]:
            ax.imshow(arr, cmap=cmap)
            ax.set_title(title)
        axes[ax_idx + 4].imshow(delta, cmap="coolwarm", vmin=-delta_limit, vmax=delta_limit)
        axes[ax_idx + 4].set_title("score delta")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{cls_name}: {Path(str(sample['img_path'])).name}")
        fig.tight_layout()
        out_path = out_dir / f"{idx:02d}_{cls_name}.png"
        fig.savefig(out_path)
        plt.close(fig)
        written.append(str(out_path))
    return written


def _write_summary(summary: dict, args: argparse.Namespace, save_path: Path, qualitative_paths: Sequence[str]) -> None:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "args": serializable_args,
        "summary": summary,
        "qualitative_examples": list(qualitative_paths),
    }
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Visualize wavelet feature distributions from AnomalyCLIP cache")
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/mvtec_anomalyclip_features"))
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/wavelet_feature_viz/mvtec"))
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--max_samples", type=int, default=40)
    parser.add_argument("--max_patches_per_sample", type=int, default=128)
    parser.add_argument("--max_total_patches", type=int, default=4000)
    parser.add_argument("--feature_layer", type=int, default=-1, help="cached patch feature layer index")
    parser.add_argument("--mask_threshold", type=float, default=0.05)
    parser.add_argument("--abnormal_patch_fraction", type=float, default=0.5)
    parser.add_argument("--embedding", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--max_qualitative", type=int, default=6)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sigma", type=float, default=4.0)

    parser.add_argument("--wavelet_mode", choices=["dual_route", "adaptive"], default="dual_route")
    parser.add_argument("--wavelet_beta", type=float, default=0.2)
    parser.add_argument("--wavelet_condition_power", type=float, default=2.0)
    parser.add_argument("--wavelet_suppress_beta", type=float, default=0.0)
    parser.add_argument("--texture_max_delta_ratio", type=float, default=0.05)
    parser.add_argument("--texture_suppression_weight", type=float, default=0.0)
    parser.add_argument("--texture_local_contrast_kernel", type=int, default=17)
    parser.add_argument("--texture_local_contrast_weight", type=float, default=0.5)
    parser.add_argument("--rank_preserve_topk_ratio", type=float, default=0.35)
    parser.add_argument("--rank_gate_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--rank_gate_temperature", type=float, default=0.05)
    parser.add_argument("--use_wavelet_confidence", action="store_true", default=True)
    parser.add_argument("--no_wavelet_confidence", dest="use_wavelet_confidence", action="store_false")
    parser.add_argument("--wavelet_confidence_power", type=float, default=1.0)
    parser.add_argument("--disable_adaptive_wavelet", action="store_true")
    parser.add_argument("--wavelet_reliability_power", type=float, default=1.0)
    parser.add_argument("--wavelet_reliability_topk_ratio", type=float, default=0.05)
    parser.add_argument("--wavelet_min_reliability", type=float, default=0.0)
    parser.add_argument("--texture_delta_reliability_power", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata(args.cache_dir)
    paths = _sample_cache_paths(args.cache_dir, args.classes)
    batch, qualitative_records = _collect_patch_batch(paths, args)
    summary = _score_summary(batch)
    summary["cache_dataset"] = metadata.get("dataset", "unknown")
    summary["cache_num_samples"] = int(metadata.get("num_samples", 0))
    summary["selected_sample_files"] = min(len(paths), args.max_samples or len(paths))
    summary["feature_layer"] = int(args.feature_layer)

    score_path = args.save_dir / "score_distribution.png"
    embedding_path = args.save_dir / "feature_embedding.png"
    calibration_effect_path = args.save_dir / "calibration_effect.png"
    summary_path = args.save_dir / "summary.json"

    _plot_score_distribution(batch, score_path)
    _plot_feature_embedding(batch, args.embedding, args.seed, embedding_path)
    _plot_calibration_effect(batch, calibration_effect_path)
    qualitative_paths = _plot_qualitative(
        qualitative_records,
        args.save_dir,
        int(metadata.get("image_size", 518)),
    )
    _write_summary(summary, args, summary_path, qualitative_paths)

    print(f"save_dir: {args.save_dir}")
    print(f"score_distribution: {score_path}")
    print(f"feature_embedding: {embedding_path}")
    print(f"calibration_effect: {calibration_effect_path}")
    print(f"qualitative_examples: {len(qualitative_paths)}")
    print(f"summary: {summary_path}")
    if "score_gap_delta" in summary:
        print(f"baseline_score_gap: {summary.get('baseline_score_gap'):.6f}")
        print(f"wavelet_score_gap: {summary.get('wavelet_score_gap'):.6f}")
        print(f"score_gap_delta: {summary.get('score_gap_delta'):.6f}")


if __name__ == "__main__":
    main()
