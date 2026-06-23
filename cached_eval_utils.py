import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from tabulate import tabulate

import AnomalyCLIP_lib
from metrics import image_level_metrics, pixel_level_metrics
from prompt_ensemble import AnomalyCLIP_PromptLearner
from wavelet_calibration import (
    compute_wavelet_reliability,
    fuse_wavelet_gates,
    structure_texture_gates_from_patch_features,
    wavelet_gate_from_patch_features,
)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auto_device(requested: str = "auto") -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_model_and_text_features(args, device: str):
    """Load AnomalyCLIP and return the generic normal/abnormal text features."""
    anomalyclip_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    model, _ = AnomalyCLIP_lib.load(
        "ViT-L/14@336px",
        device=device,
        design_details=anomalyclip_parameters,
    )
    model.eval()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anomalyclip_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)

    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)

    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(
            prompts,
            tokenized_prompts,
            compound_prompts_text,
        ).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return model, text_features[0].detach()


def build_anomaly_maps_from_patch_features(
    patch_features: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    feature_map_layer: Sequence[int],
    image_size: int,
    layer_weighting: str = "sum",
    layer_weight_temperature: float = 1.0,
    return_layer_maps: bool = False,
):
    """Rebuild AnomalyCLIP patch-level anomaly maps from cached patch features."""
    anomaly_map_list = []
    selected_patch_features = []
    first_selected_layer = feature_map_layer[0] if len(feature_map_layer) > 0 else 0
    text_features = F.normalize(text_features.float(), dim=-1)

    for idx, patch_feature in enumerate(patch_features):
        if idx >= first_selected_layer:
            patch_feature = F.normalize(patch_feature.float(), dim=-1)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
            similarity_map = AnomalyCLIP_lib.get_similarity_map(
                similarity[:, 1:, :],
                image_size,
            )
            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
            anomaly_map_list.append(anomaly_map)
            selected_patch_features.append(patch_feature)

    if len(anomaly_map_list) == 0:
        raise ValueError("No patch feature layer was selected. Check --feature_map_layer.")

    stacked_maps = torch.stack(anomaly_map_list)
    if layer_weighting == "sum":
        anomaly_map = stacked_maps.sum(dim=0)
    elif layer_weighting == "dynamic_wavelet":
        gates = [
            wavelet_gate_from_patch_features(patch_feature, output_size=image_size)
            for patch_feature in selected_patch_features
        ]
        reliabilities = [
            compute_wavelet_reliability(anomaly_map, gate).view(-1)
            for anomaly_map, gate in zip(anomaly_map_list, gates)
        ]
        weights = torch.stack(reliabilities, dim=0)
        temperature = max(float(layer_weight_temperature), 1e-6)
        weights = torch.softmax(weights / temperature, dim=0).view(
            len(anomaly_map_list),
            -1,
            1,
            1,
        )
        anomaly_map = (stacked_maps * weights).sum(dim=0) * len(anomaly_map_list)
    else:
        raise ValueError(f"unsupported layer weighting mode: {layer_weighting}")

    if return_layer_maps:
        return anomaly_map, selected_patch_features, anomaly_map_list
    return anomaly_map, selected_patch_features


def build_wavelet_gate(
    selected_patch_features: Iterable[torch.Tensor],
    image_size: int,
    fusion: str,
    wavelet_levels: int = 1,
    wavelet_level_fusion: str = "mean",
) -> torch.Tensor:
    gates = [
        wavelet_gate_from_patch_features(
            patch_feature,
            output_size=image_size,
            levels=wavelet_levels,
            level_fusion=wavelet_level_fusion,
        )
        for patch_feature in selected_patch_features
    ]
    return fuse_wavelet_gates(gates, mode=fusion)


def build_structure_texture_gate(
    selected_patch_features: Iterable[torch.Tensor],
    image_size: int,
    fusion: str,
    edge_power: float,
    wavelet_levels: int = 1,
    wavelet_level_fusion: str = "mean",
) -> torch.Tensor:
    texture_gates = [
        structure_texture_gates_from_patch_features(
            patch_feature,
            output_size=image_size,
            edge_power=edge_power,
            levels=wavelet_levels,
            level_fusion=wavelet_level_fusion,
        )[0]
        for patch_feature in selected_patch_features
    ]
    return fuse_wavelet_gates(texture_gates, mode=fusion)


def compute_image_text_prob(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)
    text_probs = image_features @ text_features.unsqueeze(0).permute(0, 2, 1)
    text_probs = (text_probs / temperature).softmax(-1)
    return text_probs[:, 0, 1]


def smooth_anomaly_map(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
    anomaly_map = anomaly_map.detach().cpu().float()
    return torch.stack(
        [torch.from_numpy(gaussian_filter(i, sigma=sigma)) for i in anomaly_map],
        dim=0,
    )


def sample_cache_paths(cache_dir: str) -> List[str]:
    sample_dir = os.path.join(cache_dir, "samples")
    if not os.path.isdir(sample_dir):
        return []
    return sorted(
        os.path.join(sample_dir, name)
        for name in os.listdir(sample_dir)
        if name.endswith(".pt")
    )


def init_results(obj_list: Sequence[str]):
    results = {}
    for obj in obj_list:
        results[obj] = {
            "gt_sp": [],
            "pr_sp": [],
            "imgs_masks": [],
            "anomaly_maps": [],
        }
    return results


def _safe_metric(results, obj: str, metric: str, aupro_steps: int = 200) -> float:
    try:
        if metric.startswith("image-"):
            return image_level_metrics(results, obj, metric)
        return pixel_level_metrics(results, obj, metric, aupro_steps=aupro_steps)
    except ValueError:
        return float("nan")


def _format_percent(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return str(np.round(value * 100, decimals=1))


def _nanmean(values: Sequence[float]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.nanmean(np.array(values, dtype=np.float64)))


def format_metrics_table(
    results,
    obj_list: Sequence[str],
    metrics: str,
    aupro_steps: int = 200,
) -> str:
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []

    for obj in obj_list:
        if len(results[obj]["gt_sp"]) == 0:
            continue

        table = [obj]
        results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"])
        results[obj]["anomaly_maps"] = (
            torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()
        )

        if metrics == "image-level":
            image_auroc = _safe_metric(results, obj, "image-auroc")
            image_ap = _safe_metric(results, obj, "image-ap")
            table.extend([_format_percent(image_auroc), _format_percent(image_ap)])
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif metrics == "pixel-level":
            pixel_auroc = _safe_metric(results, obj, "pixel-auroc", aupro_steps=aupro_steps)
            pixel_aupro = _safe_metric(results, obj, "pixel-aupro", aupro_steps=aupro_steps)
            table.append(_format_percent(pixel_auroc))
            table.append(_format_percent(pixel_aupro))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif metrics == "image-pixel-level":
            image_auroc = _safe_metric(results, obj, "image-auroc")
            image_ap = _safe_metric(results, obj, "image-ap")
            pixel_auroc = _safe_metric(results, obj, "pixel-auroc", aupro_steps=aupro_steps)
            pixel_aupro = _safe_metric(results, obj, "pixel-aupro", aupro_steps=aupro_steps)
            table.extend([_format_percent(pixel_auroc), _format_percent(pixel_aupro)])
            table.extend([_format_percent(image_auroc), _format_percent(image_ap)])
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        else:
            raise ValueError(f"unsupported metrics mode: {metrics}")

        table_ls.append(table)

    if metrics == "image-level":
        table_ls.append(
            [
                "mean",
                _format_percent(_nanmean(image_auroc_list)),
                _format_percent(_nanmean(image_ap_list)),
            ]
        )
        return tabulate(table_ls, headers=["objects", "image_auroc", "image_ap"], tablefmt="pipe")

    if metrics == "pixel-level":
        table_ls.append(
            [
                "mean",
                _format_percent(_nanmean(pixel_auroc_list)),
                _format_percent(_nanmean(pixel_aupro_list)),
            ]
        )
        return tabulate(table_ls, headers=["objects", "pixel_auroc", "pixel_aupro"], tablefmt="pipe")

    table_ls.append(
        [
            "mean",
            _format_percent(_nanmean(pixel_auroc_list)),
            _format_percent(_nanmean(pixel_aupro_list)),
            _format_percent(_nanmean(image_auroc_list)),
            _format_percent(_nanmean(image_ap_list)),
        ]
    )
    return tabulate(
        table_ls,
        headers=["objects", "pixel_auroc", "pixel_aupro", "image_auroc", "image_ap"],
        tablefmt="pipe",
    )


def selected_classes(obj_list: Sequence[str], classes: Optional[Sequence[str]]) -> List[str]:
    if classes is None or len(classes) == 0:
        return list(obj_list)
    requested = set(classes)
    return [obj for obj in obj_list if obj in requested]
