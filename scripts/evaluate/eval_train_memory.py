from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from anomalyclip.cached_eval_utils import (
    auto_device,
    build_anomaly_maps_from_patch_features,
    compute_image_text_prob,
    format_metrics_table,
    init_results,
    load_model_and_text_features,
    sample_cache_paths,
    selected_classes,
    setup_seed,
    smooth_anomaly_map,
)
from anomalyclip.config_utils import parse_args_with_config
from anomalyclip.dataset import Dataset, generate_class_info
from anomalyclip.logger import get_logger, log_run_context
from anomalyclip.normal_change_manifold import extract_spatial_tokens
from anomalyclip.utils import get_transform
from anomalyclip.wavelet_calibration import topk_pixel_score


def _sample_rows(tokens: torch.Tensor, max_rows: int, generator: torch.Generator) -> torch.Tensor:
    if max_rows <= 0 or tokens.size(0) <= max_rows:
        return tokens
    idx = torch.randperm(tokens.size(0), generator=generator)[:max_rows]
    return tokens.index_select(0, idx)


def _load_memory(path: str):
    if not path or not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu")
    return payload["memory"], payload.get("metadata", {})


def _expected_memory_metadata(args, obj_list):
    return {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "checkpoint_path": args.checkpoint_path,
        "features_list": list(args.features_list),
        "feature_map_layer": list(args.feature_map_layer),
        "classes": list(obj_list),
        "depth": int(args.depth),
        "n_ctx": int(args.n_ctx),
        "t_n_ctx": int(args.t_n_ctx),
        "dpam_layer": int(args.dpam_layer),
        "max_train_images_per_class": int(args.max_train_images_per_class),
        "tokens_per_train_image": int(args.tokens_per_train_image),
        "max_memory_tokens_per_layer": int(args.max_memory_tokens_per_layer),
        "seed": int(args.seed),
    }


def _memory_cache_matches(metadata, expected) -> bool:
    for key, value in expected.items():
        if key == "classes":
            cached_classes = set(metadata.get("classes", []))
            if not set(value).issubset(cached_classes):
                return False
            continue
        if key == "feature_map_layer":
            cached_layers = set(int(layer) for layer in metadata.get("feature_map_layer", []))
            if not set(int(layer) for layer in value).issubset(cached_layers):
                return False
            continue
        if metadata.get(key) != value:
            return False
    return True


def _parse_class_memory_cache_specs(specs):
    parsed = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                "--class_memory_cache entries must use CLASS[,CLASS...]=PATH, "
                f"got: {spec}"
            )
        classes_part, path = spec.split("=", 1)
        classes = [cls_name.strip() for cls_name in classes_part.split(",") if cls_name.strip()]
        if not classes or not path:
            raise ValueError(
                "--class_memory_cache entries must use CLASS[,CLASS...]=PATH, "
                f"got: {spec}"
            )
        parsed.append((classes, path))
    return parsed


def _parse_class_float_specs(specs, option_name: str):
    parsed = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"{option_name} entries must use CLASS[,CLASS...]=VALUE, got: {spec}"
            )
        classes_part, value_part = spec.split("=", 1)
        classes = [cls_name.strip() for cls_name in classes_part.split(",") if cls_name.strip()]
        if not classes or not value_part:
            raise ValueError(
                f"{option_name} entries must use CLASS[,CLASS...]=VALUE, got: {spec}"
            )
        value = float(value_part)
        for cls_name in classes:
            parsed[cls_name] = value
    return parsed


def _parse_class_int_list_specs(specs, option_name: str):
    parsed = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"{option_name} entries must use CLASS[,CLASS...]=INT[,INT...], got: {spec}"
            )
        classes_part, values_part = spec.split("=", 1)
        classes = [cls_name.strip() for cls_name in classes_part.split(",") if cls_name.strip()]
        values = [int(value.strip()) for value in values_part.split(",") if value.strip()]
        if not classes or not values:
            raise ValueError(
                f"{option_name} entries must use CLASS[,CLASS...]=INT[,INT...], got: {spec}"
            )
        for cls_name in classes:
            parsed[cls_name] = values
    return parsed


def _check_override_memory_metadata(metadata, args, classes, path: str) -> None:
    expected = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "checkpoint_path": args.checkpoint_path,
        "features_list": list(args.features_list),
        "depth": int(args.depth),
        "n_ctx": int(args.n_ctx),
        "t_n_ctx": int(args.t_n_ctx),
        "dpam_layer": int(args.dpam_layer),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"class memory cache metadata mismatch for {path}: "
                f"{key} expected {value!r}, got {metadata.get(key)!r}"
            )

    cached_classes = set(metadata.get("classes", []))
    missing_classes = sorted(set(classes) - cached_classes)
    if missing_classes:
        raise ValueError(f"class memory cache {path} missing classes: {missing_classes}")

    cached_layers = {int(layer) for layer in metadata.get("feature_map_layer", [])}
    requested_layers = {int(layer) for layer in args.feature_map_layer}
    missing_layers = sorted(requested_layers - cached_layers)
    if missing_layers:
        raise ValueError(f"class memory cache {path} missing feature_map_layer: {missing_layers}")


def _apply_class_memory_overrides(memory, args, obj_list):
    parsed_specs = _parse_class_memory_cache_specs(args.class_memory_cache)
    if not parsed_specs:
        return []

    requested_classes = set(obj_list)
    applied = []
    for classes, path in parsed_specs:
        active_classes = [cls_name for cls_name in classes if cls_name in requested_classes]
        if not active_classes:
            continue
        cached = _load_memory(path)
        if cached is None:
            raise FileNotFoundError(f"class memory cache not found: {path}")
        override_memory, metadata = cached
        _check_override_memory_metadata(metadata, args, active_classes, path)

        for cls_name in active_classes:
            if cls_name not in override_memory:
                raise ValueError(f"class memory cache {path} missing class memory: {cls_name}")
            available_layers = {int(layer): tensor for layer, tensor in override_memory[cls_name].items()}
            missing_layers = sorted(set(int(layer) for layer in args.feature_map_layer) - set(available_layers))
            if missing_layers:
                raise ValueError(
                    f"class memory cache {path} missing layers for {cls_name}: {missing_layers}"
                )
            memory[cls_name] = {
                int(layer): available_layers[int(layer)]
                for layer in args.feature_map_layer
            }

        applied.append(
            {
                "classes": active_classes,
                "memory_cache_path": path,
                "metadata": metadata,
            }
        )
    return applied


def _save_memory(path: str, memory, metadata) -> None:
    if not path:
        return
    cache_dir = os.path.dirname(path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    serializable = {
        cls_name: {int(layer): tensor.detach().cpu().half() for layer, tensor in layers.items()}
        for cls_name, layers in memory.items()
    }
    torch.save({"memory": serializable, "metadata": metadata}, path)


def _filter_train_records(train_data: Dataset, obj_list, max_per_class: int):
    requested = set(obj_list)
    per_class_seen = defaultdict(int)
    filtered = []
    for record in train_data.data_all:
        cls_name = record["cls_name"]
        if cls_name not in requested:
            continue
        if max_per_class > 0 and per_class_seen[cls_name] >= max_per_class:
            continue
        filtered.append(record)
        per_class_seen[cls_name] += 1
    train_data.data_all = filtered
    train_data.length = len(filtered)
    return per_class_seen


def build_train_memory(args, obj_list):
    expected_metadata = _expected_memory_metadata(args, obj_list)
    cached = None if args.rebuild_memory else _load_memory(args.memory_cache_path)
    if cached is not None:
        memory, metadata = cached
        if _memory_cache_matches(metadata, expected_metadata):
            return memory, metadata
        print(f"memory cache metadata mismatch; rebuilding: {args.memory_cache_path}")

    device = auto_device(args.device)
    model, _ = load_model_and_text_features(args, device)
    preprocess, target_transform = get_transform(args)
    train_data = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        mode="train",
    )
    planned_per_class = _filter_train_records(
        train_data,
        obj_list,
        max_per_class=int(args.max_train_images_per_class),
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

    requested = set(obj_list)
    per_class_seen = defaultdict(int)
    token_buckets = {
        cls_name: {int(layer): [] for layer in args.feature_map_layer}
        for cls_name in obj_list
    }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    model.eval()

    for items in tqdm(train_loader, desc="build memory"):
        cls_name = items["cls_name"][0]
        if cls_name not in requested:
            continue
        if args.max_train_images_per_class > 0 and per_class_seen[cls_name] >= args.max_train_images_per_class:
            continue

        per_class_seen[cls_name] += 1
        image = items["img"].to(device)
        with torch.no_grad():
            _, patch_features = model.encode_image(
                image,
                args.features_list,
                DPAM_layer=args.dpam_layer,
            )

        for layer in args.feature_map_layer:
            tokens, _, _ = extract_spatial_tokens(patch_features[int(layer)].detach().cpu().float())
            tokens = F.normalize(tokens.squeeze(0), dim=-1)
            tokens = _sample_rows(tokens, args.tokens_per_train_image, generator)
            token_buckets[cls_name][int(layer)].append(tokens)

    memory = {}
    for cls_name, layers in token_buckets.items():
        memory[cls_name] = {}
        for layer, chunks in layers.items():
            if not chunks:
                raise ValueError(f"no train memory tokens collected for class={cls_name}, layer={layer}")
            tokens = torch.cat(chunks, dim=0)
            tokens = _sample_rows(tokens, args.max_memory_tokens_per_layer, generator)
            memory[cls_name][layer] = F.normalize(tokens.float(), dim=-1).half()

    metadata = {
        **expected_metadata,
        "planned_train_images_per_class": dict(planned_per_class),
        "seen_train_images_per_class": dict(per_class_seen),
    }
    _save_memory(args.memory_cache_path, memory, metadata)
    return memory, metadata


def _nearest_memory_score(
    tokens: torch.Tensor,
    memory: torch.Tensor,
    chunk_size: int,
    topk: int = 1,
) -> torch.Tensor:
    tokens = F.normalize(tokens.float(), dim=-1)
    memory = F.normalize(memory.float(), dim=-1)
    topk = min(max(1, int(topk)), memory.size(0))
    best = torch.full((tokens.size(0), topk), -1.0, dtype=torch.float32)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, memory.size(0), chunk_size):
        sims = tokens @ memory[start : start + chunk_size].t()
        chunk_topk = torch.topk(sims, k=min(topk, sims.size(1)), dim=1).values
        best = torch.topk(torch.cat([best, chunk_topk], dim=1), k=topk, dim=1).values
    return (1.0 - best.mean(dim=1)).clamp_min(0.0)


def _minmax_per_image(anomaly_map: torch.Tensor) -> torch.Tensor:
    flat = anomaly_map.flatten(1)
    lower = flat.min(dim=1).values.view(-1, 1, 1)
    upper = flat.max(dim=1).values.view(-1, 1, 1)
    return ((anomaly_map - lower) / (upper - lower).clamp_min(1e-6)).clamp(0.0, 1.0)


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


def _should_apply_foreground_gate(sample, args) -> bool:
    if not args.use_foreground_gate:
        return False
    cls_name = sample["cls_name"]
    if args.foreground_classes is None:
        return cls_name in MVTec_OBJECT_CLASSES
    requested = set(args.foreground_classes)
    return "all" in requested or cls_name in requested


def _foreground_gate_from_image(image_path: str, image_size: int, args) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
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


def _apply_foreground_gate(sample, anomaly_map: torch.Tensor, args) -> torch.Tensor:
    if not _should_apply_foreground_gate(sample, args):
        return anomaly_map
    foreground_gate = _foreground_gate_from_image(sample["img_path"], anomaly_map.shape[-1], args)
    return anomaly_map.float() * foreground_gate.to(dtype=anomaly_map.dtype)


def _memory_map_from_sample(sample, memory, args):
    cls_name = sample["cls_name"]
    layer_maps = []
    feature_map_layer = args.class_feature_map_layer_map.get(cls_name, args.feature_map_layer)
    for layer in feature_map_layer:
        patch_feature = sample["patch_features"][int(layer)].float()
        tokens, height, width = extract_spatial_tokens(patch_feature)
        scores = _nearest_memory_score(
            tokens.squeeze(0),
            memory[cls_name][int(layer)],
            chunk_size=args.memory_chunk_size,
            topk=args.memory_topk,
        )
        score_map = scores.view(1, 1, height, width)
        score_map = F.interpolate(
            score_map,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        layer_maps.append(score_map)

    stacked = torch.stack(layer_maps, dim=0)
    if args.layer_fusion == "sum":
        anomaly_map = stacked.sum(dim=0)
    elif args.layer_fusion == "mean":
        anomaly_map = stacked.mean(dim=0)
    else:
        raise ValueError(f"unsupported layer fusion: {args.layer_fusion}")

    if args.normalize_memory_map:
        anomaly_map = _minmax_per_image(anomaly_map)
    sigma = args.class_sigma_map.get(cls_name, args.sigma)
    anomaly_map = smooth_anomaly_map(anomaly_map, sigma=sigma)
    return anomaly_map


def _load_zeroshot_text_features(args):
    if args.hybrid_mode == "memory":
        return None
    text_args = argparse.Namespace(**vars(args))
    text_args.checkpoint_path = args.hybrid_zeroshot_checkpoint_path or args.checkpoint_path
    _, text_features = load_model_and_text_features(text_args, auto_device(args.device))
    return text_features.float()


def _zeroshot_outputs_from_sample(sample, text_features: torch.Tensor, args):
    patch_features = [patch_feature.float() for patch_feature in sample["patch_features"]]
    anomaly_map, _ = build_anomaly_maps_from_patch_features(
        patch_features,
        text_features,
        args.hybrid_feature_map_layer or args.feature_map_layer,
        args.image_size,
        layer_weighting=args.hybrid_layer_weighting,
        layer_weight_temperature=args.hybrid_layer_weight_temperature,
        layer_weights=args.hybrid_layer_weights,
    )
    anomaly_map = smooth_anomaly_map(anomaly_map, sigma=args.hybrid_zeroshot_sigma)
    text_prob = compute_image_text_prob(
        sample["image_features"].float(),
        text_features,
        temperature=args.hybrid_image_temperature,
    )
    return anomaly_map, text_prob.detach().cpu()


def _combine_maps(cls_name: str, memory_map: torch.Tensor, zeroshot_map, args) -> torch.Tensor:
    if args.hybrid_mode == "memory":
        return memory_map
    selected = set(args.hybrid_classes or [])
    if args.hybrid_mode == "class_select":
        return zeroshot_map if cls_name in selected else memory_map
    if args.hybrid_mode == "class_blend" and cls_name not in selected:
        return memory_map
    if zeroshot_map is None:
        raise ValueError(f"hybrid mode {args.hybrid_mode} needs a zero-shot map")
    if args.hybrid_mode == "zeroshot":
        return zeroshot_map

    memory_for_fusion = memory_map
    zeroshot_for_fusion = zeroshot_map
    if args.hybrid_normalize_inputs:
        memory_for_fusion = _minmax_per_image(memory_for_fusion)
        zeroshot_for_fusion = _minmax_per_image(zeroshot_for_fusion)

    if args.hybrid_mode in {"blend", "class_blend"}:
        alpha = min(max(float(args.hybrid_memory_weight), 0.0), 1.0)
        return alpha * memory_for_fusion + (1.0 - alpha) * zeroshot_for_fusion
    if args.hybrid_mode == "max":
        return torch.maximum(memory_for_fusion, zeroshot_for_fusion)
    raise ValueError(f"unsupported hybrid mode: {args.hybrid_mode}")


def _image_score_from_maps(cls_name: str, final_map: torch.Tensor, zeroshot_text_prob, args) -> torch.Tensor:
    selected = set(args.hybrid_image_classes or args.hybrid_classes or [])
    use_zeroshot = (
        args.hybrid_image_score == "zeroshot"
        or (
            args.hybrid_image_score == "class_select"
            and args.hybrid_mode in {"class_select", "class_blend", "zeroshot"}
            and cls_name in selected
        )
    )
    if use_zeroshot:
        if zeroshot_text_prob is None:
            raise ValueError("--hybrid_image_score requested zero-shot scores without zero-shot text features")
        return zeroshot_text_prob
    return topk_pixel_score(
        final_map,
        topk_ratio=args.image_topk_ratio,
        normalize=args.normalize_image_score,
    )


def evaluate(args) -> None:
    setup_seed(args.seed)
    metadata_path = os.path.join(args.test_cache_dir, "metadata.pt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    test_metadata = torch.load(metadata_path, map_location="cpu")
    args.image_size = args.image_size or test_metadata["image_size"]
    args.features_list = args.features_list or test_metadata["features_list"]
    args.dpam_layer = args.dpam_layer or test_metadata.get("dpam_layer", 20)

    all_classes, _ = generate_class_info(args.dataset)
    obj_list = selected_classes(all_classes, args.classes)
    args.class_sigma_map = _parse_class_float_specs(args.class_sigma, "--class_sigma")
    args.class_feature_map_layer_map = _parse_class_int_list_specs(
        args.class_feature_map_layer,
        "--class_feature_map_layer",
    )
    memory, memory_metadata = build_train_memory(args, obj_list)
    class_memory_overrides = _apply_class_memory_overrides(memory, args, obj_list)
    zeroshot_text_features = _load_zeroshot_text_features(args)

    logger = get_logger(args.save_path)
    log_run_context(
        logger,
        args,
        title="AnomalyCLIP train-normal memory evaluation",
        extra_info={
            "test_cache_dir": args.test_cache_dir,
            "memory_cache_path": args.memory_cache_path,
            "memory_metadata": memory_metadata,
            "class_memory_cache": args.class_memory_cache,
            "class_memory_overrides": class_memory_overrides,
            "class_sigma_map": args.class_sigma_map,
            "class_feature_map_layer_map": args.class_feature_map_layer_map,
            "hybrid_zeroshot_checkpoint_path": args.hybrid_zeroshot_checkpoint_path,
        },
    )

    results = init_results(obj_list)
    evaluated = 0
    for sample_path in tqdm(sample_cache_paths(args.test_cache_dir), desc="eval memory"):
        sample = torch.load(sample_path, map_location="cpu")
        cls_name = sample["cls_name"]
        if cls_name not in results:
            continue
        memory_map = _memory_map_from_sample(sample, memory, args)
        zeroshot_map = zeroshot_text_prob = None
        needs_zeroshot = args.hybrid_mode in {"zeroshot", "blend", "max"} or (
            args.hybrid_mode in {"class_select", "class_blend"}
            and cls_name in set(args.hybrid_classes or [])
        )
        if needs_zeroshot:
            zeroshot_map, zeroshot_text_prob = _zeroshot_outputs_from_sample(
                sample,
                zeroshot_text_features,
                args,
            )
        anomaly_map = _combine_maps(cls_name, memory_map, zeroshot_map, args)
        anomaly_map = _apply_foreground_gate(sample, anomaly_map, args)
        image_score = _image_score_from_maps(cls_name, anomaly_map, zeroshot_text_prob, args)
        results[cls_name]["imgs_masks"].append(sample["img_mask"].float())
        results[cls_name]["gt_sp"].append(int(sample["anomaly"]))
        results[cls_name]["pr_sp"].extend(image_score.detach().cpu())
        results[cls_name]["anomaly_maps"].append(anomaly_map)
        evaluated += 1

    table = format_metrics_table(
        results,
        obj_list,
        args.metrics,
        aupro_steps=args.aupro_steps,
    )
    logger.info("\n%s", table)
    print(f"test_cache_dir: {args.test_cache_dir}")
    print(f"samples evaluated: {evaluated}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Evaluate MVTec with train-normal patch memory", add_help=True)
    parser.add_argument("--data_path", type=str, default="/Users/bytedance/Downloads/mvtec_anomaly_detection")
    parser.add_argument("--test_cache_dir", type=str, default="./cache/mvtec_anomalyclip_features")
    parser.add_argument("--memory_cache_path", type=str, default="./cache/mvtec_train_memory/memory.pt")
    parser.add_argument(
        "--class_memory_cache",
        type=str,
        nargs="+",
        default=None,
        help="Override selected classes with another cache, format CLASS[,CLASS...]=PATH.",
    )
    parser.add_argument("--save_path", type=str, default="./results/mvtec_train_memory")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--class_feature_map_layer",
        type=str,
        nargs="+",
        default=None,
        help="Override feature-map layers for selected classes, format CLASS[,CLASS...]=LAYER[,LAYER...].",
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--classes", type=str, nargs="+", default=None)
    parser.add_argument("--metrics", type=str, default="all", choices=["image-level", "pixel-level", "image-pixel-level", "all"])
    parser.add_argument("--aupro_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max_train_images_per_class", type=int, default=40)
    parser.add_argument("--tokens_per_train_image", type=int, default=16)
    parser.add_argument("--max_memory_tokens_per_layer", type=int, default=2048)
    parser.add_argument("--memory_chunk_size", type=int, default=2048)
    parser.add_argument("--memory_topk", type=int, default=1)
    parser.add_argument("--layer_fusion", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument(
        "--class_sigma",
        type=str,
        nargs="+",
        default=None,
        help="Override smoothing sigma for selected classes, format CLASS[,CLASS...]=VALUE.",
    )
    parser.add_argument("--image_topk_ratio", type=float, default=0.01)
    parser.add_argument("--normalize_memory_map", action="store_true")
    parser.add_argument("--normalize_image_score", action="store_true")
    parser.add_argument("--rebuild_memory", action="store_true")
    parser.add_argument("--use_foreground_gate", action="store_true")
    parser.add_argument("--foreground_classes", type=str, nargs="+", default=None)
    parser.add_argument("--foreground_outside_weight", type=float, default=0.2)
    parser.add_argument("--foreground_power", type=float, default=1.0)
    parser.add_argument("--foreground_quantile", type=float, default=0.2)
    parser.add_argument("--foreground_smooth_kernel", type=int, default=31)
    parser.add_argument("--foreground_min_contrast", type=float, default=0.03)
    parser.add_argument("--hybrid_mode", type=str, default="memory", choices=["memory", "zeroshot", "class_select", "class_blend", "blend", "max"])
    parser.add_argument("--hybrid_classes", type=str, nargs="+", default=None)
    parser.add_argument("--hybrid_image_classes", type=str, nargs="+", default=None)
    parser.add_argument("--hybrid_image_score", type=str, default="class_select", choices=["topk", "zeroshot", "class_select"])
    parser.add_argument("--hybrid_memory_weight", type=float, default=0.5)
    parser.add_argument("--hybrid_normalize_inputs", action="store_true")
    parser.add_argument("--hybrid_zeroshot_checkpoint_path", type=str, default=None)
    parser.add_argument("--hybrid_zeroshot_sigma", type=float, default=10.0)
    parser.add_argument("--hybrid_feature_map_layer", type=int, nargs="+", default=None)
    parser.add_argument("--hybrid_layer_weighting", type=str, default="sum", choices=["sum", "dynamic_wavelet"])
    parser.add_argument("--hybrid_layer_weight_temperature", type=float, default=1.0)
    parser.add_argument("--hybrid_layer_weights", type=float, nargs="+", default=None)
    parser.add_argument("--hybrid_image_temperature", type=float, default=0.07)
    return parser


if __name__ == "__main__":
    parsed_args, _ = parse_args_with_config(build_parser())
    evaluate(parsed_args)
