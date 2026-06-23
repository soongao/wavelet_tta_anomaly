import argparse
import os
from typing import Dict

import torch
from tqdm import tqdm

from cached_eval_utils import (
    auto_device,
    build_anomaly_maps_from_patch_features,
    build_structure_texture_gate,
    build_wavelet_gate,
    compute_image_text_prob,
    load_model_and_text_features,
    setup_seed,
)
from config_utils import parse_args_with_config
from dataset import Dataset
from logger import get_logger, log_run_context
from utils import get_transform


def _as_cache_tensor(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.detach().cpu().to(dtype=dtype).contiguous()


def _save_metadata(args, cache_dir: str, text_features: torch.Tensor, num_samples: int) -> None:
    metadata = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "checkpoint_path": args.checkpoint_path,
        "features_list": list(args.features_list),
        "feature_map_layer": list(args.feature_map_layer),
        "image_size": args.image_size,
        "depth": args.depth,
        "n_ctx": args.n_ctx,
        "t_n_ctx": args.t_n_ctx,
        "dpam_layer": args.dpam_layer,
        "cache_mode": "maps_only" if args.maps_only else "patch_features",
        "wavelet_fusion": args.wavelet_fusion,
        "texture_edge_power": args.texture_edge_power,
        "num_samples": num_samples,
        "text_features": text_features.detach().cpu().float(),
    }
    torch.save(metadata, os.path.join(cache_dir, "metadata.pt"))


def _sample_payload(
    items: Dict,
    image_features: torch.Tensor,
    patch_features,
    text_features: torch.Tensor,
    args,
) -> Dict:
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    anomaly_map, selected_patch_features = build_anomaly_maps_from_patch_features(
        patch_features,
        text_features,
        args.feature_map_layer,
        args.image_size,
    )
    text_prob = compute_image_text_prob(image_features, text_features)

    gt_mask = items["img_mask"]
    gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

    payload = {
        "cls_name": items["cls_name"][0],
        "cls_id": int(items["cls_id"].item() if torch.is_tensor(items["cls_id"]) else items["cls_id"]),
        "anomaly": int(items["anomaly"].item() if torch.is_tensor(items["anomaly"]) else items["anomaly"]),
        "img_path": items["img_path"][0] if isinstance(items["img_path"], (list, tuple)) else items["img_path"],
        "img_mask": _as_cache_tensor(gt_mask, torch.uint8),
        "image_features": _as_cache_tensor(image_features, torch.float16),
        "text_prob": _as_cache_tensor(text_prob, torch.float32),
        "base_anomaly_map": _as_cache_tensor(anomaly_map, torch.float16),
    }

    wavelet_gate = build_wavelet_gate(
        selected_patch_features,
        image_size=args.image_size,
        fusion=args.wavelet_fusion,
    )
    payload["wavelet_gate"] = _as_cache_tensor(wavelet_gate, torch.float16)
    texture_gate = build_structure_texture_gate(
        selected_patch_features,
        image_size=args.image_size,
        fusion=args.wavelet_fusion,
        edge_power=args.texture_edge_power,
    )
    payload["texture_gate"] = _as_cache_tensor(texture_gate, torch.float16)

    if not args.maps_only:
        payload["patch_features"] = [
            _as_cache_tensor(patch_feature, torch.float16) for patch_feature in patch_features
        ]

    return payload


def cache_features(args) -> None:
    setup_seed(args.seed)
    device = auto_device(args.device)
    os.makedirs(args.cache_dir, exist_ok=True)
    sample_dir = os.path.join(args.cache_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    logger = get_logger(args.cache_dir)
    log_run_context(
        logger,
        args,
        title="AnomalyCLIP feature cache generation",
        extra_info={"device": device, "sample_dir": sample_dir},
    )

    model, text_features = load_model_and_text_features(args, device)
    preprocess, target_transform = get_transform(args)
    test_data = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    written = 0
    kept = 0
    for idx, items in enumerate(tqdm(test_dataloader)):
        cls_name = items["cls_name"][0]
        if args.classes and cls_name not in args.classes:
            continue
        if args.max_samples is not None and kept >= args.max_samples:
            break

        cache_path = os.path.join(sample_dir, f"{idx:06d}_{cls_name}.pt")
        kept += 1
        if args.resume and os.path.exists(cache_path):
            continue

        image = items["img"].to(device)
        with torch.no_grad():
            image_features, patch_features = model.encode_image(
                image,
                args.features_list,
                DPAM_layer=args.dpam_layer,
            )
            payload = _sample_payload(
                items=items,
                image_features=image_features,
                patch_features=patch_features,
                text_features=text_features.to(device),
                args=args,
            )
        torch.save(payload, cache_path)
        written += 1

    _save_metadata(args, args.cache_dir, text_features, num_samples=kept)
    print(f"cache_dir: {args.cache_dir}")
    print(f"samples selected: {kept}")
    print(f"samples written: {written}")
    print(f"cache mode: {'maps_only' if args.maps_only else 'patch_features'}")
    logger.info(
        "cache generation finished: selected=%s written=%s mode=%s",
        kept,
        written,
        "maps_only" if args.maps_only else "patch_features",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cache AnomalyCLIP MVTec features", add_help=True)
    parser.add_argument("--data_path", type=str, default="/Users/bytedance/Downloads/mvtec_anomaly_detection")
    parser.add_argument("--checkpoint_path", type=str, default="/Users/bytedance/code/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth")
    parser.add_argument("--cache_dir", type=str, default="./cache/mvtec_anomalyclip_features")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--wavelet_fusion", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--texture_edge_power", type=float, default=1.0)
    parser.add_argument("--maps_only", action="store_true", help="cache only base anomaly maps and wavelet gates")
    parser.add_argument("--resume", action="store_true", help="skip sample files that already exist")
    parser.add_argument("--max_samples", type=int, default=None, help="optional small debug cache")
    parser.add_argument("--classes", type=str, nargs="+", default=None, help="optional class subset")
    return parser


if __name__ == "__main__":
    args, _ = parse_args_with_config(build_parser())
    cache_features(args)
