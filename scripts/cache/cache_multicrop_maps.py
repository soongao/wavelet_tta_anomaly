from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import os
from typing import Dict, List, Sequence

from PIL import Image
import torch
from tqdm import tqdm

from anomalyclip.cached_eval_utils import (
    auto_device,
    build_anomaly_maps_from_patch_features,
    load_model_and_text_features,
    setup_seed,
)
from anomalyclip.config_utils import parse_args_with_config
from anomalyclip.dataset import Dataset
from anomalyclip.logger import get_logger, log_run_context
from anomalyclip.multicrop_utils import build_multicrop_boxes, output_box_to_pil_box, stitch_crop_maps
from anomalyclip.utils import get_transform


def _batched(items: Sequence, batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _as_cache_tensor(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.detach().cpu().to(dtype=dtype).contiguous()


def _save_metadata(args, cache_dir: str, num_samples: int) -> None:
    metadata = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "checkpoint_path": args.checkpoint_path,
        "features_list": list(args.features_list),
        "feature_map_layer": list(args.feature_map_layer),
        "image_size": args.image_size,
        "crop_grid": args.crop_grid,
        "crop_ratio": args.crop_ratio,
        "num_samples": num_samples,
        "cache_mode": "stitched_crop_maps",
    }
    torch.save(metadata, os.path.join(cache_dir, "metadata.pt"))


def _build_stitched_crop_map(
    image_path: str,
    model,
    text_features: torch.Tensor,
    preprocess,
    args,
    device: str,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    boxes = build_multicrop_boxes(
        image_size=args.image_size,
        grid=args.crop_grid,
        crop_ratio=args.crop_ratio,
    )

    crops: List[torch.Tensor] = []
    for box in boxes:
        pil_box = output_box_to_pil_box(
            box,
            output_size=args.image_size,
            image_width=image.width,
            image_height=image.height,
        )
        crops.append(preprocess(image.crop(pil_box)))

    crop_maps = []
    with torch.inference_mode():
        for crop_batch in _batched(crops, args.crop_forward_batch_size):
            crop_tensor = torch.stack(crop_batch, dim=0).to(device)
            _, patch_features = model.encode_image(
                crop_tensor,
                args.features_list,
                DPAM_layer=args.dpam_layer,
            )
            batch_maps, _ = build_anomaly_maps_from_patch_features(
                patch_features,
                text_features,
                args.feature_map_layer,
                args.image_size,
                layer_weighting=args.layer_weighting,
                layer_weight_temperature=args.layer_weight_temperature,
            )
            crop_maps.extend(batch_maps.detach().cpu().float())
    return stitch_crop_maps(crop_maps, boxes, args.image_size)


def _sample_payload(items: Dict, crop_map: torch.Tensor, args) -> Dict:
    img_path = items["img_path"][0] if isinstance(items["img_path"], (list, tuple)) else items["img_path"]
    return {
        "cls_name": items["cls_name"][0],
        "img_path": img_path,
        "crop_grid": args.crop_grid,
        "crop_ratio": args.crop_ratio,
        "stitched_crop_map": _as_cache_tensor(crop_map, torch.float16),
    }


def cache_multicrop_maps(args) -> None:
    setup_seed(args.seed)
    device = auto_device(args.device)
    os.makedirs(args.cache_dir, exist_ok=True)
    sample_dir = os.path.join(args.cache_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    logger = get_logger(args.cache_dir)
    log_run_context(
        logger,
        args,
        title="AnomalyCLIP multi-crop map cache generation",
        extra_info={"device": device, "sample_dir": sample_dir},
    )

    model, text_features = load_model_and_text_features(args, device)
    text_features = text_features.to(device)
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

        crop_map = _build_stitched_crop_map(
            image_path=items["img_path"][0] if isinstance(items["img_path"], (list, tuple)) else items["img_path"],
            model=model,
            text_features=text_features,
            preprocess=preprocess,
            args=args,
            device=device,
        )
        torch.save(_sample_payload(items, crop_map, args), cache_path)
        written += 1

    _save_metadata(args, args.cache_dir, num_samples=kept)
    print(f"cache_dir: {args.cache_dir}")
    print(f"samples selected: {kept}")
    print(f"samples written: {written}")
    logger.info("multi-crop cache finished: selected=%s written=%s", kept, written)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cache AnomalyCLIP multi-crop stitched maps", add_help=True)
    parser.add_argument("--data_path", type=str, default="/Users/bytedance/Downloads/mvtec_anomaly_detection")
    parser.add_argument("--checkpoint_path", type=str, default="/Users/bytedance/code/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth")
    parser.add_argument("--cache_dir", type=str, default="./cache/mvtec_multicrop_maps")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--layer_weighting", type=str, default="sum", choices=["sum", "dynamic_wavelet"])
    parser.add_argument("--layer_weight_temperature", type=float, default=1.0)
    parser.add_argument("--crop_grid", type=int, default=2)
    parser.add_argument("--crop_ratio", type=float, default=0.75)
    parser.add_argument("--crop_forward_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true", help="skip sample files that already exist")
    parser.add_argument("--max_samples", type=int, default=None, help="optional small debug cache")
    parser.add_argument("--classes", type=str, nargs="+", default=None, help="optional class subset")
    return parser


if __name__ == "__main__":
    args, _ = parse_args_with_config(build_parser())
    cache_multicrop_maps(args)
