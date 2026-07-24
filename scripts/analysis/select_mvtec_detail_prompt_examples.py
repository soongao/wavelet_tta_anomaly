from pathlib import Path
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import AnomalyCLIP_lib
from anomalyclip.visualization import apply_ad_scoremap
from anomalyclip.prompt_ensemble import tokenize


OBJECT_PHRASES = {
    "bottle": "bottle",
    "cable": "cable",
    "capsule": "capsule",
    "carpet": "carpet texture",
    "grid": "grid pattern",
    "hazelnut": "hazelnut",
    "leather": "leather surface",
    "metal_nut": "metal nut",
    "pill": "pill",
    "screw": "screw",
    "tile": "tile surface",
    "toothbrush": "toothbrush",
    "transistor": "transistor",
    "wood": "wood surface",
    "zipper": "zipper",
}


DEFECT_PHRASES = {
    "bent": "bent deformation",
    "broken_large": "large broken area",
    "broken_small": "small broken area",
    "color": "discolored patch",
    "combined": "combined defects",
    "contamination": "contamination stain",
    "crack": "crack",
    "cut": "cut",
    "cut_inner_insulation": "cut in the inner insulation",
    "cut_lead": "cut lead",
    "cut_outer_insulation": "cut in the outer insulation",
    "faulty_imprint": "faulty imprint",
    "hole": "hole",
    "liquid": "liquid stain",
    "manipulated_front": "manipulated front surface",
    "metal_contamination": "metal contamination",
    "missing_cable": "missing cable",
    "poke": "puncture mark",
    "print": "print defect",
    "scratch": "scratch",
    "squeeze": "squeezed deformation",
    "thread": "thread-like defect",
}


def overlay_scoremap(
    image_path: str,
    scoremap: np.ndarray,
    image_size: int,
    vmin: float = None,
    vmax: float = None,
) -> Image.Image:
    source = cv2.imread(image_path)
    if source is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(cv2.resize(source, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    scoremap = np.asarray(scoremap, dtype=np.float32)
    mn = float(np.nanmin(scoremap)) if vmin is None else float(vmin)
    mx = float(np.nanmax(scoremap)) if vmax is None else float(vmax)
    scoremap = (scoremap - mn) / max(mx - mn, 1e-6)
    overlay = apply_ad_scoremap(image, scoremap)
    return Image.fromarray(overlay)


def _object_phrase(cls_name: str) -> str:
    return OBJECT_PHRASES.get(cls_name, cls_name.replace("_", " "))


def _defect_phrase(defect_type: str) -> str:
    return DEFECT_PHRASES.get(defect_type, defect_type.replace("_", " "))


def _mask_np(mask: torch.Tensor, image_size: int) -> np.ndarray:
    if mask.dim() == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif mask.dim() == 3:
        mask = mask.squeeze(0)
    mask = mask.float()
    if tuple(mask.shape[-2:]) != (image_size, image_size):
        mask = F.interpolate(mask.view(1, 1, *mask.shape[-2:]), size=(image_size, image_size), mode="nearest").squeeze()
    return (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)


def _position_phrase(mask: np.ndarray) -> str:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return "visible"
    cx = xs.mean() / max(mask.shape[1] - 1, 1)
    cy = ys.mean() / max(mask.shape[0] - 1, 1)
    horiz = "left" if cx < 0.33 else "right" if cx > 0.67 else "central"
    vert = "upper" if cy < 0.33 else "lower" if cy > 0.67 else "middle"
    if horiz == "central" and vert == "middle":
        return "central"
    if horiz == "central":
        return vert
    if vert == "middle":
        return horiz
    return f"{vert} {horiz}"


def _size_phrase(mask: np.ndarray) -> str:
    area = float(mask.sum()) / float(mask.size)
    if area < 0.002:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.04:
        return "medium-sized"
    return "large"


def _shape_orientation_phrase(mask: np.ndarray) -> Tuple[str, str]:
    ys, xs = np.where(mask > 0)
    if xs.size < 5:
        return "compact", ""
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w = max(x1 - x0 + 1, 1)
    h = max(y1 - y0 + 1, 1)
    ratio = w / h
    if ratio > 2.8:
        return "elongated", "horizontal"
    if ratio < 1 / 2.8:
        return "elongated", "vertical"
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    coords = coords - coords.mean(axis=0, keepdims=True)
    cov = coords.T @ coords / max(coords.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[-1] > 4.0 * max(eigvals[0], 1e-6):
        vx, vy = eigvecs[:, -1]
        angle = abs(math.degrees(math.atan2(vy, vx)))
        if 25 <= angle <= 65 or 115 <= angle <= 155:
            return "elongated", "diagonal"
    return "irregular", ""


def build_detail_prompt(sample: Dict, image_size: int, prompt_style: str = "full") -> Tuple[str, str, Dict]:
    cls_name = sample["cls_name"]
    defect_type = defect_type_from_path(sample["img_path"])
    obj = _object_phrase(cls_name)
    defect = _defect_phrase(defect_type)
    mask = _mask_np(sample["img_mask"], image_size=image_size)
    size = _size_phrase(mask)
    shape, orient = _shape_orientation_phrase(mask)
    position = _position_phrase(mask)
    neutral = f"a photo of a {obj}."
    attrs = [size, orient, shape, defect]
    attrs = [a for a in attrs if a]
    article = "an" if attrs and attrs[0][0].lower() in "aeiou" else "a"
    if prompt_style == "full":
        detail = f"a photo of a {obj} with {article} {' '.join(attrs)} in the {position} region."
    elif prompt_style == "appearance":
        detail = f"a photo of a {obj} with {article} {' '.join(attrs)}."
    elif prompt_style == "defect_focus":
        detail = f"a defective {obj} showing {article} {' '.join(attrs)}."
    elif prompt_style == "short":
        compact = [a for a in [orient, shape, defect] if a]
        article = "an" if compact and compact[0][0].lower() in "aeiou" else "a"
        detail = f"a {obj} with {article} {' '.join(compact)}."
    else:
        raise ValueError(f"unsupported prompt_style: {prompt_style}")
    return neutral, detail, {"defect_type": defect_type}


def load_model_for_text(metadata: Dict, device: str):
    params = {
        "Prompt_length": metadata.get("n_ctx", 12),
        "learnabel_text_embedding_depth": metadata.get("depth", 9),
        "learnabel_text_embedding_length": metadata.get("t_n_ctx", 4),
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    return model


def encode_plain_text(model, texts: Sequence[str], device: str, batch_size: int = 64) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            tokens = tokenize(batch).to(device)
            x = model.token_embedding(tokens).type(model.dtype)
            x = x + model.positional_embedding.type(model.dtype)
            x = x.permute(1, 0, 2)
            x_out = model.transformer([x, [], 0])
            x = x_out[0] if isinstance(x_out, list) else x_out
            x = x.permute(1, 0, 2)
            x = model.ln_final(x).type(model.dtype)
            out = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ model.text_projection
            feats.append(F.normalize(out.float(), dim=-1).detach().cpu())
    return torch.cat(feats, dim=0)


def build_map_from_patch_features(
    patch_features: Sequence[torch.Tensor],
    text_features: torch.Tensor,
    feature_map_layer: Sequence[int],
    image_size: int,
    layer_weighting: str = "sum",
) -> torch.Tensor:
    maps = []
    first = feature_map_layer[0] if feature_map_layer else 0
    text_features = F.normalize(text_features.float(), dim=-1)
    for idx, patch_feature in enumerate(patch_features):
        if idx < first:
            continue
        patch_feature = F.normalize(patch_feature.float(), dim=-1)
        logits = torch.einsum("bnc,dc->bnd", patch_feature, text_features)
        probs = (logits / 0.07).softmax(dim=-1)
        spatial = probs[:, 1:, 1]
        side = int(math.sqrt(spatial.shape[1]))
        score = spatial.reshape(spatial.shape[0], side, side)
        score = F.interpolate(score.unsqueeze(1), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(1)
        maps.append(score)
    stacked = torch.stack(maps)
    if layer_weighting == "sum":
        return stacked.sum(dim=0)
    if layer_weighting == "mean":
        return stacked.mean(dim=0)
    raise ValueError(f"unsupported layer_weighting: {layer_weighting}")


def overlay_gt(image_path: str, mask: np.ndarray, image_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32)
    color = np.array([255, 35, 25], dtype=np.float32)
    m = mask.astype(bool)
    arr[m] = 0.54 * arr[m] + 0.46 * color
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def mask_bg_gap(scoremap: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() == 0 or (~m).sum() == 0:
        return float("nan")
    return float(scoremap[m].mean() - scoremap[~m].mean())


def topk_hit(scoremap: np.ndarray, mask: np.ndarray, ratio: float = 0.01) -> float:
    flat = scoremap.reshape(-1)
    mask_flat = mask.reshape(-1).astype(bool)
    k = max(1, int(flat.size * ratio))
    idx = np.argpartition(flat, -k)[-k:]
    return float(mask_flat[idx].mean())


def defect_type_from_path(img_path: str) -> str:
    parts = Path(img_path).parts
    if "test" in parts:
        idx = parts.index("test")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def short_prompt(prompt: str, max_len: int = 92) -> str:
    prompt = prompt.strip()
    return prompt if len(prompt) <= max_len else prompt[: max_len - 3] + "..."


def make_contact_sheet(records: List[Dict], out_path: Path, image_size: int) -> None:
    tile = 230
    label_h = 76
    cols = 4
    rows = len(records)
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    headers = ["Image + GT", "Generic prompt", "Detailed prompt", "Detail prompt text"]
    for r, rec in enumerate(records):
        y = r * (tile + label_h)
        vmin = min(float(np.nanmin(rec["generic_map"])), float(np.nanmin(rec["detail_map"])))
        vmax = max(float(np.nanmax(rec["generic_map"])), float(np.nanmax(rec["detail_map"])))
        panels = [
            overlay_gt(rec["img_path"], rec["mask"], image_size).resize((tile, tile)),
            overlay_scoremap(rec["img_path"], rec["generic_map"], image_size, vmin=vmin, vmax=vmax).resize((tile, tile)),
            overlay_scoremap(rec["img_path"], rec["detail_map"], image_size, vmin=vmin, vmax=vmax).resize((tile, tile)),
            Image.new("RGB", (tile, tile), "white"),
        ]
        for c, panel in enumerate(panels):
            x = c * tile
            draw.text((x + 8, y + 6), headers[c], fill=(0, 0, 0), font=font)
            canvas.paste(panel, (x, y + label_h))
        text_x = 3 * tile + 8
        draw.text((text_x, y + 36), f"{rec['cls_name']} / {rec['defect_type']}", fill=(0, 0, 0), font=small)
        draw.text((text_x, y + 58), f"gap +{rec['gap_delta']:.4f}", fill=(30, 90, 30), font=small)
        prompt = short_prompt(rec["detail_prompt"], max_len=120)
        # Wrap prompt manually inside the blank tile.
        words = prompt.split()
        line = ""
        yy = y + label_h + 16
        for word in words:
            cand = (line + " " + word).strip()
            if len(cand) > 28:
                draw.text((text_x, yy), line, fill=(0, 0, 0), font=small)
                yy += 18
                line = word
            else:
                line = cand
        if line:
            draw.text((text_x, yy), line, fill=(0, 0, 0), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser("Select MVTec same-class multi-defect examples where detailed prompt helps")
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/mvtec_anomalyclip_features"))
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/detail_prompt_oracle/mvtec_same_class_detail_examples"))
    parser.add_argument("--class_name", default=None)
    parser.add_argument("--lambda_detail", type=float, default=0.3)
    parser.add_argument("--prompt_style", choices=["full", "appearance", "defect_focus", "short"], default="full")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=None)
    parser.add_argument("--layer_weighting", choices=["sum", "mean"], default="sum")
    parser.add_argument("--per_defect", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=6)
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    metadata = torch.load(args.cache_dir / "metadata.pt", map_location="cpu")
    image_size = int(metadata["image_size"])
    feature_map_layer = args.feature_map_layer or list(metadata["feature_map_layer"])
    generic_text = F.normalize(metadata["text_features"].float(), dim=-1)

    sample_paths = sorted((args.cache_dir / "samples").glob("*.pt"))
    samples = []
    for path in sample_paths:
        sample = torch.load(path, map_location="cpu")
        if int(sample["anomaly"]) != 1:
            continue
        if args.class_name and sample["cls_name"] != args.class_name:
            continue
        samples.append((path, sample))
    if not samples:
        raise RuntimeError("no samples selected")

    neutral_prompts, detail_prompts = [], []
    for _, sample in samples:
        neutral, detail, _ = build_detail_prompt(sample, image_size=image_size, prompt_style=args.prompt_style)
        neutral_prompts.append(neutral)
        detail_prompts.append(detail)
    model = load_model_for_text(metadata, device=args.device)
    detail_feats = encode_plain_text(model, detail_prompts, device=args.device)

    records = []
    for (path, sample), detail_feat, detail_prompt in zip(samples, detail_feats, detail_prompts):
        t_normal = generic_text[0]
        t_abnormal = generic_text[1]
        detail_abnormal = F.normalize((1.0 - args.lambda_detail) * t_abnormal + args.lambda_detail * detail_feat.float(), dim=-1)
        generic_pair = torch.stack([t_normal, t_abnormal], dim=0)
        detail_pair = torch.stack([t_normal, detail_abnormal], dim=0)
        generic_map = build_map_from_patch_features(
            sample["patch_features"],
            generic_pair,
            feature_map_layer=feature_map_layer,
            image_size=image_size,
            layer_weighting=args.layer_weighting,
        ).squeeze(0).detach().cpu().numpy()
        detail_map = build_map_from_patch_features(
            sample["patch_features"],
            detail_pair,
            feature_map_layer=feature_map_layer,
            image_size=image_size,
            layer_weighting=args.layer_weighting,
        ).squeeze(0).detach().cpu().numpy()
        mask = _mask_np(sample["img_mask"], image_size=image_size)
        gap_generic = mask_bg_gap(generic_map, mask)
        gap_detail = mask_bg_gap(detail_map, mask)
        m = mask.astype(bool)
        generic_mask_mean = float(generic_map[m].mean())
        detail_mask_mean = float(detail_map[m].mean())
        generic_bg_mean = float(generic_map[~m].mean())
        detail_bg_mean = float(detail_map[~m].mean())
        generic_top_hit = topk_hit(generic_map, mask)
        detail_top_hit = topk_hit(detail_map, mask)
        records.append(
            {
                "sample_path": str(path),
                "img_path": sample["img_path"],
                "cls_name": sample["cls_name"],
                "defect_type": defect_type_from_path(sample["img_path"]),
                "detail_prompt": detail_prompt,
                "mask": mask,
                "generic_map": generic_map,
                "detail_map": detail_map,
                "gap_generic": gap_generic,
                "gap_detail": gap_detail,
                "gap_delta": gap_detail - gap_generic,
                "mask_mean_delta": detail_mask_mean - generic_mask_mean,
                "bg_mean_delta": detail_bg_mean - generic_bg_mean,
                "top_hit_delta": detail_top_hit - generic_top_hit,
            }
        )

    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["cls_name"]].append(rec)

    class_summaries = []
    for cls_name, cls_records in by_class.items():
        by_defect = defaultdict(list)
        for rec in cls_records:
            if rec["gap_delta"] > 0 and rec["mask_mean_delta"] > 0 and rec["top_hit_delta"] >= 0:
                by_defect[rec["defect_type"]].append(rec)
        defect_types = [k for k, v in by_defect.items() if v]
        class_summaries.append((len(defect_types), np.mean([r["gap_delta"] for r in cls_records]), cls_name))
    class_summaries.sort(reverse=True)
    classes_to_export = [args.class_name] if args.class_name else ([c for _, _, c in class_summaries] if args.all_classes else [class_summaries[0][2]])

    all_selected = []
    per_class_selected = {}
    for chosen_cls in classes_to_export:
        by_defect = defaultdict(list)
        for rec in by_class[chosen_cls]:
            if rec["gap_delta"] > 0 and rec["mask_mean_delta"] > 0 and rec["top_hit_delta"] >= 0:
                by_defect[rec["defect_type"]].append(rec)
        selected = []
        for defect, recs in sorted(by_defect.items()):
            recs = sorted(
                recs,
                key=lambda r: (
                    r["top_hit_delta"],
                    r["gap_delta"],
                    r["mask_mean_delta"],
                    -max(r["bg_mean_delta"], 0.0),
                ),
                reverse=True,
            )
            selected.extend(recs[: args.per_defect])
        selected = sorted(selected, key=lambda r: r["gap_delta"], reverse=True)[: args.max_rows]
        selected = sorted(selected, key=lambda r: (r["defect_type"], -r["gap_delta"]))
        if selected:
            per_class_selected[chosen_cls] = selected
            all_selected.extend(selected)

    if not all_selected:
        raise RuntimeError("no positive detail examples found")

    # Save individual overlays with dataset-style names.
    overlay_root = args.save_dir / "individual"
    for rec in all_selected:
        rel = Path(rec["cls_name"]) / "test" / rec["defect_type"] / Path(rec["img_path"]).name
        vmin = min(float(np.nanmin(rec["generic_map"])), float(np.nanmin(rec["detail_map"])))
        vmax = max(float(np.nanmax(rec["generic_map"])), float(np.nanmax(rec["detail_map"])))
        for kind, amap in [("generic", rec["generic_map"]), ("detail", rec["detail_map"])]:
            out = overlay_root / kind / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            overlay_scoremap(rec["img_path"], amap, image_size, vmin=vmin, vmax=vmax).save(out)

    for cls_name, selected in per_class_selected.items():
        make_contact_sheet(selected, args.save_dir / f"{cls_name}_detail_vs_generic_contact.png", image_size=image_size)
    make_contact_sheet(all_selected[: min(len(all_selected), 80)], args.save_dir / "all_candidate_contact.png", image_size=image_size)

    csv_path = args.save_dir / "selected_examples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "cls_name",
            "defect_type",
            "img_path",
            "gap_generic",
            "gap_detail",
            "gap_delta",
            "mask_mean_delta",
            "bg_mean_delta",
            "top_hit_delta",
            "detail_prompt",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_selected:
            writer.writerow({key: rec[key] for key in fieldnames})

    summary = {
        "classes": sorted(per_class_selected.keys()),
        "num_selected": len(all_selected),
        "defect_types": sorted(set(r["defect_type"] for r in all_selected)),
        "contact_sheet": str(args.save_dir / "all_candidate_contact.png"),
        "csv": str(csv_path),
        "class_candidates": [
            {"class": cls, "positive_defect_types": n, "mean_gap_delta": float(mean)}
            for n, mean, cls in class_summaries[:10]
        ],
    }
    (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
