from pathlib import Path
import argparse
import csv
import json
import math
import sys
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import AnomalyCLIP_lib
from anomalyclip.prompt_ensemble import tokenize
from anomalyclip.visualization import apply_ad_scoremap


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
    "bent_lead": "bent lead",
    "bent_wire": "bent wire",
    "broken": "broken area",
    "broken_large": "large broken area",
    "broken_small": "small broken area",
    "broken_teeth": "broken teeth",
    "color": "discolored patch",
    "combined": "combined defects",
    "contamination": "contamination stain",
    "crack": "crack",
    "cut": "cut",
    "cut_inner_insulation": "cut in the inner insulation",
    "cut_lead": "cut lead",
    "cut_outer_insulation": "cut in the outer insulation",
    "damaged_case": "damaged case",
    "defective": "defective region",
    "fabric_border": "irregular fabric border",
    "fabric_interior": "irregular fabric interior",
    "faulty_imprint": "faulty imprint",
    "flip": "flipped part",
    "fold": "fold",
    "glue": "glue stain",
    "glue_strip": "glue strip",
    "gray_stroke": "gray stroke",
    "hole": "hole",
    "liquid": "liquid stain",
    "manipulated_front": "manipulated front surface",
    "metal_contamination": "metal contamination",
    "misplaced": "misplaced part",
    "missing_wire": "missing wire",
    "oil": "oil stain",
    "pill_type": "wrong pill type",
    "poke": "puncture mark",
    "poke_insulation": "punctured insulation",
    "print": "print defect",
    "rough": "rough region",
    "scratch": "scratch",
    "scratch_head": "scratch on the head",
    "scratch_neck": "scratch on the neck",
    "split_teeth": "split teeth",
    "squeeze": "squeezed deformation",
    "squeezed_teeth": "squeezed teeth",
    "thread": "thread-like defect",
    "thread_side": "thread on the side",
    "thread_top": "thread on the top",
}


def defect_type_from_path(img_path: str) -> str:
    parts = Path(img_path).parts
    if "test" in parts:
        idx = parts.index("test")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def object_phrase(cls_name: str) -> str:
    return OBJECT_PHRASES.get(cls_name, cls_name.replace("_", " "))


def defect_phrase(defect_type: str) -> str:
    return DEFECT_PHRASES.get(defect_type, defect_type.replace("_", " "))


def mask_np(mask: torch.Tensor, image_size: int) -> np.ndarray:
    if mask.dim() == 4:
        mask = mask.squeeze(0).squeeze(0)
    elif mask.dim() == 3:
        mask = mask.squeeze(0)
    mask = mask.float()
    if tuple(mask.shape[-2:]) != (image_size, image_size):
        mask = F.interpolate(mask.view(1, 1, *mask.shape[-2:]), size=(image_size, image_size), mode="nearest").squeeze()
    return (mask.detach().cpu().numpy() > 0.5)


def position_phrase(mask: np.ndarray) -> str:
    ys, xs = np.where(mask)
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


def size_phrase(mask: np.ndarray) -> str:
    area = float(mask.sum()) / float(mask.size)
    if area < 0.002:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.04:
        return "medium-sized"
    return "large"


def shape_orientation_phrase(mask: np.ndarray) -> Tuple[str, str]:
    ys, xs = np.where(mask)
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


def detail_prompt(sample: Dict, image_size: int) -> str:
    cls_name = sample["cls_name"]
    defect_type = defect_type_from_path(sample["img_path"])
    mask = mask_np(sample["img_mask"], image_size)
    attrs = [size_phrase(mask), shape_orientation_phrase(mask)[1], shape_orientation_phrase(mask)[0], defect_phrase(defect_type)]
    attrs = [a for a in attrs if a]
    article = "an" if attrs and attrs[0][0].lower() in "aeiou" else "a"
    return f"a photo of a {object_phrase(cls_name)} with {article} {' '.join(attrs)} in the {position_phrase(mask)} region."


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


def build_map(patch_features: Sequence[torch.Tensor], text_features: torch.Tensor, layers: Sequence[int], image_size: int) -> np.ndarray:
    first = layers[0] if layers else 0
    maps = []
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
    return torch.stack(maps).sum(dim=0).squeeze(0).detach().cpu().numpy()


def normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mn = min(float(np.nanmin(a)), float(np.nanmin(b)))
    mx = max(float(np.nanmax(a)), float(np.nanmax(b)))
    return (a - mn) / max(mx - mn, 1e-6), (b - mn) / max(mx - mn, 1e-6)


def top_mask(score: np.ndarray, ratio: float) -> np.ndarray:
    flat = score.reshape(-1)
    k = max(1, int(flat.size * ratio))
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(score.shape)


def component_recall(top: np.ndarray, gt: np.ndarray) -> float:
    labeled, n = ndimage.label(gt)
    if n == 0:
        return 0.0
    hit = 0
    for comp_id in range(1, n + 1):
        comp = labeled == comp_id
        if np.logical_and(comp, top).any():
            hit += 1
    return hit / n


def recovery_metrics(generic: np.ndarray, detail: np.ndarray, gt: np.ndarray, ratio: float) -> Dict[str, float]:
    gen_n, det_n = normalize_pair(generic, detail)
    gen_top = top_mask(gen_n, ratio)
    det_top = top_mask(det_n, ratio)
    missed = np.logical_and(gt, ~gen_top)
    recovered = np.logical_and(missed, det_top)
    gt_area = max(int(gt.sum()), 1)
    missed_area = max(int(missed.sum()), 1)
    return {
        "generic_coverage": float(np.logical_and(gen_top, gt).sum() / gt_area),
        "detail_coverage": float(np.logical_and(det_top, gt).sum() / gt_area),
        "coverage_gain": float((np.logical_and(det_top, gt).sum() - np.logical_and(gen_top, gt).sum()) / gt_area),
        "missed_recovery": float(recovered.sum() / missed_area),
        "component_recall_generic": component_recall(gen_top, gt),
        "component_recall_detail": component_recall(det_top, gt),
        "component_recall_gain": component_recall(det_top, gt) - component_recall(gen_top, gt),
    }


def overlay_scoremap(image_path: str, scoremap: np.ndarray, image_size: int) -> Image.Image:
    source = cv2.imread(image_path)
    if source is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(cv2.resize(source, (image_size, image_size)), cv2.COLOR_BGR2RGB)
    scoremap = np.clip(scoremap, 0, 1)
    return Image.fromarray(apply_ad_scoremap(image, scoremap))


def overlay_gt(image_path: str, gt: np.ndarray, image_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32)
    color = np.array([255, 35, 25], dtype=np.float32)
    arr[gt] = 0.54 * arr[gt] + 0.46 * color
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def recovered_panel(image_path: str, gt: np.ndarray, gen_top: np.ndarray, det_top: np.ndarray, image_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) * 0.55
    already = np.logical_and(gt, gen_top)
    recovered = np.logical_and(gt, np.logical_and(~gen_top, det_top))
    missed = np.logical_and(gt, np.logical_and(~gen_top, ~det_top))
    arr[already] = 0.45 * arr[already] + 0.55 * np.array([70, 130, 255])
    arr[recovered] = 0.35 * arr[recovered] + 0.65 * np.array([40, 210, 90])
    arr[missed] = 0.40 * arr[missed] + 0.60 * np.array([255, 40, 30])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_sheet(records: List[Dict], out_path: Path, image_size: int) -> None:
    tile = 230
    label_h = 66
    cols = 4
    canvas = Image.new("RGB", (cols * tile, len(records) * (tile + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    headers = ["Image + GT", "Generic", "Detailed", "Recovered GT"]
    for r, rec in enumerate(records):
        y = r * (tile + label_h)
        for c, h in enumerate(headers):
            draw.text((c * tile + 8, y + 6), h, fill=(0, 0, 0), font=font)
        gen_n, det_n = normalize_pair(rec["generic_map"], rec["detail_map"])
        gen_top = top_mask(gen_n, rec["top_ratio"])
        det_top = top_mask(det_n, rec["top_ratio"])
        panels = [
            overlay_gt(rec["img_path"], rec["gt"], image_size),
            overlay_scoremap(rec["img_path"], gen_n, image_size),
            overlay_scoremap(rec["img_path"], det_n, image_size),
            recovered_panel(rec["img_path"], rec["gt"], gen_top, det_top, image_size),
        ]
        for c, panel in enumerate(panels):
            canvas.paste(panel.resize((tile, tile)), (c * tile, y + label_h))
        draw.text((8, y + 34), f"{rec['cls_name']} / {rec['defect_type']}  rec +{rec['missed_recovery']:.2f}  cov +{rec['coverage_gain']:.2f}", fill=(20, 80, 20), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser("Find examples where detailed prompt recovers GT regions missed by generic prompt")
    parser.add_argument("--cache_dir", type=Path, default=Path("cache/mvtec_anomalyclip_features"))
    parser.add_argument("--save_dir", type=Path, default=Path("outputs/detail_prompt_oracle/mvtec_recovery_examples"))
    parser.add_argument("--lambda_detail", type=float, default=0.3)
    parser.add_argument("--top_ratio", type=float, default=0.03)
    parser.add_argument("--max_rows", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    metadata = torch.load(args.cache_dir / "metadata.pt", map_location="cpu")
    image_size = int(metadata["image_size"])
    layers = list(metadata["feature_map_layer"])
    generic_text = F.normalize(metadata["text_features"].float(), dim=-1)
    t_normal, t_abnormal = generic_text[0], generic_text[1]

    sample_paths = sorted((args.cache_dir / "samples").glob("*.pt"))
    samples = []
    prompts = []
    for path in sample_paths:
        sample = torch.load(path, map_location="cpu")
        if int(sample["anomaly"]) != 1:
            continue
        samples.append((path, sample))
        prompts.append(detail_prompt(sample, image_size))

    model = load_model_for_text(metadata, device=args.device)
    detail_feats = encode_plain_text(model, prompts, device=args.device)

    rows = []
    for (path, sample), detail_feat, prompt in zip(samples, detail_feats, prompts):
        detail_abnormal = F.normalize((1.0 - args.lambda_detail) * t_abnormal + args.lambda_detail * detail_feat.float(), dim=-1)
        generic_pair = torch.stack([t_normal, t_abnormal])
        detail_pair = torch.stack([t_normal, detail_abnormal])
        generic_map = build_map(sample["patch_features"], generic_pair, layers, image_size)
        detail_map = build_map(sample["patch_features"], detail_pair, layers, image_size)
        gt = mask_np(sample["img_mask"], image_size)
        metrics = recovery_metrics(generic_map, detail_map, gt, args.top_ratio)
        row = {
            "sample_path": str(path),
            "img_path": sample["img_path"],
            "cls_name": sample["cls_name"],
            "defect_type": defect_type_from_path(sample["img_path"]),
            "detail_prompt": prompt,
            "generic_map": generic_map,
            "detail_map": detail_map,
            "gt": gt,
            "top_ratio": args.top_ratio,
            **metrics,
        }
        rows.append(row)

    rows = [r for r in rows if r["coverage_gain"] > 0 and r["missed_recovery"] > 0]
    rows = sorted(rows, key=lambda r: (r["component_recall_gain"], r["missed_recovery"], r["coverage_gain"]), reverse=True)
    selected = rows[: args.max_rows]
    if not selected:
        raise RuntimeError("no recovery examples found")

    make_sheet(selected, args.save_dir / "detail_recovers_generic_misses_contact.png", image_size)
    csv_path = args.save_dir / "detail_recovers_generic_misses.csv"
    fields = ["cls_name", "defect_type", "img_path", "coverage_gain", "missed_recovery", "component_recall_gain", "detail_prompt"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in selected:
            writer.writerow({k: r[k] for k in fields})

    # Save individual overlays and recovered panels.
    root = args.save_dir / "individual"
    for rec in selected:
        rel = Path(rec["cls_name"]) / "test" / rec["defect_type"] / Path(rec["img_path"]).name
        gen_n, det_n = normalize_pair(rec["generic_map"], rec["detail_map"])
        gen_top = top_mask(gen_n, args.top_ratio)
        det_top = top_mask(det_n, args.top_ratio)
        outputs = {
            "generic": overlay_scoremap(rec["img_path"], gen_n, image_size),
            "detail": overlay_scoremap(rec["img_path"], det_n, image_size),
            "recovered": recovered_panel(rec["img_path"], rec["gt"], gen_top, det_top, image_size),
            "gt": overlay_gt(rec["img_path"], rec["gt"], image_size),
        }
        for kind, image in outputs.items():
            out = root / kind / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(out)

    summary = {
        "num_selected": len(selected),
        "top_ratio": args.top_ratio,
        "contact_sheet": str(args.save_dir / "detail_recovers_generic_misses_contact.png"),
        "csv": str(csv_path),
        "examples": [{k: r[k] for k in fields} for r in selected],
    }
    (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
