#!/usr/bin/env python3
"""Select paper-ready qualitative localization examples from generated overlays.

The experiment visualization script writes per-directory ``visible_examples.tsv``
files and matching overlay panel images. This selector merges those directories,
keeps examples with measured localization improvement, copies the best panels to
a single paper-candidate directory, and creates a contact sheet for quick manual
inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Candidate:
    source_dir: str
    dataset: str
    class_name: str
    rank: int
    image: str
    source_image_path: str
    visible_score: float
    delta_iou: float
    baseline_iou: float
    method_iou: float
    delta_f1: float
    baseline_f1: float
    method_f1: float
    delta_contrast: float
    background_drop: float
    anomaly_gain: float
    baseline_pixel_ap: float
    method_pixel_ap: float
    delta_pixel_ap: float
    selection_score: float
    panel_path: str
    selected_path: str = ""


def _float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in {"", "nan", "None", None}:
        return 0.0
    return float(value)


def _load_summary(source_dir: Path) -> dict:
    summary_path = source_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text())


def _infer_dataset(source_dir: Path, summary: dict) -> str:
    baseline_log = str(summary.get("baseline_log", "")).lower()
    path_text = str(source_dir).lower()
    for dataset in ("mvtec", "visa", "btad", "mpdd", "dtd_synthetic", "medical"):
        if dataset in baseline_log or dataset in path_text:
            return dataset
    return "unknown"


def _infer_class_name(source_dir: Path, summary: dict) -> str:
    selected = summary.get("selected_class")
    if selected:
        return str(selected)
    return source_dir.name


def _find_panel(source_dir: Path, rank: int, image: str) -> Optional[Path]:
    examples_dir = source_dir / "visible_localization_examples"
    if not examples_dir.is_dir():
        return None
    stem = Path(image).stem
    exact = examples_dir / f"{rank:02d}_{stem}.png"
    if exact.is_file():
        return exact
    matches = sorted(examples_dir.glob(f"{rank:02d}_*.png"))
    if matches:
        return matches[0]
    return None


def _candidate_key(candidate: Candidate) -> tuple:
    return (
        candidate.dataset,
        candidate.class_name,
        candidate.rank,
        candidate.image,
        round(candidate.visible_score, 6),
        round(candidate.delta_iou, 6),
        round(candidate.delta_f1, 6),
        round(candidate.baseline_pixel_ap, 6),
        round(candidate.method_pixel_ap, 6),
    )


def _deduplicate_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    best_by_key: dict[tuple, Candidate] = {}
    for candidate in candidates:
        key = _candidate_key(candidate)
        current = best_by_key.get(key)
        if current is None or candidate.selection_score > current.selection_score:
            best_by_key[key] = candidate
    return sorted(best_by_key.values(), key=lambda item: item.selection_score, reverse=True)


def _read_candidates(source_dir: Path) -> List[Candidate]:
    table_path = source_dir / "visible_examples.tsv"
    if not table_path.is_file():
        return []
    summary = _load_summary(source_dir)
    dataset = _infer_dataset(source_dir, summary)
    class_name = _infer_class_name(source_dir, summary)
    candidates: List[Candidate] = []
    with table_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rank = int(row["rank"])
            image = row["image"]
            panel = _find_panel(source_dir, rank, image)
            if panel is None:
                continue
            baseline_pixel_ap = _float(row, "baseline_pixel_ap")
            method_pixel_ap = _float(row, "method_pixel_ap")
            delta_pixel_ap = method_pixel_ap - baseline_pixel_ap
            visible_score = _float(row, "visible_score")
            delta_iou = _float(row, "delta_iou")
            delta_f1 = _float(row, "delta_f1")
            delta_contrast = _float(row, "delta_contrast")
            background_drop = _float(row, "background_drop")
            anomaly_gain = _float(row, "anomaly_gain")
            selection_score = (
                visible_score
                + 1.25 * delta_iou
                + 0.75 * delta_f1
                + 0.35 * max(delta_pixel_ap, 0.0)
                + 0.20 * delta_contrast
                + 0.10 * background_drop
                + 0.10 * anomaly_gain
            )
            candidates.append(
                Candidate(
                    source_dir=str(source_dir),
                    dataset=dataset,
                    class_name=class_name,
                    rank=rank,
                    image=image,
                    source_image_path=image,
                    visible_score=visible_score,
                    delta_iou=delta_iou,
                    baseline_iou=_float(row, "baseline_iou"),
                    method_iou=_float(row, "method_iou"),
                    delta_f1=delta_f1,
                    baseline_f1=_float(row, "baseline_f1"),
                    method_f1=_float(row, "method_f1"),
                    delta_contrast=delta_contrast,
                    background_drop=background_drop,
                    anomaly_gain=anomaly_gain,
                    baseline_pixel_ap=baseline_pixel_ap,
                    method_pixel_ap=method_pixel_ap,
                    delta_pixel_ap=delta_pixel_ap,
                    selection_score=selection_score,
                    panel_path=str(panel),
                )
            )
    return candidates


def _iter_source_dirs(paths: Sequence[Path]) -> Iterable[Path]:
    for path in paths:
        if (path / "visible_examples.tsv").is_file():
            yield path
            continue
        for table_path in sorted(path.glob("**/visible_examples.tsv")):
            if "paper_visualization_selection" in table_path.parts:
                continue
            yield table_path.parent


def _filter_candidates(
    candidates: Sequence[Candidate],
    min_visible_score: float,
    min_delta_iou: float,
    min_delta_f1: float,
    require_pixel_ap_gain: bool,
) -> List[Candidate]:
    filtered = []
    for candidate in candidates:
        if candidate.visible_score < min_visible_score:
            continue
        if candidate.delta_iou < min_delta_iou:
            continue
        if candidate.delta_f1 < min_delta_f1:
            continue
        if require_pixel_ap_gain and candidate.delta_pixel_ap <= 0:
            continue
        filtered.append(candidate)
    return filtered


def _balanced_select(candidates: Sequence[Candidate], total: int, per_class: int) -> List[Candidate]:
    ordered = sorted(candidates, key=lambda item: item.selection_score, reverse=True)
    selected: List[Candidate] = []
    counts: dict[str, int] = {}
    for candidate in ordered:
        key = f"{candidate.dataset}/{candidate.class_name}"
        if counts.get(key, 0) >= per_class:
            continue
        selected.append(candidate)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= total:
            break
    if len(selected) < total:
        seen = {(item.source_dir, item.rank, item.image) for item in selected}
        for candidate in ordered:
            key = (candidate.source_dir, candidate.rank, candidate.image)
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= total:
                break
    return selected


def _write_tsv(candidates: Sequence[Candidate], path: Path) -> None:
    fields = list(asdict(candidates[0]).keys()) if candidates else list(Candidate.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def _copy_selected(candidates: Sequence[Candidate], out_dir: Path) -> List[Candidate]:
    selected_dir = out_dir / "selected_panels"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for old_panel in selected_dir.glob("*.png"):
        old_panel.unlink()
    copied: List[Candidate] = []
    for idx, candidate in enumerate(candidates):
        filename = (
            f"{idx:02d}_{candidate.dataset}_{candidate.class_name}_"
            f"{Path(candidate.image).stem}.png"
        )
        target = selected_dir / filename
        shutil.copy2(candidate.panel_path, target)
        copied_candidate = Candidate(**asdict(candidate))
        copied_candidate.selected_path = str(target)
        copied.append(copied_candidate)
    return copied


def _make_contact_sheet(candidates: Sequence[Candidate], save_path: Path, thumb_width: int, columns: int) -> None:
    if not candidates:
        return
    rows = (len(candidates) + columns - 1) // columns
    label_height = 46
    thumbs = []
    for candidate in candidates:
        image = Image.open(candidate.selected_path).convert("RGB")
        ratio = thumb_width / image.width
        thumb_height = max(1, int(round(image.height * ratio)))
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumbs.append(image)
    thumb_height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (candidate, image) in enumerate(zip(candidates, thumbs)):
        row = idx // columns
        col = idx % columns
        x = col * thumb_width
        y = row * (thumb_height + label_height)
        sheet.paste(image, (x, y))
        label = (
            f"{idx:02d} {candidate.dataset}/{candidate.class_name} {candidate.image}\n"
            f"IoU {candidate.baseline_iou:.2f}->{candidate.method_iou:.2f}, "
            f"AP {candidate.baseline_pixel_ap:.2f}->{candidate.method_pixel_ap:.2f}"
        )
        draw.text((x + 6, y + thumb_height + 4), label, fill=(0, 0, 0), font=font)
    sheet.save(save_path)


def _write_report(
    candidates: Sequence[Candidate],
    raw_count: int,
    unique_count: int,
    filtered_count: int,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Paper Visualization Candidates",
        "",
        "## Selection Rule",
        "- Source panels are generated heatmap overlays that compare baseline and method maps on the original image.",
        f"- Candidate filter: visible_score >= {args.min_visible_score}, delta_iou >= {args.min_delta_iou}, delta_f1 >= {args.min_delta_f1}.",
        f"- Pixel AP gain required: {args.require_pixel_ap_gain}.",
        "- Final ranking uses visible_score plus IoU/F1 gain, positive pixel AP gain, contrast gain, background drop, and anomaly gain.",
        "",
        "## Counts",
        f"- Raw readable candidates: {raw_count}",
        f"- Unique readable candidates: {unique_count}",
        f"- Candidates after filters: {filtered_count}",
        f"- Selected candidates: {len(candidates)}",
        "",
        "## Selected Examples",
    ]
    for idx, candidate in enumerate(candidates):
        lines.append(
            f"- {idx:02d}: {candidate.dataset}/{candidate.class_name}/{candidate.image}; "
            f"visible_score={candidate.visible_score:.3f}, "
            f"IoU={candidate.baseline_iou:.3f}->{candidate.method_iou:.3f}, "
            f"F1={candidate.baseline_f1:.3f}->{candidate.method_f1:.3f}, "
            f"pixel AP={candidate.baseline_pixel_ap:.3f}->{candidate.method_pixel_ap:.3f}; "
            f"file={candidate.selected_path}"
        )
    lines.extend(
        [
            "",
            "## Paper Use Notes",
            "- Use the selected panels as a candidate pool; final manuscript figures should still be visually inspected for crop clarity and label readability.",
            "- Prefer a balanced subset across datasets/classes when composing the final figure.",
            "- The TSV/JSON files preserve the numeric evidence behind each qualitative choice.",
            "",
        ]
    )
    (args.save_dir / "selection_report.md").write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Select paper-ready localization overlays")
    parser.add_argument(
        "--source_dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/wavelet_feature_viz"),
        ],
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("outputs/wavelet_feature_viz/paper_visualization_selection"),
    )
    parser.add_argument("--total", type=int, default=24)
    parser.add_argument("--per_class", type=int, default=4)
    parser.add_argument("--min_visible_score", type=float, default=0.8)
    parser.add_argument("--min_delta_iou", type=float, default=0.10)
    parser.add_argument("--min_delta_f1", type=float, default=0.15)
    parser.add_argument("--require_pixel_ap_gain", action="store_true")
    parser.add_argument("--thumb_width", type=int, default=520)
    parser.add_argument("--columns", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    source_dirs = sorted(set(_iter_source_dirs(args.source_dirs)))
    all_candidates: List[Candidate] = []
    for source_dir in source_dirs:
        all_candidates.extend(_read_candidates(source_dir))
    raw_count = len(all_candidates)
    all_candidates = _deduplicate_candidates(all_candidates)

    filtered = _filter_candidates(
        all_candidates,
        min_visible_score=args.min_visible_score,
        min_delta_iou=args.min_delta_iou,
        min_delta_f1=args.min_delta_f1,
        require_pixel_ap_gain=args.require_pixel_ap_gain,
    )
    selected = _balanced_select(filtered, total=args.total, per_class=args.per_class)
    selected = _copy_selected(selected, args.save_dir)

    _write_tsv(all_candidates, args.save_dir / "all_candidates.tsv")
    _write_tsv(filtered, args.save_dir / "filtered_candidates.tsv")
    _write_tsv(selected, args.save_dir / "selected_examples.tsv")
    (args.save_dir / "selected_examples.json").write_text(
        json.dumps([asdict(item) for item in selected], indent=2, sort_keys=True)
    )
    _make_contact_sheet(
        selected,
        args.save_dir / "contact_sheet.png",
        thumb_width=args.thumb_width,
        columns=args.columns,
    )
    _write_report(selected, raw_count, len(all_candidates), len(filtered), args)

    print(f"source_dirs: {len(source_dirs)}")
    print(f"raw_candidates: {raw_count}")
    print(f"unique_candidates: {len(all_candidates)}")
    print(f"filtered_candidates: {len(filtered)}")
    print(f"selected: {len(selected)}")
    print(f"save_dir: {args.save_dir}")


if __name__ == "__main__":
    main()
