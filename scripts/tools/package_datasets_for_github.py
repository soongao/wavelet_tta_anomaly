#!/usr/bin/env python3
"""Package local datasets as zip archives for GitHub Release upload.

This script intentionally writes outside the repository by default so large
archives are not accidentally added to git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_DIR = Path("/private/tmp/anomalyclip_dataset_release")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    source_dir: Path
    archive_stem: str


MAIN_DATASETS = [
    DatasetSpec(
        key="mvtec",
        display_name="MVTec AD",
        source_dir=Path("/Users/bytedance/Downloads/mvtec_anomaly_detection"),
        archive_stem="mvtec_ad",
    ),
    DatasetSpec(
        key="visa",
        display_name="VisA",
        source_dir=Path("/Users/bytedance/Downloads/VisA_20220922"),
        archive_stem="visa",
    ),
    DatasetSpec(
        key="mpdd",
        display_name="MPDD",
        source_dir=Path("/Users/bytedance/Downloads/MPDD"),
        archive_stem="mpdd",
    ),
    DatasetSpec(
        key="btad",
        display_name="BTAD",
        source_dir=Path("/Users/bytedance/Downloads/BTech_Dataset_transformed"),
        archive_stem="btad",
    ),
    DatasetSpec(
        key="dtd_synthetic",
        display_name="DTD-Synthetic",
        source_dir=Path("/Users/bytedance/Downloads/DTD-Synthetic"),
        archive_stem="dtd_synthetic",
    ),
]


MEDICAL_DATASETS = [
    DatasetSpec(
        key="clinicdb",
        display_name="CVC-ClinicDB",
        source_dir=Path("/Users/bytedance/Downloads/CVC-ClinicDB"),
        archive_stem="cvc_clinicdb",
    ),
    DatasetSpec(
        key="isbi",
        display_name="ISBI",
        source_dir=Path("/Users/bytedance/Downloads/ISBI"),
        archive_stem="isbi",
    ),
    DatasetSpec(
        key="kvasir",
        display_name="Kvasir",
        source_dir=Path("/Users/bytedance/Downloads/Kvasir"),
        archive_stem="kvasir",
    ),
]


DATASET_GROUPS = {
    "main": MAIN_DATASETS,
    "medical": MEDICAL_DATASETS,
    "all": MAIN_DATASETS + MEDICAL_DATASETS,
}


def all_dataset_specs() -> dict[str, DatasetSpec]:
    return {spec.key: spec for spec in DATASET_GROUPS["all"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one zip archive per dataset. Archives larger than split-size "
            "are written as split zip parts suitable for GitHub Release assets."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["main"],
        help=(
            "Dataset group or keys to package. Groups: main, medical, all. "
            "Keys: " + ", ".join(sorted(all_dataset_specs()))
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for archives and manifests. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--split-size",
        default="1900m",
        help=(
            "Info-ZIP split size, e.g. 1900m. Use 0 to disable splitting. "
            "GitHub repository files must stay below 100MB; use Releases for these archives."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing archive parts for selected datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned archives without creating files.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Do not zip datasets; rebuild manifest.json and SHA256SUMS.txt for existing archives.",
    )
    return parser.parse_args()


def resolve_datasets(selectors: Iterable[str]) -> list[DatasetSpec]:
    specs_by_key = all_dataset_specs()
    selected: list[DatasetSpec] = []
    seen: set[str] = set()

    for selector in selectors:
        if selector in DATASET_GROUPS:
            candidates = DATASET_GROUPS[selector]
        elif selector in specs_by_key:
            candidates = [specs_by_key[selector]]
        else:
            valid = sorted([*DATASET_GROUPS.keys(), *specs_by_key.keys()])
            raise SystemExit(f"unknown dataset selector {selector!r}; valid values: {', '.join(valid)}")

        for spec in candidates:
            if spec.key not in seen:
                selected.append(spec)
                seen.add(spec.key)

    return selected


def validate_split_size(value: str) -> str | None:
    if value == "0":
        return None
    if not re.fullmatch(r"[1-9][0-9]*[kKmMgGtT]?", value):
        raise SystemExit("--split-size must be 0 or a value like 1900m")
    return value


def archive_parts(output_dir: Path, archive_stem: str) -> list[Path]:
    return sorted(output_dir.glob(f"{archive_stem}.z[0-9][0-9]")) + [output_dir / f"{archive_stem}.zip"]


def remove_existing_parts(output_dir: Path, archive_stem: str) -> None:
    for path in archive_parts(output_dir, archive_stem):
        if path.exists():
            path.unlink()


def existing_parts(output_dir: Path, archive_stem: str) -> list[Path]:
    return [path for path in archive_parts(output_dir, archive_stem) if path.exists()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_existing_dataset(spec: DatasetSpec, output_dir: Path) -> dict[str, object]:
    parts = existing_parts(output_dir, spec.archive_stem)
    if not parts:
        raise SystemExit(f"archive not found for {spec.key} in {output_dir}")
    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "source_dir": str(spec.source_dir),
        "parts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in parts
        ],
    }


def package_dataset(
    spec: DatasetSpec,
    output_dir: Path,
    split_size: str | None,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, object]:
    if not spec.source_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {spec.source_dir}")

    planned_zip = output_dir / f"{spec.archive_stem}.zip"
    command = ["zip", "-r", "-q"]
    if split_size:
        command.extend(["-s", split_size])
    command.extend(
        [
            str(planned_zip),
            spec.source_dir.name,
            "-x",
            "*/.DS_Store",
            "__MACOSX/*",
        ]
    )

    existing = existing_parts(output_dir, spec.archive_stem)
    if existing and not overwrite:
        existing_list = ", ".join(str(path) for path in existing)
        raise SystemExit(
            f"archive already exists for {spec.key}: {existing_list}. "
            "Use --overwrite to replace it."
        )

    if dry_run:
        return {
            "key": spec.key,
            "display_name": spec.display_name,
            "source_dir": str(spec.source_dir),
            "archive": str(planned_zip),
            "command": command,
            "dry_run": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    if existing:
        remove_existing_parts(output_dir, spec.archive_stem)

    subprocess.run(command, cwd=spec.source_dir.parent, check=True)

    parts = existing_parts(output_dir, spec.archive_stem)
    if not parts:
        raise RuntimeError(f"zip did not produce archive parts for {spec.key}")

    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "source_dir": str(spec.source_dir),
        "parts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in parts
        ],
    }


def write_manifests(output_dir: Path, manifest: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = []
    for dataset in manifest["datasets"]:
        for part in dataset.get("parts", []):
            path = Path(part["path"])
            lines.append(f"{part['sha256']}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    split_size = validate_split_size(args.split_size)
    specs = resolve_datasets(args.datasets)

    if shutil.which("zip") is None:
        raise SystemExit("zip command not found")

    output_dir = args.output_dir.expanduser().resolve()
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "split_size": split_size or "disabled",
        "datasets": [],
        "notes": [
            "Upload these files as GitHub Release assets, not regular git-tracked files.",
            "Confirm redistribution rights for each dataset before publishing.",
            "For split zip archives, download all parts and extract from the .zip file.",
        ],
    }

    for spec in specs:
        if args.manifest_only:
            result = describe_existing_dataset(spec, output_dir)
        else:
            result = package_dataset(
                spec=spec,
                output_dir=output_dir,
                split_size=split_size,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        manifest["datasets"].append(result)
        if args.dry_run:
            print(f"[dry-run] {spec.key}: {spec.source_dir} -> {output_dir / (spec.archive_stem + '.zip')}")
        elif args.manifest_only:
            part_count = len(result["parts"])
            total_bytes = sum(part["bytes"] for part in result["parts"])
            print(f"{spec.key}: found {part_count} file(s), {total_bytes / (1024 ** 3):.2f} GiB")
        else:
            part_count = len(result["parts"])
            total_bytes = sum(part["bytes"] for part in result["parts"])
            print(f"{spec.key}: wrote {part_count} file(s), {total_bytes / (1024 ** 3):.2f} GiB")

    if not args.dry_run:
        write_manifests(output_dir, manifest)
        print(f"manifest: {output_dir / 'manifest.json'}")
        print(f"checksums: {output_dir / 'SHA256SUMS.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
