from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

from anomalyclip.config_utils import parse_args_with_config


DATASETS = {
    "mvtec": {
        "cache_dir": "./cache/mvtec_anomalyclip_features",
        "multicrop_cache_dir": "./cache/mvtec_multicrop_maps_grid2_ratio075",
        "original_log": "results/9_12_4_multiscale/zero_shot/log.txt",
        "original_mean": (91.1, 81.4, 91.6, 96.4),
    },
    "visa": {
        "cache_dir": "./cache/visa_anomalyclip_features",
        "multicrop_cache_dir": "./cache/visa_multicrop_maps_grid2_ratio075",
        "original_log": "results/9_12_4_multiscale_visa/zero_shot/log.txt",
        "original_mean": (95.5, 86.7, 82.0, 85.3),
    },
}


CURRENT_WAVELET = {
    "wavelet_mode": "dual_route",
    "wavelet_beta": 0.20,
    "wavelet_condition_power": 2.0,
    "wavelet_suppress_beta": 0.0,
    "wavelet_fusion": "mean",
    "wavelet_levels": 2,
    "wavelet_level_fusion": "mean",
    "texture_edge_power": 1.0,
    "texture_max_delta_ratio": 0.05,
    "texture_suppression_weight": 0.0,
    "texture_local_contrast_kernel": 17,
    "texture_local_contrast_weight": 0.5,
    "rank_preserve_topk_ratio": 0.35,
    "rank_gate_mode": "hard",
    "rank_gate_temperature": 0.05,
    "use_wavelet_confidence": True,
    "wavelet_confidence_power": 1.0,
}


CURRENT_TTA = {
    "tta_mode": "wavelet_guided",
    "tta_alpha": 0.01,
    "tta_topk_ratio": 0.02,
    "tta_min_confidence": 0.20,
    "tta_anchor_layers": "mean",
    "tta_repulsion_weight": 0.10,
    "tta_abnormal_alpha_scale": 0.75,
}


CURRENT_P2I = {
    "pixel_to_image_weight": 0.10,
    "pixel_to_image_topk_ratio": 0.01,
}


CURRENT_MULTICROP = {
    "multicrop_weight": 0.50,
}


CURRENT_PROTOTYPE = {
    "proto_temperature": 0.07,
    "proto_gamma": 1.0,
    "proto_eta": 1.0,
    "proto_topk_ratio": 0.20,
    "proto_alpha0": 0.0,
    "proto_beta0": 0.01,
    "proto_tau_a": 0.15,
    "proto_update_min_abnormal_confidence": 0.06,
    "proto_wavelet_mix": 0.05,
    "proto_wavelet_mode": "boundary_aware",
    "proto_conservative_update": True,
    "proto_anchor_layers": "last",
    "proto_layer_fusion": "sum",
    "proto_percentile_low": 1.0,
    "proto_percentile_high": 99.0,
    "direct_wavelet_fusion_weight": 0.5,
}


COMPONENT_ABLATIONS = [
    {
        "name": "cached_baseline_l123",
        "description": "cached baseline, layers 1/2/3, no proposed module",
    },
    {
        "name": "wavelet_only",
        "description": "baseline + wavelet calibration",
        "use_wavelet": True,
    },
    {
        "name": "tta_only",
        "description": "baseline + wavelet-guided TTA only",
        "use_tta": True,
    },
    {
        "name": "wavelet_tta",
        "description": "baseline + wavelet calibration + wavelet-guided TTA",
        "use_wavelet": True,
        "use_tta": True,
    },
    {
        "name": "wavelet_tta_p2i",
        "description": "wavelet + TTA + pixel-to-image fusion",
        "use_wavelet": True,
        "use_tta": True,
        "use_pixel_to_image": True,
    },
    {
        "name": "wavelet_tta_multicrop",
        "description": "wavelet + TTA + multi-crop fusion",
        "use_wavelet": True,
        "use_tta": True,
        "use_multicrop": True,
    },
    {
        "name": "full_method",
        "description": "wavelet + TTA + multi-crop + pixel-to-image fusion",
        "use_wavelet": True,
        "use_tta": True,
        "use_pixel_to_image": True,
        "use_multicrop": True,
    },
]


PROTOTYPE_ABLATIONS = [
    {
        "name": "baseline",
        "description": "cached AnomalyCLIP baseline, no proposed module",
    },
    {
        "name": "clip_only_proto",
        "description": "prototype adaptation with CLIP semantic evidence only",
        "use_prototype_adaptation": True,
        "proto_wavelet_mode": "none",
    },
    {
        "name": "direct_wavelet_fusion",
        "description": "ablation: direct S0 and W map fusion, no prototype adaptation",
        "use_direct_wavelet_fusion": True,
        "proto_wavelet_mode": "boundary_aware",
    },
    {
        "name": "hf_only_proto",
        "description": "prototype adaptation using Haar high-frequency reliability only",
        "use_prototype_adaptation": True,
        "proto_wavelet_mode": "hf_only",
    },
    {
        "name": "boundary_aware_proto",
        "description": "prototype adaptation using boundary-aware wavelet reliability",
        "use_prototype_adaptation": True,
        "proto_wavelet_mode": "boundary_aware",
    },
    {
        "name": "full_no_conservative",
        "description": "full prototype adaptation without conservative confidence gating",
        "use_prototype_adaptation": True,
        "proto_wavelet_mode": "boundary_aware",
        "proto_conservative_update": False,
    },
    {
        "name": "full_conservative",
        "description": "full boundary-aware prototype adaptation with conservative update",
        "use_prototype_adaptation": True,
        "proto_wavelet_mode": "boundary_aware",
        "proto_conservative_update": True,
    },
]


INTERNAL_ABLATIONS = [
    {
        "name": "full_no_wavelet_confidence",
        "description": "full method without wavelet confidence gating",
        "use_wavelet": True,
        "use_tta": True,
        "use_pixel_to_image": True,
        "use_multicrop": True,
        "use_wavelet_confidence": False,
    },
    {
        "name": "full_no_rank_preserve",
        "description": "full method without rank-preserve top-k protection",
        "use_wavelet": True,
        "use_tta": True,
        "use_pixel_to_image": True,
        "use_multicrop": True,
        "rank_preserve_topk_ratio": 0.0,
    },
    {
        "name": "full_no_local_contrast",
        "description": "full method without local texture contrast",
        "use_wavelet": True,
        "use_tta": True,
        "use_pixel_to_image": True,
        "use_multicrop": True,
        "texture_local_contrast_kernel": 0,
        "texture_local_contrast_weight": 0.0,
    },
]


MEAN_PATTERN = re.compile(
    r"\| mean\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|"
)


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def merge_params(ablation, config=None):
    config = config or {}
    params = {}
    params.update(config.get("current_wavelet", CURRENT_WAVELET))
    params.update(config.get("current_tta", CURRENT_TTA))
    params.update(config.get("current_pixel_to_image", CURRENT_P2I))
    params.update(config.get("current_multicrop", CURRENT_MULTICROP))
    params.update(config.get("current_prototype", CURRENT_PROTOTYPE))
    params.update(ablation)
    return params


def add_wavelet_gate_args(command, params):
    command += [
        "--wavelet_mode",
        params["wavelet_mode"],
        "--wavelet_fusion",
        params["wavelet_fusion"],
        "--wavelet_levels",
        str(params["wavelet_levels"]),
        "--wavelet_level_fusion",
        params["wavelet_level_fusion"],
        "--texture_edge_power",
        str(params["texture_edge_power"]),
    ]


def add_wavelet_args(command, params):
    command.append("--use_wavelet")
    command += [
        "--wavelet_beta",
        str(params["wavelet_beta"]),
        "--wavelet_condition_power",
        str(params["wavelet_condition_power"]),
        "--wavelet_suppress_beta",
        str(params["wavelet_suppress_beta"]),
        "--texture_max_delta_ratio",
        str(params["texture_max_delta_ratio"]),
        "--texture_suppression_weight",
        str(params["texture_suppression_weight"]),
        "--texture_local_contrast_kernel",
        str(params["texture_local_contrast_kernel"]),
        "--texture_local_contrast_weight",
        str(params["texture_local_contrast_weight"]),
        "--rank_preserve_topk_ratio",
        str(params["rank_preserve_topk_ratio"]),
        "--rank_gate_mode",
        params["rank_gate_mode"],
        "--rank_gate_temperature",
        str(params["rank_gate_temperature"]),
        "--wavelet_confidence_power",
        str(params["wavelet_confidence_power"]),
    ]
    if params.get("use_wavelet_confidence", False):
        command.append("--use_wavelet_confidence")


def add_tta_args(command, params):
    command.append("--use_tta_rectification")
    command += [
        "--tta_mode",
        params["tta_mode"],
        "--tta_alpha",
        str(params["tta_alpha"]),
        "--tta_topk_ratio",
        str(params["tta_topk_ratio"]),
        "--tta_min_confidence",
        str(params["tta_min_confidence"]),
        "--tta_anchor_layers",
        params["tta_anchor_layers"],
        "--tta_repulsion_weight",
        str(params["tta_repulsion_weight"]),
        "--tta_abnormal_alpha_scale",
        str(params["tta_abnormal_alpha_scale"]),
    ]


def add_pixel_to_image_args(command, params):
    command.append("--use_pixel_to_image_fusion")
    command += [
        "--pixel_to_image_weight",
        str(params["pixel_to_image_weight"]),
        "--pixel_to_image_topk_ratio",
        str(params["pixel_to_image_topk_ratio"]),
    ]


def add_multicrop_args(command, params, dataset_config):
    command.append("--use_multicrop_fusion")
    command += [
        "--multicrop_cache_dir",
        dataset_config["multicrop_cache_dir"],
        "--multicrop_weight",
        str(params["multicrop_weight"]),
    ]


def add_prototype_args(command, params):
    if params.get("use_prototype_adaptation", False):
        command.append("--use_wavelet_prototype_adaptation")
    if params.get("use_direct_wavelet_fusion", False):
        command.append("--use_direct_wavelet_fusion")
    command += [
        "--proto_temperature",
        str(params["proto_temperature"]),
        "--proto_gamma",
        str(params["proto_gamma"]),
        "--proto_eta",
        str(params["proto_eta"]),
        "--proto_topk_ratio",
        str(params["proto_topk_ratio"]),
        "--proto_alpha0",
        str(params["proto_alpha0"]),
        "--proto_beta0",
        str(params["proto_beta0"]),
        "--proto_tau_a",
        str(params["proto_tau_a"]),
        "--proto_update_min_abnormal_confidence",
        str(params["proto_update_min_abnormal_confidence"]),
        "--proto_wavelet_mix",
        str(params["proto_wavelet_mix"]),
        "--proto_wavelet_mode",
        params["proto_wavelet_mode"],
        "--proto_anchor_layers",
        params["proto_anchor_layers"],
        "--proto_layer_fusion",
        params["proto_layer_fusion"],
        "--proto_percentile_low",
        str(params["proto_percentile_low"]),
        "--proto_percentile_high",
        str(params["proto_percentile_high"]),
        "--direct_wavelet_fusion_weight",
        str(params["direct_wavelet_fusion_weight"]),
    ]
    if not params.get("proto_conservative_update", True):
        command.append("--no_proto_conservative_update")


def make_command(dataset_name, dataset_config, ablation, save_path, args, config=None):
    params = merge_params(ablation, config=config)
    command = [
        sys.executable,
        "-B",
        "scripts/evaluate/eval_cached_calibration.py",
        "--cache_dir",
        dataset_config["cache_dir"],
        "--save_path",
        save_path,
        "--dataset",
        dataset_name,
        "--metrics",
        args.metrics,
        "--aupro_steps",
        str(args.aupro_steps),
        "--feature_map_layer",
        *[str(layer) for layer in args.feature_map_layer],
        "--sigma",
        str(args.sigma),
        "--layer_weighting",
        args.layer_weighting,
        "--layer_weight_temperature",
        str(args.layer_weight_temperature),
    ]

    needs_wavelet_gate = params.get("use_wavelet", False) or params.get("use_tta", False)
    if needs_wavelet_gate:
        add_wavelet_gate_args(command, params)
    if params.get("use_prototype_adaptation", False) or params.get("use_direct_wavelet_fusion", False):
        add_prototype_args(command, params)
    if params.get("use_wavelet", False):
        add_wavelet_args(command, params)
    if params.get("use_tta", False):
        add_tta_args(command, params)
    if params.get("use_multicrop", False):
        add_multicrop_args(command, params, dataset_config)
    if params.get("use_pixel_to_image", False):
        add_pixel_to_image_args(command, params)
    return command


def command_text(command):
    return " ".join(command)


def has_mean_row(run_dir):
    log_path = os.path.join(run_dir, "log.txt")
    if not os.path.exists(log_path):
        return False
    return parse_mean(log_path) is not None


def parse_mean(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    match = MEAN_PATTERN.search(text)
    if not match:
        return None
    return tuple(float(item) for item in match.groups())


def run_one(command, run_dir, master_log_path):
    os.makedirs(run_dir, exist_ok=True)
    console_log_path = os.path.join(run_dir, "console.log")
    start = time.time()
    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write(f"\n[{now_text()}] START {os.path.basename(run_dir)}\n")
        master_log.write(f"command: {command_text(command)}\n")
        master_log.write(f"console_log: {console_log_path}\n")

    with open(console_log_path, "w", encoding="utf-8") as console_log:
        process = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=console_log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = time.time() - start
    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write(
            f"[{now_text()}] END {os.path.basename(run_dir)} "
            f"returncode={process.returncode} elapsed_sec={elapsed:.1f}\n"
        )
    return process.returncode, elapsed


def selected_ablations(suite):
    if suite == "component":
        return COMPONENT_ABLATIONS
    if suite == "prototype":
        return PROTOTYPE_ABLATIONS
    if suite == "internal":
        return INTERNAL_ABLATIONS
    if suite == "all":
        return COMPONENT_ABLATIONS + INTERNAL_ABLATIONS
    if suite == "all_with_prototype":
        return COMPONENT_ABLATIONS + INTERNAL_ABLATIONS + PROTOTYPE_ABLATIONS
    raise ValueError(f"unknown suite: {suite}")


def configured_ablations(suite, config):
    if "ablations" in config:
        ablations = config["ablations"]
        if not isinstance(ablations, list):
            raise ValueError("'ablations' in config must be a list")
        return ablations
    if (
        "component_ablations" in config
        or "internal_ablations" in config
        or "prototype_ablations" in config
    ):
        component = config.get("component_ablations", COMPONENT_ABLATIONS)
        internal = config.get("internal_ablations", INTERNAL_ABLATIONS)
        prototype = config.get("prototype_ablations", PROTOTYPE_ABLATIONS)
        if suite == "component":
            return component
        if suite == "internal":
            return internal
        if suite == "prototype":
            return prototype
        if suite == "all":
            return component + internal
        if suite == "all_with_prototype":
            return component + internal + prototype
    return selected_ablations(suite)


def configured_datasets(config):
    datasets = config.get("datasets_config", DATASETS)
    if not isinstance(datasets, dict):
        raise ValueError("'datasets_config' in config must be a mapping")
    return datasets


def write_summary(sweep_dir, rows, args, dataset_configs):
    summary_path = os.path.join(sweep_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# AnomalyCLIP 消融实验结果\n\n")
        handle.write(f"更新时间：{now_text()}\n\n")
        handle.write(f"- suite：`{args.suite}`\n")
        handle.write(f"- metrics：`{args.metrics}`\n")
        handle.write(f"- aupro steps：`{args.aupro_steps}`\n")
        handle.write(f"- feature map layer：`{' '.join(str(x) for x in args.feature_map_layer)}`\n")
        handle.write(f"- sigma：`{args.sigma}`\n\n")
        handle.write("指标顺序：`pixel AUROC | pixel AUPRO | image AUROC | image AP`\n\n")

        for dataset_name in args.datasets:
            config = dataset_configs[dataset_name]
            original = config.get("original_mean")
            handle.write(f"## {dataset_name}\n\n")
            handle.write(f"原始 AnomalyCLIP 日志：`{config.get('original_log', 'unknown')}`\n\n")
            handle.write("| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP | 说明 |\n")
            handle.write("|:--|--:|--:|--:|--:|:--|\n")
            if original:
                handle.write(
                    f"| original_anomalyclip | {original[0]:.1f} | {original[1]:.1f} | "
                    f"{original[2]:.1f} | {original[3]:.1f} | original log |\n"
                )
            for row in rows:
                if row["dataset"] != dataset_name:
                    continue
                mean = row.get("mean")
                if mean is None:
                    handle.write(
                        f"| {row['name']} | failed | failed | failed | failed | {row['description']} |\n"
                    )
                    continue
                handle.write(
                    f"| {row['name']} | {mean[0]:.1f} | {mean[1]:.1f} | "
                    f"{mean[2]:.1f} | {mean[3]:.1f} | {row['description']} |\n"
                )
            handle.write("\n")

        handle.write("## 运行目录\n\n")
        for row in rows:
            handle.write(f"- `{row['dataset']}/{row['name']}`：`{row['run_dir']}`\n")
    return summary_path


def build_parser():
    parser = argparse.ArgumentParser("Run cached AnomalyCLIP ablation experiments")
    parser.add_argument("--suite", choices=["component", "prototype", "internal", "all", "all_with_prototype"], default="component")
    parser.add_argument("--datasets", nargs="+", default=["mvtec", "visa"])
    parser.add_argument("--save_root", default="./ablation_results")
    parser.add_argument("--metrics", default="image-pixel-level", choices=["image-level", "pixel-level", "image-pixel-level"])
    parser.add_argument("--aupro_steps", type=int, default=200)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigma", type=float, default=5)
    parser.add_argument("--layer_weighting", default="sum", choices=["sum", "dynamic_wavelet"])
    parser.add_argument("--layer_weight_temperature", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def parse_args():
    parser = build_parser()
    return parse_args_with_config(parser, default_config_path="./conf/run_ablation_experiments_conf.yaml")


def main():
    args, config = parse_args()
    dataset_configs = configured_datasets(config)
    unknown_datasets = sorted(set(args.datasets) - set(dataset_configs))
    if unknown_datasets:
        raise ValueError("unknown datasets: " + ", ".join(unknown_datasets))
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.save_root, f"{sweep_id}_{args.suite}")
    os.makedirs(sweep_dir, exist_ok=True)
    master_log_path = os.path.join(sweep_dir, "ablation_log.txt")

    ablations = configured_ablations(args.suite, config)
    with open(master_log_path, "w", encoding="utf-8") as master_log:
        master_log.write(f"ablation_start: {now_text()}\n")
        master_log.write(f"suite: {args.suite}\n")
        master_log.write(f"datasets: {args.datasets}\n")
        master_log.write(f"metrics: {args.metrics}\n")
        master_log.write(f"aupro_steps: {args.aupro_steps}\n")
        master_log.write(f"feature_map_layer: {args.feature_map_layer}\n")

    rows = []
    failures = []
    for dataset_name in args.datasets:
        dataset_config = dataset_configs[dataset_name]
        for index, ablation in enumerate(ablations, start=1):
            run_name = f"{index:02d}_{ablation['name']}"
            run_dir = os.path.join(sweep_dir, dataset_name, run_name)
            command = make_command(dataset_name, dataset_config, ablation, run_dir, args, config=config)

            if args.dry_run:
                print(command_text(command))
                continue

            if args.resume and has_mean_row(run_dir):
                print(f"[{now_text()}] skip finished: {dataset_name}/{run_name}")
                mean = parse_mean(os.path.join(run_dir, "log.txt"))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "name": ablation["name"],
                        "description": ablation["description"],
                        "run_dir": run_dir,
                        "mean": mean,
                    }
                )
                continue

            print(f"[{now_text()}] running {dataset_name}/{run_name}")
            returncode, elapsed = run_one(command, run_dir, master_log_path)
            mean = parse_mean(os.path.join(run_dir, "log.txt"))
            rows.append(
                {
                    "dataset": dataset_name,
                    "name": ablation["name"],
                    "description": ablation["description"],
                    "run_dir": run_dir,
                    "mean": mean,
                }
            )
            if returncode != 0 or mean is None:
                failures.append((dataset_name, run_name, returncode))
                print(f"[{now_text()}] failed {dataset_name}/{run_name}: returncode={returncode}")
            else:
                print(
                    f"[{now_text()}] finished {dataset_name}/{run_name} "
                    f"elapsed_sec={elapsed:.1f} mean={mean}"
                )

    summary_path = write_summary(sweep_dir, rows, args, dataset_configs) if not args.dry_run else None
    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write(f"\nablation_end: {now_text()}\n")
        if summary_path:
            master_log.write(f"summary: {summary_path}\n")
        if failures:
            master_log.write("failures:\n")
            for dataset_name, run_name, returncode in failures:
                master_log.write(f"  {dataset_name}/{run_name}: returncode={returncode}\n")
        else:
            master_log.write("failures: none\n")

    print(f"sweep_dir: {sweep_dir}")
    if summary_path:
        print(f"summary: {summary_path}")
    if failures:
        print(f"failed_runs: {len(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
