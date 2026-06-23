import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime


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


def merge_params(ablation):
    params = {}
    params.update(CURRENT_WAVELET)
    params.update(CURRENT_TTA)
    params.update(CURRENT_P2I)
    params.update(CURRENT_MULTICROP)
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


def make_command(dataset_name, dataset_config, ablation, save_path, args):
    params = merge_params(ablation)
    command = [
        sys.executable,
        "-B",
        "eval_cached_calibration.py",
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
            cwd=os.getcwd(),
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
    if suite == "internal":
        return INTERNAL_ABLATIONS
    if suite == "all":
        return COMPONENT_ABLATIONS + INTERNAL_ABLATIONS
    raise ValueError(f"unknown suite: {suite}")


def write_summary(sweep_dir, rows, args):
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
            config = DATASETS[dataset_name]
            original = config["original_mean"]
            handle.write(f"## {dataset_name}\n\n")
            handle.write(f"原始 AnomalyCLIP 日志：`{config['original_log']}`\n\n")
            handle.write("| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP | 说明 |\n")
            handle.write("|:--|--:|--:|--:|--:|:--|\n")
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


def parse_args():
    parser = argparse.ArgumentParser("Run cached AnomalyCLIP ablation experiments")
    parser.add_argument("--suite", choices=["component", "internal", "all"], default="component")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS.keys()), default=["mvtec", "visa"])
    parser.add_argument("--save_root", default="./ablation_results")
    parser.add_argument("--metrics", default="image-pixel-level", choices=["image-level", "pixel-level", "image-pixel-level"])
    parser.add_argument("--aupro_steps", type=int, default=200)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigma", type=float, default=5)
    parser.add_argument("--layer_weighting", default="sum", choices=["sum", "dynamic_wavelet"])
    parser.add_argument("--layer_weight_temperature", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.save_root, f"{sweep_id}_{args.suite}")
    os.makedirs(sweep_dir, exist_ok=True)
    master_log_path = os.path.join(sweep_dir, "ablation_log.txt")

    ablations = selected_ablations(args.suite)
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
        dataset_config = DATASETS[dataset_name]
        for index, ablation in enumerate(ablations, start=1):
            run_name = f"{index:02d}_{ablation['name']}"
            run_dir = os.path.join(sweep_dir, dataset_name, run_name)
            command = make_command(dataset_name, dataset_config, ablation, run_dir, args)

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

    summary_path = write_summary(sweep_dir, rows, args) if not args.dry_run else None
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
