import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from config_utils import parse_args_with_config


PARAMETER_SETS = [
    {
        "name": "current_best_msml_p2i010",
        "tta_mode": "wavelet_guided",
        "wavelet_beta": 0.20,
        "wavelet_condition_power": 2.0,
        "wavelet_suppress_beta": 0.0,
        "texture_max_delta_ratio": 0.05,
        "texture_suppression_weight": 0.0,
        "texture_local_contrast_kernel": 17,
        "texture_local_contrast_weight": 0.5,
        "rank_preserve_topk_ratio": 0.35,
        "rank_gate_mode": "hard",
        "rank_gate_temperature": 0.05,
        "use_wavelet_confidence": True,
        "wavelet_confidence_power": 1.0,
        "wavelet_levels": 2,
        "wavelet_level_fusion": "mean",
        "wavelet_min_reliability": 0.0,
        "texture_delta_reliability_power": 0.0,
        "use_image_to_pixel_gate": False,
        "image_to_pixel_weight": 0.0,
        "image_to_pixel_power": 1.0,
        "image_to_pixel_min_gate": 0.0,
        "use_pixel_to_image_fusion": True,
        "pixel_to_image_weight": 0.10,
        "pixel_to_image_topk_ratio": 0.01,
        "pixel_to_image_normalize": False,
        "layer_weighting": "sum",
        "layer_weight_temperature": 1.0,
        "sigma": 5,
        "tta_alpha": 0.01,
        "tta_topk_ratio": 0.02,
        "tta_min_confidence": 0.20,
        "tta_min_confidence_margin": 0.0,
        "tta_anchor_layers": "mean",
        "tta_repulsion_weight": 0.10,
        "tta_abnormal_alpha_scale": 0.75,
    },
    {
        "name": "current_best_with_layer_consistency",
        "tta_mode": "wavelet_guided",
        "wavelet_beta": 0.20,
        "wavelet_condition_power": 2.0,
        "wavelet_suppress_beta": 0.0,
        "texture_max_delta_ratio": 0.05,
        "texture_suppression_weight": 0.0,
        "texture_local_contrast_kernel": 17,
        "texture_local_contrast_weight": 0.5,
        "rank_preserve_topk_ratio": 0.35,
        "rank_gate_mode": "hard",
        "rank_gate_temperature": 0.05,
        "use_wavelet_confidence": True,
        "wavelet_confidence_power": 1.0,
        "wavelet_levels": 2,
        "wavelet_level_fusion": "mean",
        "wavelet_min_reliability": 0.0,
        "texture_delta_reliability_power": 0.0,
        "use_image_to_pixel_gate": False,
        "image_to_pixel_weight": 0.0,
        "image_to_pixel_power": 1.0,
        "image_to_pixel_min_gate": 0.0,
        "use_pixel_to_image_fusion": True,
        "pixel_to_image_weight": 0.10,
        "pixel_to_image_topk_ratio": 0.01,
        "pixel_to_image_normalize": False,
        "layer_weighting": "sum",
        "layer_weight_temperature": 1.0,
        "sigma": 5,
        "tta_alpha": 0.01,
        "tta_topk_ratio": 0.02,
        "tta_min_confidence": 0.20,
        "tta_min_confidence_margin": 0.0,
        "tta_anchor_layers": "mean",
        "tta_repulsion_weight": 0.10,
        "tta_abnormal_alpha_scale": 0.75,
        "use_layer_consistency": True,
        "layer_consistency_topk_ratio": 0.10,
        "layer_consistency_min_layers": 2,
        "layer_consistency_boost": 0.03,
        "layer_consistency_suppress": 0.0,
        "layer_consistency_condition_power": 1.0,
    },
    {
        "name": "current_best_with_wavelet_residual",
        "tta_mode": "wavelet_guided",
        "wavelet_beta": 0.20,
        "wavelet_condition_power": 2.0,
        "wavelet_suppress_beta": 0.0,
        "texture_max_delta_ratio": 0.05,
        "texture_suppression_weight": 0.0,
        "texture_local_contrast_kernel": 17,
        "texture_local_contrast_weight": 0.5,
        "rank_preserve_topk_ratio": 0.35,
        "rank_gate_mode": "hard",
        "rank_gate_temperature": 0.05,
        "use_wavelet_confidence": True,
        "wavelet_confidence_power": 1.0,
        "wavelet_levels": 2,
        "wavelet_level_fusion": "mean",
        "wavelet_min_reliability": 0.0,
        "texture_delta_reliability_power": 0.0,
        "use_image_to_pixel_gate": False,
        "image_to_pixel_weight": 0.0,
        "image_to_pixel_power": 1.0,
        "image_to_pixel_min_gate": 0.0,
        "use_pixel_to_image_fusion": True,
        "pixel_to_image_weight": 0.10,
        "pixel_to_image_topk_ratio": 0.01,
        "pixel_to_image_normalize": False,
        "layer_weighting": "sum",
        "layer_weight_temperature": 1.0,
        "sigma": 5,
        "tta_alpha": 0.01,
        "tta_topk_ratio": 0.02,
        "tta_min_confidence": 0.20,
        "tta_min_confidence_margin": 0.0,
        "tta_anchor_layers": "mean",
        "tta_repulsion_weight": 0.10,
        "tta_abnormal_alpha_scale": 0.75,
        "use_wavelet_residual": True,
        "wavelet_residual_weight": 0.03,
        "wavelet_residual_topk_ratio": 0.35,
        "wavelet_residual_condition_power": 2.0,
        "wavelet_residual_max_delta_ratio": 0.03,
        "wavelet_residual_local_contrast_kernel": 17,
        "wavelet_residual_local_contrast_weight": 0.5,
        "wavelet_residual_rank_gate_mode": "hard",
        "wavelet_residual_confidence_power": 1.0,
    },
    {
        "name": "wg_tta_mid_repulsion",
        "tta_mode": "wavelet_guided",
        "wavelet_beta": 0.20,
        "wavelet_condition_power": 2.0,
        "wavelet_suppress_beta": 0.0,
        "texture_max_delta_ratio": 0.05,
        "texture_suppression_weight": 0.0,
        "texture_local_contrast_kernel": 17,
        "texture_local_contrast_weight": 0.7,
        "rank_preserve_topk_ratio": 0.20,
        "rank_gate_mode": "soft",
        "rank_gate_temperature": 0.04,
        "use_wavelet_confidence": True,
        "wavelet_confidence_power": 1.5,
        "wavelet_levels": 1,
        "wavelet_level_fusion": "mean",
        "wavelet_min_reliability": 0.0,
        "texture_delta_reliability_power": 0.5,
        "use_image_to_pixel_gate": False,
        "image_to_pixel_weight": 0.0,
        "image_to_pixel_power": 1.0,
        "image_to_pixel_min_gate": 0.0,
        "use_pixel_to_image_fusion": False,
        "pixel_to_image_weight": 0.0,
        "pixel_to_image_topk_ratio": 0.01,
        "pixel_to_image_normalize": False,
        "layer_weighting": "sum",
        "layer_weight_temperature": 1.0,
        "sigma": 5,
        "tta_alpha": 0.04,
        "tta_topk_ratio": 0.02,
        "tta_min_confidence": 0.20,
        "tta_min_confidence_margin": 0.04,
        "tta_anchor_layers": "last",
        "tta_repulsion_weight": 0.25,
        "tta_abnormal_alpha_scale": 1.0,
    },
    {
        "name": "wg_tta_abnormal_light",
        "tta_mode": "wavelet_guided",
        "wavelet_beta": 0.25,
        "wavelet_condition_power": 2.0,
        "wavelet_suppress_beta": 0.0,
        "texture_max_delta_ratio": 0.05,
        "texture_suppression_weight": 0.0,
        "texture_local_contrast_kernel": 17,
        "texture_local_contrast_weight": 0.5,
        "rank_preserve_topk_ratio": 0.30,
        "rank_gate_mode": "soft",
        "rank_gate_temperature": 0.05,
        "use_wavelet_confidence": True,
        "wavelet_confidence_power": 1.0,
        "wavelet_levels": 1,
        "wavelet_level_fusion": "mean",
        "wavelet_min_reliability": 0.05,
        "texture_delta_reliability_power": 1.0,
        "use_image_to_pixel_gate": False,
        "image_to_pixel_weight": 0.0,
        "image_to_pixel_power": 1.0,
        "image_to_pixel_min_gate": 0.0,
        "use_pixel_to_image_fusion": False,
        "pixel_to_image_weight": 0.0,
        "pixel_to_image_topk_ratio": 0.01,
        "pixel_to_image_normalize": False,
        "layer_weighting": "dynamic_wavelet",
        "layer_weight_temperature": 0.5,
        "sigma": 5,
        "tta_alpha": 0.04,
        "tta_topk_ratio": 0.02,
        "tta_min_confidence": 0.25,
        "tta_min_confidence_margin": 0.04,
        "tta_anchor_layers": "last",
        "tta_repulsion_weight": 0.25,
        "tta_abnormal_alpha_scale": 0.5,
    },
    {
        "name": "wg_tta_high_confidence",
        "tta_mode": "wavelet_guided",
        "wavelet_beta": 0.20,
        "wavelet_condition_power": 3.0,
        "wavelet_suppress_beta": 0.0,
        "texture_max_delta_ratio": 0.03,
        "texture_suppression_weight": 0.0,
        "texture_local_contrast_kernel": 25,
        "texture_local_contrast_weight": 0.7,
        "rank_preserve_topk_ratio": 0.20,
        "rank_gate_mode": "soft",
        "rank_gate_temperature": 0.03,
        "use_wavelet_confidence": True,
        "wavelet_confidence_power": 2.0,
        "wavelet_levels": 1,
        "wavelet_level_fusion": "mean",
        "wavelet_min_reliability": 0.05,
        "texture_delta_reliability_power": 1.0,
        "use_image_to_pixel_gate": False,
        "image_to_pixel_weight": 0.0,
        "image_to_pixel_power": 1.0,
        "image_to_pixel_min_gate": 0.0,
        "use_pixel_to_image_fusion": False,
        "pixel_to_image_weight": 0.0,
        "pixel_to_image_topk_ratio": 0.01,
        "pixel_to_image_normalize": False,
        "layer_weighting": "sum",
        "layer_weight_temperature": 1.0,
        "sigma": 5,
        "tta_alpha": 0.03,
        "tta_topk_ratio": 0.015,
        "tta_min_confidence": 0.35,
        "tta_min_confidence_margin": 0.05,
        "tta_anchor_layers": "last",
        "tta_repulsion_weight": 0.20,
        "tta_abnormal_alpha_scale": 0.75,
    },
    {
        "name": "legacy_tta_control",
        "tta_mode": "legacy",
        "wavelet_beta": 0.25,
        "wavelet_condition_power": 2.0,
        "wavelet_suppress_beta": 0.0,
        "texture_max_delta_ratio": 0.05,
        "texture_suppression_weight": 0.0,
        "texture_local_contrast_kernel": 0,
        "texture_local_contrast_weight": 0.0,
        "rank_preserve_topk_ratio": 0.0,
        "rank_gate_mode": "hard",
        "rank_gate_temperature": 0.05,
        "use_wavelet_confidence": False,
        "wavelet_confidence_power": 1.0,
        "wavelet_levels": 1,
        "wavelet_level_fusion": "mean",
        "wavelet_min_reliability": 0.0,
        "texture_delta_reliability_power": 0.0,
        "use_image_to_pixel_gate": False,
        "image_to_pixel_weight": 0.0,
        "image_to_pixel_power": 1.0,
        "image_to_pixel_min_gate": 0.0,
        "use_pixel_to_image_fusion": False,
        "pixel_to_image_weight": 0.0,
        "pixel_to_image_topk_ratio": 0.01,
        "pixel_to_image_normalize": False,
        "layer_weighting": "sum",
        "layer_weight_temperature": 1.0,
        "sigma": 5,
        "tta_alpha": 0.04,
        "tta_topk_ratio": 0.02,
        "tta_min_confidence": 0.20,
        "tta_min_confidence_margin": 0.0,
        "tta_anchor_layers": "last",
        "tta_repulsion_weight": 0.0,
        "tta_abnormal_alpha_scale": 1.0,
    },
]


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_command(args, params, save_path):
    if args.runner == "cached":
        command = [
            sys.executable,
            "-B",
            "eval_cached_calibration.py",
            "--cache_dir",
            args.cache_dir,
            "--save_path",
            save_path,
        ]
    else:
        command = [
            sys.executable,
            "-B",
            "test.py",
            "--data_path",
            args.data_path,
            "--checkpoint_path",
            args.checkpoint_path,
            "--save_path",
            save_path,
        ]

    command += [
        "--metrics",
        args.metrics,
        "--aupro_steps",
        str(args.aupro_steps),
        "--feature_map_layer",
        *[str(layer) for layer in args.feature_map_layer],
        "--sigma",
        str(params.get("sigma", args.sigma)),
        "--layer_weighting",
        params["layer_weighting"],
        "--layer_weight_temperature",
        str(params["layer_weight_temperature"]),
        "--use_wavelet",
        "--wavelet_mode",
        args.wavelet_mode,
        "--wavelet_beta",
        str(params["wavelet_beta"]),
        "--wavelet_condition_power",
        str(params["wavelet_condition_power"]),
        "--wavelet_suppress_beta",
        str(params["wavelet_suppress_beta"]),
        "--wavelet_fusion",
        args.wavelet_fusion,
        "--wavelet_levels",
        str(params.get("wavelet_levels", args.wavelet_levels)),
        "--wavelet_level_fusion",
        params.get("wavelet_level_fusion", args.wavelet_level_fusion),
        "--texture_edge_power",
        str(args.texture_edge_power),
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
        params.get("rank_gate_mode", "hard"),
        "--rank_gate_temperature",
        str(params.get("rank_gate_temperature", 0.05)),
        "--use_tta_rectification",
        "--tta_mode",
        params["tta_mode"],
        "--tta_alpha",
        str(params["tta_alpha"]),
        "--tta_topk_ratio",
        str(params["tta_topk_ratio"]),
        "--tta_min_confidence",
        str(params["tta_min_confidence"]),
        "--tta_min_confidence_margin",
        str(params.get("tta_min_confidence_margin", 0.0)),
        "--tta_anchor_layers",
        params.get("tta_anchor_layers", "last"),
        "--tta_repulsion_weight",
        str(params["tta_repulsion_weight"]),
        "--tta_abnormal_alpha_scale",
        str(params["tta_abnormal_alpha_scale"]),
    ]
    if args.runner == "cached" and params.get("use_multicrop_fusion", False):
        command += [
            "--multicrop_cache_dir",
            params.get("multicrop_cache_dir", args.multicrop_cache_dir),
        ]

    if params.get("use_wavelet_confidence", False):
        command.append("--use_wavelet_confidence")
    if params.get("use_image_to_pixel_gate", False):
        command.append("--use_image_to_pixel_gate")
    if params.get("use_pixel_to_image_fusion", False):
        command.append("--use_pixel_to_image_fusion")
    if params.get("pixel_to_image_normalize", False):
        command.append("--pixel_to_image_normalize")
    if params.get("use_layer_consistency", False):
        command.append("--use_layer_consistency")
    if params.get("use_wavelet_residual", False):
        command.append("--use_wavelet_residual")
    if params.get("use_low_rank_residual", False):
        command.append("--use_low_rank_residual")
    if params.get("use_multicrop_fusion", False):
        command.append("--use_multicrop_fusion")
    if params.get("low_rank_no_center", False):
        command.append("--low_rank_no_center")
    command += [
        "--wavelet_confidence_power",
        str(params["wavelet_confidence_power"]),
        "--wavelet_min_reliability",
        str(params.get("wavelet_min_reliability", 0.0)),
        "--texture_delta_reliability_power",
        str(params.get("texture_delta_reliability_power", 0.0)),
        "--layer_consistency_topk_ratio",
        str(params.get("layer_consistency_topk_ratio", 0.10)),
        "--layer_consistency_min_layers",
        str(params.get("layer_consistency_min_layers", 2)),
        "--layer_consistency_boost",
        str(params.get("layer_consistency_boost", 0.03)),
        "--layer_consistency_suppress",
        str(params.get("layer_consistency_suppress", 0.0)),
        "--layer_consistency_condition_power",
        str(params.get("layer_consistency_condition_power", 1.0)),
        "--wavelet_residual_weight",
        str(params.get("wavelet_residual_weight", 0.03)),
        "--wavelet_residual_topk_ratio",
        str(params.get("wavelet_residual_topk_ratio", 0.35)),
        "--wavelet_residual_condition_power",
        str(params.get("wavelet_residual_condition_power", 2.0)),
        "--wavelet_residual_max_delta_ratio",
        str(params.get("wavelet_residual_max_delta_ratio", 0.03)),
        "--wavelet_residual_local_contrast_kernel",
        str(params.get("wavelet_residual_local_contrast_kernel", 17)),
        "--wavelet_residual_local_contrast_weight",
        str(params.get("wavelet_residual_local_contrast_weight", 0.5)),
        "--wavelet_residual_rank_gate_mode",
        params.get("wavelet_residual_rank_gate_mode", "hard"),
        "--wavelet_residual_rank_gate_temperature",
        str(params.get("wavelet_residual_rank_gate_temperature", 0.05)),
        "--wavelet_residual_confidence_power",
        str(params.get("wavelet_residual_confidence_power", 1.0)),
        "--low_rank_weight",
        str(params.get("low_rank_weight", 0.03)),
        "--low_rank_rank_ratio",
        str(params.get("low_rank_rank_ratio", 0.15)),
        "--low_rank_max_rank",
        str(params.get("low_rank_max_rank", 32)),
        "--low_rank_sample_patches",
        str(params.get("low_rank_sample_patches", 128)),
        "--low_rank_fusion",
        params.get("low_rank_fusion", "mean"),
        "--low_rank_condition_power",
        str(params.get("low_rank_condition_power", 2.0)),
        "--low_rank_max_delta_ratio",
        str(params.get("low_rank_max_delta_ratio", 0.03)),
        "--low_rank_clip_topk_ratio",
        str(params.get("low_rank_clip_topk_ratio", 0.35)),
        "--low_rank_clip_gate_mode",
        params.get("low_rank_clip_gate_mode", "hard"),
        "--low_rank_clip_gate_temperature",
        str(params.get("low_rank_clip_gate_temperature", 0.05)),
        "--low_rank_layer_selection",
        params.get("low_rank_layer_selection", "all"),
        "--multicrop_weight",
        str(params.get("multicrop_weight", 0.25)),
        "--image_to_pixel_weight",
        str(params.get("image_to_pixel_weight", 0.0)),
        "--image_to_pixel_power",
        str(params.get("image_to_pixel_power", 1.0)),
        "--image_to_pixel_min_gate",
        str(params.get("image_to_pixel_min_gate", 0.0)),
        "--pixel_to_image_weight",
        str(params.get("pixel_to_image_weight", 0.0)),
        "--pixel_to_image_topk_ratio",
        str(params.get("pixel_to_image_topk_ratio", 0.01)),
    ]

    if args.tta_update_abnormal:
        command.append("--tta_update_abnormal")

    return command


def command_text(command):
    return " ".join(command)


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
    return process.returncode


def has_finished(run_dir):
    log_path = os.path.join(run_dir, "log.txt")
    if not os.path.exists(log_path):
        return False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        return "| mean" in handle.read()


def build_parser():
    parser = argparse.ArgumentParser("Run AnomalyCLIP wavelet+TTA parameter sweep")
    parser.add_argument("--runner", choices=["test", "cached"], default="test")
    parser.add_argument("--cache_dir", default="./cache/mvtec_anomalyclip_features")
    parser.add_argument("--multicrop_cache_dir", default="./cache/mvtec_multicrop_maps")
    parser.add_argument("--data_path", default="/Users/bytedance/Downloads/mvtec_anomaly_detection")
    parser.add_argument("--checkpoint_path", default="/Users/bytedance/code/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth")
    parser.add_argument("--save_root", default="./sweep_results")
    parser.add_argument("--metrics", default="image-pixel-level", choices=["image-level", "pixel-level", "image-pixel-level"])
    parser.add_argument("--aupro_steps", type=int, default=200)
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sigma", type=float, default=5)
    parser.add_argument("--wavelet_mode", default="dual_route", choices=["dual_route", "adaptive"])
    parser.add_argument("--wavelet_fusion", default="mean", choices=["mean", "sum"])
    parser.add_argument("--wavelet_levels", type=int, default=1)
    parser.add_argument("--wavelet_level_fusion", default="mean", choices=["mean", "sum"])
    parser.add_argument("--texture_edge_power", type=float, default=1.0)
    parser.add_argument("--tta_update_abnormal", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip runs whose log.txt already has a mean row")
    parser.add_argument("--dry_run", action="store_true", help="print commands without running them")
    return parser


def parse_args():
    parser = build_parser()
    return parse_args_with_config(parser, default_config_path="./conf/run_param_sweep_conf.yaml")


def main():
    args, config = parse_args()
    parameter_sets = config.get("parameter_sets", PARAMETER_SETS)
    if not isinstance(parameter_sets, list):
        raise ValueError("'parameter_sets' in config must be a list")
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.save_root, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    master_log_path = os.path.join(sweep_dir, "sweep_log.txt")

    with open(master_log_path, "w", encoding="utf-8") as master_log:
        master_log.write(f"sweep_start: {now_text()}\n")
        master_log.write(f"runner: {args.runner}\n")
        master_log.write(f"aupro_steps: {args.aupro_steps}\n")
        master_log.write(f"metrics: {args.metrics}\n")
        master_log.write(f"feature_map_layer: {args.feature_map_layer}\n")
        master_log.write(f"default_sigma: {args.sigma}\n")

    failures = []
    for idx, params in enumerate(parameter_sets, start=1):
        run_name = f"{idx:02d}_{params['name']}"
        run_dir = os.path.join(sweep_dir, run_name)
        command = make_command(args, params, run_dir)

        if args.dry_run:
            print(command_text(command))
            continue

        if args.resume and has_finished(run_dir):
            print(f"[{now_text()}] skip finished: {run_name}")
            continue

        print(f"[{now_text()}] running {run_name}")
        returncode = run_one(command, run_dir, master_log_path)
        if returncode != 0:
            failures.append((run_name, returncode))
            print(f"[{now_text()}] failed {run_name}: returncode={returncode}")
        else:
            print(f"[{now_text()}] finished {run_name}")

    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write(f"\nsweep_end: {now_text()}\n")
        if failures:
            master_log.write("failures:\n")
            for run_name, returncode in failures:
                master_log.write(f"  {run_name}: returncode={returncode}\n")
        else:
            master_log.write("failures: none\n")

    print(f"sweep_dir: {sweep_dir}")
    if failures:
        print(f"failed_runs: {len(failures)}")


if __name__ == "__main__":
    main()
