from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import ast
import os
import re
import shlex
from typing import Any, Dict, Iterable, List, Optional, Tuple

from anomalyclip.config_utils import parse_args_with_config, write_yaml_config


RESULT_ROOTS = [
    "my_results_mvtec",
    "my_results_mvtec_dual_route_best",
    "my_results_mvtec_dual_route_guarded",
    "my_results_mvtec_wavelet_tta_conservative",
    "sweep_results",
    "cached_results",
    "ablation_results",
    "results",
]

SCRIPT_FILES = {
    "test.py": "scripts/evaluate/test.py",
    "eval_cached_calibration.py": "scripts/evaluate/eval_cached_calibration.py",
    "run_param_sweep.py": "scripts/experiments/run_param_sweep.py",
    "run_ablation_experiments.py": "scripts/experiments/run_ablation_experiments.py",
    "cache_mvtec_features.py": "scripts/cache/cache_mvtec_features.py",
    "cache_multicrop_maps.py": "scripts/cache/cache_multicrop_maps.py",
    "train.py": "scripts/train.py",
    "test_one_example.py": "scripts/evaluate/test_one_example.py",
    "export_configs.py": "scripts/tools/export_configs.py",
}


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    return value


def _literal_node(node: ast.AST, default: Any = None, constants: Optional[Dict[str, Any]] = None) -> Any:
    if node is None:
        return default
    if constants and isinstance(node, ast.Name) and node.id in constants:
        return _yaml_safe(constants[node.id])
    try:
        return _yaml_safe(ast.literal_eval(node))
    except Exception:
        return default


def _script_tree(script: str) -> Optional[ast.Module]:
    script_path = SCRIPT_FILES.get(script)
    if not script_path:
        return None
    path = PROJECT_ROOT / script_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=str(path))


def _module_constant(script: str, name: str, default: Any = None) -> Any:
    return _module_constants(script).get(name, default)


def _module_constants(script: str) -> Dict[str, Any]:
    tree = _script_tree(script)
    if tree is None:
        return {}
    constants: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                value = _literal_node(node.value, constants=constants)
                if value is not None:
                    constants[target.id] = value
    return constants


def _keyword(call: ast.Call, name: str) -> Optional[ast.AST]:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _option_dest(options: List[str], call: ast.Call) -> Optional[str]:
    dest_node = _keyword(call, "dest")
    if dest_node is not None:
        return _literal_node(dest_node)
    long_options = [option for option in options if option.startswith("--")]
    if not long_options:
        return None
    return long_options[0].lstrip("-").replace("-", "_")


class _ArgumentCollector(ast.NodeVisitor):
    def __init__(self, constants: Optional[Dict[str, Any]] = None) -> None:
        self.specs: Dict[str, Dict[str, Any]] = {}
        self.constants = constants or {}

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            options = [
                _literal_node(arg, constants=self.constants)
                for arg in node.args
                if isinstance(_literal_node(arg, constants=self.constants), str)
            ]
            dest = _option_dest(options, node)
            if dest and dest not in {"help", "config"}:
                default_node = _keyword(node, "default")
                action = _literal_node(_keyword(node, "action"), constants=self.constants) if _keyword(node, "action") else None
                nargs = _literal_node(_keyword(node, "nargs"), constants=self.constants) if _keyword(node, "nargs") else None
                if default_node is not None:
                    default = _literal_node(default_node, constants=self.constants)
                elif action == "store_true":
                    default = False
                elif action == "store_false":
                    default = True
                else:
                    default = None
                self.specs[dest] = {
                    "default": default,
                    "action": action,
                    "nargs": nargs,
                }
        self.generic_visit(node)


def _parser_arg_specs(script: str) -> Dict[str, Dict[str, Any]]:
    tree = _script_tree(script)
    if tree is None:
        return {}
    constants = _module_constants(script)
    collector = _ArgumentCollector(constants=constants)
    collector.visit(tree)
    return collector.specs


def _parser_defaults(script: str) -> Dict[str, Any]:
    return {key: spec["default"] for key, spec in _parser_arg_specs(script).items()}


def _safe_name(path: str) -> str:
    stem = path.strip("./")
    stem = stem.replace(os.sep, "__")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    stem = stem.strip("._")
    return stem or "config"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"True", "False"}:
        return value == "True"
    if value in {"None", "null"}:
        return None
    if value.startswith("[") or value.startswith("{") or value.startswith("("):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, tuple):
                return list(parsed)
            return parsed
        except Exception:
            return value
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", value) or re.fullmatch(
            r"[-+]?\d+[eE][-+]?\d+",
            value,
        ):
            return float(value)
    except Exception:
        return value
    return value


def _parse_logged_args(text: str) -> Dict[str, Any]:
    match = re.search(r"\nargs:\n(?P<body>.*?)(?:\nextra:\n|\n===== end run context =====)", text, re.S)
    if not match:
        return {}
    args: Dict[str, Any] = {}
    for line in match.group("body").splitlines():
        if not line.startswith("  ") or ": " not in line:
            continue
        key, value = line.strip().split(": ", 1)
        args[key] = _parse_scalar(value)
    return args


def _parse_logged_command(text: str) -> Tuple[Optional[str], List[str]]:
    match = re.search(r"^command:\s+(.+)$", text, re.M)
    if not match:
        return None, []
    try:
        tokens = shlex.split(match.group(1))
    except ValueError:
        return None, []
    script_index = None
    for index, token in enumerate(tokens):
        if token.endswith(".py"):
            script_index = index
            break
    if script_index is None:
        return None, []
    return os.path.basename(tokens[script_index]), tokens[script_index + 1 :]


def _parse_args_from_command(script: str, command_args: List[str]) -> Dict[str, Any]:
    specs = _parser_arg_specs(script)
    if not specs:
        return {}
    args = _parser_defaults(script)
    index = 0
    while index < len(command_args):
        token = command_args[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        spec = specs.get(key)
        if spec is None:
            index += 1
            continue
        action = spec.get("action")
        nargs = spec.get("nargs")
        if action == "store_true":
            args[key] = True
            index += 1
            continue
        if action == "store_false":
            args[key] = False
            index += 1
            continue
        values: List[Any] = []
        index += 1
        while index < len(command_args) and not command_args[index].startswith("--"):
            values.append(_parse_scalar(command_args[index]))
            index += 1
            if nargs not in {"+", "*"}:
                break
        args[key] = values if nargs in {"+", "*"} else (values[0] if values else None)
    return args


def _find_logs(roots: Iterable[str]) -> List[str]:
    logs: List[str] = []
    for root in roots:
        if not os.path.exists(root):
            continue
        if os.path.isfile(root):
            if os.path.basename(root) == "log.txt":
                logs.append(root)
            continue
        for dirpath, _, filenames in os.walk(root):
            if "log.txt" in filenames:
                logs.append(os.path.join(dirpath, "log.txt"))
    return sorted(set(logs))


def _config_for_log(log_path: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    script, command_args = _parse_logged_command(text)
    logged_args = _parse_logged_args(text)

    args = dict(logged_args)
    command_parsed_args: Dict[str, Any] = {}
    if script:
        command_parsed_args = _parse_args_from_command(script, command_args)
        command_parsed_args.update(args)
        args = command_parsed_args

    run_dir = os.path.dirname(log_path)
    if not script:
        script = "unknown"
    if not args:
        return (
            run_dir,
            {
                "script": script,
                "source": {
                    "log_path": log_path,
                    "run_dir": run_dir,
                    "status": "no args block or parseable command found",
                },
                "args": {},
            },
        )

    args.pop("config", None)
    return (
        run_dir,
        {
            "script": script,
            "source": {
                "log_path": log_path,
                "run_dir": run_dir,
                "status": "generated from logged args",
            },
            "args": args,
        },
    )


def _write_default_config(script: str, output_path: str, extra: Optional[Dict[str, Any]] = None) -> str:
    defaults = _parser_defaults(script)
    if not defaults:
        raise ValueError(f"no parser available for {script}")
    data: Dict[str, Any] = {
        "script": SCRIPT_FILES.get(script, script),
        "args": defaults,
    }
    if extra:
        data.update(extra)
    write_yaml_config(output_path, data)
    return output_path


def write_default_configs(conf_dir: str) -> List[str]:
    paths = []
    paths.append(
        _write_default_config(
            "test.py",
            os.path.join(conf_dir, "test_conf.yaml"),
        )
    )
    paths.append(
        _write_default_config(
            "eval_cached_calibration.py",
            os.path.join(conf_dir, "eval_cached_calibration_conf.yaml"),
        )
    )
    paths.append(
        _write_default_config(
            "run_param_sweep.py",
            os.path.join(conf_dir, "run_param_sweep_conf.yaml"),
            {"parameter_sets": _module_constant("run_param_sweep.py", "PARAMETER_SETS", [])},
        )
    )
    paths.append(
        _write_default_config(
            "run_ablation_experiments.py",
            os.path.join(conf_dir, "run_ablation_experiments_conf.yaml"),
            {
                "datasets_config": _module_constant("run_ablation_experiments.py", "DATASETS", {}),
                "current_wavelet": _module_constant("run_ablation_experiments.py", "CURRENT_WAVELET", {}),
                "current_tta": _module_constant("run_ablation_experiments.py", "CURRENT_TTA", {}),
                "current_pixel_to_image": _module_constant("run_ablation_experiments.py", "CURRENT_P2I", {}),
                "current_multicrop": _module_constant("run_ablation_experiments.py", "CURRENT_MULTICROP", {}),
                "component_ablations": _module_constant("run_ablation_experiments.py", "COMPONENT_ABLATIONS", []),
                "internal_ablations": _module_constant("run_ablation_experiments.py", "INTERNAL_ABLATIONS", []),
            },
        )
    )
    for script in [
        "cache_mvtec_features.py",
        "cache_multicrop_maps.py",
        "train.py",
        "test_one_example.py",
        "export_configs.py",
    ]:
        paths.append(
            _write_default_config(
                script,
                os.path.join(conf_dir, f"{script[:-3]}_conf.yaml"),
            )
        )
    return paths


def write_result_configs(conf_dir: str, roots: Iterable[str]) -> List[str]:
    written = []
    for log_path in _find_logs(roots):
        item = _config_for_log(log_path)
        if item is None:
            continue
        run_dir, data = item
        output_path = os.path.join(conf_dir, "results", f"{_safe_name(run_dir)}_conf.yaml")
        write_yaml_config(output_path, data)
        written.append(output_path)
    return written


def build_parser():
    parser = argparse.ArgumentParser("Export YAML configs for AnomalyCLIP runs")
    parser.add_argument("--conf_dir", default="./conf")
    parser.add_argument("--results", nargs="*", default=RESULT_ROOTS)
    parser.add_argument("--skip_defaults", action="store_true")
    parser.add_argument("--skip_results", action="store_true")
    return parser


def parse_args():
    args, _ = parse_args_with_config(build_parser())
    return args


def main():
    args = parse_args()
    written = []
    if not args.skip_defaults:
        written.extend(write_default_configs(args.conf_dir))
    if not args.skip_results:
        written.extend(write_result_configs(args.conf_dir, args.results))
    print(f"written_configs: {len(written)}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
