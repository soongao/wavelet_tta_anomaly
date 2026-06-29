
import logging
import os
import platform
import shlex
import subprocess
import sys
from typing import Any, Dict, Optional


def _run_git_command(args):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_format_value(val)}" for key, val in value.items()) + "}"
    return str(value)


def _format_mapping(mapping: Dict[str, Any]) -> str:
    if not mapping:
        return "  none"
    return "\n".join(
        f"  {key}: {_format_value(value)}" for key, value in sorted(mapping.items())
    )


def get_logger(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log.txt')
    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                    datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.propagate = False
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def log_run_context(logger, args, title: str, extra_info: Optional[Dict[str, Any]] = None):
    """Write enough run metadata to reproduce an experiment later."""
    git_commit = _run_git_command(["rev-parse", "--short", "HEAD"]) or "unknown"
    git_status = _run_git_command(["status", "--short"]) or ""
    git_status_lines = git_status.splitlines()
    if len(git_status_lines) == 0:
        git_status_text = "clean"
    else:
        shown_lines = git_status_lines[:50]
        git_status_text = "\n".join(f"  {line}" for line in shown_lines)
        if len(git_status_lines) > len(shown_lines):
            git_status_text += f"\n  ... {len(git_status_lines) - len(shown_lines)} more files"

    try:
        import torch

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception:
        torch_info = {
            "torch_version": "unavailable",
            "cuda_available": "unknown",
            "cuda_device_count": "unknown",
        }

    command = shlex.join([sys.executable] + sys.argv)
    arg_info = vars(args) if hasattr(args, "__dict__") else {}
    extra_info = extra_info or {}

    logger.info(
        "\n===== %s =====\n"
        "command: %s\n"
        "cwd: %s\n"
        "python: %s\n"
        "platform: %s\n"
        "%s\n"
        "git_commit: %s\n"
        "git_status:\n%s\n"
        "args:\n%s\n"
        "extra:\n%s\n"
        "===== end run context =====",
        title,
        command,
        os.getcwd(),
        sys.version.replace("\n", " "),
        platform.platform(),
        _format_mapping(torch_info),
        git_commit,
        git_status_text,
        _format_mapping(arg_info),
        _format_mapping(extra_info),
    )
