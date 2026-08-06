import argparse
import os
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


CONFIG_ARG = "--config"


def load_yaml_config(config_path: Optional[str], required: bool = False) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        if required:
            raise FileNotFoundError(f"config file not found: {config_path}")
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {config_path}")
    return data


def write_yaml_config(config_path: str, data: Dict[str, Any]) -> None:
    directory = os.path.dirname(config_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            data,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def add_config_argument(
    parser: argparse.ArgumentParser,
    default: Optional[str] = None,
) -> argparse.ArgumentParser:
    if any(CONFIG_ARG in action.option_strings for action in parser._actions):
        return parser
    parser.add_argument(CONFIG_ARG, default=default, help="path to a YAML config file")
    return parser


def _parser_dests(parser: argparse.ArgumentParser) -> Iterable[str]:
    for action in parser._actions:
        if action.dest in ("help", "config"):
            continue
        yield action.dest


def _add_bool_disable_aliases(parser: argparse.ArgumentParser) -> None:
    existing_options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    for action in list(parser._actions):
        if action.dest in ("help", "config"):
            continue
        if not isinstance(action, argparse._StoreTrueAction):
            continue
        dashed = f"--no-{action.dest.replace('_', '-')}"
        underscored = f"--no_{action.dest}"
        aliases = [
            option
            for option in (dashed, underscored)
            if option not in existing_options
        ]
        if not aliases:
            continue
        parser.add_argument(
            *aliases,
            dest=action.dest,
            action="store_false",
            help=argparse.SUPPRESS,
        )
        existing_options.update(aliases)


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Iterable[str]] = None,
    default_config_path: Optional[str] = None,
    allow_unknown_config_keys: bool = False,
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    add_config_argument(parser, default=default_config_path)
    _add_bool_disable_aliases(parser)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(CONFIG_ARG, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = pre_args.config
    if config_path is None and default_config_path and os.path.exists(default_config_path):
        config_path = default_config_path

    config = load_yaml_config(config_path)
    config_args = config.get("args", {})
    if config_args is None:
        config_args = {}
    if not isinstance(config_args, dict):
        raise ValueError("'args' in config must be a mapping")

    if not allow_unknown_config_keys:
        known_dests = set(_parser_dests(parser))
        unknown = sorted(set(config_args) - known_dests)
        if unknown:
            raise ValueError(
                "unknown config args for this parser: " + ", ".join(unknown)
            )

    if config_args:
        parser.set_defaults(**deepcopy(config_args))
    parser.set_defaults(config=config_path)
    args = parser.parse_args(argv)
    return args, config


def parser_default_args(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for action in parser._actions:
        if action.dest in ("help", "config"):
            continue
        if action.default is argparse.SUPPRESS:
            continue
        defaults[action.dest] = deepcopy(action.default)
    return defaults
