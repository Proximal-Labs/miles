import argparse
import enum
import functools
import inspect
from pathlib import Path
from typing import Any

import yaml


def snapshot_parser(parser: argparse.ArgumentParser) -> dict[str, Any]:
    action_indexes = {id(action): index for index, action in enumerate(parser._actions)}
    destinations = dict.fromkeys([action.dest for action in parser._actions] + list(parser._defaults))
    return {
        "parser": _encode(
            {
                "description": parser.description,
                "epilog": parser.epilog,
                "allow_abbrev": parser.allow_abbrev,
                "prefix_chars": parser.prefix_chars,
                "fromfile_prefix_chars": parser.fromfile_prefix_chars,
                "argument_default": parser.argument_default,
                "conflict_handler": parser.conflict_handler,
                "exit_on_error": parser.exit_on_error,
            }
        ),
        "actions": [
            {"class": _qualified_name(type(action)), "attributes": _encode(vars(action) | {"container": None})}
            for action in parser._actions
        ],
        "defaults": _encode(parser._defaults),
        "effective_defaults": _encode({dest: parser.get_default(dest) for dest in destinations}),
        "groups": [
            {
                "title": group.title,
                "description": group.description,
                "actions": [action_indexes[id(action)] for action in group._group_actions],
            }
            for group in parser._action_groups
        ],
        "mutually_exclusive_groups": [
            {
                "required": group.required,
                "actions": [action_indexes[id(action)] for action in group._group_actions],
            }
            for group in parser._mutually_exclusive_groups
        ],
    }


def dump_snapshot(value: Any) -> str:
    return yaml.safe_dump(_encode(value), sort_keys=True, allow_unicode=True, width=120)


def _encode(value: Any) -> Any:
    if isinstance(value, str) and value == argparse.SUPPRESS:
        return {"$argparse": "SUPPRESS"}
    if isinstance(value, enum.Enum):
        return {"$enum": _qualified_name(type(value)), "name": value.name}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Path):
        return {"$path": str(value)}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, tuple):
        return {"$tuple": [_encode(item) for item in value]}
    if isinstance(value, range):
        return {"$range": [value.start, value.stop, value.step]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"Snapshot mappings require string keys: {type(value)}")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, functools.partial):
        return {"$partial": _encode(value.func), "args": _encode(value.args), "keywords": _encode(value.keywords)}
    if isinstance(value, type) or inspect.isfunction(value) or inspect.isbuiltin(value):
        return {"$callable": _qualified_name(value)}
    raise TypeError(f"Unsupported snapshot value type: {_qualified_name(type(value))}")


def _qualified_name(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"
