import argparse
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from tools.argument_snapshot.schema import snapshot_parser


@dataclass(frozen=True)
class _Scenario:
    backend: str
    arguments: tuple[str, ...] = ()
    legacy: bool = False
    custom: bool = False


def capture_scenarios(selected: list[str] | None = None) -> dict[str, Any]:
    scenarios = {
        "megatron": _Scenario(backend="megatron"),
        "fsdp": _Scenario(backend="fsdp"),
        "backend_probe": _Scenario(backend="probe"),
        "fully_async": _Scenario(backend="megatron", arguments=("--fully-async",)),
        "legacy": _Scenario(backend="megatron", legacy=True),
        "custom_megatron": _Scenario(backend="megatron", custom=True),
        "custom_fsdp": _Scenario(backend="fsdp", custom=True),
    }
    for name, flag in {
        "rollout": "--rollout-function-path",
        "generate": "--custom-generate-function-path",
        "inference": "--custom-inference-engine-provider-path",
    }.items():
        scenarios[f"hook_{name}"] = _Scenario(
            backend="megatron", arguments=(flag, "tools.argument_snapshot.scenarios._Hook")
        )
        scenarios[f"legacy_hook_{name}"] = _Scenario(
            backend="megatron", arguments=(flag, "tools.argument_snapshot.scenarios._Hook"), legacy=True
        )
    scenarios["missing_hook_module"] = _Scenario(
        backend="megatron", arguments=("--custom-generate-function-path", "miles_snapshot_missing.module")
    )
    scenarios["hook_without_arguments"] = _Scenario(
        backend="megatron", arguments=("--custom-generate-function-path", "builtins.str")
    )
    scenarios["invalid_hook_path"] = _Scenario(
        backend="megatron", arguments=("--custom-generate-function-path", "invalid_snapshot_path")
    )
    scenarios["hook_function"] = _Scenario(
        backend="megatron",
        arguments=("--custom-generate-function-path", "tools.argument_snapshot.scenarios._hook_function"),
    )
    scenarios["hook_fsdp"] = _Scenario(
        backend="fsdp",
        arguments=("--custom-inference-engine-provider-path", "tools.argument_snapshot.scenarios._Hook"),
    )
    scenarios["megatron_repeat"] = _Scenario(backend="megatron")
    names = list(scenarios) if selected is None else selected
    if unknown := set(names) - scenarios.keys():
        raise ValueError(f"Unknown snapshot scenarios: {sorted(unknown)}; available: {list(scenarios)}")
    return {name: _capture_scenario(scenarios[name]) for name in names}


def _capture_scenario(scenario: _Scenario) -> dict[str, Any]:
    arguments = ["--rollout-batch-size", "2", "--train-backend", "fsdp" if scenario.backend == "fsdp" else "megatron"]
    arguments.extend(scenario.arguments)
    with _environment(arguments=arguments, legacy=scenario.legacy):
        parser = _build_parser(scenario)
        parsed = {"minimal": vars(parser.parse_args(arguments))}
        variants = {
            "lora_disabled": ["--no-sglang-lora-use-virtual-experts"],
            "sglang_alias": ["--sglang-tp-size", "2"],
            "eval_true": ["--eval-sglang-enable-metrics"],
            "eval_false": ["--no-eval-sglang-enable-metrics"],
        }
        for name, extra in variants.items():
            parsed[name] = vars(parser.parse_args(arguments + extra))
        return {"argv": arguments, "legacy": scenario.legacy, "schema": snapshot_parser(parser), "parsed": parsed}


def _build_parser(scenario: _Scenario) -> argparse.ArgumentParser:
    from miles.utils.arguments import get_miles_extra_args_provider

    provider = get_miles_extra_args_provider(_custom_arguments if scenario.custom else None)
    if scenario.backend == "probe":
        return provider(argparse.ArgumentParser())
    if scenario.backend == "fsdp":
        from miles.backends.fsdp_utils.arguments import build_fsdp_parser

        return build_fsdp_parser(extra_args_provider=provider)
    return _capture_megatron(provider)


def _capture_megatron(
    provider: Callable[[argparse.ArgumentParser], argparse.ArgumentParser]
) -> argparse.ArgumentParser:
    from miles.backends.megatron_utils.arguments import parse_args

    original = argparse.ArgumentParser.parse_known_args
    completed_parser: argparse.ArgumentParser | None = None

    def _provide(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        nonlocal completed_parser
        completed_parser = provider(parser)
        return completed_parser

    def _intercept(
        parser: argparse.ArgumentParser,
        args: list[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        if parser is completed_parser:
            raise _ParserCaptured(parser)
        return original(parser, args=args, namespace=namespace)

    argparse.ArgumentParser.parse_known_args = _intercept
    try:
        parse_args(extra_args_provider=_provide)
    except _ParserCaptured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_known_args = original
    raise RuntimeError("Megatron did not parse the completed provider parser")


class _ParserCaptured(Exception):
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        super().__init__("Captured completed Megatron parser at parsing boundary")
        self.parser = parser


def _custom_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--snapshot-custom", type=int, default=17)
    for action in parser._actions:
        if "--padded-vocab-size" in action.option_strings:
            action.default = 1024
            break
    else:
        parser.add_argument("--padded-vocab-size", type=int, default=1024)
    return parser


class _Hook:
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--snapshot-hook", type=int, default=23)


def _hook_function() -> None:
    pass


_hook_function.add_arguments = _Hook.add_arguments


@contextmanager
def _environment(*, arguments: list[str], legacy: bool) -> Iterator[None]:
    original_argv = sys.argv
    values = {
        "MILES_USE_LEGACY_ROLLOUT_V1": str(int(legacy)),
        "PROMETHEUS_PORT": "9090",
        "MILES_SCRIPT_ENV_REPORT": "",
    }
    original_environment = {name: os.environ.get(name) for name in values}
    sys.argv = ["argument-snapshot", *arguments]
    os.environ.update(values)
    try:
        yield
    finally:
        sys.argv = original_argv
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
