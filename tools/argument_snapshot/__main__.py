import argparse
import difflib
import importlib
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Annotated, Any

import typer
from tools.argument_snapshot.schema import dump_snapshot, snapshot_parser

_app = typer.Typer()


@_app.command("generate")
def _generate(
    factory: Annotated[str, typer.Option()],
    scenario: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    sys.stdout.write(_capture(factory=factory, scenarios=scenario))


@_app.command("compare")
def _compare(
    factory: Annotated[str, typer.Option()],
    baseline: Annotated[Path, typer.Option()],
    scenario: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    expected = baseline.read_text()
    actual = _capture(factory=factory, scenarios=scenario)
    if actual != expected:
        sys.stdout.writelines(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=str(baseline),
                tofile="current",
            )
        )
        raise typer.Exit(code=1)


def _capture(*, factory: str, scenarios: list[str] | None) -> str:
    module_name, separator, attribute = factory.partition(":")
    if not separator:
        raise typer.BadParameter("Factory must be module:callable")
    with redirect_stdout(sys.stderr):
        module = importlib.import_module(module_name)
        capture = vars(module)[attribute]
        value: Any = capture() if scenarios is None else capture(selected=scenarios)
        if isinstance(value, argparse.ArgumentParser):
            value = snapshot_parser(value)
        return dump_snapshot(value)


if __name__ == "__main__":
    _app()
