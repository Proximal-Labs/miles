"""The public persistence flag routes to CPU collection, with validation before launch."""

import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.proximal_publication.test_training_deployment import _run

from miles_plugins.proximal.collect_batch import validate_collection_request


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    run = _run()
    path = tmp_path / "run.json"
    path.write_text(run.model_dump_json())
    monkeypatch.setenv("PROXIMAL_RUN_CONFIG", str(path))
    monkeypatch.setenv("PROXIMAL_SERVING_CONFIG", str(root / "examples/proximal/e2e/serving.stage-a.json"))
    monkeypatch.setenv("PROXIMAL_TRAINING_CONFIG", str(root / "examples/proximal/gsm8k/training.json"))
    for name in ("miles_plugins.proximal.modal_training", "miles_plugins.proximal.serving_app"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("miles_plugins.proximal.modal_training")
    calls = []
    monkeypatch.setattr(module, "collect", SimpleNamespace(remote=lambda *args: calls.append(args) or "collection"))
    monkeypatch.setattr(
        module, "train", SimpleNamespace(remote=lambda: pytest.fail("Collection must not start training GPUs"))
    )
    yield module, calls
    for name in ("miles_plugins.proximal.modal_training", "miles_plugins.proximal.serving_app"):
        sys.modules.pop(name, None)


def test_flag_starts_cpu_collection_and_never_trainer(launcher):
    module, calls = launcher
    module.main(
        collect_rollouts=1024, rollouts_persist_to_volume=True, fresh=True, yes_rollouts=True, yes_publish=True
    )
    assert calls == [(1024, True, "", True, True, True)]
    tree = ast.parse(Path(module.__file__).read_text())
    collect = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "collect")
    decorator = collect.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    assert "gpu" not in {keyword.arg for keyword in decorator.keywords}


def test_modal_cli_exposes_the_requested_persistence_flag(launcher):
    from click.testing import CliRunner
    from modal.cli.entry_point import entrypoint_cli

    _, calls = launcher
    result = CliRunner().invoke(entrypoint_cli, ["run", "-m", "miles_plugins.proximal.modal_training", "--help"])
    assert result.exit_code == 0, result.output
    assert "--rollouts-persist-to-volume" in result.output
    assert "--collect-rollouts" in result.output
    assert not calls


@pytest.mark.parametrize(
    "change",
    [
        {"collect_rollouts": 0},
        {"collect_rollouts": -1},
        {"collect_rollouts": 1023},
        {"rollouts_persist_to_volume": False},
        {"yes_rollouts": False},
        {"yes_publish": False},
        {"fresh": False},
    ],
)
def test_invalid_collection_never_allocates_any_remote_worker(launcher, change):
    module, calls = launcher
    args = dict(
        collect_rollouts=1024, rollouts_persist_to_volume=True, fresh=True, yes_rollouts=True, yes_publish=True
    )
    with pytest.raises(ValueError):
        module.main(**(args | change))
    assert not calls


def test_existing_policy_is_explicit_and_cannot_change_run(config, policy):
    assert (
        validate_collection_request(
            config, samples=1024, fresh=False, policy_json=policy.model_dump_json(), persist_to_volume=True
        )
        == policy
    )
    changed = policy.model_copy(update={"run_id": "another-run"})
    with pytest.raises(ValueError, match="another run/base"):
        validate_collection_request(
            config, samples=1024, fresh=False, policy_json=changed.model_dump_json(), persist_to_volume=True
        )
