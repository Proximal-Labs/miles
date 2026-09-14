"""Record the two-node Sokoban recipe and check its resource and batch contract."""

import shlex

from tests.fast.launch_scripts.py_harness import (
    format_recording,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot


def test_async_sokoban_launch(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "examples/experimental/nemo-gym/run_nemotron35_sokoban.py")
    module.execute(module.ScriptArgs())
    submitted = [command for command in recording.commands if "ray job submit" in command]
    assert len(submitted) == 1
    tokens = shlex.split(submitted[0])
    assert any(token.endswith("/train_async.py") for token in tokens)
    assert "--colocate" not in tokens and "--load" not in tokens
    assert "--fully-async" in tokens and "--use-rollout-logprobs" in tokens
    assert "--wandb-key" not in tokens and "frozen-wandb-api-key" not in submitted[0]
    expected = {
        "--actor-num-nodes": "1",
        "--actor-num-gpus-per-node": "8",
        "--rollout-num-gpus": "8",
        "--rollout-batch-size": "8",
        "--n-samples-per-prompt": "16",
        "--global-batch-size": "128",
        "--lr": "3e-07",
        "--num-rollout": "1000",
        "--save-interval": "200",
        "--rollout-max-response-len": "65536",
        "--sglang-context-length": "81920",
        "--mtp-loss-scaling-factor": "0",
    }
    for flag, value in expected.items():
        assert tokens[tokens.index(flag) + 1] == value
    snapshot = (
        REPO_ROOT
        / "tests/snapshots/launch_scripts/py/examples/experimental/nemo-gym/run_nemotron35_sokoban.py/execute.txt"
    )
    assert_matches_snapshot(snapshot, format_recording(recording, sandbox=tmp_path), "async_sokoban::execute")
