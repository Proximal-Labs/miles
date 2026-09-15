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
    recorded = "\n".join(line.rstrip() for line in format_recording(recording, sandbox=tmp_path).splitlines()) + "\n"
    assert_matches_snapshot(snapshot, recorded, "async_sokoban::execute")


def test_qwen_sokoban_preserves_training_settings(monkeypatch):
    freeze_environment(monkeypatch)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "examples/experimental/nemo-gym/run_nemotron35_sokoban.py")
    module.execute(module.ScriptArgs(model_family="qwen36", model_name="Qwen3.6-35B-A3B"))
    command = next(c for c in recording.commands if "ray job submit" in c)
    tokens = shlex.split(command)
    assert "--mtp-num-layers" not in tokens
    assert "--enable-mtp-training" not in tokens
    assert "--sglang-speculative-algorithm" not in tokens
    assert "--colocate" not in tokens and "--load" not in tokens
    assert "--use-rollout-routing-replay" in tokens
    assert "--fully-async" in tokens
    expected = {
        "--ref-load": "/root/models/Qwen3.6-35B-A3B_torch_dist",
        "--hf-checkpoint": "/root/models/Qwen3.6-35B-A3B",
        "--megatron-to-hf-mode": "raw",
        "--global-batch-size": "128",
        "--n-samples-per-prompt": "16",
        "--rollout-batch-size": "8",
        "--lr": "3e-07",
        "--rollout-max-response-len": "65536",
        "--sglang-context-length": "81920",
        "--num-rollout": "1000",
        "--save-interval": "200",
        "--tensor-model-parallel-size": "2",
        "--pipeline-model-parallel-size": "2",
        "--expert-model-parallel-size": "2",
        "--rollout-num-gpus-per-engine": "1",
        "--max-weight-staleness": "2",
        "--custom-rm-path": "sokoban_reward.reward_func",
        "--wandb-project": "nemotron35-sokoban",
    }
    for flag, value in expected.items():
        assert tokens[tokens.index(flag) + 1] == value
    assert "--no-load-optim" in tokens and "--no-load-rng" in tokens and "--finetune" in tokens
    assert "frozen-wandb-api-key" not in command
