import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.run_inkling_small_sft import ScriptArgs, execute


@pytest.mark.parametrize("marker,expected_calls", [(b"release\n", 0), (b"", 1), (b"123", 1), (None, 1)])
def test_prepare_checks_volume_before_gpu_submission(monkeypatch, marker, expected_calls):
    pytest.importorskip("modal")
    import tools.modal_inkling_sft as launcher

    events = []

    def read_file(path):
        events.append(("read", path))
        if marker is None:
            raise FileNotFoundError(path)
        yield marker

    monkeypatch.setattr(launcher, "volume", SimpleNamespace(read_file=read_file))
    monkeypatch.setattr(launcher, "train", SimpleNamespace(remote=lambda config: events.append(("gpu", config))))
    launcher.main(json.dumps({"mode": "prepare", "model_dir": "/mnt/inkling/custom-models"}))
    assert events[0] == ("read", "custom-models/Inkling-Small_torch_dist/latest_checkpointed_iteration.txt")
    assert len(events) == 1 + expected_calls


def test_prepare_volume_error_does_not_allocate_gpus(monkeypatch):
    pytest.importorskip("modal")
    import tools.modal_inkling_sft as launcher

    def read_file(path):
        raise ConnectionError("Volume unavailable")

    monkeypatch.setattr(launcher, "volume", SimpleNamespace(read_file=read_file))
    monkeypatch.setattr(
        launcher, "train", SimpleNamespace(remote=lambda config: pytest.fail("Unexpected GPU allocation"))
    )
    with pytest.raises(ConnectionError, match="Volume unavailable"):
        launcher.main('{"mode":"prepare"}')


@pytest.mark.parametrize("cuda_version", ["13.0", "13.1", "12.9", None])
def test_b300_preflight_allows_cuda_13_experiment(monkeypatch, capsys, cuda_version):
    pytest.importorskip("modal")
    from tools.modal_inkling_sft import _DEFAULT_IMAGE, _gpu_preflight

    assert ScriptArgs().image == _DEFAULT_IMAGE
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(
            device_count=lambda: 8,
            get_device_properties=lambda index: SimpleNamespace(name="NVIDIA B300", total_memory=288 * 2**30),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    if cuda_version in (None, "12.9"):
        with pytest.raises(RuntimeError, match="requires torch CUDA >=13.0"):
            _gpu_preflight()
    else:
        _gpu_preflight()
        assert ("EXPERIMENTAL: CUDA 13.0" in capsys.readouterr().out) == (cuda_version == "13.0")


def test_lora_resume_keeps_base_checkpoint_and_restores_adapter_optimizer(monkeypatch):
    import miles.utils.external_utils.command_utils as U

    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(U, "get_default_wandb_args", lambda *args, **kwargs: "")
    adapter = "/mnt/inkling/checkpoints/test/iter_0000010/adapter"
    execute(resume=True, lora_adapter_path=adapter, run_id="test")
    command = calls[0]["train_args"]
    assert "--load /mnt/inkling/models/Inkling-Small_torch_dist " in command
    assert f"--lora-adapter-path {adapter} " in command
    assert "--lora-rank 32 --lora-alpha 32" in command
    assert "--target-modules all-linear --experts-shared-outer-loras" in command
    assert "--no-load-optim" not in command
    assert "--finetune" not in command
    assert "--start-rollout-id" not in command
    assert "offload" not in command
    assert "--optimizer adam " in command
    assert "--optimizer muon" not in command
    assert "--optimizer dist_muon" not in command
    assert "--data-source-path miles.rollout.inkling_sft_data_source.InklingSFTDataSource" in command
    assert calls[0]["train_script"] == "train.py"


def test_fresh_sft_starts_at_first_rollout(monkeypatch):
    import miles.utils.external_utils.command_utils as U

    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(U, "get_default_wandb_args", lambda *args, **kwargs: "")
    execute(run_id="fresh")
    assert "--start-rollout-id 0 " in calls[0]["train_args"]
    assert "--distributed-timeout-minutes 30 " in calls[0]["train_args"]
    assert "--num-epoch 10 " in calls[0]["train_args"]
    assert "--lr 1e-05 --min-lr 1e-06 " in calls[0]["train_args"]
    assert "--lr-decay-style cosine --lr-warmup-init 0 --lr-warmup-fraction 0.01 " in calls[0]["train_args"]
    assert "--rollout-batch-size 32 --global-batch-size 32 " in calls[0]["train_args"]
    assert "--micro-batch-size 1 " in calls[0]["train_args"]
    execute(run_id="longer-warmup", distributed_timeout_minutes=40, num_epoch=2, global_batch_size=4)
    assert "--distributed-timeout-minutes 40 " in calls[1]["train_args"]
    assert "--num-epoch 2 " in calls[1]["train_args"]
    assert "--rollout-batch-size 4 --global-batch-size 4 " in calls[1]["train_args"]
    assert "--lr-warmup-fraction 0.05 " in calls[1]["train_args"]


def test_lora_rank_cannot_silently_disable_adapters():
    with pytest.raises(ValueError, match="LoRA rank"):
        ScriptArgs(lora_rank=0)


@pytest.mark.parametrize(
    "mode,period,enabled", [("train", 1, True), ("train", 2, True), ("train", 0, False), ("smoke", 1, False)]
)
def test_environment_evaluation_flags(monkeypatch, mode, period, enabled):
    import miles.utils.external_utils.command_utils as U

    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(U, "get_default_wandb_args", lambda *args, **kwargs: "")
    execute(mode=mode, eval_config="/eval.json", eval_every_n_epochs=period)
    argv = calls[0]["train_args"]
    assert ("--inkling-eval-config /eval.json" in argv) == enabled
    assert (f"--inkling-eval-every-n-epochs {period}" in argv) == enabled


def test_disabled_evaluation_does_not_read_config_or_attach_secret(monkeypatch):
    from scripts.run_inkling_small_sft import launch

    import miles.utils.external_utils.command_utils as U

    commands = []
    monkeypatch.setattr(U, "exec_command_cpu", commands.append)
    launch(mode="train", eval_config="/does-not-exist.json", eval_every_n_epochs=0)
    assert "INKLING_EVAL_SECRET" not in commands[0]
    assert "_eval_config" not in commands[0]


def test_negative_evaluation_interval_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        ScriptArgs(eval_every_n_epochs=-1)


def test_eval_rollouts_per_env_override(monkeypatch):
    import miles.utils.external_utils.command_utils as U

    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(U, "get_default_wandb_args", lambda *args, **kwargs: "")
    execute(mode="train", eval_config="/eval.json", eval_rollouts_per_env=3)
    assert "--inkling-eval-rollouts-per-env 3 " in calls[0]["train_args"]
    execute(mode="train", eval_config="/eval.json", eval_rollouts_per_env=3, eval_every_n_epochs=0)
    assert "--inkling-eval-" not in calls[1]["train_args"]
    with pytest.raises(ValueError, match="eval_rollouts_per_env"):
        ScriptArgs(eval_rollouts_per_env=0)


@pytest.mark.parametrize("num_nodes", [1, 2])
def test_modal_resume_skips_incomplete_newer_adapter(tmp_path, num_nodes):
    pytest.importorskip("modal")
    from tools.modal_inkling_sft import _resume_adapter

    complete = tmp_path / "iter_0000010" / "adapter"
    incomplete = tmp_path / "iter_0000020" / "adapter"
    for directory in (complete, incomplete):
        directory.mkdir(parents=True)
        for rank in range(8 * num_nodes):
            (directory / f"adapter_megatron_rank{rank}.pt").write_bytes(b"shard")
            if directory == complete or rank < 8 * num_nodes - 1:
                (directory / f"training_state_rank{rank}.pt").write_bytes(b"state")
    args = SimpleNamespace(lora_adapter_path=None, save_dir=str(tmp_path), num_nodes=num_nodes, num_gpus_per_node=8)
    (tmp_path / "rollout").mkdir()
    (tmp_path / "rollout/global_dataset_state_dict_10.pt").write_bytes(b"cursor")
    assert _resume_adapter(args) == str(complete)
    args.lora_adapter_path = str(incomplete)
    with pytest.raises(FileNotFoundError, match="No complete native LoRA"):
        _resume_adapter(args)


@pytest.mark.parametrize(
    "overrides,dp",
    [
        ({}, 2),
        ({"pipeline_model_parallel_size": 4, "decoder_last_pipeline_num_layers": 12}, 1),
        ({"tensor_model_parallel_size": 8}, 1),
    ],
)
def test_two_node_parallelism_reaches_miles(monkeypatch, overrides, dp):
    import miles.utils.external_utils.command_utils as U

    calls = []
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(U, "get_default_wandb_args", lambda *args, **kwargs: "")
    execute(num_nodes=2, **overrides)
    args = calls[0]["config"]
    command = calls[0]["train_args"]
    assert args.data_parallel_size == dp
    assert "--actor-num-nodes 2 " in command
    assert f"--tensor-model-parallel-size {args.tensor_model_parallel_size} " in command
    assert f"--pipeline-model-parallel-size {args.pipeline_model_parallel_size} " in command
    assert "--expert-model-parallel-size 4 " in command
    if args.decoder_last_pipeline_num_layers is not None:
        assert "--decoder-last-pipeline-num-layers 12 " in command


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_nodes": 3},
        {"pipeline_model_parallel_size": 0},
        {"tensor_model_parallel_size": 3},
        {"expert_model_parallel_size": 16},
        {"pipeline_model_parallel_size": 4},
        {"decoder_last_pipeline_num_layers": 43},
        {"decoder_first_pipeline_num_layers": 0},
        {"num_nodes": 2, "global_batch_size": 3},
    ],
)
def test_invalid_topology_rejected_before_launch(overrides):
    with pytest.raises(ValueError):
        ScriptArgs(**overrides)


@pytest.mark.parametrize("mode,expected", [("prepare", "single"), ("smoke", "cluster"), ("train", "cluster")])
def test_two_nodes_only_allocated_for_training(monkeypatch, mode, expected):
    pytest.importorskip("modal")
    import tools.modal_inkling_sft as launcher

    calls = []
    monkeypatch.setattr(launcher, "_prepared_checkpoint_cached", lambda config: False)
    monkeypatch.setattr(launcher, "train", SimpleNamespace(remote=lambda config: calls.append("single")))
    monkeypatch.setattr(launcher, "run_cluster", SimpleNamespace(remote=lambda config: calls.append("cluster")))
    launcher.main(json.dumps({"mode": mode, "num_nodes": 2}))
    assert calls == [expected]


@pytest.mark.parametrize("failure", [False, True])
def test_remote_coordinator_releases_both_containers(monkeypatch, failure):
    pytest.importorskip("modal")
    import tools.modal_inkling_sft as launcher

    state, events = {}, []

    def get(**kwargs):
        events.append(("get", kwargs))
        if failure:
            raise RuntimeError("cancelled")

    call = SimpleNamespace(get=get, cancel=lambda **kwargs: events.append(("cancel", kwargs)))
    monkeypatch.setattr(launcher.modal.Dict, "ephemeral", lambda: nullcontext(state))
    monkeypatch.setattr(launcher, "train_two_nodes", SimpleNamespace(spawn=lambda *args: call))
    if failure:
        with pytest.raises(RuntimeError, match="cancelled"):
            launcher.run_cluster.local("{}")
        assert state["error"]
        assert events[-2] == ("get", {"timeout": 180})
    else:
        launcher.run_cluster.local("{}")
    assert events[-1] == ("cancel", {"terminate_containers": True})


@pytest.mark.parametrize(
    "changed",
    [
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "decoder_last_pipeline_num_layers",
    ],
)
def test_resume_rejects_changed_parallelism(monkeypatch, tmp_path, changed):
    pytest.importorskip("modal")
    import tools.modal_inkling_sft as launcher

    args = ScriptArgs(
        num_nodes=2,
        run_id="resume",
        mode="train",
        output_dir=str(tmp_path),
        data_dir=str(tmp_path),
        model_dir=str(tmp_path),
        resume=True,
    )
    destination = tmp_path / args.run_id
    destination.mkdir()
    saved = asdict(args)
    saved[changed] = 12 if changed == "decoder_last_pipeline_num_layers" else 8
    (destination / "launch.json").write_text(json.dumps(saved))
    (tmp_path / "train.prepared.jsonl").touch()
    Path(args.torch_dist).mkdir()
    (Path(args.torch_dist) / "latest_checkpointed_iteration.txt").touch()
    monkeypatch.setattr(launcher, "_gpu_preflight", lambda: None)
    with pytest.raises(ValueError, match=f"changed {changed}"):
        launcher.train.local(json.dumps(asdict(args)))
