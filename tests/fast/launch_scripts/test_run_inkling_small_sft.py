from types import SimpleNamespace
import sys
import json

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
    monkeypatch.setattr(launcher, "train", SimpleNamespace(remote=lambda config: pytest.fail("Unexpected GPU allocation")))
    with pytest.raises(ConnectionError, match="Volume unavailable"):
        launcher.main('{"mode":"prepare"}')


@pytest.mark.parametrize("cuda_version", ["13.0", "13.1", "12.9", None])
def test_b300_preflight_allows_cuda_13_experiment(monkeypatch, capsys, cuda_version):
    pytest.importorskip("modal")
    from tools.modal_inkling_sft import _gpu_preflight, _DEFAULT_IMAGE

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
    execute(run_id="longer-warmup", distributed_timeout_minutes=40)
    assert "--distributed-timeout-minutes 40 " in calls[1]["train_args"]


def test_lora_rank_cannot_silently_disable_adapters():
    with pytest.raises(ValueError, match="LoRA rank"):
        ScriptArgs(lora_rank=0)


def test_modal_resume_skips_incomplete_newer_adapter(tmp_path):
    pytest.importorskip("modal")
    from tools.modal_inkling_sft import _resume_adapter

    complete = tmp_path / "iter_0000010" / "adapter"
    incomplete = tmp_path / "iter_0000020" / "adapter"
    for directory in (complete, incomplete):
        directory.mkdir(parents=True)
        for rank in range(8):
            (directory / f"adapter_megatron_rank{rank}.pt").write_bytes(b"shard")
            if directory == complete or rank < 7:
                (directory / f"training_state_rank{rank}.pt").write_bytes(b"state")
    args = SimpleNamespace(lora_adapter_path=None, save_dir=str(tmp_path), num_gpus_per_node=8)
    (tmp_path / "rollout").mkdir()
    (tmp_path / "rollout/global_dataset_state_dict_10.pt").write_bytes(b"cursor")
    assert _resume_adapter(args) == str(complete)
    args.lora_adapter_path = str(incomplete)
    with pytest.raises(FileNotFoundError, match="No complete native LoRA"):
        _resume_adapter(args)
