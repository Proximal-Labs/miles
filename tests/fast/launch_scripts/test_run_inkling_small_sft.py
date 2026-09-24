from types import SimpleNamespace

import pytest

from scripts.run_inkling_small_sft import ScriptArgs, execute


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
    assert "offload" not in command
    assert "--optimizer dist_muon" in command
    assert "--data-source-path miles.rollout.inkling_sft_data_source.InklingSFTDataSource" in command
    assert calls[0]["train_script"] == "train.py"


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
