"""Fresh-base serving bootstrap and detached training preparation, all on CPU."""

from argparse import Namespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from tests.integration.proximal_async.test_offline_batch import populate

from miles_plugins.proximal.initial_policy import prepare_base_policy, verify_base_policy
from miles_plugins.proximal.offline_batch import freeze_batch, train_argv, validate_train_args
from miles_plugins.proximal.snapshot import prepare_snapshot


def base_config(config, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    tensors = {
        f"model.layers.0.{parent}.{name}.weight": torch.ones((6, 4), dtype=torch.bfloat16)
        for parent, names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for name in names
    }
    save_file(tensors, str(base / "model.safetensors"))
    return config.model_copy(update={"tokenizer_path": base})


async def test_fresh_batch_needs_no_native_checkpoint_and_keeps_zero_policy_proof(config, policy, attempt, tmp_path):
    config = base_config(config, tmp_path)
    snapshot = prepare_base_policy(config, output=tmp_path / "initial")
    policy = policy.model_copy(update={"snapshot": snapshot.reference})
    assert verify_base_policy(config, policy, snapshot.directory) == snapshot
    tensors = load_file(str(snapshot.directory / "adapter_model.safetensors"))
    assert len(tensors) == 14
    assert all(torch.count_nonzero(tensor) == 0 for tensor in tensors.values())
    await populate(config, policy, attempt, 2)
    bundle = tmp_path / "batch"
    batch = freeze_batch(
        config=config,
        source_root=config.artifact_directory / config.run_id,
        group_ids=("g0", "g1"),
        policy=policy,
        num_samples=4,
        out=bundle,
        base_policy=snapshot.directory,
    )
    command = train_argv(bundle, batch, None, fresh=True)
    assert "--proximal-frozen-fresh" in command
    assert "--lora-adapter-path" not in command and "--proximal-frozen-checkpoint" not in command
    assert command[command.index("--num-rollout") + 1] == "1"
    assert command[command.index("--start-rollout-id") + 1] == "0"
    args = Namespace(
        debug_train_only=True,
        rollout_global_dataset=False,
        rollout_function_path="miles_plugins.proximal.offline_batch.FrozenBatchRolloutFn",
        proximal_frozen_batch=bundle,
        proximal_frozen_fresh=True,
        num_rollout=1,
        start_rollout_id=0,
        global_batch_size=4,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        use_rollout_logprobs=True,
        use_tis=False,
        load=str(config.tokenizer_path),
        hf_checkpoint=str(config.tokenizer_path),
        data_source_path="miles.rollout.data_source.RolloutDataSourceWithBuffer",
        train_backend="megatron",
        lora_rank=config.research.lora.rank,
        lora_alpha=config.research.lora.alpha,
        target_modules=list(config.research.lora.target_modules),
        lora_dropout=0,
        lora_adapter_path=None,
        lora_A_init_method="xavier",
        lora_B_init_method="zero",
        save=str(tmp_path / "new-checkpoint"),
        save_interval=1,
    )
    validate_train_args(args, batch, None)
    args.lora_adapter_path = str(snapshot.directory)
    with pytest.raises(ValueError, match="never loads the serving adapter"):
        validate_train_args(args, batch, None)
    with pytest.raises(ValueError, match="Choose exactly one"):
        train_argv(bundle, batch, None)


def test_nonzero_or_different_policy_cannot_be_used_for_fresh_training(config, policy, tmp_path):
    config = base_config(config, tmp_path)
    snapshot = prepare_base_policy(config, output=tmp_path / "initial")
    adapter = tmp_path / "initial/zero-adapter"
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    next(tensor for name, tensor in tensors.items() if ".lora_B." in name).fill_(0.01)
    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    changed = prepare_snapshot(adapter, metadata=snapshot.manifest.metadata, output_root=tmp_path / "changed")
    policy = policy.model_copy(update={"snapshot": changed.reference})
    with pytest.raises(ValueError, match="zero-delta"):
        verify_base_policy(config, policy, changed.directory)
    with pytest.raises(ValueError, match="initial base policy"):
        verify_base_policy(config, policy.model_copy(update={"version": 2}), changed.directory)


def test_missing_or_quantized_targets_fail_before_publication(config, tmp_path):
    config = base_config(config, tmp_path)
    save_file(
        {"model.layers.0.mlp.gate_proj.weight": torch.ones((6, 4), dtype=torch.int8)},
        str(config.tokenizer_path / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="unquantized"):
        prepare_base_policy(config, output=tmp_path / "initial")
    assert not (tmp_path / "initial").exists()
