"""A publishing Volume reload cannot invalidate the trainer's staged batch."""

import hashlib
from argparse import Namespace

import pytest
from tests.integration.proximal_async.test_initial_policy import base_config
from tests.integration.proximal_async.test_offline_batch import populate
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles_plugins.proximal.e2e.batch_sweep_inputs import SweepPhase, SweepPlan
from miles_plugins.proximal.e2e.batch_sweep_recovery import stage_batch
from miles_plugins.proximal.e2e.state_gpu_replay import ReferenceReplay
from miles_plugins.proximal.initial_policy import prepare_base_policy
from miles_plugins.proximal.offline_batch import freeze_batch


async def test_both_replay_updates_survive_disappearing_volume(config, policy, attempt, tmp_path):
    config = base_config(config, tmp_path)
    snapshot = prepare_base_policy(config, output=tmp_path / "initial")
    policy = policy.model_copy(update={"snapshot": snapshot.reference})
    await populate(config, policy, attempt, 2)
    volume = tmp_path / "volume-batch"
    freeze_batch(
        config=config,
        source_root=config.artifact_directory / config.run_id,
        group_ids=("g0", "g1"),
        policy=policy,
        num_samples=4,
        out=volume,
        base_policy=snapshot.directory,
    )
    sha = hashlib.sha256((volume / "batch.json").read_bytes()).hexdigest()
    plan = SweepPlan(
        experiment_id="staging",
        batch_path="batch",
        batch_sha256=sha,
        samples=4,
        nodes=1,
        phases=(SweepPhase(name="mlp", updates=2, target_modules=("linear_fc1",), resume=None),),
        recipe=("--optimizer", "adam", "--lr", "4e-5", "--seed", "42"),
    )
    local = tmp_path / "local-batch"
    stage_batch(volume, local, plan)
    volume.rename(tmp_path / "unmounted")
    assert hashlib.sha256((local / "batch.json").read_bytes()).hexdigest() == sha
    args = Namespace(
        verification_batch=local, debug_train_only=True, num_rollout=2, start_rollout_id=0, lora_adapter_path=None
    )
    replay = ReferenceReplay(RolloutFnConstructorInput(args=args, data_source=None))
    first = replay(RolloutFnTrainInput(rollout_id=0))
    second = replay(RolloutFnTrainInput(rollout_id=1))
    assert [s.tokens for g in first.samples for s in g] == [s.tokens for g in second.samples for s in g]
    assert len(second.samples) == 2
    args.start_rollout_id, args.lora_adapter_path = 1, "/verified/native"
    resumed = ReferenceReplay(RolloutFnConstructorInput(args=args, data_source=None))
    with pytest.raises(ValueError, match="exactly two"):
        resumed(RolloutFnTrainInput(rollout_id=0))
    assert len(resumed(RolloutFnTrainInput(rollout_id=1)).samples) == 2
