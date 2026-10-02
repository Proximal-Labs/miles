"""Real frozen bundles: one chain step trains once on its own base-policy batch, inside the lag window."""

import hashlib
from argparse import Namespace

import pytest
from tests.integration.proximal_async.test_initial_policy import base_config
from tests.integration.proximal_async.test_offline_batch import populate
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles_plugins.proximal.buffer import accepted
from miles_plugins.proximal.e2e.batch_chain_inputs import ChainArm, ChainBatch, ChainPlan, validate_batches
from miles_plugins.proximal.e2e.batch_chain_replay import ChainReplay
from miles_plugins.proximal.initial_policy import prepare_base_policy
from miles_plugins.proximal.offline_batch import freeze_batch


def frozen(config, policy, snapshot, mount, name, group_ids):
    out = mount / name
    freeze_batch(
        config=config,
        source_root=config.artifact_directory / config.run_id,
        group_ids=group_ids,
        policy=policy,
        num_samples=len(group_ids) * config.research.group_size,
        out=out,
        base_policy=snapshot.directory,
    )
    return ChainBatch(path=name, sha256=hashlib.sha256((out / "batch.json").read_bytes()).hexdigest())


def chain(config, batches):
    return ChainPlan(
        experiment_id="chain",
        samples=config.research.group_size,
        nodes=1,
        batches=tuple(batches),
        arms=(ChainArm(name="mlp", target_modules=("linear_fc1", "linear_fc2")),),
        recipe=("--optimizer", "adam", "--lr", "4e-5", "--seed", "42", "--lr-decay-style", "constant"),
        gate="none",
        gate_timeout_action="stop",
    )


@pytest.mark.parametrize("max_policy_lag", [1, 2])
async def test_steps_train_once_each_on_disjoint_batches_inside_the_lag_window(
    config, policy, attempt, tmp_path, max_policy_lag
):
    config = base_config(config, tmp_path)
    config = config.model_copy(
        update={"research": config.research.model_copy(update={"max_policy_lag": max_policy_lag})}
    )
    snapshot = prepare_base_policy(config, output=tmp_path / "initial")
    policy = policy.model_copy(update={"snapshot": snapshot.reference})
    await populate(config, policy, attempt, 3)
    mount = tmp_path / "volume"
    batches = [frozen(config, policy, snapshot, mount, f"batch-{step}", (f"g{step}",)) for step in range(3)]
    if max_policy_lag < 2:
        with pytest.raises(ValueError, match="max_policy_lag"):
            validate_batches(mount, chain(config, batches), config)
        return
    assert len(validate_batches(mount, chain(config, batches), config)) == 3
    # Distinct manifests that still overlap in one stored group (identical bytes would already be a replay).
    pairs = [
        frozen(config, policy, snapshot, mount, name, ids)
        for name, ids in (("ab", ("g0", "g1")), ("bc", ("g1", "g2")))
    ]
    overlapping = chain(config, pairs).model_copy(update={"samples": 2 * config.research.group_size})
    with pytest.raises(ValueError, match="share"):
        validate_batches(mount, overlapping, config)

    seen = []
    for step in range(3):
        args = Namespace(
            chain_batch=mount / f"batch-{step}",
            debug_train_only=True,
            num_rollout=step + 1,
            start_rollout_id=step,
            lora_adapter_path="/verified/native" if step else None,
            save=str(tmp_path / f"save-{step}"),
        )
        replay = ChainReplay(RolloutFnConstructorInput(args=args, data_source=None))
        with pytest.raises(ValueError, match="exactly once"):
            replay(RolloutFnTrainInput(rollout_id=step + 1))
        groups = replay(RolloutFnTrainInput(rollout_id=step)).samples
        assert len(groups) == 1 and len(groups[0]) == config.research.group_size
        seen.append({accepted(sample).attempt.group_id for sample in groups[0]})
        with pytest.raises(ValueError, match="exactly once"):
            replay(RolloutFnTrainInput(rollout_id=step))
        replay.save(step)
        assert (tmp_path / f"save-{step}" / "rollout" / f"proximal_{step}.json").is_file()
    assert seen == [{"g0"}, {"g1"}, {"g2"}]

    args = Namespace(
        chain_batch=mount / "batch-1",
        debug_train_only=True,
        num_rollout=2,
        start_rollout_id=1,
        lora_adapter_path=None,
        save=str(tmp_path / "save"),
    )
    with pytest.raises(ValueError, match="previous step's native adapter"):
        ChainReplay(RolloutFnConstructorInput(args=args, data_source=None))
