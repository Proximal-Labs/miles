"""Four new samples combine with four preselected old failures, without relabeling."""

import hashlib
import shutil

import pytest
from tests.integration.proximal_async.test_batch_assembly import make_bundle
from tests.integration.proximal_async.test_offline_batch import args_for, recovery
from tests.integration.proximal_async.test_run_state import context as _context

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles_plugins.proximal.buffer import accepted
from miles_plugins.proximal.offline_batch import (
    BatchSelection,
    FrozenBatchRolloutFn,
    GroupMembers,
    TrainingGroup,
    freeze_batch,
    load_training_group,
    publish_batch,
    read_checkpoint,
    regroup_batch,
    train_argv,
    validate_batch,
)

context = _context


@pytest.fixture
def group8(config):
    return config.model_copy(
        update={"research": config.research.model_copy(update={"group_size": 8}), "max_in_flight_samples": 8}
    )


def retry_config(config):
    return config.model_copy(update={"research": config.research.model_copy(update={"group_size": 4})})


def plan(old="old-0", new="new-0"):
    return (
        TrainingGroup(
            members=(
                GroupMembers(source_group_id=old, sample_offsets=(0, 1, 2, 3)),
                GroupMembers(source_group_id=new, sample_offsets=(0, 1, 2, 3)),
            )
        ),
    )


async def test_four_plus_four_becomes_one_native_group_preserving_attempts(
    group8,
    policy,
    attempt,
    tmp_path,
    context,
    store_dsn,
    pg_bin,
    monkeypatch,
):
    old = await make_bundle(group8, policy, attempt, tmp_path / "old", prefix="old", groups=1, mixed=False)
    new = await make_bundle(retry_config(group8), policy, attempt, tmp_path / "new", prefix="new", groups=1)
    before = {
        s.group_ids[0]: hashlib.sha256((s.bundle / "groups" / f"{s.group_ids[0]}.bin").read_bytes()).hexdigest()
        for s in (old, new)
    }
    bundle = tmp_path / "regrouped"
    batch = regroup_batch(
        selections=(old, new), groups=plan(), num_samples=8, require_nonzero_reward_variance=True, out=bundle
    )
    assert len(batch.groups) == 2 and len(batch.training_groups) == 1
    assert {g.header.group_id: g.payload_sha256 for g in batch.groups} == before
    checkpoint_path = recovery(group8, context, tmp_path, store_dsn, pg_bin)
    checkpoint = read_checkpoint(checkpoint_path, batch)
    durable = tmp_path / "durable"
    publish_batch(bundle, durable, commit=lambda: None)
    for path in (bundle, old.bundle, new.bundle, group8.artifact_directory):
        shutil.rmtree(path)
    monkeypatch.delenv(group8.store_dsn_env)
    assert validate_batch(durable) == batch
    args = args_for(batch, durable, checkpoint_path, checkpoint, tmp_path / "next")
    fn = FrozenBatchRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))
    output = fn(RolloutFnTrainInput(rollout_id=1))
    assert len(output.samples) == 1 and len(output.samples[0]) == 8
    flat, metadata = postprocess_rollout_data(args, output.samples, train_parallel_config={"dp_size": 1})
    data = convert_samples_to_train_data(args, flat, metadata, None, None)
    assert data["raw_reward"] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    assert data["sample_indices"] == list(range(8))
    assert data["tokens"] == [[1, 2, 3]] * 8 and data["loss_masks"] == [[1, 1]] * 8
    assert data["rollout_log_probs"][-1] == pytest.approx([-0.2, -0.4])
    assert [accepted(s).attempt.group_id for s in flat] == ["old-0"] * 4 + ["new-0"] * 4
    assert [accepted(s).attempt.sample_index for s in flat] == [0, 1, 2, 3] * 2
    assert [accepted(s).attempt.attempt_id for s in flat] == [f"old-0-{i}" for i in range(4)] + [
        f"new-0-{i}" for i in range(4)
    ]
    command = train_argv(durable, batch, checkpoint_path)
    assert command[command.index("--rollout-batch-size") + 1] == "1"


@pytest.mark.parametrize("field,value", [("environment_id", 99), ("image_id", 99), ("source_commit_sha", "f" * 40)])
async def test_regroup_rejects_different_task_image_or_commit(group8, policy, attempt, tmp_path, field, value):
    old = await make_bundle(group8, policy, attempt, tmp_path / "old", prefix="old", groups=1, mixed=False)
    new_config = retry_config(group8)
    new_config = new_config.model_copy(
        update={
            "dataset": new_config.dataset.model_copy(
                update={
                    "tasks": (new_config.dataset.tasks[0].model_copy(update={field: value}),),
                }
            )
        }
    )
    new = await make_bundle(new_config, policy, attempt, tmp_path / "new", prefix="new", groups=1)
    with pytest.raises(ValueError, match="one exact task"):
        regroup_batch(
            selections=(old, new),
            groups=plan(),
            num_samples=8,
            require_nonzero_reward_variance=True,
            out=tmp_path / "bad",
        )
    assert not (tmp_path / "bad/batch.json").exists()


@pytest.mark.parametrize("indices", [(0, 0, 1, 2), (0, 1, 2, 4), (0, 1, 2)])
async def test_regroup_rejects_duplicate_missing_or_wrong_count_members(group8, policy, attempt, tmp_path, indices):
    old = await make_bundle(group8, policy, attempt, tmp_path / "old", prefix="old", groups=1, mixed=False)
    new = await make_bundle(retry_config(group8), policy, attempt, tmp_path / "new", prefix="new", groups=1)
    groups = (
        TrainingGroup(members=(plan()[0].members[0], GroupMembers(source_group_id="new-0", sample_offsets=indices))),
    )
    with pytest.raises(ValueError):
        regroup_batch(
            selections=(old, new),
            groups=groups,
            num_samples=8,
            require_nonzero_reward_variance=True,
            out=tmp_path / "bad",
        )
    assert not (tmp_path / "bad/batch.json").exists()


async def test_zero_plus_zero_stays_ineligible_but_source_bytes_survive(group8, policy, attempt, tmp_path):
    old = await make_bundle(group8, policy, attempt, tmp_path / "old", prefix="old", groups=1, mixed=False)
    new = await make_bundle(
        retry_config(group8), policy, attempt, tmp_path / "new", prefix="new", groups=1, mixed=False
    )
    with pytest.raises(ValueError, match="zero-variance"):
        regroup_batch(
            selections=(old, new),
            groups=plan(),
            num_samples=8,
            require_nonzero_reward_variance=True,
            out=tmp_path / "bad",
        )
    assert validate_batch(old.bundle).num_samples == 8
    assert validate_batch(new.bundle).num_samples == 4


async def test_offsets_are_not_confused_with_original_global_sample_indices(group8, policy, attempt, tmp_path):
    old = await make_bundle(group8, policy, attempt, tmp_path / "old", prefix="old", groups=2, mixed=False)
    new = await make_bundle(retry_config(group8), policy, attempt, tmp_path / "new", prefix="new", groups=2)
    old = old.model_copy(update={"group_ids": ("old-1",)})
    new = new.model_copy(update={"group_ids": ("new-1",)})
    groups = plan("old-1", "new-1")
    out = tmp_path / "assembled"
    batch = regroup_batch(
        selections=(old, new), groups=groups, num_samples=8, require_nonzero_reward_variance=True, out=out
    )
    samples = load_training_group(out, batch, groups[0])
    assert [accepted(sample).attempt.sample_index for sample in samples] == [8, 9, 10, 11, 4, 5, 6, 7]


async def test_full_1024_packet_combines_old_rescued_and_new_groups(group8, policy, attempt, tmp_path):
    mixed = await make_bundle(group8, policy, attempt, tmp_path / "mixed", prefix="mixed", groups=96)
    zeros = await make_bundle(group8, policy, attempt, tmp_path / "zeros", prefix="zero", groups=30, mixed=False)
    original = tmp_path / "original"
    old_ids = (*mixed.group_ids, *zeros.group_ids)
    freeze_batch(
        config=group8,
        source_root=group8.artifact_directory / group8.run_id,
        group_ids=old_ids,
        policy=policy,
        num_samples=126 * 8,
        out=original,
    )
    retries = await make_bundle(retry_config(group8), policy, attempt, tmp_path / "retries", prefix="retry", groups=30)
    new_config = group8.model_copy(
        update={
            "dataset": group8.dataset.model_copy(
                update={"tasks": (group8.dataset.tasks[0].model_copy(update={"environment_id": 99}),)}
            )
        }
    )
    new = await make_bundle(new_config, policy, attempt, tmp_path / "new", prefix="new", groups=44)
    groups = (
        *(
            TrainingGroup(members=(GroupMembers(source_group_id=g, sample_offsets=tuple(range(8))),))
            for g in mixed.group_ids
        ),
        *(plan(f"zero-{i}", f"retry-{i}")[0] for i in range(30)),
        *(
            TrainingGroup(members=(GroupMembers(source_group_id=g, sample_offsets=tuple(range(8))),))
            for g in new.group_ids[:2]
        ),
    )
    out = tmp_path / "assembled"
    batch = regroup_batch(
        selections=(
            BatchSelection(bundle=original, group_ids=old_ids),
            new.model_copy(update={"group_ids": new.group_ids[:2]}),
            retries,
        ),
        groups=groups,
        num_samples=1024,
        require_nonzero_reward_variance=True,
        out=out,
    )
    assert len(batch.groups) == 158 and len(batch.training_groups) == 128
    assert len(batch.additional_sources) == 2
    durable = tmp_path / "durable"
    publish_batch(out, durable, commit=lambda: None)
    for path in (out, original, mixed.bundle, zeros.bundle, retries.bundle, new.bundle, group8.artifact_directory):
        shutil.rmtree(path)
    assert validate_batch(durable) == batch
    samples = [sample for group in groups for sample in load_training_group(durable, batch, group)]
    assert len(samples) == len({accepted(s).attempt.attempt_id for s in samples}) == 1024
    assert [accepted(s).grade.reward for s in samples[768:776]] == [0.0] * 5 + [1.0, 0.0, 1.0]
    assert accepted(samples[-1]).attempt.task.environment_id == 99
