"""Cross-dataset assembly keeps paid evidence intact through Miles conversion."""

import hashlib
import json
import shutil

import pytest
from tests.integration.proximal_async.test_buffer import sample_for
from tests.integration.proximal_async.test_offline_batch import args_for, recovery
from tests.integration.proximal_async.test_run_state import context as _context

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles_plugins.proximal.buffer import accepted
from miles_plugins.proximal.contracts import digest, training_contract
from miles_plugins.proximal.offline_batch import (
    AssembledBatch,
    BatchSelection,
    FrozenBatchRolloutFn,
    assemble_batch,
    freeze_batch,
    load_batch_group,
    publish_batch,
    read_checkpoint,
    validate_batch,
)
from miles_plugins.proximal.store import open_store

context = _context


async def make_bundle(config, policy, attempt, root, *, prefix, groups, mixed=True, attempt_prefix=None):
    store = await open_store(config)
    ids = tuple(f"{prefix}-{i}" for i in range(groups))
    try:
        for group_index, group in enumerate(ids):
            samples = []
            for i in range(config.research.group_size):
                member = attempt.model_copy(
                    update={
                        "attempt_id": f"{attempt_prefix or prefix}-{group_index}-{i}",
                        "group_id": group,
                        "sample_index": i,
                        "dataset_sha256": digest(config.dataset),
                        "task": config.dataset.tasks[0],
                    }
                )
                sample, proof = sample_for(member)
                sample.index = group_index * config.research.group_size + i
                sample.group_index = group_index
                sample.reward = float(i % 2) if mixed else 0.0
                proof = proof.model_copy(update={"grade": proof.grade.model_copy(update={"reward": sample.reward})})
                sample.metadata["proximal_accepted"] = proof.model_dump_json()
                samples.append(sample)
            await store.add_group(group, policy, samples)
    finally:
        await store.close()
    freeze_batch(
        config=config,
        source_root=config.artifact_directory / config.run_id,
        group_ids=ids,
        policy=policy,
        num_samples=groups * config.research.group_size,
        out=root,
    )
    return BatchSelection(bundle=root, group_ids=ids)


def another_dataset(config):
    return config.model_copy(
        update={
            "dataset": config.dataset.model_copy(
                update={
                    "tasks": (config.dataset.tasks[0].model_copy(update={"environment_id": 99}),),
                }
            )
        }
    )


async def test_96_plus_32_groups_of_8_survive_offline_native_conversion(
    config,
    policy,
    attempt,
    tmp_path,
    context,
    store_dsn,
    pg_bin,
    monkeypatch,
):
    config = config.model_copy(
        update={
            "research": config.research.model_copy(update={"group_size": 8}),
            "max_in_flight_samples": 8,
        }
    )
    second = another_dataset(config)
    old = await make_bundle(config, policy, attempt, tmp_path / "old", prefix="old", groups=96)
    new = await make_bundle(second, policy, attempt, tmp_path / "new", prefix="new", groups=32)
    original = {
        g: hashlib.sha256((s.bundle / "groups" / f"{g}.bin").read_bytes()).hexdigest()
        for s in (old, new)
        for g in s.group_ids
    }
    bundle = tmp_path / "assembled"
    batch = assemble_batch(selections=(old, new), num_samples=1024, require_nonzero_reward_variance=True, out=bundle)
    assert isinstance(batch, AssembledBatch)
    assert batch.source.dataset == config.dataset
    assert batch.additional_sources[0].dataset == second.dataset
    assert {g.header.group_id: g.payload_sha256 for g in batch.groups} == original
    checkpoint_path = recovery(config, context, tmp_path, store_dsn, pg_bin)
    checkpoint = read_checkpoint(checkpoint_path, batch)
    committed = tmp_path / "committed"
    commits = []
    publish_batch(bundle, committed, commit=lambda: commits.append((committed / "batch.json").exists()))
    assert commits == [False, True]
    for path in (bundle, old.bundle, new.bundle, config.artifact_directory):
        shutil.rmtree(path)
    monkeypatch.delenv(config.store_dsn_env)
    assert validate_batch(committed) == batch
    args = args_for(batch, committed, checkpoint_path, checkpoint, tmp_path / "next-checkpoint")
    fn = FrozenBatchRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))
    output = fn(RolloutFnTrainInput(rollout_id=1))
    flat, metadata = postprocess_rollout_data(args, output.samples, train_parallel_config={"dp_size": 8})
    train = convert_samples_to_train_data(args, flat, metadata, None, None)
    assert len(train["tokens"]) == 1024
    assert train["sample_indices"] == list(range(1024))
    assert train["tokens"] == [[1, 2, 3]] * 1024
    assert train["loss_masks"] == [[1, 1]] * 1024
    assert train["raw_reward"] == [0.0, 1.0] * 512
    assert train["rollout_log_probs"][1023] == pytest.approx([-0.2, -0.4])
    assert accepted(flat[0]).attempt.dataset_sha256 == digest(config.dataset)
    assert accepted(flat[-1]).attempt.dataset_sha256 == digest(second.dataset)
    assert accepted(flat[0]).attempt.attempt_id == "old-0-0"
    assert accepted(flat[-1]).attempt.attempt_id == "new-31-7"


@pytest.fixture
async def selections(config, policy, attempt, tmp_path):
    old = await make_bundle(config, policy, attempt, tmp_path / "old", prefix="old", groups=1)
    new = await make_bundle(another_dataset(config), policy, attempt, tmp_path / "new", prefix="new", groups=1)
    return old, new


@pytest.mark.parametrize("field", ["policy", "harness", "sampling", "lora", "project", "tokenizer", "thinking", "lag"])
async def test_mismatched_sources_fail_before_completion(selections, tmp_path, field):
    path = selections[1].bundle / "batch.json"
    data = json.loads(path.read_text())
    source = data["source"]
    if field == "policy":
        data["policy"]["snapshot"]["sha256"] = "f" * 64
        data["groups"][0]["header"]["policy"] = data["policy"]
    else:
        if field == "harness":
            source["harness"]["max_turns"] += 1
        elif field == "sampling":
            source["research"]["sampling"]["max_tokens"] += 1
        elif field == "lora":
            source["research"]["lora"]["rank"] += 1
        elif field == "project":
            source["dataset"]["project_id"] += 1
        elif field == "tokenizer":
            source["tokenizer_path"] += "-different"
        elif field == "thinking":
            source["enable_thinking"] = not source["enable_thinking"]
        elif field == "lag":
            source["research"]["max_policy_lag"] += 1
        from miles_plugins.proximal.contracts import RunConfig

        changed = RunConfig.model_validate_json(json.dumps(source))
        data["groups"][0]["header"]["contract_sha256"] = digest(training_contract(changed))
    path.write_text(json.dumps(data))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError):
        assemble_batch(selections=selections, num_samples=4, require_nonzero_reward_variance=True, out=out)
    assert not out.exists()


@pytest.mark.parametrize("failure", ["duplicate", "missing", "count", "corrupt", "overlap"])
async def test_bad_selection_never_overwrites_input_or_publishes_completion(selections, tmp_path, failure):
    old, new = selections
    original = (old.bundle / "batch.json").read_bytes()
    count, out = 4, tmp_path / "rejected"
    if failure == "duplicate":
        old = old.model_copy(update={"group_ids": (*old.group_ids, *old.group_ids)})
        count = 6
    elif failure == "missing":
        new = new.model_copy(update={"group_ids": ("absent",)})
    elif failure == "count":
        count = 8
    elif failure == "corrupt":
        (new.bundle / "groups/new-0.bin").write_bytes(b"bad")
    elif failure == "overlap":
        out = old.bundle
    with pytest.raises(ValueError):
        assemble_batch(selections=(old, new), num_samples=count, require_nonzero_reward_variance=True, out=out)
    assert (old.bundle / "batch.json").read_bytes() == original
    if failure != "overlap":
        assert not (out / "batch.json").exists()


async def test_zero_variance_is_explicit_and_revalidated_on_consumption(config, policy, attempt, tmp_path):
    old = await make_bundle(config, policy, attempt, tmp_path / "old", prefix="old", groups=1)
    new = await make_bundle(
        another_dataset(config), policy, attempt, tmp_path / "new", prefix="new", groups=1, mixed=False
    )
    with pytest.raises(ValueError, match="zero-variance"):
        assemble_batch(
            selections=(old, new), num_samples=4, require_nonzero_reward_variance=True, out=tmp_path / "bad"
        )
    out = tmp_path / "allowed"
    batch = assemble_batch(selections=(old, new), num_samples=4, require_nonzero_reward_variance=False, out=out)
    strict = batch.model_copy(update={"require_nonzero_reward_variance": True})
    with pytest.raises(ValueError, match="zero-variance"):
        load_batch_group(out, strict, strict.groups[-1])
    # The input manifests and sample bytes never acquire a new dataset identity.
    assert validate_batch(old.bundle).source.dataset == config.dataset


async def test_duplicate_attempt_across_distinct_source_groups_is_rejected(config, policy, attempt, tmp_path):
    old = await make_bundle(config, policy, attempt, tmp_path / "old", prefix="old", groups=1, attempt_prefix="same")
    new = await make_bundle(
        another_dataset(config), policy, attempt, tmp_path / "new", prefix="new", groups=1, attempt_prefix="same"
    )
    with pytest.raises(ValueError, match="Repeated rollout attempt"):
        assemble_batch(
            selections=(old, new), num_samples=4, require_nonzero_reward_variance=True, out=tmp_path / "bad"
        )
    assert not (tmp_path / "bad/batch.json").exists()
