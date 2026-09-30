"""Durable complete groups become real Miles train data with all services offline."""

import asyncio
import json
import shutil
from argparse import Namespace

import httpx
import pytest
from tests.integration.proximal_async.test_buffer import entry, versioned
from tests.integration.proximal_async.test_e2e_stage_a import serve
from tests.integration.proximal_async.test_e2e_stage_a import stage_a as _stage_a
from tests.integration.proximal_async.test_run_state import context as _context
from tests.integration.proximal_async.test_run_state import native_save

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles_plugins.proximal import state_checkpoints
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.buffer import accepted
from miles_plugins.proximal.capture_server import CaptureServer, capture_tokenizer
from miles_plugins.proximal.collect_batch import collect_batch, collect_persisted
from miles_plugins.proximal.contracts import RunStateArtifacts, digest, sampling_args, training_contract
from miles_plugins.proximal.e2e.fake_pool import FakePool
from miles_plugins.proximal.e2e.fake_trainer import Publisher
from miles_plugins.proximal.e2e.stub_platform import StubPlatform
from miles_plugins.proximal.offline_batch import (
    ROLLOUT,
    FrozenBatchRolloutFn,
    freeze_batch,
    load_group,
    oldest_groups,
    publish_batch,
    read_checkpoint,
    train_argv,
    training_groups,
    validate_batch,
    validate_train_args,
)
from miles_plugins.proximal.store import open_store

context = _context
stage_a = _stage_a


async def populate(config, policy, attempt, count):
    store = await open_store(config)
    try:
        for i in range(count):
            samples = entry(attempt, policy, group=f"g{i}", group_index=i).group
            samples[1].reward = 1.0
            proof = accepted(samples[1])
            proof = proof.model_copy(update={"grade": proof.grade.model_copy(update={"reward": 1.0})})
            samples[1].metadata["proximal_accepted"] = proof.model_dump_json()
            await store.add_group(f"g{i}", policy, samples)
    finally:
        await store.close()


def freeze(config, policy, root, out, count=2):
    return freeze_batch(
        config=config,
        source_root=root,
        group_ids=tuple(f"g{i}" for i in range(count)),
        policy=policy,
        num_samples=count * config.research.group_size,
        out=out,
    )


def args_for(batch, bundle, checkpoint_path, checkpoint, save):
    return Namespace(
        debug_train_only=True,
        rollout_global_dataset=False,
        rollout_function_path=ROLLOUT,
        global_batch_size=batch.num_samples,
        rollout_batch_size=len(training_groups(batch)),
        n_samples_per_prompt=batch.source.research.group_size,
        **sampling_args(batch.source.research.sampling),
        num_rollout=checkpoint.step + 2,
        start_rollout_id=checkpoint.step + 1,
        use_rollout_logprobs=True,
        use_tis=False,
        tis_clip=None,
        load=str(batch.source.tokenizer_path),
        hf_checkpoint=str(batch.source.tokenizer_path),
        data_source_path="miles.rollout.data_source.RolloutDataSourceWithBuffer",
        train_backend="megatron",
        lora_rank=batch.source.research.lora.rank,
        lora_alpha=batch.source.research.lora.alpha,
        lora_dropout=0,
        target_modules=list(batch.source.research.lora.target_modules),
        actor_num_nodes=1,
        actor_num_gpus_per_node=checkpoint.native.world_size,
        tp=2,
        proximal_frozen_batch=bundle,
        proximal_frozen_checkpoint=checkpoint_path,
        lora_adapter_path=str(checkpoint_path / "checkpoint/adapter"),
        save=str(save),
        save_interval=1,
        optimizer="adam",
        disable_rollout_trim_samples=False,
        use_dynamic_global_batch_size=False,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )


def recovery(config, context, tmp_path, store_dsn, pg_bin):
    working = tmp_path / "native"
    native_save(working, step=0)
    context = context.model_copy(
        update={
            "contract_sha256": digest(training_contract(config)),
            "train_args": ("--optimizer", "adam", "--lr", "0.00004"),
        }
    )
    identity = state_checkpoints.take(
        0,
        checkpoints=working,
        dsn=store_dsn,
        snapshot_root=tmp_path / "volume",
        pg_bin=pg_bin,
        context=context,
        launch_id="one",
        parent=None,
        commit=lambda: None,
    )
    return tmp_path / "volume/checkpoints" / identity


async def test_1024_samples_survive_source_removal_and_native_miles_conversion(
    config, policy, attempt, tmp_path, context, store_dsn, pg_bin, monkeypatch
):
    # 512 complete groups of 2, deliberately small sequences for a CPU witness.
    await populate(config, policy, attempt, 512)
    bundle = tmp_path / "batch"
    batch = freeze(config, policy, config.artifact_directory / config.run_id, bundle, count=512)
    checkpoint_path = recovery(config, context, tmp_path, store_dsn, pg_bin)
    checkpoint = read_checkpoint(checkpoint_path, batch)
    mount, committed = tmp_path / "batch-mount", tmp_path / "batch-committed"
    commits = []

    def commit():
        shutil.copytree(mount, committed, dirs_exist_ok=True)
        commits.append((committed / "batch.json").exists())

    publish_batch(bundle, mount, commit=commit)
    assert commits == [False, True]
    # Lose both the collector's working bundle and the uncommitted mount state.
    shutil.rmtree(bundle)
    shutil.rmtree(mount)
    bundle = committed
    shutil.rmtree(config.artifact_directory)
    monkeypatch.delenv(config.store_dsn_env)
    for name in ("PX_TEST_KEY", "CAPTURE_TEST_KEY", "CAPTURE_PLATFORM_TEST_KEY", "FLEET_TEST_KEY"):
        monkeypatch.delenv(name)
    assert validate_batch(bundle).num_samples == 1024
    args = args_for(batch, bundle, checkpoint_path, checkpoint, tmp_path / "new-checkpoint")
    rollout = FrozenBatchRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))
    output = rollout(RolloutFnTrainInput(rollout_id=1))
    flat, metadata = postprocess_rollout_data(args, output.samples, train_parallel_config={"dp_size": 16})
    train = convert_samples_to_train_data(args, flat, metadata, None, None)
    assert len(train["tokens"]) == 1024
    assert train["sample_indices"] == list(range(1024))
    assert train["tokens"] == [[1, 2, 3]] * 1024
    assert train["loss_masks"] == [[1, 1]] * 1024
    assert train["raw_reward"] == [0.0, 1.0] * 512
    assert train["rollout_log_probs"][0] == pytest.approx([-0.2, -0.4])
    assert train["rewards"] == pytest.approx([-(2**-0.5), 2**-0.5] * 512, abs=2e-6)
    assert accepted(flat[-1]).attempt.group_id == "g511"
    with pytest.raises(ValueError, match="implicit replay"):
        rollout(RolloutFnTrainInput(rollout_id=2))
    command = train_argv(bundle, batch, checkpoint_path)
    assert command[command.index("--global-batch-size") + 1] == "1024"
    assert (
        int(command[command.index("--num-rollout") + 1]) - int(command[command.index("--start-rollout-id") + 1]) == 1
    )
    args.global_batch_size = 512
    with pytest.raises(ValueError, match="global-batch-size"):
        validate_train_args(args, batch, checkpoint)


async def test_legacy_payload_and_corruption_fail_closed(config, policy, attempt, tmp_path):
    await populate(config, policy, attempt, 2)
    root = config.artifact_directory / config.run_id
    for path in (root / "groups").glob("*.json"):
        path.unlink()  # Existing runs before #32 have .bin only.
    bundle = tmp_path / "batch"
    batch = freeze(config, policy, root, bundle)
    assert [g.header.group_id for g in batch.groups] == ["g0", "g1"]
    assert load_group(bundle, batch.groups[0], config)[1].reward == 1.0
    (bundle / "groups/g1.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_batch(bundle)


async def test_volume_freeze_needs_no_hardlinks_and_rejects_manifest_replacement(
    config, policy, attempt, tmp_path, monkeypatch
):
    await populate(config, policy, attempt, 2)
    source = config.artifact_directory / config.run_id
    bundle = tmp_path / "batch"

    def no_hardlinks(*args, **kwargs):
        raise PermissionError("Volume does not support hardlinks")

    monkeypatch.setattr("os.link", no_hardlinks)
    freeze(config, policy, source, bundle)
    original = (bundle / "batch.json").read_bytes()
    freeze(config, policy, source, bundle)  # Same manifest is an idempotent retry.
    with pytest.raises(ValueError, match="checksum/size"):
        freeze(config, policy, source, bundle, count=1)
    assert (bundle / "batch.json").read_bytes() == original
    assert validate_batch(bundle).num_samples == 4


@pytest.mark.parametrize("ids,samples", [(("g0", "g0"), 4), (("g0",), 4), (("../g0", "g1"), 4)])
async def test_invalid_selection_has_no_completion(config, policy, attempt, tmp_path, ids, samples):
    await populate(config, policy, attempt, 2)
    with pytest.raises(ValueError):
        freeze_batch(
            config=config,
            source_root=config.artifact_directory / config.run_id,
            group_ids=ids,
            policy=policy,
            num_samples=samples,
            out=tmp_path / "out",
        )
    assert not (tmp_path / "out/batch.json").exists()


async def test_selection_from_volume_indexes_needs_no_database(config, policy, attempt, tmp_path, monkeypatch):
    await populate(config, policy, attempt, 3)
    source = config.artifact_directory / config.run_id
    monkeypatch.delenv(config.store_dsn_env)
    assert oldest_groups(config, source, policy, 4) == ("g0", "g1")
    with pytest.raises(ValueError, match="only 3"):
        oldest_groups(config, source, policy, 8)
    with pytest.raises(ValueError, match="only 0"):
        oldest_groups(config, source, versioned(policy, 2), 4)


async def test_wrong_policy_and_reward_evidence_rejected(config, policy, attempt, tmp_path):
    await populate(config, policy, attempt, 2)
    with pytest.raises(ValueError, match="exact behavior policy"):
        freeze(config, versioned(policy, 2), config.artifact_directory / config.run_id, tmp_path / "wrong")
    store = await open_store(config)
    samples = entry(attempt, policy, group="bad").group
    samples[1].reward = 99.0  # Valid codec and checksum, invalid verifier evidence.
    try:
        await store.add_group("bad", policy, samples)
    finally:
        await store.close()
    with pytest.raises(ValueError, match="reward lacks"):
        freeze_batch(
            config=config,
            source_root=config.artifact_directory / config.run_id,
            group_ids=("bad",),
            policy=policy,
            num_samples=2,
            out=tmp_path / "bad",
        )


async def test_native_resume_refuses_state_loss_and_input_overwrite(
    config, policy, attempt, tmp_path, context, store_dsn, pg_bin
):
    config = config.model_copy(update={"tokenizer_path": tmp_path / "base"})
    await populate(config, policy, attempt, 2)
    bundle = tmp_path / "batch"
    batch = freeze(config, policy, config.artifact_directory / config.run_id, bundle)
    checkpoint_path = recovery(config, context, tmp_path, store_dsn, pg_bin)
    checkpoint = read_checkpoint(checkpoint_path, batch)
    changes = [
        {"optimizer": "muon"},
        {"actor_num_gpus_per_node": 4},
        {"tp": 1},
        {"no_load_optim": True},
        {"no_save_rng": True},
        {"finetune": True},
        {"proximal_frozen_fresh": True},
        {"lora_adapter_path": str(tmp_path / "serving-export")},
        *({"save": str(path)} for path in (bundle, bundle / "output", checkpoint_path, config.tokenizer_path)),
    ]
    for change in changes:
        args = args_for(batch, bundle, checkpoint_path, checkpoint, tmp_path / "new-checkpoint")
        validate_train_args(args, batch, checkpoint)
        vars(args).update(change)
        with pytest.raises(ValueError):
            validate_train_args(args, batch, checkpoint)
    optimizer = checkpoint_path / "checkpoint/adapter/training_state_rank0.pt"
    optimizer.write_bytes(optimizer.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum|size"):
        train_argv(bundle, batch, checkpoint_path)


@pytest.mark.parametrize("persist_to_volume", [False, True])
async def test_collect_then_stop_services_and_read_batch(stage_a, tmp_path, monkeypatch, persist_to_volume):
    run, path, ports = stage_a
    if persist_to_volume:
        run = run.model_copy(update={"artifact_storage": RunStateArtifacts(kind="run_state")})
        path.write_text(run.model_dump_json())
    mount, durable = tmp_path / "volume-mount", tmp_path / "committed-volume"

    def commit():
        shutil.copytree(mount, durable, dirs_exist_ok=True)

    tokenizer = capture_tokenizer(run.tokenizer_path, run.tito_model)
    authorization = authorize_run(run, yes_rollouts=True, yes_publish=True)
    store = await open_store(run)
    servers = []
    async with httpx.AsyncClient(timeout=30) as backend, httpx.AsyncClient(timeout=30) as agent_http:
        pool = FakePool(run, tokenizer=tokenizer, api_key="fleet-secret")
        capture = CaptureServer.beside_trainer(authorization, tokenizer=tokenizer, client=backend, store=store)
        stub = StubPlatform(
            run, api_key="platform-secret", capture_key="capture-platform-secret", reward="mixed", client=agent_http
        )
        try:
            for app, port in ((stub.app, ports[0]), (capture.app, ports[1]), (pool.app, ports[2])):
                servers.append(await serve(app, port))
            policy = await Publisher(
                authorization, backend, store, adapters=[tmp_path / "adapters/adapter-0"], mode="local"
            ).publish(1)
            if persist_to_volume:
                collection = collect_persisted(
                    authorization,
                    config_path=path,
                    policy=policy,
                    num_samples=4,
                    out=tmp_path / "collected",
                    snapshot_root=mount,
                    collection_root=mount / "collections/one",
                    commit=commit,
                )
            else:
                collection = collect_batch(
                    authorization, config_path=path, policy=policy, num_samples=4, out=tmp_path / "collected"
                )
            result = await asyncio.wait_for(collection, 60)
        finally:
            for server, task in servers:
                server.should_exit = True
                await task
            await store.close()
    shutil.rmtree(run.artifact_directory)
    monkeypatch.delenv(run.store_dsn_env)
    assert len(json.loads((tmp_path / "collected/selection.json").read_text())) == 2
    if persist_to_volume:
        # Only committed bytes survive losing the entire original CPU container.
        shutil.rmtree(mount)
        shutil.rmtree(tmp_path / "collected")
        bundle = durable / "collections/one/batch"
        assert len(list((durable / "artifacts" / run.run_id / "accepted").glob("*/accepted.json"))) == 4
        assert not list((durable / "artifacts" / run.run_id / "accepted").glob("*/failed.json"))
        assert len(list((durable / "artifacts" / run.run_id / "groups").glob("*.json"))) == 2
    else:
        bundle = tmp_path / "collected"
    assert validate_batch(bundle) == result
    assert result.num_samples == 4
