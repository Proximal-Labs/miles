"""CPU-only finite collection using the existing platform producer and buffer.

Requires an already committed policy, running inference/capture service, and a
Postgres store. Does not create a trainer, publish weights or provision replicas.
"""

import argparse
import asyncio
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import TypeAdapter

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles_plugins.proximal.authorization import AuthorizedRun, authorize_run, require_authorization, secret_env
from miles_plugins.proximal.buffer import accepted, validate_group
from miles_plugins.proximal.contracts import (
    Contract,
    Policy,
    Positive,
    RunConfig,
    RunStateArtifacts,
    SafeId,
    digest,
    read_run_config,
    training_contract,
)
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.offline_batch import FrozenBatch, freeze_batch, publish_batch, validate_batch
from miles_plugins.proximal.options import BUFFER
from miles_plugins.proximal.rollout import PlatformRolloutFn
from miles_plugins.proximal.snapshot import Digest
from miles_plugins.proximal.state_artifacts import copy_verified, describe, initialize
from miles_plugins.proximal.state_writer import StateWriter
from miles_plugins.proximal.storage import write_atomic, write_immutable
from miles_plugins.proximal.store import open_store


class CollectionInvocation(Contract):
    collection_id: SafeId
    contract_sha256: Digest
    num_samples: Positive
    # None explicitly selects fresh-base initialization.
    policy: Policy | None


def claim_collection(
    authorization: AuthorizedRun,
    *,
    collection_id: str,
    samples: int,
    policy: Policy | None,
    snapshot_root: Path,
    commit: Callable[[], None],
) -> bool:
    """Persist a retry fuse before side effects. True means already completed.

    One run owner is required; this is not a distributed lease. Interrupted work
    is recoverable from captures/groups, but cannot safely restart from an empty
    local database with new request IDs.
    """
    config = require_authorization(authorization)
    identity = TypeAdapter(SafeId).validate_python(collection_id)
    validate_collection_request(
        config,
        samples=samples,
        fresh=policy is None,
        policy_json="" if policy is None else policy.model_dump_json(),
        persist_to_volume=True,
    )
    record = CollectionInvocation(
        collection_id=identity, contract_sha256=digest(training_contract(config)), num_samples=samples, policy=policy
    )
    marker = snapshot_root / "collection-invocations" / f"{identity}.json"
    collection = snapshot_root / "collections" / identity
    if marker.exists():
        if CollectionInvocation.model_validate_json(marker.read_bytes()) != record:
            raise ValueError("Collection ID reused with different inputs")
        if not (collection / "batch/batch.json").is_file():
            raise ValueError("Interrupted collection: refusing to relaunch paid rollouts; reconcile durable artifacts")
        batch = validate_batch(collection / "batch")
        if (
            not isinstance(batch, FrozenBatch)
            or digest(training_contract(batch.source)) != record.contract_sha256
            or batch.num_samples != samples
            or (policy is not None and batch.policy != policy)
        ):
            raise ValueError("Completed collection differs from its invocation")
        if policy is None:
            from miles_plugins.proximal.initial_policy import verify_base_policy

            verify_base_policy(config, batch.policy, collection / "batch/base_policy")
        return True
    if collection.exists():
        raise ValueError("Collection already exists without an invocation record; reconcile it explicitly")
    with tempfile.TemporaryDirectory() as staging:
        source = Path(staging) / "invocation.json"
        source.write_text(record.model_dump_json())
        copy_verified(source, marker, describe(source, relative=marker.name))
    commit()  # No paid request or publication is allowed before this succeeds.
    return False


def validate_collection_request(
    config: RunConfig, *, samples: int, fresh: bool, policy_json: str, persist_to_volume: bool
) -> Policy | None:
    """Shared free validation on the local launch host and CPU worker."""
    if not persist_to_volume:
        raise ValueError("Detached collection requires --rollouts-persist-to-volume")
    if config.research.unused_groups != "retry":
        raise ValueError("Finite collection requires retrying failed groups")
    if samples <= 0 or samples % config.research.group_size or fresh == bool(policy_json):
        raise ValueError("Choose complete groups and exactly one of --fresh or --policy-file")
    policy = None if fresh else Policy.model_validate_json(policy_json)
    if policy is not None and (policy.run_id != config.run_id or policy.base_model != config.base_model):
        raise ValueError("Policy belongs to another run/base")
    return policy


async def collect_batch(
    authorization: AuthorizedRun,
    *,
    config_path: Path,
    policy: Policy,
    num_samples: int,
    out: Path,
    publisher: StateWriter | None = None,
    base_policy: Path | None = None,
) -> FrozenBatch:
    config = require_authorization(authorization)
    if read_run_config(config_path) != config:
        raise ValueError("Collection config changed after authorization")
    if num_samples <= 0 or num_samples % config.research.group_size:
        raise ValueError("Sample count must be a positive multiple of group_size")
    if policy.run_id != config.run_id or policy.base_model != config.base_model:
        raise ValueError("Collection policy belongs to another run/base")
    if isinstance(config.artifact_storage, RunStateArtifacts):
        if publisher is None or not publisher.thread.is_alive() or publisher.run_id != config.run_id:
            raise ValueError("Run-state collection requires its run's active artifact publisher")
        if publisher.artifacts != config.artifact_directory:
            raise ValueError("Collector and publisher must use the same local artifact directory")
    if out.exists():
        raise ValueError("Use a new collection directory; freeze selection.json to recover completed work")
    args = argparse.Namespace(
        proximal_config=str(config_path),
        proximal_yes_rollouts=True,
        proximal_yes_publish=True,
        rollout_submission_granularity="sample",
        n_samples_per_prompt=config.research.group_size,
        async_unused_samples_handler=config.research.unused_groups,
        rollout_sample_filter_path=None,
        rollout_batch_size=1,
        rollout_global_dataset=True,
        async_max_concurrent_samples=config.max_in_flight_samples,
        custom_async_data_buffer_path=BUFFER,
        save=None,
        load=None,
    )
    # Drain one group at a time to bound collector memory. Miles owns concurrency,
    # retries, validation, backpressure, capture release and logical cancellation.
    source = PlatformTaskSource(args, num_groups=num_samples // config.research.group_size)
    store = await open_store(config)
    rollout = PlatformRolloutFn(RolloutFnConstructorInput(args=args, data_source=source))
    ids: list[str] = []
    try:
        if await store.current_policy() != policy:
            raise ValueError("Collection requires the explicitly selected policy to be current")
        write_immutable(out / "source.json", config.model_dump_json().encode())
        write_immutable(out / "policy.json", policy.model_dump_json().encode())
        for index in range(num_samples // config.research.group_size):
            if await store.current_policy() != policy:
                raise ValueError("Policy changed during fixed-policy collection")
            result = await rollout(RolloutFnTrainInput(rollout_id=index, weight_version=policy.version))
            if not isinstance(result, RolloutFnTrainOutput) or len(result.samples) != 1:
                raise ValueError("Collector expected one complete group")
            group = result.samples[0]
            if validate_group(config, group) != policy:
                raise ValueError("Collection selected a different behavior policy; use a dedicated collection run")
            ids.append(accepted(group[0]).attempt.group_id)
            write_atomic(out / "selection.json", json.dumps(ids).encode())
            # On a mounted Volume this persists the selection as well as group data.
            await asyncio.to_thread(store.sync.commit)
            print(f"Collected {len(ids) * config.research.group_size}/{num_samples} samples", flush=True)
        batch = await asyncio.to_thread(
            freeze_batch,
            config=config,
            source_root=config.artifact_directory / config.run_id,
            group_ids=tuple(ids),
            policy=policy,
            num_samples=num_samples,
            out=out,
            base_policy=base_policy,
        )
        await asyncio.to_thread(store.sync.commit)
        return batch
    finally:
        try:
            await rollout.close()
        finally:
            await store.close()


async def collect_persisted(
    authorization: AuthorizedRun,
    *,
    config_path: Path,
    policy: Policy,
    num_samples: int,
    out: Path,
    snapshot_root: Path,
    collection_root: Path,
    commit: Callable[[], None],
    base_policy: Path | None = None,
    train_args: tuple[str, ...] | None = None,
) -> FrozenBatch:
    """The CPU run owns the same publisher as training; no native checkpoint needed."""
    config = require_authorization(authorization)
    if not isinstance(config.artifact_storage, RunStateArtifacts):
        raise ValueError("Persisted collection stages locally with the run-state publisher")
    if num_samples <= 0 or num_samples % config.research.group_size:
        raise ValueError("Sample count must be a positive multiple of group_size")
    if (
        read_run_config(config_path) != config
        or policy.run_id != config.run_id
        or policy.base_model != config.base_model
    ):
        raise ValueError("Collection config/policy changed after authorization")
    if collection_root.exists() or out.exists():
        raise ValueError("Use a new collection ID; previously persisted rollouts remain available")
    # These small files make even interrupted collection recoverable without the DB.
    with tempfile.TemporaryDirectory() as staging:
        records = [("source.json", config.model_dump_json()), ("policy.json", policy.model_dump_json())]
        if train_args is not None:
            records.append(("training-args.json", json.dumps(train_args)))
        for name, value in records:
            source = Path(staging) / name
            source.write_text(value)
            copy_verified(source, collection_root / name, describe(source, relative=name))
    if base_policy is not None:
        from miles_plugins.proximal.initial_policy import copy_base_policy, verify_base_policy

        copy_base_policy(verify_base_policy(config, policy, base_policy), collection_root / "base_policy")
    await asyncio.to_thread(commit)
    dsn = secret_env(config.store_dsn_env)
    await asyncio.to_thread(initialize, dsn)
    publisher = StateWriter(
        dsn=dsn,
        run_id=config.run_id,
        artifacts=config.artifact_directory,
        snapshot_root=snapshot_root,
        commit=commit,
    )
    publisher.start()
    try:
        batch = await collect_batch(
            authorization,
            config_path=config_path,
            policy=policy,
            num_samples=num_samples,
            out=out,
            publisher=publisher,
            base_policy=base_policy,
        )
    finally:
        # collect_batch stops its producer first. Drain before the local DB closes.
        await asyncio.to_thread(publisher.close)
    await asyncio.to_thread(publish_batch, out, collection_root / "batch", commit=commit)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yes-rollouts", action="store_true", required=True)
    parser.add_argument(
        "--yes-publish",
        action="store_true",
        required=True,
        help="Existing run capability requires both consents; collection never publishes",
    )
    args = parser.parse_args()
    config = read_run_config(args.config)
    authorization = authorize_run(config, yes_rollouts=args.yes_rollouts, yes_publish=args.yes_publish)
    asyncio.run(
        collect_batch(
            authorization,
            config_path=args.config,
            policy=Policy.model_validate_json(args.policy.read_bytes()),
            num_samples=args.samples,
            out=args.out,
        )
    )


if __name__ == "__main__":
    main()
