"""CPU-only finite collection using the existing platform producer and buffer.

Requires an already committed policy, running inference/capture service, and a
Postgres store. Does not create a trainer, publish weights or provision replicas.
"""

import argparse
import asyncio
import json
from pathlib import Path

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles_plugins.proximal.authorization import AuthorizedRun, authorize_run, require_authorization
from miles_plugins.proximal.buffer import accepted, validate_group
from miles_plugins.proximal.contracts import Policy, RunStateArtifacts, read_run_config
from miles_plugins.proximal.data_source import PlatformTaskSource
from miles_plugins.proximal.offline_batch import FrozenBatch, freeze_batch
from miles_plugins.proximal.options import BUFFER
from miles_plugins.proximal.rollout import PlatformRolloutFn
from miles_plugins.proximal.storage import write_atomic, write_immutable
from miles_plugins.proximal.store import open_store


async def collect_batch(
    authorization: AuthorizedRun, *, config_path: Path, policy: Policy, num_samples: int, out: Path
) -> FrozenBatch:
    config = require_authorization(authorization)
    if read_run_config(config_path) != config:
        raise ValueError("Collection config changed after authorization")
    if num_samples <= 0 or num_samples % config.research.group_size:
        raise ValueError("Sample count must be a positive multiple of group_size")
    if policy.run_id != config.run_id or policy.base_model != config.base_model:
        raise ValueError("Collection policy belongs to another run/base")
    if isinstance(config.artifact_storage, RunStateArtifacts):
        raise ValueError("Standalone collection needs shared_disk or a mounted modal_volume, not a trainer outbox")
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
    source = PlatformTaskSource(args)
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
        )
        await asyncio.to_thread(store.sync.commit)
        return batch
    finally:
        try:
            await rollout.close()
        finally:
            await store.close()


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
