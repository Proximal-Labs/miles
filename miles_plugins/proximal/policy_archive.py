"""One pending serving snapshot, drained by the existing run-state writer.

HTTP readiness is local to one replica. Fleet policy selection waits for the full
Volume archive so a cold replica can recover it. Native training checkpoints use
the existing independent recovery path.
"""

import asyncio
from pathlib import Path

from miles_plugins.proximal.authorization import AuthorizedRun, require_authorization
from miles_plugins.proximal.contracts import Policy, RunConfig
from miles_plugins.proximal.modal_volume import authorize_volume_publication, modal_publish_snapshot
from miles_plugins.proximal.snapshot import read_snapshot, snapshot_relative_path
from miles_plugins.proximal.storage import write_immutable
from miles_plugins.proximal.store import open_store


def pending_path(config: RunConfig) -> Path:
    return config.artifact_directory / config.run_id / "publication" / "pending-policy.json"


def enqueue(authorization: AuthorizedRun, policy: Policy) -> None:
    config = require_authorization(authorization)
    if policy.run_id != config.run_id or policy.base_model != config.base_model:
        raise ValueError("Policy archive belongs to another run")
    write_immutable(pending_path(config), policy.model_dump_json().encode())


def publish_pending(authorization: AuthorizedRun) -> None:
    config = require_authorization(authorization)
    path = pending_path(config)
    if not path.exists():
        return
    policy = Policy.model_validate_json(path.read_bytes())
    if policy.run_id != config.run_id or policy.base_model != config.base_model:
        raise ValueError("Policy archive belongs to another run")
    snapshot = read_snapshot(path.parent / snapshot_relative_path(policy.snapshot), policy.snapshot)
    if (
        snapshot.manifest.metadata.base_model != policy.base_model
        or snapshot.manifest.metadata.run_id != policy.run_id
        or snapshot.manifest.metadata.checkpoint_iteration != policy.version - 1
    ):
        raise ValueError("Policy archive metadata mismatch")
    publication = authorize_volume_publication(config.volume, yes_publish=True)
    modal_publish_snapshot(publication, snapshot)

    async def commit() -> None:
        store = await open_store(config)
        try:
            await store.commit_policy(policy)
        finally:
            await store.close()

    asyncio.run(commit())
    path.unlink()  # Only acknowledgement of durable bytes AND policy commit frees the slot.
