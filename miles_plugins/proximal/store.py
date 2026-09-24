"""Durable rollout store: a small Postgres index plus immutable payload files.

Miles's producer hands each finished group to the async DataBuffer; the buffer
persists it here, and the trainer's batch query selects from here. Which groups a
training run has already consumed is trainer state, saved with Miles's checkpoint
(see data_source.py), so it rolls back together with the weights on resume.

The policy table is the single authority for which immutable adapter each policy
version names. A resumed run abandons versions published after its checkpoint;
groups sampled from abandoned weights are never selected again.
"""

import asyncio
import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS_V2,
    decode_samples_and_merge_input_sample,
    encode_samples,
)
from miles.utils.types import Sample
from miles_plugins.proximal.authorization import secret_env
from miles_plugins.proximal.contracts import Contract, Policy, RunConfig, SafeId, digest, training_contract
from miles_plugins.proximal.storage import write_immutable

if TYPE_CHECKING:
    import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS proximal_policies (
    training_run_id text NOT NULL,
    version integer NOT NULL,
    snapshot_sha256 text NOT NULL,
    policy jsonb NOT NULL,
    abandoned boolean NOT NULL DEFAULT false,
    published_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (training_run_id, version, snapshot_sha256)
);
CREATE UNIQUE INDEX IF NOT EXISTS proximal_policies_live
    ON proximal_policies (training_run_id, version) WHERE NOT abandoned;
CREATE TABLE IF NOT EXISTS proximal_rollout_groups (
    training_run_id text NOT NULL,
    group_id text NOT NULL,
    policy_version integer NOT NULL,
    policy_sha256 text NOT NULL,
    group_index bigint NOT NULL,
    contract_sha256 text NOT NULL,
    payload_path text NOT NULL,
    payload_sha256 text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (training_run_id, group_id)
);
CREATE INDEX IF NOT EXISTS proximal_rollout_groups_fresh
    ON proximal_rollout_groups (training_run_id, policy_version, created_at);
CREATE TABLE IF NOT EXISTS proximal_rollout_rejections (
    training_run_id text NOT NULL,
    group_id text NOT NULL,
    filter_path text NOT NULL,
    reason text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (training_run_id, group_id, filter_path),
    FOREIGN KEY (training_run_id, group_id)
        REFERENCES proximal_rollout_groups (training_run_id, group_id)
);
"""

_SELECT = """
SELECT g.group_id, g.policy_version, g.payload_path, g.payload_sha256
FROM proximal_rollout_groups g
JOIN proximal_policies p
  ON p.training_run_id = g.training_run_id
 AND p.version = g.policy_version
 AND p.snapshot_sha256 = g.policy_sha256
 AND NOT p.abandoned
WHERE g.training_run_id = %s
  AND g.contract_sha256 = %s
  AND g.policy_version BETWEEN %s AND %s
  AND NOT (g.group_id = ANY(%s))
  AND NOT EXISTS (
      SELECT 1 FROM proximal_rollout_rejections r
      WHERE r.training_run_id = g.training_run_id AND r.group_id = g.group_id
        AND r.filter_path = %s
  )
ORDER BY g.created_at, g.group_id
"""


class PolicyConflict(ValueError):
    """A live version already names different weights; resume must rewind first."""


class SampleIdentity(Contract):
    index: int
    group_index: int


class StoredGroup(Contract):
    """What the payload file holds besides Miles's own sample codec bytes."""

    group_id: SafeId
    contract_sha256: str
    policy: Policy
    identities: tuple[SampleIdentity, ...]


@dataclass(frozen=True)
class PayloadSync:
    """Cross-container visibility for the payload mount (e.g. a Modal Volume).

    ``commit`` runs after a payload is written and before its index row exists,
    so a reader that sees the row can see the file after ``reload``.
    """

    commit: Callable[[], None]
    reload: Callable[[], None]


# For a shared disk: writes are immediately visible to every reader.
LOCAL_DISK = PayloadSync(commit=lambda: None, reload=lambda: None)


@dataclass(frozen=True)
class GroupRow:
    group_id: str
    policy_version: int
    payload_path: str
    payload_sha256: str


class RolloutStore:
    """Borrowed async connection; the composition root opens and closes it."""

    def __init__(
        self,
        connection: "psycopg.AsyncConnection[tuple[object, ...]]",
        *,
        run_id: str,
        contract_sha256: str,
        root: Path,
        sync: PayloadSync,
    ):
        self.connection, self.run_id, self.root, self.sync = connection, run_id, root, sync
        # Groups are selectable only by a trainer with the identical training contract.
        self.contract_sha256 = contract_sha256
        # One connection runs one statement at a time; put/get share an event loop.
        self._lock = asyncio.Lock()

    @classmethod
    async def open(
        cls, dsn: str, *, run_id: str, contract_sha256: str, root: Path, sync: PayloadSync
    ) -> "RolloutStore":
        import psycopg  # Runtime dependency of the training process only.

        connection = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
        store = cls(connection, run_id=run_id, contract_sha256=contract_sha256, root=root, sync=sync)
        async with store._lock:
            await connection.execute(SCHEMA)
        return store

    async def close(self) -> None:
        await self.connection.close()

    # ------------------------------ policies ------------------------------

    async def rewind(self, *, keep_through: int) -> int:
        """Abandon every live version after ``keep_through``; returns how many."""
        async with self._lock:
            cursor = await self.connection.execute(
                "UPDATE proximal_policies SET abandoned = true"
                " WHERE training_run_id = %s AND version > %s AND NOT abandoned",
                (self.run_id, keep_through),
            )
            return cursor.rowcount

    async def commit_policy(self, policy: Policy) -> None:
        """Record a verified, servable version. Idempotent for identical weights."""
        if policy.run_id != self.run_id:
            raise ValueError("Policy belongs to another training run")
        import psycopg.errors

        async with self._lock:
            async with self.connection.transaction():
                cursor = await self.connection.execute(
                    "SELECT max(version) FROM proximal_policies WHERE training_run_id = %s AND NOT abandoned",
                    (self.run_id,),
                )
                row = await cursor.fetchone()
                latest = row[0] if row is not None else None
                if isinstance(latest, int) and policy.version > latest + 1:
                    raise PolicyConflict("Policy versions must advance one at a time")
                try:
                    await self.connection.execute(
                        "INSERT INTO proximal_policies (training_run_id, version, snapshot_sha256, policy)"
                        " VALUES (%s, %s, %s, %s)"
                        " ON CONFLICT (training_run_id, version, snapshot_sha256)"
                        " DO UPDATE SET abandoned = false",
                        (self.run_id, policy.version, policy.snapshot.sha256, policy.model_dump_json()),
                    )
                except psycopg.errors.UniqueViolation as exc:
                    raise PolicyConflict(
                        f"Version {policy.version} is already live with different weights; rewind on resume"
                    ) from exc

    async def current_policy(self) -> Policy | None:
        async with self._lock:
            cursor = await self.connection.execute(
                "SELECT policy::text FROM proximal_policies WHERE training_run_id = %s AND NOT abandoned"
                " ORDER BY version DESC LIMIT 1",
                (self.run_id,),
            )
            row = await cursor.fetchone()
        return None if row is None else Policy.model_validate_json(str(row[0]))

    async def policy(self, version: int) -> Policy | None:
        """The live policy for ``version``, or None if absent or abandoned."""
        async with self._lock:
            cursor = await self.connection.execute(
                "SELECT policy::text FROM proximal_policies"
                " WHERE training_run_id = %s AND version = %s AND NOT abandoned",
                (self.run_id, version),
            )
            row = await cursor.fetchone()
        return None if row is None else Policy.model_validate_json(str(row[0]))

    # ------------------------------- groups -------------------------------

    def _payload_path(self, group_id: str) -> Path:
        return self.root / self.run_id / "groups" / f"{group_id}.bin"

    async def add_group(self, group_id: str, policy: Policy, samples: Sequence[Sample]) -> None:
        """Persist one validated group: payload file first, index row last."""
        identities = []
        for sample in samples:
            if sample.index is None or sample.group_index is None:
                raise ValueError("Stored samples need Miles sample identities")
            identities.append(SampleIdentity(index=sample.index, group_index=sample.group_index))
        header = StoredGroup(
            group_id=group_id, contract_sha256=self.contract_sha256, policy=policy, identities=tuple(identities)
        )
        header_bytes = header.model_dump_json().encode()
        # V2 fields: the codec's default v1 allowlist omits the reward.
        body = encode_samples(list(samples), {}, fields=COMPUTED_FIELDS_V2)
        payload = len(header_bytes).to_bytes(8, "big") + header_bytes + body
        digest = hashlib.sha256(payload).hexdigest()
        path = self._payload_path(group_id)
        await asyncio.to_thread(write_immutable, path, payload)
        await asyncio.to_thread(self.sync.commit)  # Visible to other containers before the row exists.
        async with self._lock:
            cursor = await self.connection.execute(
                "INSERT INTO proximal_rollout_groups (training_run_id, group_id, policy_version, policy_sha256,"
                " group_index, contract_sha256, payload_path, payload_sha256)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (training_run_id, group_id) DO NOTHING",
                (
                    self.run_id,
                    group_id,
                    policy.version,
                    policy.snapshot.sha256,
                    identities[0].group_index,
                    self.contract_sha256,
                    str(path),
                    digest,
                ),
            )
            if cursor.rowcount == 0:
                existing = await self.connection.execute(
                    "SELECT payload_sha256 FROM proximal_rollout_groups WHERE training_run_id = %s AND group_id = %s",
                    (self.run_id, group_id),
                )
                row = await existing.fetchone()
                if row is None or row[0] != digest:
                    raise ValueError(f"Group {group_id} was already stored with different contents")

    async def reject_group(self, group_id: str, *, filter_path: str, reason: str | None) -> bool:
        """Record a fixed verdict; return whether this is its first insertion."""
        async with self._lock:
            cursor = await self.connection.execute(
                "INSERT INTO proximal_rollout_rejections (training_run_id, group_id, filter_path, reason)"
                " VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (self.run_id, group_id, filter_path, reason),
            )
            return cursor.rowcount == 1

    async def select(
        self, *, min_version: int, max_version: int, exclude: Iterable[str], limit: int, filter_path: str | None
    ) -> list[GroupRow]:
        """The batch query: fresh, live-lineage, unconsumed groups, oldest first."""
        async with self._lock:
            cursor = await self.connection.execute(
                _SELECT + " LIMIT %s",
                (self.run_id, self.contract_sha256, min_version, max_version, list(exclude), filter_path, limit),
            )
            rows = await cursor.fetchall()
        return [GroupRow(str(r[0]), int(str(r[1])), str(r[2]), str(r[3])) for r in rows]

    async def count(
        self, *, min_version: int, max_version: int, exclude: Iterable[str], filter_path: str | None
    ) -> int:
        async with self._lock:
            cursor = await self.connection.execute(
                "SELECT count(*) FROM (" + _SELECT + ") eligible",
                (self.run_id, self.contract_sha256, min_version, max_version, list(exclude), filter_path),
            )
            row = await cursor.fetchone()
        return 0 if row is None else int(str(row[0]))

    async def _read_payload(self, row: GroupRow) -> bytes:
        path = Path(row.payload_path)
        for attempt in range(2):
            try:
                payload = await asyncio.to_thread(path.read_bytes)
            except FileNotFoundError:
                payload = None
            if payload is not None and hashlib.sha256(payload).hexdigest() == row.payload_sha256:
                return payload
            if attempt == 0:
                # This container's view of the mount may predate the writer's commit.
                await asyncio.to_thread(self.sync.reload)
        raise ValueError(f"Stored payload for group {row.group_id} is missing or fails its checksum")

    async def load(self, row: GroupRow) -> tuple[StoredGroup, list[Sample]]:
        payload = await self._read_payload(row)
        size = int.from_bytes(payload[:8], "big")
        header = StoredGroup.model_validate_json(payload[8 : 8 + size])
        if (
            header.group_id != row.group_id
            or header.policy.version != row.policy_version
            or header.contract_sha256 != self.contract_sha256
        ):
            raise ValueError(f"Stored payload for group {row.group_id} names a different group, policy or contract")
        samples = []
        body = payload[8 + size :]
        reply = decode_samples_and_merge_input_sample(body, Sample(), fields=COMPUTED_FIELDS_V2)
        if len(reply.samples) != len(header.identities):
            raise ValueError("Stored payload sample count differs from its identities")
        for sample, identity in zip(reply.samples, header.identities, strict=True):
            sample.index, sample.group_index = identity.index, identity.group_index
            samples.append(sample)
        return header, samples


def payload_sync(config: RunConfig) -> PayloadSync:
    storage = config.artifact_storage
    if storage.kind == "shared_disk":
        return LOCAL_DISK
    if storage.kind == "modal_volume":
        import modal  # Optional dependency, only for Volume-backed runs.

        volume = modal.Volume.from_name(
            storage.volume.volume_name, environment_name=storage.volume.environment_name, create_if_missing=False
        )
        return PayloadSync(commit=volume.commit, reload=volume.reload)
    assert_never(storage)


async def open_store(config: RunConfig) -> RolloutStore:
    """The one way processes open the store: contract digest and payload sync from config."""
    return await RolloutStore.open(
        secret_env(config.store_dsn_env),
        run_id=config.run_id,
        contract_sha256=digest(training_contract(config)),
        root=config.artifact_directory,
        sync=payload_sync(config),
    )
