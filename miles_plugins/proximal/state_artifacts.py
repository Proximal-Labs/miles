"""The run's acknowledged artifact outbox, on its existing local Postgres.

Only the composition root drains it. Payload commit precedes completion-record
commit; acknowledgement follows both. Workers retain their already-paid result
while waiting, so a lost commit acknowledgement never resubmits inference.
"""

import hashlib
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import AfterValidator, Field

from miles_plugins.proximal.contracts import Contract, SafeId
from miles_plugins.proximal.snapshot import Digest

logger = logging.getLogger(__name__)
SCHEMA = """
CREATE TABLE IF NOT EXISTS proximal_artifact_outbox (
    training_run_id text NOT NULL,
    record_id text NOT NULL,
    record jsonb NOT NULL,
    committed boolean NOT NULL DEFAULT false,
    enqueued_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (training_run_id, record_id)
);
CREATE INDEX IF NOT EXISTS proximal_artifact_outbox_pending
ON proximal_artifact_outbox (training_run_id, enqueued_at) WHERE NOT committed;
"""


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {"..", ".", ""} for p in value.split("/")) or "\\" in value:
        raise ValueError("Artifact path must be a normalized relative path")
    return value


RelativePath = Annotated[str, AfterValidator(_relative)]


class StateFile(Contract):
    path: RelativePath
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: Digest


class ArtifactRecord(Contract):
    run_id: SafeId
    record_id: SafeId
    payload: StateFile
    completion: StateFile


def describe(path: Path, *, relative: str) -> StateFile:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular artifact: {path}")
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return StateFile(path=relative, size_bytes=path.stat().st_size, sha256=sha)


def verify(path: Path, file: StateFile) -> None:
    if describe(path, relative=file.path) != file:
        raise ValueError(f"Artifact fails its checksum/size: {path}")


def copy_verified(source: Path, target: Path, file: StateFile) -> None:
    """Bounded-memory copy and rename; never treat a partial old file as complete.

    This deliberately does not use local write_immutable's hardlink on a Volume.
    A committed conflicting destination fails closed. Uncommitted temporary files
    are invisible to consumers and can be retried.
    """
    if target.exists() or target.is_symlink():
        verify(target, file)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".copy-", delete=False) as output:
        temporary = Path(output.name)
        try:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
            verify(temporary, file)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def initialize(dsn: str) -> None:
    import psycopg  # Only the trainer requires the database driver.

    with psycopg.connect(dsn) as connection:
        connection.execute(SCHEMA)


def publish_pending(
    *,
    dsn: str,
    run_id: str,
    artifacts: Path,
    snapshot_root: Path,
    commit: Callable[[], None],
    batch_bytes: int = 256 * 1024 * 1024,
) -> int:
    """One bounded batch. A single larger record is streamed alone, never buffered.

    Caller serializes this with checkpoint commits. Rows are acknowledged only
    after both explicit commits succeed. Retrying identical records is harmless.
    """
    import psycopg  # Optional on the local launch/config-validation host.

    with psycopg.connect(dsn) as connection:
        rows = connection.execute(
            "SELECT record::text FROM proximal_artifact_outbox WHERE training_run_id = %s AND NOT committed"
            " ORDER BY enqueued_at, record_id LIMIT 64",
            (run_id,),
        ).fetchall()
        records: list[ArtifactRecord] = []
        size = 0
        for row in rows:
            record = ArtifactRecord.model_validate_json(row[0])
            if record.run_id != run_id:
                raise ValueError("Outbox record belongs to another run")
            addition = record.payload.size_bytes + record.completion.size_bytes
            if records and size + addition > batch_bytes:
                break
            records.append(record)
            size += addition
        if not records:
            return 0
        for completion in (False, True):
            for record in records:
                file = record.completion if completion else record.payload
                # The paths are relative to this run, never to the whole Volume.
                copy_verified(artifacts / run_id / file.path, snapshot_root / "artifacts" / run_id / file.path, file)
            commit()
        for record in records:
            connection.execute(
                "UPDATE proximal_artifact_outbox SET committed = true WHERE training_run_id = %s AND record_id = %s",
                (run_id, record.record_id),
            )
        logger.info("Committed %d run-state artifacts (%d bytes) for %s", len(records), size, run_id)
        return len(records)


def backlog(dsn: str, run_id: str) -> tuple[int, int, float]:
    """Count, queued bytes and oldest age; not a count of training-eligible groups."""
    import psycopg

    with psycopg.connect(dsn) as connection:
        row = connection.execute(
            "SELECT count(*), coalesce(sum((record->'payload'->>'size_bytes')::bigint + "
            "(record->'completion'->>'size_bytes')::bigint), 0), "
            "coalesce(extract(epoch FROM clock_timestamp() - min(enqueued_at)), 0) "
            "FROM proximal_artifact_outbox WHERE training_run_id = %s AND NOT committed",
            (run_id,),
        ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1]), float(row[2])
