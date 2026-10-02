"""Publish validated snapshots to an existing Modal Volume, with manifest last."""

import hashlib
import io
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ConfigDict

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.adapter_delta import delta_relative_path, prepare_delta, read_delta
from miles_plugins.proximal.sharded_snapshot import ShardedSnapshot, part_relative_path, sharded_relative_path
from miles_plugins.proximal.snapshot import (
    Nonempty,
    PreparedSnapshot,
    SnapshotReference,
    read_snapshot,
    snapshot_relative_path,
)

if TYPE_CHECKING:
    import modal

_AUTHORITY = object()


class VolumeDestination(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    volume_name: Nonempty
    environment_name: Nonempty


@dataclass(frozen=True, init=False)
class AuthorizedVolumePublication:
    destination: VolumeDestination

    def __init__(self, destination: VolumeDestination, *, _authority: object):
        if _authority is not _AUTHORITY:
            raise PermissionError("Use authorize_volume_publication with explicit publication consent")
        object.__setattr__(self, "destination", destination)


def authorize_volume_publication(destination: VolumeDestination, *, yes_publish: bool) -> AuthorizedVolumePublication:
    if yes_publish is not True:
        raise PermissionError("Publishing requires --yes-publish")
    return AuthorizedVolumePublication(destination, _authority=_AUTHORITY)


class PublishedSnapshot(FrozenStrictBaseModel):
    destination: VolumeDestination
    snapshot: SnapshotReference


def _existing_file_matches(volume: "modal.Volume", path: str, *, sha256: str) -> bool:
    digest = hashlib.sha256()
    try:
        for chunk in volume.read_file(path):
            digest.update(chunk)
    except FileNotFoundError:
        return False
    if digest.hexdigest() != sha256:
        raise ValueError(f"Refusing to replace conflicting immutable Volume file: {path}")
    return True


def _publish_files(
    authorization: AuthorizedVolumePublication,
    directory: Path,
    remote_root: str,
    files: Sequence[tuple[str, str]],
    marker: tuple[str, str],
) -> None:
    # Modal is an optional SDK boundary; preparation must work without it.
    import modal

    destination = authorization.destination
    volume = modal.Volume.from_name(
        destination.volume_name, environment_name=destination.environment_name, create_if_missing=False
    )
    missing = [
        (name, sha) for name, sha in files if not _existing_file_matches(volume, f"{remote_root}/{name}", sha256=sha)
    ]
    if missing:
        with volume.batch_upload(force=False) as upload:
            for name, _ in missing:
                upload.put_file(directory / name, f"{remote_root}/{name}")
        for name, sha in missing:
            if not _existing_file_matches(volume, f"{remote_root}/{name}", sha256=sha):
                raise ValueError(f"Uploaded serving file is not visible: {name}")
    # Completion becomes visible only after every payload file has committed and verified.
    name, sha = marker
    if not _existing_file_matches(volume, f"{remote_root}/{name}", sha256=sha):
        with volume.batch_upload(force=False) as upload:
            upload.put_file(directory / name, f"{remote_root}/{name}")
        if not _existing_file_matches(volume, f"{remote_root}/{name}", sha256=sha):
            raise ValueError("Uploaded completion manifest is not visible")


def modal_publish_snapshot(
    authorization: AuthorizedVolumePublication, snapshot: PreparedSnapshot
) -> PublishedSnapshot:
    """Upload only missing files; retries verify existing bytes, never overwrite.

    A returned artifact reference says nothing about replica load/readiness.
    Concurrent publishers of the same snapshot may fail on create; retrying is safe.
    """
    if not isinstance(authorization, AuthorizedVolumePublication):
        raise PermissionError("Expected an authorized Volume publication")
    snapshot = read_snapshot(snapshot.directory, snapshot.reference)
    _publish_files(
        authorization,
        snapshot.directory,
        "/" + snapshot_relative_path(snapshot.reference),
        [(file.name, file.sha256) for file in snapshot.manifest.files],
        ("manifest.json", snapshot.reference.sha256),
    )
    return PublishedSnapshot(destination=authorization.destination, snapshot=snapshot.reference)


def modal_publish_delta_snapshot(
    authorization: AuthorizedVolumePublication,
    snapshot: PreparedSnapshot,
    *,
    base: PreparedSnapshot | None,
    depth: int,
) -> int:
    """Return the published chain depth; zero denotes a complete anchor snapshot."""
    if not isinstance(authorization, AuthorizedVolumePublication):
        raise PermissionError("Expected an authorized Volume publication")
    with tempfile.TemporaryDirectory(prefix=".delta-", dir=snapshot.directory.parent) as temporary:
        delta = None if base is None else prepare_delta(snapshot, base, depth=depth, output=Path(temporary) / "bundle")
        if delta is None:
            modal_publish_snapshot(authorization, snapshot)
            return 0
        delta = read_delta(delta.directory, snapshot.reference)
        marker_sha = hashlib.sha256((delta.directory / "transport.json").read_bytes()).hexdigest()
        _publish_files(
            authorization,
            delta.directory,
            "/" + delta_relative_path(snapshot.reference),
            [(file.name, file.sha256) for file in delta.manifest.files],
            ("transport.json", marker_sha),
        )
        return depth


def modal_publish_shard(authorization: AuthorizedVolumePublication, snapshot: ShardedSnapshot, *, rank: int) -> None:
    """Each rank uploads and verifies only its assigned byte ranges."""
    if not isinstance(authorization, AuthorizedVolumePublication):
        raise PermissionError("Expected an authorized Volume publication")
    if not 0 <= rank < snapshot.manifest.world_size:
        raise ValueError("Invalid upload rank")
    import modal

    volume = modal.Volume.from_name(
        authorization.destination.volume_name,
        environment_name=authorization.destination.environment_name,
        create_if_missing=False,
    )
    root = "/" + sharded_relative_path(snapshot.manifest.snapshot)
    missing = []
    for name, parts in snapshot.manifest.files.items():
        for part in parts:
            if part.rank == rank:
                remote = f"{root}/{part_relative_path(name, part)}"
                if not _existing_file_matches(volume, remote, sha256=part.sha256):
                    missing.append((name, part, remote))
    if missing:
        with volume.batch_upload(force=False) as upload:
            for name, part, remote in missing:
                with (snapshot.directory / name).open("rb") as stream:
                    stream.seek(part.offset)
                    data = stream.read(part.size_bytes)
                if len(data) != part.size_bytes or hashlib.sha256(data).hexdigest() != part.sha256:
                    raise ValueError("Shared staging file does not match this rank's assignment")
                upload.put_file(io.BytesIO(data), remote)
        for _, part, remote in missing:
            if not _existing_file_matches(volume, remote, sha256=part.sha256):
                raise ValueError("Uploaded part is not visible")


def modal_complete_sharded(authorization: AuthorizedVolumePublication, snapshot: ShardedSnapshot) -> None:
    """Caller must first collect successful upload/readback verdicts from every rank."""
    if not isinstance(authorization, AuthorizedVolumePublication):
        raise PermissionError("Expected an authorized Volume publication")
    raw = snapshot.marker.read_bytes()
    if raw != snapshot.manifest.model_dump_json().encode():
        raise ValueError("Sharded publication marker differs from the assigned manifest")
    _publish_files(
        authorization,
        snapshot.marker.parent,
        "/" + sharded_relative_path(snapshot.manifest.snapshot),
        [],
        ("parts.json", hashlib.sha256(raw).hexdigest()),
    )
