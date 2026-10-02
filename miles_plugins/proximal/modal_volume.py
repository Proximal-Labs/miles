"""Publish validated snapshots to an existing Modal Volume, with manifest last."""

import hashlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ConfigDict

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.snapshot import (
    Nonempty,
    PreparedSnapshot,
    SnapshotReference,
    manifest_bytes,
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
    # Modal is an optional SDK boundary; local preparation must work without it.
    import modal

    destination = authorization.destination
    volume = modal.Volume.from_name(
        destination.volume_name, environment_name=destination.environment_name, create_if_missing=False
    )
    remote_root = "/" + snapshot_relative_path(snapshot.reference)
    missing = [
        file
        for file in snapshot.manifest.files
        if not _existing_file_matches(volume, f"{remote_root}/{file.name}", sha256=file.sha256)
    ]
    if missing:
        with volume.batch_upload(force=False) as upload:
            for file in missing:
                upload.put_file(snapshot.directory / file.name, f"{remote_root}/{file.name}")
        for file in missing:
            if not _existing_file_matches(volume, f"{remote_root}/{file.name}", sha256=file.sha256):
                raise ValueError(f"Uploaded serving file is not visible: {file.name}")
    # Completion marker is uploaded only after the serving files have committed.
    if not _existing_file_matches(volume, f"{remote_root}/manifest.json", sha256=snapshot.reference.sha256):
        with volume.batch_upload(force=False) as upload:
            upload.put_file(io.BytesIO(manifest_bytes(snapshot.manifest)), f"{remote_root}/manifest.json")
        if not _existing_file_matches(volume, f"{remote_root}/manifest.json", sha256=snapshot.reference.sha256):
            raise ValueError("Uploaded completion manifest is not visible")
    return PublishedSnapshot(destination=destination, snapshot=snapshot.reference)
