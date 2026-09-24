"""CPU-only, immutable serving snapshots from completed PEFT adapter exports.

Checkpoint files are copied as bytes, never deserialized with torch/pickle.
The manifest digest identifies the metadata AND every serving file. This is
artifact integrity, not proof of numerical train/serve equivalence.
"""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, Field(min_length=1)]
AdapterFileName = Literal["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
_WEIGHTS: tuple[AdapterFileName, ...] = ("adapter_model.safetensors", "adapter_model.bin")
_MANIFEST = "manifest.json"


class BaseModelIdentity(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: Nonempty
    revision: Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]


class SnapshotMetadata(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    run_id: Nonempty
    # Miles passes rollout_id to the checkpoint hook; this is NOT an optimizer-step count.
    checkpoint_iteration: Annotated[int, Field(ge=0)]
    base_model: BaseModelIdentity


class SnapshotFile(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: AdapterFileName
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Digest


class SnapshotManifest(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1] = 1
    metadata: SnapshotMetadata
    files: tuple[SnapshotFile, ...]

    @model_validator(mode="after")
    def _check_files(self) -> "SnapshotManifest":
        names = [file.name for file in self.files]
        if len(names) != 2 or names[0] != "adapter_config.json" or names[1] not in _WEIGHTS:
            raise ValueError("Expected adapter_config.json and exactly one unsharded adapter weight file")
        return self


class SnapshotReference(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    sha256: Digest


# PEFT is an external format with additional fields. Narrow the fields we validate
# here; preserve and hash the original bytes, including all other PEFT settings.
class _PEFTConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    peft_type: Literal["LORA"]
    r: Annotated[int, Field(gt=0)]
    base_model_name_or_path: str | None = None


@dataclass(frozen=True)
class PreparedSnapshot:
    directory: Path
    reference: SnapshotReference
    manifest: SnapshotManifest


def manifest_bytes(manifest: SnapshotManifest) -> bytes:
    return (json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n").encode()


def snapshot_relative_path(reference: SnapshotReference) -> str:
    return f"snapshots/{reference.sha256}"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")


def _validate_adapter(directory: Path, base_model: BaseModelIdentity) -> None:
    config = _PEFTConfig.model_validate_json((directory / "adapter_config.json").read_bytes())
    if config.base_model_name_or_path is not None and config.base_model_name_or_path != base_model.name:
        raise ValueError("PEFT base_model_name_or_path conflicts with the declared base model")


def read_snapshot(directory: Path, reference: SnapshotReference) -> PreparedSnapshot:
    """Revalidate the manifest and all file bytes before upload or engine loading."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected a snapshot directory: {directory}")
    _regular_file(directory / _MANIFEST)
    raw = (directory / _MANIFEST).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("Snapshot manifest digest mismatch")
    manifest = SnapshotManifest.model_validate_json(raw)
    if raw != manifest_bytes(manifest):
        raise ValueError("Snapshot manifest is not canonically encoded")
    if {p.name for p in directory.iterdir()} != {_MANIFEST, *(file.name for file in manifest.files)}:
        raise ValueError("Snapshot contains unexpected or missing files")
    for file in manifest.files:
        path = directory / file.name
        _regular_file(path)
        if path.stat().st_size != file.size_bytes or _file_digest(path) != file.sha256:
            raise ValueError(f"Snapshot file integrity mismatch: {file.name}")
    _validate_adapter(directory, manifest.metadata.base_model)
    return PreparedSnapshot(directory=directory, reference=reference, manifest=manifest)


def prepare_snapshot(adapter_directory: Path, *, metadata: SnapshotMetadata, output_root: Path) -> PreparedSnapshot:
    """Copy a completed export into a content-addressed local bundle.

    The caller owns the source checkpoint's lifetime and must finish export first.
    Native resume/optimizer shards are deliberately not serving artifacts.
    """
    if adapter_directory.is_symlink() or not adapter_directory.is_dir():
        raise ValueError("Adapter export directory does not exist or is a symlink")
    weights = [name for name in _WEIGHTS if (adapter_directory / name).exists()]
    if len(weights) != 1:
        raise ValueError("Need exactly one complete PEFT adapter_model.bin or adapter_model.safetensors export")
    names: tuple[AdapterFileName, ...] = ("adapter_config.json", weights[0])
    for name in names:
        _regular_file(adapter_directory / name)
    _validate_adapter(adapter_directory, metadata.base_model)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=output_root) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        files = []
        for name in names:
            shutil.copyfile(adapter_directory / name, staged / name)
            files.append(
                SnapshotFile(name=name, size_bytes=(staged / name).stat().st_size, sha256=_file_digest(staged / name))
            )
        manifest = SnapshotManifest(metadata=metadata, files=tuple(files))
        raw = manifest_bytes(manifest)
        reference = SnapshotReference(sha256=hashlib.sha256(raw).hexdigest())
        (staged / _MANIFEST).write_bytes(raw)
        read_snapshot(staged, reference)
        destination = output_root / snapshot_relative_path(reference)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.rename(staged, destination)
            except OSError:
                # Concurrent identical exports may win the rename. Verify below;
                # a corrupt/partial existing destination is never overwritten.
                if not destination.exists():
                    raise
        return read_snapshot(destination, reference)
