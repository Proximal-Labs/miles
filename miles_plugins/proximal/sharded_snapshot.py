"""Byte shards for parallel upload from ranks sharing one node's staging files."""

import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.snapshot import Digest, SnapshotReference, _regular_file, snapshot_relative_path

PayloadKind = Literal["snapshot", "delta"]
PayloadFile = Literal[
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "manifest.json",
    "model-00000-of-00001.safetensors",
    "model.safetensors.index.json",
    "transport.json",
]


class Part(FrozenStrictBaseModel):
    offset: Annotated[int, Field(ge=0)]
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Digest
    rank: Annotated[int, Field(ge=0)]


class ShardedManifest(FrozenStrictBaseModel):
    schema_version: Literal[1] = 1
    kind: PayloadKind
    snapshot: SnapshotReference
    world_size: Annotated[int, Field(ge=1)]
    files: dict[PayloadFile, tuple[Part, ...]]

    @model_validator(mode="after")
    def _validate_parts(self) -> "ShardedManifest":
        names = set(self.files)
        full_names = {"adapter_config.json", "manifest.json"}
        if self.kind == "snapshot":
            valid = names in (full_names | {"adapter_model.safetensors"}, full_names | {"adapter_model.bin"})
        else:
            valid = names == {"model-00000-of-00001.safetensors", "model.safetensors.index.json", "transport.json"}
        if not valid:
            raise ValueError("Invalid sharded payload file set")
        for parts in self.files.values():
            if not parts:
                raise ValueError("Missing file parts")
            offset = 0
            for part in parts:
                if part.offset != offset or part.rank >= self.world_size:
                    raise ValueError("Overlapping, missing or invalid shard assignment")
                offset += part.size_bytes
        return self


@dataclass(frozen=True)
class ShardedSnapshot:
    directory: Path
    manifest: ShardedManifest
    marker: Path


def sharded_relative_path(reference: SnapshotReference) -> str:
    return f"sharded/{reference.sha256}"


def prepare_sharded(
    directory: Path, reference: SnapshotReference, *, kind: PayloadKind, world_size: int, marker: Path
) -> ShardedSnapshot:
    """Balance each file over all upload ranks; retain canonical snapshot/delta bytes."""
    if world_size < 1:
        raise ValueError("Need at least one upload rank")
    files = {}
    for path in sorted(directory.iterdir()):
        _regular_file(path)
        size = path.stat().st_size
        # Each nonempty file has at most world_size parts; small files can leave ranks idle.
        chunk_size = max(1, (size + world_size - 1) // world_size)
        parts = []
        with path.open("rb") as stream:
            for rank in range(world_size):
                offset = stream.tell()
                data = stream.read(chunk_size)
                if data:
                    parts.append(
                        dict(offset=offset, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), rank=rank)
                    )
        files[path.name] = parts
    manifest = ShardedManifest.model_validate(dict(kind=kind, snapshot=reference, world_size=world_size, files=files))
    marker.write_text(manifest.model_dump_json())
    return ShardedSnapshot(directory, manifest, marker)


def part_relative_path(name: str, part: Part) -> str:
    return f"parts/{name}/{part.rank}-{part.sha256}"


def _restore_sharded(mount: Path, reference: SnapshotReference, output: Path) -> PayloadKind:
    source = mount / sharded_relative_path(reference)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Expected a sharded snapshot directory")
    _regular_file(source / "parts.json")
    manifest = ShardedManifest.model_validate_json((source / "parts.json").read_bytes())
    if manifest.snapshot != reference:
        raise ValueError("Sharded snapshot reference mismatch")
    output.mkdir()
    for name, parts in manifest.files.items():
        with (output / name).open("wb") as destination:
            for part in parts:
                path = source / part_relative_path(name, part)
                _regular_file(path)
                if not path.resolve().is_relative_to(source.resolve()) or path.stat().st_size != part.size_bytes:
                    raise ValueError("Invalid snapshot part path or size")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        destination.write(block)
                if digest.hexdigest() != part.sha256:
                    raise ValueError("Snapshot part integrity mismatch")
    return manifest.kind


@contextmanager
def snapshot_transport(mount: Path, cache: Path, reference: SnapshotReference) -> Iterator[tuple[PayloadKind, Path]]:
    full = mount / snapshot_relative_path(reference)
    delta = mount / f"deltas/{reference.sha256}"
    if (full / "manifest.json").exists():
        yield "snapshot", full
    elif (delta / "transport.json").exists():
        yield "delta", delta
    else:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".shards-", dir=cache) as temporary:
            directory = Path(temporary) / "bundle"
            kind = _restore_sharded(mount, reference, directory)
            yield kind, directory
