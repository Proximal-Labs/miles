"""Byte ranges for full-snapshot HTTP uploads from training ranks."""

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.snapshot import Digest, SnapshotReference, _regular_file

PayloadFile = Literal[
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "manifest.json",
]


class Part(FrozenStrictBaseModel):
    offset: Annotated[int, Field(ge=0)]
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Digest
    rank: Annotated[int, Field(ge=0)]


class ShardedManifest(FrozenStrictBaseModel):
    schema_version: Literal[1] = 1
    snapshot: SnapshotReference
    world_size: Annotated[int, Field(ge=1, le=4096)]
    files: dict[PayloadFile, tuple[Part, ...]]

    @model_validator(mode="after")
    def _validate_parts(self) -> "ShardedManifest":
        names = set(self.files)
        full_names = {"adapter_config.json", "manifest.json"}
        valid = names in (full_names | {"adapter_model.safetensors"}, full_names | {"adapter_model.bin"})
        if not valid:
            raise ValueError("Invalid sharded payload file set")
        for parts in self.files.values():
            if not parts or len(parts) > self.world_size:
                raise ValueError("Missing file parts")
            offset = 0
            for part in parts:
                if part.offset != offset or part.rank >= self.world_size:
                    raise ValueError("Overlapping, missing or invalid shard assignment")
                offset += part.size_bytes
        return self


def sharded_relative_path(reference: SnapshotReference) -> str:
    return f"sharded/{reference.sha256}"


def prepare_sharded(directory: Path, reference: SnapshotReference, *, world_size: int) -> ShardedManifest:
    """Balance each file over all upload ranks; retain canonical snapshot bytes."""
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
    return ShardedManifest.model_validate(dict(snapshot=reference, world_size=world_size, files=files))


def part_relative_path(name: str, part: Part) -> str:
    return f"parts/{name}/{part.rank}-{part.sha256}"


def restore_sharded(mount: Path, manifest: ShardedManifest, output: Path) -> None:
    source = mount / sharded_relative_path(manifest.snapshot)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Expected a sharded snapshot directory")
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
