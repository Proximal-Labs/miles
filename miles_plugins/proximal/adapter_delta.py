"""SGLang disk-delta encoding, materialized as immutable PEFT snapshots."""

import hashlib
import json
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import safetensors.numpy
import zstandard
from pydantic import Field

from miles.utils.disk_delta import _tensor_locations, checksum, make_tensor_reader
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.sharded_snapshot import snapshot_transport
from miles_plugins.proximal.snapshot import (
    BaseModelIdentity,
    Digest,
    PreparedSnapshot,
    SnapshotManifest,
    SnapshotReference,
    _file_digest,
    _regular_file,
    manifest_bytes,
    read_snapshot,
    snapshot_relative_path,
)

# ponytail: bounded replay of seven deltas; tune anchor frequency after network benchmarks.
MAX_DELTA_DEPTH = 7
_SHARD: Literal["model-00000-of-00001.safetensors"] = "model-00000-of-00001.safetensors"
_INDEX: Literal["model.safetensors.index.json"] = "model.safetensors.index.json"
_WEIGHTS = "adapter_model.safetensors"


class DeltaFile(FrozenStrictBaseModel):
    name: Literal["model-00000-of-00001.safetensors", "model.safetensors.index.json"]
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Digest


class DeltaManifest(FrozenStrictBaseModel):
    schema_version: Literal[1] = 1
    base: SnapshotReference
    depth: Annotated[int, Field(ge=1, le=MAX_DELTA_DEPTH)]
    target: SnapshotManifest
    files: Annotated[tuple[DeltaFile, ...], Field(min_length=2, max_length=2)]


@dataclass(frozen=True)
class PreparedDelta:
    directory: Path
    manifest: DeltaManifest


def delta_relative_path(reference: SnapshotReference) -> str:
    return f"deltas/{reference.sha256}"


def _header(path: Path) -> bytes:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        size = struct.unpack("<Q", prefix)[0]
        if size > path.stat().st_size - 8:
            raise ValueError("Invalid safetensors header length")
        return prefix + stream.read(size)


def prepare_delta(
    target: PreparedSnapshot, base: PreparedSnapshot, *, depth: int, output: Path
) -> PreparedDelta | None:
    """Return a smaller, compatible tensor delta, or select a full publication."""
    target = read_snapshot(target.directory, target.reference)
    base = read_snapshot(base.directory, base.reference)
    if not 1 <= depth <= MAX_DELTA_DEPTH:
        return None
    if (
        target.manifest.metadata.base_model != base.manifest.metadata.base_model
        or target.manifest.metadata.run_id != base.manifest.metadata.run_id
        or target.manifest.metadata.checkpoint_iteration != base.manifest.metadata.checkpoint_iteration + 1
        or target.manifest.files[0] != base.manifest.files[0]
        or target.manifest.files[1].name != _WEIGHTS
        or base.manifest.files[1].name != _WEIGHTS
        or target.manifest.files[1].size_bytes != base.manifest.files[1].size_bytes
        or _header(target.directory / _WEIGHTS) != _header(base.directory / _WEIGHTS)
    ):
        return None
    read_base = make_tensor_reader(str(base.directory))
    read_target = make_tensor_reader(str(target.directory))
    deltas = {}
    checksums = {}
    for name in _tensor_locations(str(target.directory)):
        new = read_target(name)
        diff = new ^ read_base(name)
        if np.any(diff):
            deltas[name] = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
            checksums[name] = checksum("adler32", new)
    output.mkdir(parents=True)
    # Safetensors randomizes metadata key order; canonicalize it so retries have identical bytes.
    blob = safetensors.numpy.save(deltas, metadata=checksums or None)
    header_size = struct.unpack("<Q", blob[:8])[0]
    header = json.dumps(json.loads(blob[8 : 8 + header_size]), sort_keys=True, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    with (output / _SHARD).open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(memoryview(blob)[8 + header_size :])
    index = {
        "metadata": {
            "version": f"{target.manifest.metadata.checkpoint_iteration + 1:06d}",
            "base_version": f"{base.manifest.metadata.checkpoint_iteration + 1:06d}",
            "delta_encoding": "xor",
            "compression_format": "zstd",
            "checksum_format": "adler32",
        },
        "weight_map": dict.fromkeys(deltas, _SHARD),
    }
    (output / _INDEX).write_text(json.dumps(index, sort_keys=True))
    manifest = DeltaManifest(
        base=base.reference,
        depth=depth,
        target=target.manifest,
        files=tuple(
            DeltaFile(name=name, size_bytes=(output / name).stat().st_size, sha256=_file_digest(output / name))
            for name in (_INDEX, _SHARD)
        ),
    )
    raw = manifest.model_dump_json().encode()
    if sum(file.size_bytes for file in manifest.files) + len(raw) >= target.manifest.files[1].size_bytes:
        return None
    (output / "transport.json").write_bytes(raw)
    return PreparedDelta(output, manifest)


def read_delta(directory: Path, reference: SnapshotReference) -> PreparedDelta:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Expected a delta directory")
    _regular_file(directory / "transport.json")
    manifest = DeltaManifest.model_validate_json((directory / "transport.json").read_bytes())
    if hashlib.sha256(manifest_bytes(manifest.target)).hexdigest() != reference.sha256:
        raise ValueError("Delta target manifest digest mismatch")
    if tuple(file.name for file in manifest.files) != (_INDEX, _SHARD):
        raise ValueError("Invalid delta file list")
    if {p.name for p in directory.iterdir()} != {"transport.json", _INDEX, _SHARD}:
        raise ValueError("Unexpected or missing delta files")
    for file in manifest.files:
        path = directory / file.name
        _regular_file(path)
        if path.stat().st_size != file.size_bytes or _file_digest(path) != file.sha256:
            raise ValueError("Delta file integrity mismatch")
    return PreparedDelta(directory, manifest)


def _apply_delta(delta: PreparedDelta, base: PreparedSnapshot, staged: Path) -> None:
    """SGLang's XOR/Zstd wire format, applied only to a disposable copy of the base."""
    target = delta.manifest.target
    if (
        target.metadata.base_model != base.manifest.metadata.base_model
        or target.metadata.run_id != base.manifest.metadata.run_id
        or target.metadata.checkpoint_iteration != base.manifest.metadata.checkpoint_iteration + 1
        or target.files[0] != base.manifest.files[0]
        or target.files[1].name != _WEIGHTS
        or base.manifest.files[1].name != _WEIGHTS
        or target.files[1].size_bytes != base.manifest.files[1].size_bytes
    ):
        raise ValueError("Incompatible delta base")
    index = json.loads((delta.directory / _INDEX).read_bytes())
    tensors = safetensors.numpy.load_file(str(delta.directory / _SHARD))
    expected_metadata = {
        "version": f"{target.metadata.checkpoint_iteration + 1:06d}",
        "base_version": f"{base.manifest.metadata.checkpoint_iteration + 1:06d}",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "adler32",
    }
    if index != {"metadata": expected_metadata, "weight_map": dict.fromkeys(tensors, _SHARD)}:
        raise ValueError("Invalid delta index or version order")
    with safetensors.safe_open(str(delta.directory / _SHARD), framework="numpy") as shard:
        checksums = shard.metadata() or {}
    if set(checksums) != set(tensors):
        raise ValueError("Missing delta tensor checksums")
    locations = _tensor_locations(str(staged))
    for name, compressed in tensors.items():
        if name not in locations or compressed.dtype != np.uint8 or compressed.ndim != 1:
            raise ValueError("Invalid delta tensor")
        path, offset, size, _, _ = locations[name]
        # Check the declared frame size before allocating; max_output_size alone is not a bound.
        compressed_bytes = compressed.tobytes()
        if zstandard.frame_content_size(compressed_bytes) != size:
            raise ValueError("Delta decompressed size mismatch")
        diff = zstandard.ZstdDecompressor().decompress(compressed_bytes, max_output_size=size, allow_extra_data=False)
        with open(path, "r+b") as stream:
            stream.seek(offset)
            new = np.frombuffer(stream.read(size), dtype=np.uint8) ^ np.frombuffer(diff, dtype=np.uint8)
            if checksum("adler32", new) != checksums[name]:
                raise ValueError("Delta tensor checksum mismatch")
            stream.seek(offset)
            stream.write(new.tobytes())
    (staged / "manifest.json").write_bytes(manifest_bytes(target))


def materialize_snapshot(
    mount: Path,
    cache: Path,
    reference: SnapshotReference,
    base_model: BaseModelIdentity,
    *,
    remaining: int = MAX_DELTA_DEPTH,
) -> PreparedSnapshot:
    """Resolve a cold replica from a full anchor; validate before an atomic cache rename."""
    destination = cache / snapshot_relative_path(reference)
    if destination.exists():
        cached = read_snapshot(destination, reference)
        if cached.manifest.metadata.base_model != base_model:
            raise ValueError("Cached snapshot base model does not match this replica")
        return cached
    with snapshot_transport(mount, cache, reference) as (kind, directory):
        delta = None
        if kind == "snapshot":
            source = read_snapshot(directory, reference)
        else:
            if remaining <= 0:
                raise ValueError("Delta chain exceeds replay limit")
            delta = read_delta(directory, reference)
            if delta.manifest.depth > remaining or delta.manifest.target.metadata.base_model != base_model:
                raise ValueError("Invalid delta depth or base model")
            source = materialize_snapshot(mount, cache, delta.manifest.base, base_model, remaining=remaining - 1)
        if source.manifest.metadata.base_model != base_model:
            raise ValueError("Snapshot base model does not match this replica")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".load-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "bundle"
            shutil.copytree(source.directory, staged)
            if delta is not None:
                _apply_delta(delta, source, staged)
            read_snapshot(staged, reference)
            staged.rename(destination)
    return read_snapshot(destination, reference)
