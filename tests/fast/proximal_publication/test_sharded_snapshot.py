import json

import pytest

from miles_plugins.proximal.sharded_snapshot import (
    ShardedManifest,
    part_relative_path,
    prepare_sharded,
    restore_sharded,
)
from miles_plugins.proximal.snapshot import prepare_snapshot, read_snapshot


def test_full_shards_reconstruct_and_reject_corruption(tmp_path, adapter, metadata):
    full = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "export")
    plan = prepare_sharded(full.directory, full.reference, world_size=4)
    root = tmp_path / "received"
    directory = root / "sharded" / full.reference.sha256
    directory.mkdir(parents=True)
    paths = []
    for name, parts in plan.files.items():
        for part in parts:
            path = directory / part_relative_path(name, part)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((full.directory / name).read_bytes()[part.offset : part.offset + part.size_bytes])
            paths.append(path)
    restore_sharded(root, plan, tmp_path / "restored")
    assert read_snapshot(tmp_path / "restored", full.reference).manifest == full.manifest
    paths[0].write_bytes(b"x" * paths[0].stat().st_size)
    with pytest.raises(ValueError, match="integrity"):
        restore_sharded(root, plan, tmp_path / "bad")
    for key, value in (("offset", 1), ("rank", 4)):
        raw = json.loads(plan.model_dump_json())
        raw["files"]["adapter_model.bin"][0][key] = value
        with pytest.raises(ValueError, match="assignment"):
            ShardedManifest.model_validate_json(json.dumps(raw))
