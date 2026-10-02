import json

import pytest
from tests.fast.proximal_publication.test_publication import _authorization
from tests.fast.proximal_publication.test_publication import volume as _volume

from miles_plugins.proximal.adapter_delta import materialize_snapshot
from miles_plugins.proximal.modal_volume import modal_complete_sharded, modal_publish_shard
from miles_plugins.proximal.sharded_snapshot import ShardedManifest, prepare_sharded, sharded_relative_path
from miles_plugins.proximal.snapshot import prepare_snapshot, snapshot_relative_path

volume = _volume


def test_shard_integrity_and_incomplete_publication_fail_before_cache_install(tmp_path, adapter, metadata, volume):
    (adapter / "adapter_model.bin").write_bytes(bytes(range(256)) * 4096)
    full = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "export")
    plan = prepare_sharded(
        full.directory, full.reference, kind="snapshot", world_size=8, marker=tmp_path / "parts.json"
    )
    assert [part.rank for part in plan.manifest.files["adapter_model.bin"]] == list(range(8))
    for rank in range(8):
        modal_publish_shard(_authorization(), plan, rank=rank)
    assert not any(path.endswith("parts.json") for path in volume.files)
    mount = tmp_path / "volume"

    def refresh():
        for name, raw in volume.files.items():
            path = mount / name.lstrip("/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

    refresh()
    with pytest.raises(ValueError, match="regular"):
        materialize_snapshot(mount, tmp_path / "cache", full.reference, metadata.base_model)
    assert not (tmp_path / "cache" / snapshot_relative_path(full.reference)).exists()
    modal_complete_sharded(_authorization(), plan)
    refresh()
    cached = materialize_snapshot(mount, tmp_path / "cache", full.reference, metadata.base_model)
    assert cached.manifest == full.manifest
    assert (cached.directory / "adapter_model.bin").read_bytes() == (adapter / "adapter_model.bin").read_bytes()
    part = next(
        path
        for path in (mount / sharded_relative_path(full.reference)).rglob("*")
        if path.is_file() and "adapter_model.bin" in path.parts
    )
    original = part.read_bytes()
    part.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="integrity"):
        materialize_snapshot(mount, tmp_path / "corrupt-cache", full.reference, metadata.base_model)
    assert not (tmp_path / "corrupt-cache" / snapshot_relative_path(full.reference)).exists()


def test_sharded_manifest_rejects_gaps_and_bad_rank_ownership(tmp_path, adapter, metadata):
    full = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "export")
    plan = prepare_sharded(
        full.directory, full.reference, kind="snapshot", world_size=8, marker=tmp_path / "parts.json"
    )
    for key, value in (("offset", 1), ("rank", 8)):
        raw = json.loads(plan.manifest.model_dump_json())
        raw["files"]["adapter_model.bin"][0][key] = value
        with pytest.raises(ValueError, match="assignment"):
            ShardedManifest.model_validate_json(json.dumps(raw))
    with pytest.raises(PermissionError):
        modal_publish_shard(object(), plan, rank=0)
    with pytest.raises(PermissionError):
        modal_complete_sharded(object(), plan)
