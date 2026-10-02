"""Lossless LoRA deltas using SGLang's per-tensor disk-delta wire format."""

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import httpx
import numpy as np
import pytest
import safetensors.numpy
from tests.fast.proximal_publication.test_publication import _authorization
from tests.fast.proximal_publication.test_publication import volume as _volume

from miles_plugins.proximal.adapter_delta import (
    MAX_DELTA_DEPTH,
    delta_relative_path,
    materialize_snapshot,
    prepare_delta,
    read_delta,
)
from miles_plugins.proximal.modal_volume import modal_publish_delta_snapshot
from miles_plugins.proximal.replica import ReplicaConfig, ReplicaLoRALoader, authorize_replica_load
from miles_plugins.proximal.snapshot import prepare_snapshot, read_snapshot, snapshot_relative_path


volume = _volume


def snapshots(tmp_path, metadata, count=3):
    adapter = tmp_path / "export"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA","r":32}')
    rng = np.random.default_rng(17)
    a = rng.integers(0, 256, 32768, dtype=np.uint8)
    b = rng.integers(0, 256, 32768, dtype=np.uint8)
    result = []
    for i in range(count):
        a[i * 32 : (i + 1) * 32] ^= 1
        safetensors.numpy.save_file(
            {"layer.lora_A.weight": a, "layer.lora_B.weight": b}, str(adapter / "adapter_model.safetensors")
        )
        result.append(
            prepare_snapshot(
                adapter,
                metadata=metadata.model_copy(update={"checkpoint_iteration": i}),
                output_root=tmp_path / "publisher",
            )
        )
    return result


def mount_volume(volume, mount):
    for name, data in volume.files.items():
        path = mount / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_publish_retry_cold_and_warm_replica_replay(tmp_path, metadata, volume):
    first, second, third = snapshots(tmp_path, metadata)
    authorization = _authorization()
    assert modal_publish_delta_snapshot(authorization, first, base=None, depth=1) == 0
    volume.fail_after = len(volume.uploaded) + 1
    with pytest.raises(ConnectionError):
        modal_publish_delta_snapshot(authorization, second, base=first, depth=1)
    assert not any(name.endswith("transport.json") for name in volume.files)
    volume.fail_after = None
    assert modal_publish_delta_snapshot(authorization, second, base=first, depth=1) == 1
    before = len(volume.uploaded)
    assert modal_publish_delta_snapshot(authorization, second, base=first, depth=1) == 1
    assert len(volume.uploaded) == before
    assert modal_publish_delta_snapshot(authorization, third, base=second, depth=2) == 2
    assert volume.uploaded[-1].endswith("transport.json")
    assert not any(
        second.reference.sha256 in path and path.endswith("adapter_model.safetensors") for path in volume.files
    )
    mount = tmp_path / "volume"
    mount_volume(volume, mount)
    loaded = []

    def engine(request):
        body = json.loads(request.content)
        snapshot = third if body["lora_name"].endswith(third.reference.sha256) else second
        verified = read_snapshot(Path(body["lora_path"]), snapshot.reference)
        assert verified.manifest == snapshot.manifest
        loaded.append(snapshot.reference)
        return httpx.Response(200, json={"success": True, "loaded_adapters": {body["lora_name"]: body["lora_path"]}})

    with httpx.Client(transport=httpx.MockTransport(engine)) as client:
        for index, versions in enumerate(((second, third), (third,))):
            replica = ReplicaLoRALoader(
                authorize_replica_load(
                    ReplicaConfig(
                        base_model=metadata.base_model, served_model_name="test", backend_url="http://localhost"
                    ),
                    yes_load=True,
                ),
                volume_mount=mount,
                local_cache=tmp_path / f"replica-{index}",
                reload_volume=lambda: None,
                client=client,
            )
            for snapshot in versions:
                assert replica.ensure_loaded(snapshot.reference).snapshot == snapshot.reference
                replica.ensure_loaded(snapshot.reference)
            read_snapshot(tmp_path / f"replica-{index}" / snapshot_relative_path(first.reference), first.reference)
    assert loaded == [second.reference, third.reference, third.reference]
    assert read_snapshot(first.directory, first.reference) == first


def test_full_fallback_for_restart_layout_change_and_anchor_limit(tmp_path, metadata, volume):
    first, second = snapshots(tmp_path, metadata, count=2)
    assert modal_publish_delta_snapshot(_authorization(), first, base=None, depth=1) == 0
    assert modal_publish_delta_snapshot(_authorization(), second, base=first, depth=MAX_DELTA_DEPTH + 1) == 0
    changed = second.manifest.metadata.model_copy(update={"checkpoint_iteration": 2})
    adapter = tmp_path / "export"
    safetensors.numpy.save_file(
        {"different.weight": np.zeros(100, dtype=np.uint8)}, str(adapter / "adapter_model.safetensors")
    )
    third = prepare_snapshot(adapter, metadata=changed, output_root=tmp_path / "publisher")
    assert modal_publish_delta_snapshot(_authorization(), third, base=second, depth=1) == 0
    assert not any("/deltas/" in path for path in volume.files)


def test_corruption_fails_without_mutating_base_or_creating_target(tmp_path, metadata, volume):
    first, second = snapshots(tmp_path, metadata, count=2)
    modal_publish_delta_snapshot(_authorization(), first, base=None, depth=1)
    modal_publish_delta_snapshot(_authorization(), second, base=first, depth=1)
    mount = tmp_path / "volume"
    mount_volume(volume, mount)
    delta_dir = mount / delta_relative_path(second.reference)
    # Even an internally consistent transport wrapper cannot substitute a bad tensor delta.
    shard = delta_dir / "model-00000-of-00001.safetensors"
    payload = safetensors.numpy.load_file(str(shard))
    with safetensors.safe_open(str(shard), framework="numpy") as source:
        checksums = source.metadata()
    checksums[next(iter(checksums))] = "00000000"
    safetensors.numpy.save_file(payload, str(shard), metadata=checksums)
    marker = delta_dir / "transport.json"
    raw = json.loads(marker.read_bytes())
    raw["files"][1].update(size_bytes=shard.stat().st_size, sha256=hashlib.sha256(shard.read_bytes()).hexdigest())
    marker.write_text(json.dumps(raw))
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="checksum mismatch"):
        materialize_snapshot(mount, cache, second.reference, metadata.base_model)
    assert not (cache / snapshot_relative_path(second.reference)).exists()
    read_snapshot(cache / snapshot_relative_path(first.reference), first.reference)
    # A corrupt base cannot be used, even if its directory has the requested digest name.
    (cache / snapshot_relative_path(first.reference) / "adapter_model.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity mismatch"):
        materialize_snapshot(mount, cache, second.reference, metadata.base_model)


def test_unchanged_tensors_omitted_and_incompressible_delta_falls_back(tmp_path, metadata):
    first, second = snapshots(tmp_path, metadata, count=2)
    delta = prepare_delta(second, first, depth=1, output=tmp_path / "delta")
    assert delta is not None
    index = json.loads((delta.directory / "model.safetensors.index.json").read_bytes())
    assert index["weight_map"] == {"layer.lora_A.weight": "model-00000-of-00001.safetensors"}
    adapter = tmp_path / "export"
    rng = np.random.default_rng(41)
    safetensors.numpy.save_file(
        {name: rng.integers(0, 256, 32768, dtype=np.uint8) for name in ("layer.lora_A.weight", "layer.lora_B.weight")},
        str(adapter / "adapter_model.safetensors"),
    )
    third = prepare_snapshot(
        adapter, metadata=metadata.model_copy(update={"checkpoint_iteration": 2}), output_root=tmp_path / "publisher"
    )
    assert prepare_delta(third, second, depth=2, output=tmp_path / "large-delta") is None
    (delta.directory / "model.safetensors.index.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity mismatch"):
        read_delta(delta.directory, second.reference)


def test_sglang_can_apply_the_lora_delta(tmp_path, metadata):
    """Optional interop check against an actual SGLang checkout, without importing its GPU runtime."""
    source = os.environ.get("SGLANG_LOCAL_CHECKPOINT_SOURCE")
    if source is None:
        pytest.skip("Set SGLANG_LOCAL_CHECKPOINT_SOURCE for the SGLang interoperability check")
    spec = importlib.util.spec_from_file_location("sglang_local_checkpoint", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first, second = snapshots(tmp_path, metadata, count=2)
    source_dir = tmp_path / "sglang-source"
    delta = prepare_delta(second, first, depth=1, output=source_dir / "weight_v000002")
    assert delta is not None
    shutil.copytree(first.directory, source_dir / "weight_v000001")
    local = tmp_path / "sglang-local"
    module.pull(str(local), str(first.directory), str(source_dir), 2)
    expected = (second.directory / "adapter_model.safetensors").read_bytes()
    assert (local / "adapter_model.safetensors").read_bytes() == expected
    module.pull(str(local), str(first.directory), str(source_dir), 2)
    assert (local / "adapter_model.safetensors").read_bytes() == expected


def test_multiple_changed_tensors_produce_identical_retry_bytes(tmp_path, metadata):
    first, second = snapshots(tmp_path, metadata, count=2)
    adapter = tmp_path / "export"
    tensors = safetensors.numpy.load_file(str(adapter / "adapter_model.safetensors"))
    tensors["layer.lora_B.weight"][0] ^= 1
    safetensors.numpy.save_file(tensors, str(adapter / "adapter_model.safetensors"))
    second = prepare_snapshot(adapter, metadata=second.manifest.metadata, output_root=tmp_path / "publisher")
    deltas = [prepare_delta(second, first, depth=1, output=tmp_path / f"delta-{i}") for i in range(8)]
    assert all(delta is not None for delta in deltas)
    assert all(delta.manifest == deltas[0].manifest for delta in deltas)
    assert all(
        (delta.directory / "transport.json").read_bytes() == (deltas[0].directory / "transport.json").read_bytes()
        for delta in deltas
    )


def test_unchanged_adapter_replays_without_tensor_payload(tmp_path, metadata):
    first = snapshots(tmp_path, metadata, count=1)[0]
    second = prepare_snapshot(
        first.directory,
        metadata=first.manifest.metadata.model_copy(update={"checkpoint_iteration": 1}),
        output_root=tmp_path / "publisher",
    )
    mount = tmp_path / "volume"
    shutil.copytree(first.directory, mount / snapshot_relative_path(first.reference))
    delta = prepare_delta(second, first, depth=1, output=mount / delta_relative_path(second.reference))
    assert delta is not None
    assert safetensors.numpy.load_file(str(delta.directory / "model-00000-of-00001.safetensors")) == {}
    assert (
        materialize_snapshot(mount, tmp_path / "cache", second.reference, metadata.base_model).manifest
        == second.manifest
    )
