"""CPU-only fixtures independent of the root GPU/Ray fixture imports."""

import json

import pytest

from miles_plugins.proximal.snapshot import BaseModelIdentity, SnapshotMetadata


@pytest.fixture
def metadata():
    return SnapshotMetadata(
        run_id="test-run",
        checkpoint_iteration=3,
        base_model=BaseModelIdentity(name="test/base", revision="a" * 40),
    )


@pytest.fixture
def adapter(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 64}))
    # Opaque fixture bytes exercise transport/integrity, not model compatibility.
    (path / "adapter_model.bin").write_bytes(b"adapter version A")
    (path / "adapter_megatron_rank0.pt").write_bytes(b"native resume data must not be uploaded")
    return path
