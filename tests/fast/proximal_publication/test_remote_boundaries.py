"""Guard the small set of authorized remote mutation sites in this integration."""

import ast
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from miles_plugins.proximal.modal_volume import VolumeDestination, modal_publish_snapshot
from miles_plugins.proximal.replica import ReplicaConfig, ReplicaLoRALoader
from miles_plugins.proximal.snapshot import prepare_snapshot


def test_raw_destination_does_not_authorize_publication(tmp_path, adapter, metadata):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "out")
    forged = SimpleNamespace(destination=VolumeDestination(volume_name="v", environment_name="dev"))
    with pytest.raises(PermissionError, match="authorized"):
        modal_publish_snapshot(forged, snapshot)


def test_raw_config_does_not_authorize_engine_mutation(tmp_path, metadata):
    config = ReplicaConfig(
        base_model=metadata.base_model, served_model_name="served", backend_url="http://127.0.0.1:1"
    )
    with httpx.Client(trust_env=False) as client:
        with pytest.raises(PermissionError, match="authorized"):
            ReplicaLoRALoader(
                SimpleNamespace(config=config),
                volume_mount=tmp_path / "volume",
                local_cache=tmp_path / "cache",
                reload_volume=lambda: pytest.fail("No reload without authorization"),
                client=client,
            )


def test_remote_mutations_stay_in_authorized_adapters():
    plugin = Path(__file__).resolve().parents[3] / "miles_plugins" / "proximal"
    mutations = []
    for path in plugin.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            assert method not in {"deploy", "spawn", "remote", "ephemeral", "remove_file", "unload_lora_adapter"}
            if method in {"batch_upload", "post"}:
                mutations.append((path.name, method))
            if method == "from_name":
                [create] = [kw.value for kw in node.keywords if kw.arg == "create_if_missing"]
                assert isinstance(create, ast.Constant) and create.value is False
    assert sorted(mutations) == [
        ("modal_volume.py", "batch_upload"),
        ("modal_volume.py", "batch_upload"),
        ("replica.py", "post"),
    ]
