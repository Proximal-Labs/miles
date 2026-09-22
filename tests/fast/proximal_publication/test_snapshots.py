import argparse
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from miles.utils.function_registry import load_function
from miles_plugins.proximal.checkpoint import SnapshotExportConfig, proximal_snapshot_post_save
from miles_plugins.proximal.snapshot import BaseModelIdentity, SnapshotMetadata, prepare_snapshot, read_snapshot


def test_snapshots_are_content_addressed_and_exclude_resume_state(tmp_path, adapter, metadata):
    first = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    again = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    assert first == again
    assert {p.name for p in first.directory.iterdir()} == {"manifest.json", "adapter_config.json", "adapter_model.bin"}
    (adapter / "adapter_model.bin").write_bytes(b"adapter version B")
    second = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    assert first.reference != second.reference
    assert (first.directory / "adapter_model.bin").read_bytes() == b"adapter version A"
    changed_metadata = metadata.model_copy(update={"checkpoint_iteration": 4})
    third = prepare_snapshot(adapter, metadata=changed_metadata, output_root=tmp_path / "exports")
    assert third.reference != second.reference


@pytest.mark.parametrize("file", ["adapter_model.bin", "manifest.json"])
def test_corrupt_existing_bundle_is_rejected_not_replaced(tmp_path, adapter, metadata, file):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    (snapshot.directory / file).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    assert (snapshot.directory / file).read_bytes() == b"corrupt"


def test_missing_export_does_not_fall_back_to_native_checkpoint(tmp_path, adapter, metadata):
    (adapter / "adapter_model.bin").unlink()
    with pytest.raises(ValueError, match="complete PEFT"):
        prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    assert not (tmp_path / "exports").exists()


def test_ambiguous_weights_and_symlinks_are_rejected(tmp_path, adapter, metadata):
    (adapter / "adapter_model.safetensors").write_bytes(b"another representation")
    with pytest.raises(ValueError, match="exactly one"):
        prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    (adapter / "adapter_model.safetensors").unlink()
    (adapter / "adapter_model.bin").unlink()
    (adapter / "adapter_model.bin").symlink_to(adapter / "adapter_megatron_rank0.pt")
    with pytest.raises(ValueError, match="non-symlink"):
        prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")


def test_base_model_conflict_and_empty_weights_fail_before_output(tmp_path, adapter, metadata):
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 64, "base_model_name_or_path": "different/base"})
    )
    with pytest.raises(ValueError, match="base model"):
        prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    assert not (tmp_path / "exports").exists()
    (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 64}))
    (adapter / "adapter_model.bin").write_bytes(b"")
    with pytest.raises(ValidationError):
        prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")


def test_snapshot_rejects_unlisted_files(tmp_path, adapter, metadata):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "exports")
    (snapshot.directory / "injected.py").write_text("raise RuntimeError('never executed')")
    with pytest.raises(ValueError, match="unexpected"):
        read_snapshot(snapshot.directory, snapshot.reference)


def test_required_identity_is_strict_and_immutable(metadata):
    with pytest.raises(ValidationError):
        BaseModelIdentity(name="model", revision="main")
    with pytest.raises(ValidationError):
        SnapshotMetadata.model_validate({"run_id": "r", "base_model": metadata.base_model})
    with pytest.raises(ValidationError):
        metadata.checkpoint_iteration = 4


def test_real_post_save_entrypoint_exports_adapter_without_merged_hf_fallback(tmp_path, adapter, metadata):
    config = SnapshotExportConfig(
        run_id=metadata.run_id, base_model=metadata.base_model, output_root=tmp_path / "exports"
    )
    args = argparse.Namespace(
        proximal_snapshot_config=config, train_backend="megatron", lora_rank=64, use_critic=False, save=str(tmp_path)
    )
    proximal_snapshot_post_save(args, 7, str(adapter.parent), "/unused/merged-hf")
    [manifest_file] = list(config.output_root.glob("snapshots/*/manifest.json"))
    assert json.loads(manifest_file.read_text())["metadata"]["checkpoint_iteration"] == 7
    (adapter / "adapter_model.bin").unlink()
    with pytest.raises(ValueError, match="complete PEFT"):
        proximal_snapshot_post_save(args, 8, str(adapter.parent), "/unused/merged-hf")


def test_hook_configuration_validates_before_training(tmp_path, metadata):
    parser = argparse.ArgumentParser()
    hook = load_function("miles_plugins.proximal.checkpoint.proximal_snapshot_post_save")
    hook.add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([])
    config = SnapshotExportConfig(run_id="r", base_model=metadata.base_model, output_root=tmp_path / "out")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    args = parser.parse_args(["--proximal-snapshot-config", str(config_path)])
    args.train_backend = "megatron"
    args.lora_rank, args.use_critic, args.save = 64, False, str(tmp_path)
    hook.validate_args(args)
    args.train_backend = "fsdp"
    with pytest.raises(ValueError, match="Megatron"):
        hook.validate_args(args)
    args.train_backend = "megatron"
    args.use_critic = True
    with pytest.raises(ValueError, match="actor-only"):
        hook.validate_args(args)
    args.use_critic, args.lora_rank = False, 0
    with pytest.raises(ValueError, match="LoRA training"):
        hook.validate_args(args)
    assert not config.output_root.exists()


def test_cli_preparation_runs_in_fresh_cpu_process(tmp_path, adapter, metadata):
    config = SnapshotExportConfig(run_id="r", base_model=metadata.base_model, output_root=tmp_path / "out")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "miles_plugins.proximal",
            "prepare",
            "--adapter-directory",
            str(adapter),
            "--config",
            str(config_path),
            "--checkpoint-iteration",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert len(json.loads(result.stdout)["sha256"]) == 64
    imports = subprocess.run(
        [
            sys.executable,
            "-c",
            "import miles_plugins.proximal.__main__; import sys; "
            "assert not {'torch', 'ray', 'sglang', 'modal'} & sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert imports.returncode == 0


@pytest.mark.parametrize("command, flag", [("publish-modal", "--yes-publish"), ("load-replica", "--yes-load")])
def test_remote_cli_commands_require_explicit_consent(command, flag):
    result = subprocess.run(
        [sys.executable, "-m", "miles_plugins.proximal", command],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert flag in result.stderr
