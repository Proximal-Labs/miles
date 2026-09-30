from pathlib import Path

import pytest

from miles_plugins.proximal.e2e import snapshots


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_copy_new_skips_other_writers_scratch_directories(tmp_path):
    """Run 012: the publisher's .export-* staging vanished mid-copy and killed the snapshotter."""
    source, target = tmp_path / "artifacts", tmp_path / "snapshot"
    write(source / "run" / "groups" / "g.bin")
    write(source / "run" / "publication" / ".export-abc" / "adapter_model.safetensors")

    snapshots._copy_new(source, target)

    assert (target / "run" / "groups" / "g.bin").is_file()
    assert not (target / "run" / "publication").exists()


def test_copy_new_tolerates_a_file_that_vanishes_mid_copy(tmp_path, monkeypatch):
    source, target = tmp_path / "artifacts", tmp_path / "snapshot"
    write(source / "a.bin")
    write(source / "gone.bin")
    real_copy = snapshots.shutil.copy2

    def copy2(src, dst):
        if Path(src).name == "gone.bin":
            Path(dst).write_text("partial")
            raise FileNotFoundError(src)
        return real_copy(src, dst)

    monkeypatch.setattr(snapshots.shutil, "copy2", copy2)

    snapshots._copy_new(source, target)

    assert (target / "a.bin").read_text() == "x"
    assert not (target / "gone.bin").exists()  # No partial copy left to be mistaken for state.


def test_copy_new_still_copies_only_missing_files(tmp_path):
    source, target = tmp_path / "artifacts", tmp_path / "snapshot"
    write(source / "p.bin", "new")
    write(target / "p.bin", "old")

    snapshots._copy_new(source, target)

    assert (target / "p.bin").read_text() == "old"  # Write-once payloads are never rewritten.


@pytest.mark.parametrize("name", [".staging", ".0000002.tmp"])
def test_any_dot_directory_is_scratch(tmp_path, name):
    source, target = tmp_path / "artifacts", tmp_path / "snapshot"
    write(source / name / "f")

    snapshots._copy_new(source, target)

    assert not (target / name).exists()


def test_recovery_configuration_imports_without_trainer_dependencies():
    """modal run loads config locally before its GPU image supplies torch/psycopg/SGLang."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class NoTrainerDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'psycopg', 'sglang', 'ray'}:
            raise AssertionError('Trainer-only dependency loaded on the launch host: ' + fullname)
sys.meta_path.insert(0, NoTrainerDependencies())
from miles_plugins.proximal.training import TrainingDeployment
from miles_plugins.proximal.state_checkpoints import RecoveryContext
""",
        ],
        check=True,
    )
