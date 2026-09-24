import io
from contextlib import contextmanager
from pathlib import Path

import pytest

from miles_plugins.proximal.modal_volume import (
    AuthorizedVolumePublication,
    VolumeDestination,
    authorize_volume_publication,
    modal_publish_snapshot,
)
from miles_plugins.proximal.snapshot import prepare_snapshot


modal = pytest.importorskip("modal")


class FakeVolume:
    """SDK boundary fixture with partial upload failure; not a Modal consistency test."""

    def __init__(self):
        self.files = {}
        self.uploaded = []
        self.fail_after = None
        self.corrupt_upload = False

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    @contextmanager
    def batch_upload(self, force=False):
        assert force is False
        pending = []

        class Upload:
            def put_file(self, source, destination):
                pending.append((source, destination))

        yield Upload()
        for source, destination in pending:
            if self.fail_after is not None and len(self.uploaded) == self.fail_after:
                raise ConnectionError("Interrupted upload")
            assert destination not in self.files
            data = source.getvalue() if isinstance(source, io.BytesIO) else Path(source).read_bytes()
            self.files[destination] = b"corrupt" if self.corrupt_upload else data
            self.uploaded.append(destination)


@pytest.fixture
def volume(monkeypatch):
    volume = FakeVolume()

    def lookup(name, *, environment_name, create_if_missing):
        assert (name, environment_name, create_if_missing) == ("test-volume", "test", False)
        return volume

    monkeypatch.setattr(modal.Volume, "from_name", lookup)
    return volume


def _authorization():
    return authorize_volume_publication(
        VolumeDestination(volume_name="test-volume", environment_name="test"), yes_publish=True
    )


def test_manifest_is_last_and_republication_is_idempotent(tmp_path, adapter, metadata, volume):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "out")
    result = modal_publish_snapshot(_authorization(), snapshot)
    assert result.snapshot == snapshot.reference
    assert len(volume.uploaded) == 3
    assert volume.uploaded[-1].endswith("/manifest.json")
    assert modal_publish_snapshot(_authorization(), snapshot) == result
    assert len(volume.uploaded) == 3


def test_partial_upload_can_resume_without_publishing_incomplete_manifest(tmp_path, adapter, metadata, volume):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "out")
    volume.fail_after = 1
    with pytest.raises(ConnectionError):
        modal_publish_snapshot(_authorization(), snapshot)
    assert len(volume.files) == 1
    assert not any(path.endswith("manifest.json") for path in volume.files)
    volume.fail_after = None
    modal_publish_snapshot(_authorization(), snapshot)
    assert len(volume.uploaded) == 3


def test_corrupt_upload_does_not_publish_a_manifest(tmp_path, adapter, metadata, volume):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "out")
    volume.corrupt_upload = True
    with pytest.raises(ValueError, match="conflicting"):
        modal_publish_snapshot(_authorization(), snapshot)
    assert not any(path.endswith("manifest.json") for path in volume.files)


def test_conflicting_published_bytes_are_never_replaced(tmp_path, adapter, metadata, volume):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "out")
    modal_publish_snapshot(_authorization(), snapshot)
    path = next(path for path in volume.files if path.endswith("adapter_model.bin"))
    volume.files[path] = b"changed by another writer"
    with pytest.raises(ValueError, match="conflicting"):
        modal_publish_snapshot(_authorization(), snapshot)
    assert volume.files[path] == b"changed by another writer"


def test_corrupt_local_snapshot_fails_before_volume_lookup(tmp_path, adapter, metadata, monkeypatch):
    snapshot = prepare_snapshot(adapter, metadata=metadata, output_root=tmp_path / "out")
    (snapshot.directory / "adapter_model.bin").write_bytes(b"changed")
    monkeypatch.setattr(modal.Volume, "from_name", lambda *a, **k: pytest.fail("Network lookup before validation"))
    with pytest.raises(ValueError, match="mismatch"):
        modal_publish_snapshot(_authorization(), snapshot)


def test_publication_authorization_cannot_be_constructed_without_consent():
    destination = VolumeDestination(volume_name="test-volume", environment_name="test")
    with pytest.raises(PermissionError, match="yes-publish"):
        authorize_volume_publication(destination, yes_publish=False)
    with pytest.raises(PermissionError):
        AuthorizedVolumePublication(destination, _authority=object())
