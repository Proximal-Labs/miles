import hashlib
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, dump_snapshot

_REPO_ROOT = Path(__file__).resolve().parents[4]


class ConfigSnapshotRecord(FrozenStrictBaseModel):
    directory: str
    source: ProcessIdentity
    boundary: str
    index: int
    run_uuid: str
    replacements: dict[str, dict[str, Any]]
    config: Any

    @property
    def key(self) -> str:
        return f"{self.source.to_name()}/{self.boundary}-{self.index:04d}.yaml"

    @property
    def target(self) -> Path:
        return Path(self.directory) / self.key


class _SnapshotAttempt(FrozenStrictBaseModel):
    test: str
    completed: bool


class ConfigSnapshotStorage:
    def __init__(self, *, directory: Path | None = None) -> None:
        self.directory = directory or Path(os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR) or _REPO_ROOT / ".snapshot-records")
        self._process_directory = self.directory / "raw" / uuid.uuid4().hex

    def write(self, record: ConfigSnapshotRecord) -> None:
        self._process_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path=self._process_directory / f"{record.boundary}-{record.index:04d}.json",
            text=record.model_dump_json(),
        )

    def read(self) -> list[ConfigSnapshotRecord]:
        return [
            ConfigSnapshotRecord.model_validate_json(path.read_text())
            for path in sorted((self.directory / "raw").glob("*/*.json"))
        ]

    @classmethod
    @contextmanager
    def for_test(cls, *, test: str) -> Iterator["ConfigSnapshotStorage"]:
        previous = os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR)
        root = Path(previous or tempfile.mkdtemp(prefix="miles-snapshot-records-"))
        storage = cls(directory=root / Path(test).stem / uuid.uuid4().hex)
        storage.finish(test=test, completed=False)
        os.environ[SNAPSHOT_RECORD_DIR_ENV_VAR] = str(storage.directory)
        try:
            yield storage
        finally:
            if previous is None:
                os.environ.pop(SNAPSHOT_RECORD_DIR_ENV_VAR, None)
            else:
                os.environ[SNAPSHOT_RECORD_DIR_ENV_VAR] = previous

    def finish(self, *, test: str, completed: bool) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            path=self.directory / "attempt.json", text=_SnapshotAttempt(test=test, completed=completed).model_dump_json()
        )

    @classmethod
    def read_expected(cls, *, test: str, targets: set[Path]) -> dict[Path, str]:
        manifest = cls._manifest_path(test)
        previous = cls._read_manifest(test)
        return {
            target: content
            for target in targets | previous | {manifest}
            if (content := cls.read_golden(target)) is not None
        }

    @classmethod
    def with_manifest(cls, *, test: str, snapshots: dict[Path, str]) -> dict[Path, str]:
        if not snapshots and cls.read_golden(cls._manifest_path(test)) is None:
            return {}
        return {**snapshots, cls._manifest_path(test): dump_snapshot(sorted(str(path) for path in snapshots))}

    @classmethod
    def write_expected(cls, *, test: str, snapshots: dict[Path, str]) -> None:
        obsolete = cls._read_manifest(test) - snapshots.keys()
        for target, content in snapshots.items():
            cls.write_golden(target=target, content=content)
        for target in obsolete:
            (_REPO_ROOT / target).unlink(missing_ok=True)

    @staticmethod
    def snapshot_directory(*, directory: Path | None, name: str, component: str, instance: str | None) -> str:
        if not name:
            raise ValueError("Snapshot name is required")
        name_path = Path(name)
        if name_path.is_absolute() or ".." in name_path.parts:
            raise ValueError("Snapshot name must be a nonempty relative path without '..'")
        root = directory if directory is not None else _REPO_ROOT / "tests/snapshots/runtime_config"
        target = (root / name_path / component / (instance or "default")).resolve()
        return str(target.relative_to(_REPO_ROOT) if target.is_relative_to(_REPO_ROOT) else target)

    @staticmethod
    def read_golden(target: Path) -> str | None:
        path = _REPO_ROOT / target
        return path.read_text() if path.exists() else None

    @staticmethod
    def write_golden(*, target: Path, content: str) -> None:
        path = _REPO_ROOT / target
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path=path, text=content)

    @staticmethod
    def _manifest_path(test: str) -> Path:
        name = f"{Path(test).stem}-{hashlib.sha256(test.encode()).hexdigest()[:16]}.yaml"
        return Path("tests/snapshots/runtime_config/_manifests") / name

    @classmethod
    def _read_manifest(cls, test: str) -> set[Path]:
        content = cls.read_golden(cls._manifest_path(test))
        return {Path(item) for item in yaml.safe_load(content)} if content is not None else set()
