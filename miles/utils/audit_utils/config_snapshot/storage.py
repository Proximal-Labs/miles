import os
import uuid
from pathlib import Path
from typing import Any

from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR

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
