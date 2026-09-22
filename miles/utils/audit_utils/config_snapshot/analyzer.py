import copy
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotRecord, ConfigSnapshotStorage
from miles.utils.test_utils.snapshot import SNAPSHOT_UPDATE_ENV_VAR, dump_snapshot

logger = logging.getLogger(__name__)


class ConfigSnapshotMismatch(AssertionError):
    pass


@dataclass(frozen=True)
class ConfigSnapshotAnalyzer:
    storage: ConfigSnapshotStorage
    test: str

    @classmethod
    @contextmanager
    def for_test(cls, *, test: str) -> Iterator["ConfigSnapshotAnalyzer"]:
        with ConfigSnapshotStorage.for_test(test=test) as storage:
            yield cls(storage=storage, test=test)

    def finish(self, *, returncode: int) -> None:
        self.storage.finish(test=self.test, completed=returncode == 0)
        if returncode != 0:
            return
        try:
            self.assert_matches()
        except Exception as error:
            logger.exception("Snapshot analysis failed; raw dumps: %s", self.storage.directory)
            raise ConfigSnapshotMismatch(f"Snapshot analysis failed; raw dumps: {self.storage.directory}\n{error}") from error

    def assert_matches(self) -> None:
        actual = self.storage.with_manifest(test=self.test, snapshots=self.build())
        if os.environ.get(SNAPSHOT_UPDATE_ENV_VAR):
            self.storage.write_expected(test=self.test, snapshots=actual)
            return

        expected = self.storage.read_expected(test=self.test, targets=set(actual))
        failures = []
        for target in sorted(actual.keys() | expected.keys()):
            if target not in actual:
                failures.append(f"Missing snapshot: {target}")
            elif target not in expected:
                failures.append(f"Missing baseline: {target}")
            elif actual[target] != expected[target]:
                failures.append(f"Snapshot mismatch: {target}")
        if failures:
            raise ConfigSnapshotMismatch("\n".join(failures))

    def build(self) -> dict[Path, str]:
        snapshots: dict[Path, str] = {}
        for record in self.storage.read():
            actual = _render_record(record)
            if record.target in snapshots and snapshots[record.target] != actual:
                raise ValueError(f"Processes disagree on snapshot {record.target}")
            snapshots[record.target] = actual
        return snapshots


def _render_record(record: ConfigSnapshotRecord) -> str:
    replacements = record.replacements.get(record.key, {})
    value = _replace_run_uuid(
        {
            "boundary": record.boundary,
            "config": record.config,
        },
        run_uuid=record.run_uuid,
    )
    return dump_snapshot(
        {
            "normalization": {"run_uuid": "$RUN_UUID", "replacements": replacements},
            "snapshot": _normalize(value, replacements=replacements),
        }
    )


def _normalize(value: Any, *, replacements: dict[str, Any]) -> Any:
    result = copy.deepcopy(value)
    for pointer, replacement in replacements.items():
        if not pointer.startswith("/"):
            raise ValueError(f"Expected an absolute JSON pointer, got {pointer!r}")
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
        target = result
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        key = int(parts[-1]) if isinstance(target, list) else parts[-1]
        target[key]
        target[key] = replacement
    return result


def _replace_run_uuid(value: Any, *, run_uuid: str) -> Any:
    if isinstance(value, str):
        return value.replace(run_uuid, "$RUN_UUID") if run_uuid else value
    if isinstance(value, dict):
        return {key: _replace_run_uuid(item, run_uuid=run_uuid) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_run_uuid(item, run_uuid=run_uuid) for item in value]
    return value
