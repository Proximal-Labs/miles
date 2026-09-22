import re
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import TypeAdapter

from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotRecord, ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.env_report.redaction import redact_arg, redact_env_vars, redact_server_info
from miles.utils.test_utils.snapshot import snapshot_values


@dataclass
class _SnapshotState:
    directory: str
    storage: ConfigSnapshotStorage
    run_uuid: str
    source: ProcessIdentity
    replacements: dict[str, dict[str, Any]]
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class ConfigSnapshotDumper:
    _state: ClassVar[_SnapshotState | None] = None

    @classmethod
    def configure(cls, *, args: Namespace, source: ProcessIdentity) -> None:
        cls._state = None
        if args.config_snapshot_dir is None:
            return

        replacements = (
            {}
            if args.config_snapshot_normalize is None
            else TypeAdapter(dict[str, dict[str, Any]]).validate_json(Path(args.config_snapshot_normalize).read_text())
        )
        cls._state = _SnapshotState(
            directory=ConfigSnapshotStorage.snapshot_directory(
                directory=Path(args.config_snapshot_dir) if args.config_snapshot_dir is not None else None,
                name=args.config_snapshot_name,
                component=args.deploy_component,
                instance=args.deploy_instance_id,
            ),
            storage=ConfigSnapshotStorage(),
            run_uuid=args.run_uuid,
            source=source,
            replacements=replacements,
        )

    @classmethod
    def dump(cls, *, boundary: str, config: Any) -> None:
        if (state := cls._state) is None:
            return
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", state.source.to_name()) or not re.fullmatch(r"[a-z_]+", boundary):
            raise ValueError(f"Invalid snapshot identity: {state.source!r}/{boundary!r}")

        index = state.counts[boundary]
        state.counts[boundary] += 1
        record = ConfigSnapshotRecord(
            directory=state.directory,
            source=state.source,
            boundary=boundary,
            index=index,
            run_uuid=state.run_uuid,
            replacements=state.replacements,
            config=_redact(snapshot_values(config)),
        )
        state.storage.write(record)


def _redact(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if not isinstance(value, dict):
        return value

    return {
        name: _redact(
            redact_env_vars(item)
            if name in {"env", "env_vars", "train_env_vars"} and isinstance(item, dict)
            else redact_arg(name, item)
        )
        for name, item in redact_server_info(value).items()
    }
