from collections.abc import Sequence
from itertools import pairwise

from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent


def check(events: Sequence[Event]) -> list[str]:
    versions: dict[tuple[str | None, str, str], dict[int, dict[str, str]]] = {}
    applicability: dict[tuple[str | None, str, str], tuple[bool, int]] = {}
    for event in events:
        if not isinstance(event, InferenceEngineWeightChecksumEvent):
            continue
        assert (
            event.version_epoch and event.weight_version is not None and event.engine_snapshots
        ), "Movement evidence lacks identified weight publication"
        assert (
            event.lora_enabled is not None and event.update_weights_interval is not None
        ), "Movement evidence lacks training configuration"
        for snapshot in event.engine_snapshots:
            model = (event.trainer_model_id, snapshot.model_name, event.version_epoch)
            policy = (event.lora_enabled, event.update_weights_interval)
            assert applicability.setdefault(model, policy) == policy, f"Movement applicability changed: {model}"
            by_version = versions.setdefault(model, {})
            assert (
                by_version.setdefault(event.weight_version, snapshot.tensors) == snapshot.tensors
            ), f"Same-version weights disagree: {model}/{event.weight_version}"

    issues: list[str] = []
    for model, by_version in versions.items():
        lora_enabled, interval = applicability[model]
        if lora_enabled or interval != 1:
            continue
        for before, after in pairwise(sorted(by_version)):
            previous, current = by_version[before], by_version[after]
            if previous.keys() != current.keys():
                issues.append(f"Checksum tensor set changed: {model}/{before}->{after}")
                continue
            unchanged = sorted(name for name in current if current[name] == previous[name])
            if unchanged:
                issues.append(f"Unchanged tensor checksums: {model}/{before}->{after}: {unchanged}")
    return issues
