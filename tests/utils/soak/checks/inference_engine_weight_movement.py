from itertools import pairwise

from miles.utils.audit_utils.event_logger.models import Event, InferenceEngineWeightChecksumEvent


def check(events: list[Event]) -> list[str]:
    versions: dict[tuple[str | None, str, str | None], dict[int, dict[str, str]]] = {}
    policies: dict[tuple[str | None, str, str | None], tuple[tuple[str, ...], int]] = {}
    for event in events:
        if not isinstance(event, InferenceEngineWeightChecksumEvent) or event.weight_version is None:
            continue
        assert event.movement_skip_reasons is not None, "Versioned checksum lacks movement applicability"
        for snapshot in event.engine_snapshots:
            model = (event.trainer_model_id, snapshot.model_name, event.version_epoch)
            policy = (tuple(sorted(event.movement_skip_reasons)), event.movement_max_steps)
            if model in policies:
                assert policies[model] == policy, f"Checksum movement policy changed for {model}"
            policies[model] = policy
            by_version = versions.setdefault(model, {})
            if event.weight_version in by_version:
                assert (
                    by_version[event.weight_version] == snapshot.tensors
                ), f"Same-version weights disagree for {model}"
            else:
                by_version[event.weight_version] = snapshot.tensors

    issues: list[str] = []
    for model, by_version in versions.items():
        skip_reasons, max_steps = policies[model]
        if skip_reasons:
            continue
        ordered = sorted(by_version)
        last_changed = dict.fromkeys(by_version[ordered[0]], 0)
        for ordinal, (before, after) in enumerate(pairwise(ordered), start=1):
            previous, current = by_version[before], by_version[after]
            if previous.keys() != current.keys():
                issues.append(f"Checksum tensor set changed for {model} between versions {before} and {after}")
                break
            for tensor, checksum in current.items():
                if checksum != previous[tensor]:
                    last_changed[tensor] = ordinal
                elif ordinal - last_changed[tensor] == max_steps:
                    issues.append(
                        f"Checksum tensor {tensor!r} did not change for model={model[0]!r}, "
                        f"model_name={model[1]!r}, epoch={model[2]!r}, version={after}, "
                        f"threshold={max_steps} published-version transitions"
                    )
    return issues
