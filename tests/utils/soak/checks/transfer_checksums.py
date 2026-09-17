from collections.abc import Sequence

from miles.utils.audit_utils.event_logger.models import (
    Event,
    InferenceEngineWeightChecksumEvent,
    WeightTransferChecksumEvent,
)


def assert_transfer_checksums(
    events: Sequence[Event], *, publication_keys: set[tuple[str | None, str, int, str]]
) -> None:
    sent: dict[tuple[str, str, str, int], dict[str, str]] = {}
    manifests: dict[tuple[str, str, str, int], tuple[str, frozenset[str]]] = {}
    duplicates: set[tuple[str, str, str, int]] = set()
    for event in events:
        if not isinstance(event, WeightTransferChecksumEvent):
            continue
        key = (event.update_id, event.cell_id, event.workers_hash, event.receiver_rank)
        manifest = (event.receiver_session_id, frozenset(event.expected_names))
        if manifests.setdefault(key, manifest) != manifest:
            duplicates.add(key)
        tensors = sent.setdefault(key, {})
        if tensors.keys() & event.tensors.keys():
            duplicates.add(key)
        tensors.update(event.tensors)

    for event in events:
        if not isinstance(event, InferenceEngineWeightChecksumEvent):
            continue
        if (
            event.trainer_model_id,
            event.version_epoch,
            event.weight_version,
            event.update_id,
        ) not in publication_keys:
            continue
        assert event.transfer_mode is not None, "Checksum evidence lacks transfer protocol"
        if event.transfer_mode != "p2p":
            continue
        for snapshot in event.engine_snapshots:
            assert snapshot.received_update_id == event.update_id, "Receiver raw snapshot belongs to another update"
            assert snapshot.received_tensors, "Receiver raw checksums are missing"
            prefix = (event.update_id, snapshot.cell_id, snapshot.workers_hash)
            actual_sent: dict[str, str] = {}
            for key, tensors in sent.items():
                if key[:3] != prefix:
                    continue
                assert key not in duplicates, f"Duplicate or mixed-incarnation send evidence: {key}"
                _, names = manifests[key]
                assert names and tensors.keys() == names, f"Incomplete send-buffer coverage: {key}"
                actual_sent.update({f"rank{key[3]}/{name}": checksum for name, checksum in tensors.items()})
            assert actual_sent == snapshot.received_tensors, f"Raw P2P send/receive checksums differ: {prefix}"
