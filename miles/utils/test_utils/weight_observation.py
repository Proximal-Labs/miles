import hashlib
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import WeightTransferChecksumEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _WeightObservation:
    update_id: str
    target_incarnations: Mapping[str, str]


_observation: ContextVar[_WeightObservation | None] = ContextVar("weight_observation", default=None)


@contextmanager
def observe_weight_update(*, enabled: bool, update_id: str, target_incarnations: Mapping[str, str]) -> Iterator[None]:
    token = _observation.set(_WeightObservation(update_id, target_incarnations) if enabled else None)
    try:
        yield
    finally:
        _observation.reset(token)


def observation_update_id() -> str | None:
    return context.update_id if (context := _observation.get()) is not None else None


def observe_transfer(
    *,
    parameters: Mapping[str, torch.Tensor],
    names: Sequence[str],
    cell_id: str,
    receiver_rank: int,
    receiver_session_id: str,
    expected_names: Sequence[str],
) -> None:
    if (context := _observation.get()) is None or not is_event_logger_initialized():
        return
    try:
        assert len(names) == len(set(names)), "P2P bucket contains duplicate tensor names"
        tensors = {}
        for name in names:
            tensor = parameters[name]
            assert tensor.device.type == "cpu", "Expected registered CPU send buffers"
            tensors[name] = hashlib.sha256(
                tensor.detach().contiguous().flatten().view(torch.uint8).numpy()
            ).hexdigest()
        get_event_logger().log(
            WeightTransferChecksumEvent,
            dict(
                update_id=context.update_id,
                cell_id=cell_id,
                workers_hash=context.target_incarnations[cell_id],
                receiver_rank=receiver_rank,
                receiver_session_id=receiver_session_id,
                expected_names=list(expected_names),
                tensors=tensors,
            ),
            print_log=False,
        )
    except Exception:
        logger.exception("Could not observe P2P send buffers")
