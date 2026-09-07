"""Transfer protocol contract and factory."""

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement

if TYPE_CHECKING:
    from miles.backends.training_utils.weight_update.protocols.p2p import _P2PInferenceCellUpdater


@dataclass(frozen=True)
class UpdatableEngine:
    cell_id: str
    api_client: SGLangApiClient
    gpu_count: int
    gpu_offset: int
    workers_hash: str

    def __post_init__(self) -> None:
        assert self.gpu_count > 0, f"engine {self.cell_id} serves on {self.gpu_count!r} GPUs, which cannot be updated"


@dataclass(frozen=True)
class UpdatableEngines:
    engines: list[UpdatableEngine]

    @property
    def snapshot_cell_id_to_hashes(self) -> dict[str, str]:
        return {engine.cell_id: engine.workers_hash for engine in self.engines}


class WeightTransferProtocol(ABC):
    """Moves HF-named weight buckets from training ranks to rollout engines.

    ``connect`` makes every pairing decision once: it sets ``is_sender`` and
    whatever send channels the protocol needs. The updater then drives
    ``send_bucket`` on sender ranks only; streamed adapter tensors are ordinary
    bucket entries (``{lora_name}:{hf_key}`` names).
    """

    required_placement: ClassVar[WeightUpdatePlacement] = WeightUpdatePlacement(gather_pp=False)
    supports_lora: ClassVar[bool] = False
    use_weight_update_session: ClassVar[bool] = True
    needs_base_resync_for_lora: bool = False

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.rollout_engines: Sequence[SGLangApiClient] | None = None
        self.is_sender: bool | None = None
        self.cell_updaters: list[_P2PInferenceCellUpdater] = []
        self.group_name = "miles"
        self.update_weight_metrics: dict[str, float] = {}

    @abstractmethod
    def connect(
        self,
        engines: Sequence[UpdatableEngine],
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None: ...

    def begin_sync(
        self,
        weight_version: int,
        iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]],
    ) -> bool:
        """Hook before the session frame; return False to skip this round.
        The return value must be identical on every rank."""
        return True

    @abstractmethod
    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None: ...

    def after_base_weights(self) -> None:  # noqa: B027 — optional hook
        """Hook after the base-weight stream completes (e.g. await in-flight writes)."""

    def synchronize_cell_errors(self) -> None:  # noqa: B027 — optional hook
        """Hook after the base-weight stream, to agree across the trainer ranks on which cells lost the update."""

    def finalize(self, weight_version: int) -> None:  # noqa: B027 — optional hook
        """Hook after all sends (e.g. publish + engine reload)."""

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics


def get_weight_transfer_protocol(args: Namespace) -> WeightTransferProtocol:
    if args.colocate:
        from miles.backends.training_utils.weight_update.protocols.cuda_ipc import UpdateWeightFromTensor

        return UpdateWeightFromTensor(args)
    if args.update_weight_transfer_mode == "broadcast":
        from miles.backends.training_utils.weight_update.protocols.broadcast import UpdateWeightFromDistributed

        return UpdateWeightFromDistributed(args)
    if args.update_weight_transfer_mode == "disk-delta":
        from miles.backends.training_utils.weight_update.protocols.delta import UpdateWeightFromDiskDelta

        return UpdateWeightFromDiskDelta(args)
    if args.update_weight_transfer_mode == "p2p":
        from miles.backends.training_utils.weight_update.protocols.p2p import UpdateWeightP2P

        return UpdateWeightP2P(args)
    raise ValueError(f"Unknown --update-weight-transfer-mode {args.update_weight_transfer_mode!r}")
