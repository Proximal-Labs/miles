"""Transfer protocol contract and factory."""

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

import torch
from pydantic import JsonValue

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


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
    initial_weight_version: int = 0

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.rollout_engines: Sequence[SGLangApiClient] | None = None
        self.is_sender: bool | None = None
        self.group_name = "miles"
        self.update_weight_metrics: dict[str, float] = {}

    def configure_lora(self, config: dict[str, JsonValue]) -> None:  # noqa: B027
        """Receive the training backend's authoritative PEFT configuration."""

    @abstractmethod
    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
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

    def finalize(self, weight_version: int) -> None:  # noqa: B027 — optional hook
        """Hook after all sends (e.g. publish + engine reload)."""

    def after_engines_resumed(self) -> None:  # noqa: B027 — optional hook
        """Hook once the engines have applied every bucket and resumed generation."""

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics


def get_weight_transfer_protocol(args: Namespace) -> WeightTransferProtocol:
    if path := getattr(args, "custom_weight_transfer_protocol_path", None):
        from miles.utils.function_registry import load_function

        protocol = load_function(path)(args)
        if not isinstance(protocol, WeightTransferProtocol):
            raise TypeError("Custom transfer must implement WeightTransferProtocol")
        return protocol
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
