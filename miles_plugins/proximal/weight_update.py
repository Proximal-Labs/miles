"""Existing WeightUpdater -> complete PEFT snapshot -> Volume -> serving policy.

All ranks join HF tensor gathers. Only rank zero uploads; its final verdict is
broadcast so a failed upload cannot strand sibling ranks at the next barrier.

A version becomes selectable only after a replica verified it and the rollout
store recorded it. The first publication of a process rewinds the store to the
resumed checkpoint, abandoning versions whose weights the resume discarded.
"""

import asyncio
import json
import tempfile
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import httpx
import torch
import torch.distributed as dist
from pydantic import JsonValue
from safetensors.torch import save_file

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_weight_name
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.clients import ServingPoolClient
from miles_plugins.proximal.contracts import Policy, read_run_config
from miles_plugins.proximal.modal_volume import authorize_volume_publication, modal_publish_snapshot
from miles_plugins.proximal.snapshot import SnapshotMetadata, prepare_snapshot
from miles_plugins.proximal.store import open_store


class ModalVolumeTransfer(WeightTransferProtocol):
    required_placement = WeightUpdatePlacement(gather_pp=True)
    supports_lora = True
    use_weight_update_session = False

    @staticmethod
    def validate_args(args: Namespace) -> None:
        from miles_plugins.proximal.options import validate_args

        validate_args(args)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.config = read_run_config(args.proximal_config)
        self.authorization = authorize_run(
            self.config, yes_rollouts=args.proximal_yes_rollouts, yes_publish=args.proximal_yes_publish
        )
        self.initial_weight_version = args.start_rollout_id or 0
        self._rewound = False
        self._tensors: dict[str, torch.Tensor] = {}
        self._error: str | None = None
        self._peft_config_json: str | None = None

    def configure_lora(self, config: dict[str, JsonValue]) -> None:
        # JSON SDK boundary, authored once by the Megatron LoRA adapter.
        if config.get("peft_type") != "LORA" or config.get("r") != self.args.lora_rank:
            raise ValueError("Trainer supplied an incompatible PEFT configuration")
        self._peft_config_json = json.dumps(
            config | {"base_model_name_or_path": self.config.base_model.name}, sort_keys=True
        )

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        if rollout_engines or not placement.is_full_gather or parallel_state.indep_dp.size != 1:
            raise ValueError(
                "Modal publication requires full adapter gather, one actor cell, and no local rollout engines"
            )
        self.rollout_engines = rollout_engines
        self.is_sender = dist.get_rank() == 0

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        self._tensors.clear()
        self._error = None
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # Keep participating in gathers on a local validation/copy failure. Report
        # it collectively at finalize, before any rank advances its version.
        if self._error is not None:
            return
        try:
            for name, tensor in bucket:
                prefix, separator, hf_name = name.partition(":")
                if prefix != LORA_ADAPTER_NAME or not separator or not is_lora_weight_name(hf_name):
                    raise ValueError(f"Expected a single HF adapter tensor, got {name!r}")
                if hf_name in self._tensors:
                    raise ValueError(f"Duplicate gathered adapter tensor: {hf_name}")
                self._tensors[hf_name] = tensor.detach().to("cpu").contiguous().clone()
        except Exception as exc:
            self._error = f"{type(exc).__name__}: adapter staging failed"

    def finalize(self, weight_version: int) -> None:
        verdict: list[str | None] = [self._error]
        if self.is_sender and verdict[0] is None:
            try:
                asyncio.run(self._publish(weight_version))
            except Exception as exc:
                # Do not distribute provider exceptions that might contain credentials.
                verdict[0] = f"{type(exc).__name__}: immutable policy publication failed"
        dist.broadcast_object_list(verdict, src=0, group=get_gloo_group())  # type: ignore[no-untyped-call]  # Upstream group accessor.
        self._tensors.clear()
        if verdict[0] is not None:
            raise RuntimeError(verdict[0])

    async def _publish(self, version: int) -> None:
        if not self._tensors:
            raise ValueError("No adapter tensors were gathered")
        root = self.config.artifact_directory / self.config.run_id / "publication"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix=".export-") as directory:
            adapter = Path(directory)
            if self._peft_config_json is None:
                raise ValueError("WeightUpdater must supply the training backend's adapter configuration")
            (adapter / "adapter_config.json").write_text(self._peft_config_json)
            save_file(dict(sorted(self._tensors.items())), str(adapter / "adapter_model.safetensors"))
            snapshot = prepare_snapshot(
                adapter,
                metadata=SnapshotMetadata(
                    run_id=self.config.run_id, checkpoint_iteration=version - 1, base_model=self.config.base_model
                ),
                output_root=root,
            )
        publication = authorize_volume_publication(self.config.volume, yes_publish=True)
        await asyncio.to_thread(modal_publish_snapshot, publication, snapshot)
        policy = Policy(
            run_id=self.config.run_id, version=version, snapshot=snapshot.reference, base_model=self.config.base_model
        )
        async with httpx.AsyncClient(timeout=self.config.request_timeout_seconds) as client:
            await ServingPoolClient(self.authorization, client).prepare(policy)
        store = await open_store(self.config)
        try:
            if not self._rewound:
                # Versions after the resumed checkpoint named weights this process discarded.
                await store.rewind(keep_through=self.initial_weight_version)
                self._rewound = True
            await store.commit_policy(policy)
        finally:
            await store.close()
