"""WeightUpdater -> HTTP to every replica -> durable archive -> selectable policy.

All ranks join HF gathers. HTTP scatters full-snapshot byte ranges to every rank;
plain Volume publication retains the existing rank-zero path. Recovery archives
are drained by the run-state writer before a policy becomes fleet-selectable.

A version becomes selectable only after a replica verified it and the rollout
store recorded it. The first publication of a process rewinds the store to the
resumed checkpoint, abandoning versions whose weights the resume discarded.
"""

import asyncio
import json
import logging
import tempfile
import time
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
from miles.utils.lora.utils import LORA_ADAPTER_NAME, is_lora_weight_name
from miles_plugins.proximal import http_fleet, policy_archive
from miles_plugins.proximal.adapter_layout import adapter_layout_problem
from miles_plugins.proximal.authorization import authorize_run
from miles_plugins.proximal.clients import ServingPoolClient
from miles_plugins.proximal.contracts import Policy, RunStateArtifacts, read_run_config
from miles_plugins.proximal.modal_volume import authorize_volume_publication, modal_publish_snapshot
from miles_plugins.proximal.serving import lora_serving_targets
from miles_plugins.proximal.sharded_snapshot import ShardedManifest, prepare_sharded
from miles_plugins.proximal.snapshot import PreparedSnapshot, SnapshotMetadata, prepare_snapshot
from miles_plugins.proximal.store import open_store

logger = logging.getLogger(__name__)


def peft_config_json(config: dict[str, JsonValue], *, rank: int, base_model_name: str) -> str:
    """The published adapter_config.json, from the trainer's PEFT configuration."""
    # JSON SDK boundary, authored once by the Megatron LoRA adapter.
    if config.get("peft_type") != "LORA" or config.get("r") != rank:
        raise ValueError("Trainer supplied an incompatible PEFT configuration")
    return json.dumps(config | {"base_model_name_or_path": base_model_name}, sort_keys=True)


def staged_adapter_tensor(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """One gathered ``miles_lora:<hf name>`` tensor as its HF name and a host copy."""
    prefix, separator, hf_name = name.partition(":")
    if prefix != LORA_ADAPTER_NAME or not separator or not is_lora_weight_name(hf_name):
        raise ValueError(f"Expected a single HF adapter tensor, got {name!r}")
    return hf_name, tensor.detach().to("cpu").contiguous().clone()


def write_adapter(directory: Path, *, tensors: dict[str, torch.Tensor], config_json: str) -> None:
    """A complete PEFT adapter: adapter_config.json and one safetensors file."""
    (directory / "adapter_config.json").write_text(config_json)
    save_file(dict(sorted(tensors.items())), str(directory / "adapter_model.safetensors"))


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
        # Replicas load adapters into SGLang engines launched with lora_serving_targets; Miles's
        # resolved adapter targets may name modules differently, so publish the served ones.
        served: list[JsonValue] = list(lora_serving_targets(self.config))
        self._peft_config_json = peft_config_json(
            config | {"target_modules": served}, rank=self.args.lora_rank, base_model_name=self.config.base_model.name
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
        if self.config.weight_sync_transport == "http" and not isinstance(
            self.config.artifact_storage, RunStateArtifacts
        ):
            raise ValueError(
                "HTTP publication requires the Modal run-state writer for background recovery archives; "
                'outside the Modal training launcher set "weight_sync_transport": "volume" in the run config'
            )
        if self.config.weight_sync_transport == "http" and self.config.weight_sync_pool is None:
            raise ValueError("HTTP sync requires the Modal training launcher to supply weight_sync_pool")
        self.rollout_engines = rollout_engines
        self.is_sender = dist.get_rank() == 0

    def begin_sync(
        self, weight_version: int, iter_buckets: Callable[..., Iterator[list[tuple[str, torch.Tensor]]]]
    ) -> bool:
        self._tensors.clear()
        self._error = None
        if self.config.weight_sync_transport == "http":
            error = None
            if self.is_sender:
                deadline = time.monotonic() + self.config.request_timeout_seconds
                while policy_archive.pending_path(self.config).exists():
                    if time.monotonic() >= deadline:
                        error = "Previous policy archive is still pending; refusing another update"
                        break
                    time.sleep(0.1)
            self._check_rank_errors(error)
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # Keep participating in gathers on a local validation/copy failure. Report
        # it collectively at finalize, before any rank advances its version.
        if self._error is not None:
            return
        try:
            for name, tensor in bucket:
                hf_name, staged = staged_adapter_tensor(name, tensor)
                if hf_name in self._tensors:
                    raise ValueError(f"Duplicate gathered adapter tensor: {hf_name}")
                if not torch.isfinite(staged).all():
                    # A policy with NaN/inf weights must never become selectable.
                    self._error = f"Adapter tensor {hf_name} is not finite; refusing to publish it"
                    return
                self._tensors[hf_name] = staged
        except Exception as exc:
            self._error = f"{type(exc).__name__}: adapter staging failed"

    @staticmethod
    def _check_rank_errors(error: str | None) -> None:
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=get_gloo_group())  # type: ignore[no-untyped-call]
        if any(errors):
            raise RuntimeError(f"Policy publication failed: {errors}")

    def finalize(self, weight_version: int) -> None:
        error = self._error
        if self.is_sender and error is None:
            error = adapter_layout_problem(
                {name: tuple(tensor.shape) for name, tensor in self._tensors.items()},
                serving_targets=lora_serving_targets(self.config),
                rank=self.args.lora_rank,
            )
        try:
            self._check_rank_errors(error)
            if self.config.weight_sync_transport == "http":
                self._publish_http(weight_version)
            else:
                if self.is_sender:
                    try:
                        asyncio.run(self._publish(weight_version))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: immutable policy publication failed"
                self._check_rank_errors(error)
        finally:
            self._tensors.clear()

    def _publish_http(self, version: int) -> None:
        started = time.monotonic()
        plans: list[ShardedManifest | None] = [None]
        fleets: list[tuple[http_fleet.ReplicaUpload, ...]] = [()]
        outgoing: list[list[tuple[int, bytes]]] | None = None
        error = None
        if self.is_sender:
            try:
                snapshot = self._export_snapshot(version)
                prepared_manifest = prepare_sharded(
                    snapshot.directory, snapshot.reference, world_size=dist.get_world_size()
                )
                plans[0] = prepared_manifest
                outgoing = [[] for _ in range(dist.get_world_size())]
                index = 0
                for name, parts in prepared_manifest.files.items():
                    with (snapshot.directory / name).open("rb") as stream:
                        for part in parts:
                            stream.seek(part.offset)
                            outgoing[part.rank].append((index, stream.read(part.size_bytes)))
                            index += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: adapter export failed"
        self._check_rank_errors(error)
        dist.broadcast_object_list(plans, src=0, group=get_gloo_group())  # type: ignore[no-untyped-call]
        manifest = plans[0]
        assert manifest is not None
        incoming: list[list[tuple[int, bytes]] | None] = [None]
        # Ranks receive their own bytes; no cross-machine shared filesystem is assumed.
        dist.scatter_object_list(incoming, outgoing, src=0, group=get_gloo_group())  # type: ignore[no-untyped-call]
        del outgoing
        try:
            if self.is_sender:
                try:
                    upstreams = http_fleet.discover_replicas(self.config)
                    fleets[0] = http_fleet.begin_uploads(self.authorization, manifest, upstreams)
                except Exception as exc:
                    logger.exception("HTTP fleet discovery or handshake failed")
                    error = f"{type(exc).__name__}: HTTP fleet discovery or handshake failed"
            self._check_rank_errors(error)
            dist.broadcast_object_list(fleets, src=0, group=get_gloo_group())  # type: ignore[no-untyped-call]
            try:
                assert incoming[0] is not None
                http_fleet.send_to_replicas(self.authorization, manifest, fleets[0], incoming[0])
            except Exception as exc:
                logger.exception("HTTP shard upload failed on rank %s", dist.get_rank())
                error = f"rank {dist.get_rank()}: {type(exc).__name__}: HTTP shard upload failed"
            self._check_rank_errors(error)
            if self.is_sender:
                try:
                    http_fleet.send_to_replicas(self.authorization, manifest, fleets[0], None)
                    if http_fleet.discover_replicas(self.config) != tuple(target.upstream for target in fleets[0]):
                        raise ValueError("Serving fleet changed during HTTP publication; retry the update")
                    asyncio.run(self._rewind())
                    policy_archive.enqueue(
                        self.authorization,
                        Policy(
                            run_id=self.config.run_id,
                            version=version,
                            snapshot=manifest.snapshot,
                            base_model=self.config.base_model,
                        ),
                    )
                    logger.info(
                        "HTTP adapter delivered: version=%s snapshot=%s replicas=%s seconds=%.3f; awaiting archive",
                        version,
                        manifest.snapshot.sha256,
                        len(fleets[0]),
                        time.monotonic() - started,
                    )
                except Exception as exc:
                    logger.exception("HTTP fleet preparation failed")
                    error = f"{type(exc).__name__}: HTTP fleet preparation failed"
            self._check_rank_errors(error)
        except BaseException:
            if self.is_sender:
                http_fleet.cancel_uploads(self.authorization, manifest, fleets[0])
            raise

    async def _rewind(self) -> None:
        if not self._rewound:
            store = await open_store(self.config)
            try:
                await store.rewind(keep_through=self.initial_weight_version)
                self._rewound = True
            finally:
                await store.close()

    def _export_snapshot(self, version: int) -> PreparedSnapshot:
        if not self._tensors:
            raise ValueError("No adapter tensors were gathered")
        root = self.config.artifact_directory / self.config.run_id / "publication"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix=".export-") as directory:
            adapter = Path(directory)
            if self._peft_config_json is None:
                raise ValueError("WeightUpdater must supply the training backend's adapter configuration")
            write_adapter(adapter, tensors=self._tensors, config_json=self._peft_config_json)
            snapshot = prepare_snapshot(
                adapter,
                metadata=SnapshotMetadata(
                    run_id=self.config.run_id, checkpoint_iteration=version - 1, base_model=self.config.base_model
                ),
                output_root=root,
            )
        return snapshot

    async def _publish(self, version: int) -> None:
        snapshot = self._export_snapshot(version)
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
