"""Real CPU Gloo + WeightUpdater; substitute only the external publisher/HTTP."""

import asyncio
import json
from argparse import Namespace

import httpx
import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.utils import distributed_utils
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.lora import LORA_ADAPTER_NAME
from miles_plugins.proximal import weight_update
from miles_plugins.proximal.options import TRANSFER
from miles_plugins.proximal.store import open_store


LAYER = "base_model.model.model.layers.0"
MODULES = tuple(f"self_attn.{m}" for m in ("q_proj", "k_proj", "v_proj", "o_proj")) + tuple(
    f"mlp.{m}" for m in ("gate_proj", "up_proj", "down_proj")
)


class CpuAdapterIterator:
    weight_update_selector = "all"

    def __init__(self, args, model, *, required_placement, **kwargs):
        self.placement = required_placement

    def iter_hf_weights(self, weights, *, include_base, adapters, materialize):
        assert include_base is False
        assert adapters == [(LORA_ADAPTER_NAME, None)]
        if materialize:
            # One decoder layer, every module the run config targets, as Megatron-Bridge exports it.
            yield [
                (f"{LORA_ADAPTER_NAME}:{LAYER}.{module}.lora_{side}.weight", tensor)
                for module in MODULES
                for side, tensor in (("A", torch.ones(2, 3)), ("B", torch.zeros(3, 2)))
            ]


def current_version(config):
    async def read():
        store = await open_store(config)
        try:
            policy = await store.current_policy()
            return None if policy is None else policy.version
        finally:
            await store.close()

    return asyncio.run(read())


class MtpAdapterIterator(CpuAdapterIterator):
    def iter_hf_weights(self, weights, **kwargs):
        yield from super().iter_hf_weights(weights, **kwargs)
        # An MTP layer's adapter: SGLang would file it under decoder layer 0.
        yield [(f"{LORA_ADAPTER_NAME}:base_model.model.mtp.layers.0.mlp.gate_proj.lora_A.weight", torch.ones(2, 3))]


class NonFiniteAdapterIterator(CpuAdapterIterator):
    def iter_hf_weights(self, weights, **kwargs):
        for bucket in super().iter_hf_weights(weights, **kwargs):
            yield [(name, tensor.fill_(float("nan")) if ".lora_B." in name else tensor) for name, tensor in bucket]


def transfer_args(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    return Namespace(
        proximal_config=str(path),
        proximal_yes_rollouts=True,
        proximal_yes_publish=True,
        start_rollout_id=0,
        lora_rank=2,
        colocate=False,
        custom_weight_transfer_protocol_path=TRANSFER,
        update_weight_transfer_mode="broadcast",
        check_lora_weight_equal=False,
    )


def single_rank():
    group = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        **{key: group for key in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
    )


def make_updater(args, iterator_factory):
    return WeightUpdater(
        args,
        [],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        iterator_factory=iterator_factory,
        parallel_state=single_rank(),
        is_lora=True,
        lora_sync_config={"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": ["q_proj"]},
    )


def test_weight_update_exports_real_tensors_then_commits_version(config, tmp_path, monkeypatch):
    args = transfer_args(config, tmp_path)
    events = []
    fail = [False]

    def publish(authorization, snapshot):
        tensors = load_file(str(snapshot.directory / "adapter_model.safetensors"))
        assert len(tensors) == 2 * len(MODULES)
        assert next(tensor for name, tensor in tensors.items() if ".lora_A." in name).shape == (2, 3)
        events.append(("upload", snapshot.reference.sha256))

    def http(request):
        # The serving pool warms and verifies the uploaded version before the store commits it.
        assert request.url.path == "/policies/prepare"
        if fail[0]:
            return httpx.Response(400, json={"error": "test publication rejected"})
        body = json.loads(request.content)
        sha = body["snapshot"]["sha256"]
        assert events[-1] == ("upload", sha)
        events.append(("prepare", sha))
        return httpx.Response(
            200,
            json={
                "snapshot": body["snapshot"],
                "base_model": body["base_model"],
                "request_model": f"{config.base_model.name}:miles-{sha}",
            },
        )

    client_cls = httpx.AsyncClient
    monkeypatch.setattr(weight_update, "modal_publish_snapshot", publish)
    monkeypatch.setattr(
        weight_update.httpx, "AsyncClient", lambda **kwargs: client_cls(transport=httpx.MockTransport(http))
    )
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.group.WORLD)

    try:
        updater = make_updater(args, CpuAdapterIterator)
        updater.connect_rollout_engines([])
        updater.update_weights()
        assert updater.weight_version == 1
        fail[0] = True
        with pytest.raises(RuntimeError, match="publication failed"):
            updater.update_weights()
        assert updater.weight_version == 1  # Failed publication never advances the trainer.
        fail[0] = False
        updater.update_weights()
        assert updater.weight_version == 2
        assert len([e for e in events if e[0] == "prepare"]) == 2
        assert current_version(config) == 2
        # A restart from step 0 republishes version 1 and abandons version 2.
        restarted = make_updater(args, CpuAdapterIterator)
        restarted.connect_rollout_engines([])
        restarted.update_weights()
        assert restarted.weight_version == 1
        assert current_version(config) == 1
    finally:
        dist.destroy_process_group()


def test_weight_update_refuses_a_non_finite_adapter(config, tmp_path, monkeypatch):
    """A policy with NaN weights never reaches the Volume, serving or the store."""
    uploads = []
    monkeypatch.setattr(weight_update, "modal_publish_snapshot", lambda authorization, snapshot: uploads.append(1))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.group.WORLD)

    try:
        updater = make_updater(transfer_args(config, tmp_path), NonFiniteAdapterIterator)
        updater.connect_rollout_engines([])
        with pytest.raises(RuntimeError, match="lora_B.weight is not finite"):
            updater.update_weights()
        assert uploads == []
        assert current_version(config) is None
    finally:
        dist.destroy_process_group()


def test_weight_update_refuses_an_adapter_sglang_would_mis_serve(config, tmp_path, monkeypatch):
    """An adapter tensor outside the decoder layers never reaches the Volume, serving or the store."""
    uploads = []
    monkeypatch.setattr(weight_update, "modal_publish_snapshot", lambda authorization, snapshot: uploads.append(1))
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.group.WORLD)

    try:
        updater = make_updater(transfer_args(config, tmp_path), MtpAdapterIterator)
        updater.connect_rollout_engines([])
        with pytest.raises(RuntimeError, match="outside the text decoder layers"):
            updater.update_weights()
        assert uploads == []
        assert current_version(config) is None
    finally:
        dist.destroy_process_group()
