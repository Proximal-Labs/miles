"""Real CPU Gloo + WeightUpdater; substitute only the external publisher/HTTP."""

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


class CpuAdapterIterator:
    weight_update_selector = "all"

    def __init__(self, args, model, *, required_placement, **kwargs):
        self.placement = required_placement

    def iter_hf_weights(self, weights, *, include_base, adapters, materialize):
        assert include_base is False
        assert adapters == [(LORA_ADAPTER_NAME, None)]
        if materialize:
            yield [
                (
                    f"{LORA_ADAPTER_NAME}:base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                    torch.ones(2, 3),
                ),
                (
                    f"{LORA_ADAPTER_NAME}:base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                    torch.zeros(3, 2),
                ),
            ]


def test_weight_update_exports_real_tensors_then_commits_version(config, tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    args = Namespace(
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
    group = GroupInfo(rank=0, size=1, group=None)
    parallel = ParallelState(
        **{key: group for key in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
    )
    events = []
    fail = [False]

    def publish(authorization, snapshot):
        tensors = load_file(str(snapshot.directory / "adapter_model.safetensors"))
        assert len(tensors) == 2
        assert next(tensor for name, tensor in tensors.items() if ".lora_A." in name).shape == (2, 3)
        events.append(("upload", snapshot.reference.sha256))

    def http(request):
        if fail[0]:
            return httpx.Response(400, json={"error": "test publication rejected"})
        policy = json.loads(request.content)
        assert events[-1] == ("upload", policy["snapshot"]["sha256"])
        events.append(("commit", policy["version"]))
        return httpx.Response(200, json=policy)

    client_cls = httpx.AsyncClient
    monkeypatch.setattr(weight_update, "modal_publish_snapshot", publish)
    monkeypatch.setattr(
        weight_update.httpx, "AsyncClient", lambda **kwargs: client_cls(transport=httpx.MockTransport(http))
    )
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    monkeypatch.setattr(distributed_utils, "GLOO_GROUP", dist.group.WORLD)
    try:
        updater = WeightUpdater(
            args,
            [],
            weights_getter=lambda: {},
            model_name="qwen3",
            quantization_config=None,
            iterator_factory=CpuAdapterIterator,
            parallel_state=parallel,
            is_lora=True,
            lora_sync_config={"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": ["q_proj"]},
        )
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
        assert [value for operation, value in events if operation == "commit"] == [1, 2]
    finally:
        dist.destroy_process_group()
