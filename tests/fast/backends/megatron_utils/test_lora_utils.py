"""LoRA detection, adapter parameters, and training checkpoint state."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.backends.megatron_utils.lora.utils as lora_utils
from miles.backends.megatron_utils.lora.utils import (
    _is_adapter_param_name,
    is_lora_enabled,
    load_lora_adapter,
    save_lora_checkpoint,
)
from miles.utils.lora.utils import LORA_ADAPTER_NAME, is_lora_weight_name

# ---------------------------------------------------------------------------
# is_lora_enabled
# ---------------------------------------------------------------------------


class TestIsLoraEnabled:
    def test_enabled_by_rank(self):
        args = Namespace(lora_rank=32, lora_adapter_path=None)
        assert is_lora_enabled(args) is True

    def test_enabled_by_adapter_path(self):
        args = Namespace(lora_rank=0, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_enabled_by_both(self):
        args = Namespace(lora_rank=16, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_disabled(self):
        args = Namespace(lora_rank=0, lora_adapter_path=None)
        assert is_lora_enabled(args) is False

    def test_disabled_missing_attrs(self):
        args = Namespace()
        assert is_lora_enabled(args) is False


# ---------------------------------------------------------------------------
# is_lora_weight_name / _is_adapter_param_name
# ---------------------------------------------------------------------------


class TestIsLoraWeightName:
    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_A.default.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_B.default.weight",
        ],
    )
    def test_positive(self, name):
        assert is_lora_weight_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    def test_negative(self, name):
        assert is_lora_weight_name(name) is False


class TestIsAdapterParamName:
    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.lora_A.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
        ],
    )
    def test_positive(self, name):
        assert _is_adapter_param_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.weight",
            "module.decoder.layers.0.mlp.linear_fc1.weight",
            "module.embedding.word_embeddings.weight",
        ],
    )
    def test_negative(self, name):
        assert _is_adapter_param_name(name) is False


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"


class _AdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.lora_A = torch.nn.Parameter(torch.zeros(1, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))


def _single_rank(monkeypatch):
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(rank=0), pp=SimpleNamespace(rank=0)),
    )


def test_load_lora_adapter_rejects_a_shard_that_does_not_match_the_adapter(tmp_path, monkeypatch):
    """A silently partial load resumes training on a half-initialized adapter."""
    _single_rank(monkeypatch)
    torch.save({"lora_A": torch.ones(1, 2), "stale_lora_C": torch.ones(1)}, tmp_path / "adapter_megatron_rank0.pt")

    with pytest.raises(RuntimeError, match=r"missing=\['lora_B'\], unexpected=\['stale_lora_C'\]"):
        lora_utils.load_lora_adapter([_AdapterModel()], str(tmp_path))


def test_load_lora_adapter_rejects_shards_saved_under_another_layout(tmp_path, monkeypatch):
    """Falling through to fresh adapter weights would hide a resharding mistake."""
    _single_rank(monkeypatch)
    torch.save({"lora_A": torch.ones(1, 2)}, tmp_path / "adapter_megatron_rank1.pt")

    with pytest.raises(FileNotFoundError, match="none for global rank 0"):
        lora_utils.load_lora_adapter([_AdapterModel()], str(tmp_path))


class TestSaveLoraCheckpointTrainingState:
    def _save(self, tmp_path, *, no_save_optim, scheduler=None, optimizer=None):
        publisher = SimpleNamespace(write_adapter=lambda *_: None)

        adapter = torch.nn.Parameter(torch.ones(2))
        model = [SimpleNamespace(named_parameters=lambda: [("layers.0.self_attention.lora_A.weight", adapter)])]
        args = Namespace(megatron_to_hf_mode="bridge", no_save_optim=no_save_optim)
        optimizer = optimizer or SimpleNamespace(state_dict=lambda: {"step": 7})
        save_lora_checkpoint(
            model,
            args,
            str(tmp_path / "checkpoint"),
            publisher=publisher,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            iteration=3,
        )
        return sorted(path.name for path in (tmp_path / "checkpoint").iterdir())

    @staticmethod
    def _state(tmp_path):
        return torch.load(tmp_path / "checkpoint" / "training_state_rank0.pt", weights_only=False)

    def test_training_state_is_written_by_default(self, tmp_path):
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, no_save_optim=False, scheduler=scheduler)

        assert files == ["adapter_megatron_rank0.pt", "native_checkpoint.json", "training_state_rank0.pt"]
        state = self._state(tmp_path)
        assert state["optimizer"] == {"step": 7}
        assert state["opt_param_scheduler"] == {"lr": 0.5}
        assert state["iteration"] == 3

    def test_no_save_optim_drops_the_optimizer_and_keeps_the_resume_metadata(self, tmp_path):
        """--no-save-optim is about optimizer state; losing the step and the LR schedule with it
        would silently restart a resumed run from iteration 0."""
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, no_save_optim=True, scheduler=scheduler)

        assert files == ["adapter_megatron_rank0.pt", "native_checkpoint.json", "training_state_rank0.pt"]
        state = self._state(tmp_path)
        assert state["optimizer"] is None
        assert state["opt_param_scheduler"] == {"lr": 0.5}
        assert state["iteration"] == 3

    def test_a_distributed_optimizer_saves_its_parameter_state(self, tmp_path, monkeypatch):
        """DistributedOptimizer.state_dict() leaves out the fp32 main params and the Adam moments."""
        shard = {"param": torch.ones(3), "exp_avg": torch.full((3,), 0.1), "exp_avg_sq": torch.full((3,), 0.01)}
        part = _DistributedPart({"per_bucket_numel": [3], 0: {torch.float32: [[shard]]}})
        optimizer = SimpleNamespace(chained_optimizers=[part], state_dict=lambda: {"step": 7})

        self._save(tmp_path, no_save_optim=False, optimizer=optimizer)

        (saved,) = self._state(tmp_path)["optimizer_parameter_state"]
        assert saved["per_bucket_numel"] == [3]
        for key, tensor in shard.items():
            assert torch.equal(saved[0][torch.float32][0][0][key], tensor)
        assert saved[0][torch.float32][0][0]["padding"] is False  # the shape Megatron's loader reads


class TestLoadTrainingState:
    @staticmethod
    def _recorder():
        loaded = []
        return loaded, SimpleNamespace(load_state_dict=loaded.append)

    def _write(self, tmp_path, optimizer_state):
        torch.save(
            {"iteration": 3, "optimizer": optimizer_state, "opt_param_scheduler": {"lr": 0.5}},
            tmp_path / "training_state_rank0.pt",
        )

    def test_an_optimizer_free_checkpoint_still_restores_the_step_and_the_schedule(self, tmp_path):
        self._write(tmp_path, None)
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == (3, False, True)
        assert optimizer_loads == []
        assert scheduler_loads == [{"lr": 0.5}]

    def test_a_full_checkpoint_restores_the_optimizer(self, tmp_path):
        self._write(tmp_path, {"step": 7})
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == (3, True, True)
        assert optimizer_loads == [{"step": 7}]
        assert scheduler_loads == [{"lr": 0.5}]

    def test_a_checkpoint_without_scheduler_state_reports_the_schedule_as_not_restored(self, tmp_path):
        """The caller then advances the fresh scheduler by the iteration itself."""
        torch.save(
            {"iteration": 3, "optimizer": None, "opt_param_scheduler": None},
            tmp_path / "training_state_rank0.pt",
        )
        _, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == (3, False, False)
        assert scheduler_loads == []

    def test_an_adapter_without_training_state_restores_nothing(self, tmp_path):
        _, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == (None, False, False)
        assert scheduler_loads == []


class _DistributedPart:
    """A Megatron DistributedOptimizer as the LoRA checkpoint sees it."""

    def __init__(self, state=None, events=None):
        self.state = state
        self.events = [] if events is None else events

    def get_parameter_state_dp_reshardable(self):
        return self.state

    def load_parameter_state_from_dp_reshardable(self, state):
        self.events.append(("parameter_state", state))


class TestLoadDistributedOptimizerState:
    """Without its parameter state, a DistributedOptimizer resumes from torch.empty moments."""

    @staticmethod
    def _write(tmp_path, **extra):
        torch.save(
            {"iteration": 3, "optimizer": {"step": 7}, "opt_param_scheduler": None, **extra},
            tmp_path / "training_state_rank0.pt",
        )

    @staticmethod
    def _optimizer(events):
        return SimpleNamespace(
            chained_optimizers=[_DistributedPart(events=events)],
            load_state_dict=lambda state: events.append(("optimizer", state)),
        )

    def test_the_parameter_state_is_restored_after_the_optimizer_state(self, tmp_path):
        self._write(tmp_path, optimizer_parameter_state=[{"exp_avg_sq": torch.ones(2)}])
        events = []

        assert lora_utils._load_training_state(tmp_path, self._optimizer(events), None) == (3, True, False)

        assert [event[0] for event in events] == ["optimizer", "parameter_state"]
        assert torch.equal(events[1][1]["exp_avg_sq"], torch.ones(2))

    def test_saved_elements_are_marked_real_parameters_for_the_loader(self, tmp_path):
        """radixark/Megatron-LM (Sep 2026) indexes element['padding'] on load; the getter omits it."""
        shard = {"param": torch.ones(2), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)}
        state = {"per_bucket_numel": [2], "per_bucket_numel_unpadded": [2], 0: {torch.float32: [[shard]]}}
        self._write(tmp_path, optimizer_parameter_state=[state])
        events = []

        lora_utils._load_training_state(tmp_path, self._optimizer(events), None)

        (element,) = events[1][1][0][torch.float32][0]
        assert element["padding"] is False
        assert torch.equal(element["param"], torch.ones(2))

    def test_a_checkpoint_without_parameter_state_is_refused(self, tmp_path):
        self._write(tmp_path)
        events = []

        with pytest.raises(RuntimeError, match="--no-load-optim"):
            lora_utils._load_training_state(tmp_path, self._optimizer(events), None)
        assert events == []

    def test_no_load_optim_resumes_from_a_checkpoint_without_parameter_state(self, tmp_path):
        self._write(tmp_path)
        events = []

        assert lora_utils._load_training_state(tmp_path, self._optimizer(events), None, load_optimizer=False) == (
            3,
            False,
            False,
        )
        assert events == []


def test_loading_an_adapter_refreshes_the_optimizer_main_params(tmp_path, monkeypatch):
    """The fp32 main params were copied before the load; the first step() would write them back."""
    rank0 = SimpleNamespace(rank=0)
    monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: SimpleNamespace(tp=rank0, pp=rank0))
    name = "layers.0.self_attention.lora_A.weight"
    torch.save({name: torch.ones(2)}, tmp_path / "adapter_megatron_rank0.pt")
    torch.save({"iteration": 3, "optimizer": {"step": 7}}, tmp_path / "training_state_rank0.pt")
    param = torch.nn.Parameter(torch.zeros(2))
    events = []
    optimizer = SimpleNamespace(
        reload_model_params=lambda: events.append(("reload", param.detach().clone())),
        load_state_dict=lambda state: events.append(("optimizer", state)),
    )

    load_lora_adapter([SimpleNamespace(named_parameters=lambda: [(name, param)])], str(tmp_path), optimizer=optimizer)

    assert [event[0] for event in events] == ["reload", "optimizer"]
    assert torch.equal(events[0][1], torch.ones(2))


class TestLoadTrainingStateOptimizerGate:
    """--no-load-optim must keep the fresh optimizer without losing the step or the LR schedule."""

    @staticmethod
    def _recorder():
        loaded = []
        return loaded, SimpleNamespace(load_state_dict=loaded.append)

    @staticmethod
    def _write_training_state(tmp_path):
        torch.save(
            {"iteration": 11, "optimizer": {"step": 7}, "opt_param_scheduler": {"lr": 0.5}},
            tmp_path / "training_state_rank0.pt",
        )

    def test_no_load_optim_skips_the_optimizer_and_keeps_the_rest(self, tmp_path):
        self._write_training_state(tmp_path)
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler, load_optimizer=False) == (
            11,
            False,
            True,
        )
        assert optimizer_loads == []
        assert scheduler_loads == [{"lr": 0.5}]

    def test_load_lora_adapter_forwards_the_flag(self, tmp_path, monkeypatch):
        rank0 = SimpleNamespace(rank=0)
        monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: SimpleNamespace(tp=rank0, pp=rank0))
        name = "layers.0.self_attention.lora_A.weight"
        torch.save({name: torch.ones(2)}, tmp_path / "adapter_megatron_rank0.pt")
        self._write_training_state(tmp_path)
        model = [SimpleNamespace(named_parameters=lambda: [(name, torch.nn.Parameter(torch.zeros(2)))])]
        optimizer_loads, optimizer = self._recorder()
        optimizer.reload_model_params = lambda: None
        scheduler_loads, scheduler = self._recorder()

        loaded, iteration, optimizer_restored, scheduler_restored = load_lora_adapter(
            model,
            str(tmp_path),
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            load_optimizer=False,
        )

        assert (loaded, iteration, optimizer_restored, scheduler_restored) == (True, 11, False, True)
        assert optimizer_loads == []
        assert scheduler_loads == [{"lr": 0.5}]


def test_native_checkpoint_rng_round_trip_restores_next_draw():
    import random

    import numpy as np

    from miles.backends.megatron_utils.lora.checkpoint_state import restore_rng, rng_state

    saved = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(8))
    random.random()
    np.random.rand()
    torch.rand(8)
    restore_rng(saved)
    actual = (random.random(), np.random.rand(), torch.rand(8))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])


def test_native_resume_matches_next_adam_update_on_cpu(tmp_path, monkeypatch):
    """A real optimizer, tensors, scheduler and random gradient; no GPU equivalence claim."""
    rank0 = SimpleNamespace(rank=0)
    monkeypatch.setattr(
        lora_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(
            effective_dp=rank0,
            cp=rank0,
            tp=rank0,
            pp=rank0,
        ),
    )
    args = Namespace(
        megatron_to_hf_mode="raw",
        lora_rank=2,
        lora_alpha=2,
        target_modules=["linear_qkv"],
        lora_adapter_targets=["linear_qkv"],
        lora_dropout=0.0,
        experts_shared_outer_loras=False,
    )

    def training_objects():
        model = _AdapterModel()
        with torch.no_grad():
            model.lora_A.fill_(1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.reload_model_params = lambda: None  # CPU Adam has no separate master parameter copy.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return model, optimizer, scheduler

    def update(model, optimizer, scheduler):
        optimizer.zero_grad()
        (model.lora_A * torch.rand_like(model.lora_A)).square().sum().backward()
        optimizer.step()
        scheduler.step()

    torch.manual_seed(19)
    model, optimizer, scheduler = training_objects()
    update(model, optimizer, scheduler)
    save_lora_checkpoint(
        [model], args, str(tmp_path), publisher=None, optimizer=optimizer, opt_param_scheduler=scheduler, iteration=1
    )
    update(model, optimizer, scheduler)
    expected = model.lora_A.detach().clone()
    resumed, resumed_optim, resumed_scheduler = training_objects()
    assert load_lora_adapter(
        [resumed], str(tmp_path), optimizer=resumed_optim, opt_param_scheduler=resumed_scheduler
    ) == (True, 1, True, True)
    update(resumed, resumed_optim, resumed_scheduler)
    assert torch.equal(resumed.lora_A, expected)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for key, value in optimizer.state[model.lora_A].items():
        assert torch.equal(resumed_optim.state[resumed.lora_A][key], value)
