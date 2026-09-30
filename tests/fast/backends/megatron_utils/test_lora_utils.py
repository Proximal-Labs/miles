"""Unit tests for miles.backends.megatron_utils.lora.utils.

Tests cover module name conversion, LoRA detection helpers, parameter identification,
exclude-module parsing, and LoRA sync config building — all without GPU.
"""

import sys
import types
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

import miles.backends.megatron_utils.lora.utils as lora_utils
from miles.backends.megatron_utils.lora.utils import (
    _get_lora_class_name,
    _is_adapter_param_name,
    build_lora_sync_config,
    convert_target_modules_to_hf,
    convert_target_modules_to_megatron,
    is_lora_enabled,
    load_lora_adapter,
    parse_exclude_modules,
    save_lora_checkpoint,
)
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_weight_name

# ---------------------------------------------------------------------------
# _get_lora_class_name
# ---------------------------------------------------------------------------


class TestGetLoraClassName:
    def test_none_returns_canonical(self):
        assert _get_lora_class_name(None) == "CanonicalLoRA"

    def test_type_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA) == "FakeLoRA"

    def test_instance_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA()) == "FakeLoRA"


# ---------------------------------------------------------------------------
# convert_target_modules_to_megatron
# ---------------------------------------------------------------------------


def _make_lora_type(name: str):
    """Helper to create a mock lora_type whose class name matches *name*."""
    mock = MagicMock()
    type(mock).__name__ = name
    return mock


class TestConvertTargetModulesToMegatron:
    def test_gdn_hf_names_collapse_to_fused_in_proj(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(["in_proj_qkvz", "in_proj_ba", "out_proj"], lora_type=lora)
        assert result == ["in_proj", "out_proj"]

    # --- "all-linear" variants ------------------------------------------------

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_canonical(self, shorthand):
        result = convert_target_modules_to_megatron(shorthand, lora_type=None)
        assert result == [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_standard(self, shorthand):
        lora_type = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(shorthand, lora_type=lora_type)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_single_element_list(self, shorthand):
        result = convert_target_modules_to_megatron([shorthand], lora_type=None)
        assert len(result) == 7  # CanonicalLoRA has 7 modules

    # --- HF -> Megatron conversion (standard LoRA) ----------------------------

    def test_hf_to_megatron_standard_dedup(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=lora)
        assert result == ["linear_qkv"]

    def test_hf_to_megatron_standard_all_modules(self):
        lora = _make_lora_type("LoRA")
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=lora)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    # --- HF -> Megatron conversion (CanonicalLoRA) ----------------------------

    def test_hf_to_megatron_canonical_split(self):
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=None)
        assert result == ["linear_q", "linear_k", "linear_v"]

    def test_hf_to_megatron_canonical_gate_up(self):
        result = convert_target_modules_to_megatron(["gate_proj", "up_proj"], lora_type=None)
        assert result == ["linear_fc1_gate", "linear_fc1_up"]

    # --- Already in Megatron format -------------------------------------------

    def test_megatron_format_passthrough(self):
        modules = ["linear_qkv", "linear_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    def test_megatron_format_passthrough_canonical(self):
        modules = ["linear_q", "linear_k", "linear_v"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    # --- Single string input --------------------------------------------------

    def test_single_hf_string_input(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron("o_proj", lora_type=lora)
        assert result == ["linear_proj"]


# ---------------------------------------------------------------------------
# convert_target_modules_to_hf
# ---------------------------------------------------------------------------


class TestConvertTargetModulesToHf:
    def test_standard_linear_qkv(self):
        assert convert_target_modules_to_hf(["linear_qkv"]) == ["q_proj", "k_proj", "v_proj"]

    def test_standard_linear_proj(self):
        assert convert_target_modules_to_hf(["linear_proj"]) == ["o_proj"]

    def test_standard_linear_fc1(self):
        assert convert_target_modules_to_hf(["linear_fc1"]) == ["gate_proj", "up_proj"]

    def test_standard_linear_fc2(self):
        assert convert_target_modules_to_hf(["linear_fc2"]) == ["down_proj"]

    def test_gdn_in_proj_expands_to_sglang_modules(self):
        assert convert_target_modules_to_hf(["in_proj"]) == ["in_proj_qkvz", "in_proj_ba"]

    @pytest.mark.parametrize(
        "module,expected",
        [
            ("out_proj", ["out_proj"]),  # same-name passthrough
            ("language_model.decoder.layers.*.self_attention.out_proj", ["out_proj"]),  # wildcard path
            ("language_model.decoder.layers.0.self_attention.out_proj", ["out_proj"]),  # dotted path, passthrough
            (
                "language_model.decoder.layers.0.self_attention.linear_qkv",  # dotted path, table-mapped
                ["q_proj", "k_proj", "v_proj"],
            ),
        ],
    )
    def test_paths_reduce_to_leaf_before_mapping(self, module, expected):
        assert convert_target_modules_to_hf([module]) == expected

    def test_canonical_split_modules(self):
        result = convert_target_modules_to_hf(["linear_q", "linear_k", "linear_v"])
        assert result == ["q_proj", "k_proj", "v_proj"]

    def test_canonical_fc1_gate_up(self):
        result = convert_target_modules_to_hf(["linear_fc1_gate", "linear_fc1_up"])
        assert result == ["gate_proj", "up_proj"]

    def test_unknown_module_passthrough(self):
        assert convert_target_modules_to_hf(["some_custom_module"]) == ["some_custom_module"]

    def test_roundtrip_canonical_all_linear(self):
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=None)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def test_roundtrip_standard_all_linear(self):
        lora = _make_lora_type("LoRA")
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=lora)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


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
# parse_exclude_modules
# ---------------------------------------------------------------------------


class TestParseExcludeModules:
    def test_none(self):
        args = Namespace(exclude_modules=None)
        assert parse_exclude_modules(args) == []

    def test_single_module_string(self):
        args = Namespace(exclude_modules="o_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert result == ["linear_proj"]

    def test_comma_separated(self):
        args = Namespace(exclude_modules="o_proj, down_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_list_input(self):
        args = Namespace(exclude_modules=["o_proj", "down_proj"])
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_missing_attr(self):
        args = Namespace()
        assert parse_exclude_modules(args) == []


# ---------------------------------------------------------------------------
# build_lora_sync_config
# ---------------------------------------------------------------------------


class TestBuildLoraSyncConfig:
    def test_basic_config(self):
        args = Namespace(
            lora_rank=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        )
        config = build_lora_sync_config(args)
        assert config["peft_type"] == "LORA"
        assert config["r"] == 32
        assert config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.0
        assert config["bias"] == "none"
        assert config["task_type"] == "CAUSAL_LM"
        assert set(config["target_modules"]) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_no_target_modules_uses_default(self):
        args = Namespace(lora_rank=16, lora_alpha=16, lora_dropout=0.0, target_modules=None)
        config = build_lora_sync_config(args)
        assert len(config["target_modules"]) == 7

    def test_canonical_target_modules(self):
        args = Namespace(
            lora_rank=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["linear_q", "linear_k"],
        )
        config = build_lora_sync_config(args)
        assert config["target_modules"] == ["q_proj", "k_proj"]
        assert config["r"] == 8


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"


class TestBuildLoraSyncConfigUnderMultiLora:
    @staticmethod
    def _args(**overrides):
        return Namespace(
            **{
                "lora_rank": 8,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "target_modules": ["linear_qkv"],
                "multi_lora": False,
                **overrides,
            }
        )

    def test_a_single_adapter_on_an_inkling_checkpoint_publishes_the_shorthand(self, monkeypatch):
        """The engine was launched to auto-detect its targets, so the adapter must say the same."""
        monkeypatch.setattr(
            "miles.backends.megatron_utils.lora.utils.sglang_lora_target_all_sentinel", lambda _a: True
        )

        assert build_lora_sync_config(self._args())["target_modules"] == "all-linear"


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
    def _save(self, tmp_path, monkeypatch, *, no_save_optim, scheduler=None, optimizer=None):
        rank0 = SimpleNamespace(rank=0)
        monkeypatch.setattr(
            lora_utils, "get_parallel_state", lambda: SimpleNamespace(effective_dp=rank0, cp=rank0, tp=rank0, pp=rank0)
        )
        # the HF PEFT export is best-effort and needs a real bridge; fail it fast
        bridge = types.ModuleType("megatron.bridge")
        bridge.AutoBridge = SimpleNamespace(
            from_hf_pretrained=Mock(side_effect=RuntimeError("no bridge in this test"))
        )
        monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)

        adapter = torch.nn.Parameter(torch.ones(2))
        model = [SimpleNamespace(named_parameters=lambda: [("layers.0.self_attention.lora_A.weight", adapter)])]
        args = Namespace(
            hf_checkpoint="/nonexistent",
            # the bridge export path; raw mode writes the rank-sharded config instead
            megatron_to_hf_mode="bridge",
            target_modules=None,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            no_save_optim=no_save_optim,
        )
        optimizer = optimizer or SimpleNamespace(state_dict=lambda: {"step": 7})
        save_lora_checkpoint(
            model, args, str(tmp_path), optimizer=optimizer, opt_param_scheduler=scheduler, iteration=3
        )
        return sorted(path.name for path in tmp_path.iterdir())

    @staticmethod
    def _state(tmp_path):
        return torch.load(tmp_path / "training_state_rank0.pt", weights_only=False)

    def test_training_state_is_written_by_default(self, tmp_path, monkeypatch):
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, monkeypatch, no_save_optim=False, scheduler=scheduler)

        assert files == ["adapter_megatron_rank0.pt", "native_checkpoint.json", "training_state_rank0.pt"]
        state = self._state(tmp_path)
        assert state["optimizer"] == {"step": 7}
        assert state["opt_param_scheduler"] == {"lr": 0.5}
        assert state["iteration"] == 3

    def test_no_save_optim_drops_the_optimizer_and_keeps_the_resume_metadata(self, tmp_path, monkeypatch):
        """--no-save-optim is about optimizer state; losing the step and the LR schedule with it
        would silently restart a resumed run from iteration 0."""
        scheduler = SimpleNamespace(state_dict=lambda: {"lr": 0.5})
        files = self._save(tmp_path, monkeypatch, no_save_optim=True, scheduler=scheduler)

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

        self._save(tmp_path, monkeypatch, no_save_optim=False, optimizer=optimizer)

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

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == 3
        assert optimizer_loads == []
        assert scheduler_loads == [{"lr": 0.5}]

    def test_a_full_checkpoint_restores_the_optimizer(self, tmp_path):
        self._write(tmp_path, {"step": 7})
        optimizer_loads, optimizer = self._recorder()
        scheduler_loads, scheduler = self._recorder()

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler) == 3
        assert optimizer_loads == [{"step": 7}]
        assert scheduler_loads == [{"lr": 0.5}]


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

        assert lora_utils._load_training_state(tmp_path, self._optimizer(events), None) == 3

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

        assert lora_utils._load_training_state(tmp_path, self._optimizer(events), None, load_optimizer=False) == 3
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

        assert lora_utils._load_training_state(tmp_path, optimizer, scheduler, load_optimizer=False) == 11
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

        loaded, iteration = load_lora_adapter(
            model,
            str(tmp_path),
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            load_optimizer=False,
        )

        assert (loaded, iteration) == (True, 11)
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
    save_lora_checkpoint([model], args, str(tmp_path), optimizer=optimizer, opt_param_scheduler=scheduler, iteration=1)
    update(model, optimizer, scheduler)
    expected = model.lora_A.detach().clone()
    resumed, resumed_optim, resumed_scheduler = training_objects()
    assert load_lora_adapter(
        [resumed], str(tmp_path), optimizer=resumed_optim, opt_param_scheduler=resumed_scheduler
    ) == (True, 1)
    update(resumed, resumed_optim, resumed_scheduler)
    assert torch.equal(resumed.lora_A, expected)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    for key, value in optimizer.state[model.lora_A].items():
        assert torch.equal(resumed_optim.state[resumed.lora_A][key], value)
