"""Exercise actor sync lifecycle without importing the GPU-only Megatron stack."""

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _actor_method(name, **dependencies):
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/actor.py"
    tree = ast.parse(path.read_text())
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    method = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(dependencies)
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace[name]


def test_train_only_raw_lora_never_initializes_inference_sync():
    updater = Mock(side_effect=AssertionError("No inference sync in offline training"))
    actor = SimpleNamespace(args=SimpleNamespace(debug_train_only=True, megatron_to_hf_mode="raw"))
    _actor_method("_init_training_state", WeightUpdater=updater)(actor)
    assert actor.weight_updater is None
    updater.assert_not_called()


@pytest.mark.parametrize("mode", ["raw", "bridge"])
def test_rollout_lora_retains_bridge_requirement_and_sync(mode):
    updater = Mock()
    actor = SimpleNamespace(
        args=SimpleNamespace(debug_train_only=False, colocate=False, megatron_to_hf_mode=mode, model_name="inkling"),
        hf_config=SimpleNamespace(),
        model=object(),
        _get_actor_weights=Mock(),
    )
    init = _actor_method(
        "_init_training_state",
        WeightUpdater=updater,
        lora_rollout_enabled=lambda args: True,
        build_lora_sync_config=lambda args: "lora-config",
        get_hf_weight_iterator=Mock(),
        get_parallel_state=lambda: "parallel-state",
    )
    if mode == "raw":
        with pytest.raises(AssertionError, match="requires.*bridge"):
            init(actor)
        updater.assert_not_called()
    else:
        init(actor)
        updater.assert_called_once()
        assert actor.weight_updater is updater.return_value


@pytest.mark.parametrize("updater", [None, Mock()])
def test_reconfigure_preserves_training_groups_without_requiring_inference(updater):
    reconfigure = Mock()
    actor = SimpleNamespace(weight_updater=updater, _indep_dp_store_addr="store")
    method = _actor_method(
        "reconfigure_indep_dp",
        reconfigure_indep_dp_group=reconfigure,
        get_parallel_state=lambda: "parallel-state",
        dist=SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 8),
    )
    method(actor, "new-group")
    reconfigure.assert_called_once()
    if updater is not None:
        updater.conn_status.mark_trainer_stale.assert_called_once()
