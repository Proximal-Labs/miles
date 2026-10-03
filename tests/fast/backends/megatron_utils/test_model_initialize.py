import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch

if TYPE_CHECKING:
    from miles.backends.megatron_utils.model import LoadCheckpointOutput


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    if attrs is not None:
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


class _DummyDDP:
    pass


class _DummyModel:
    pass


class _DummyOptimizer:
    pass


class _DummyChainedOptimizer:
    pass


class _DummyDistributedOptimizer:
    pass


class _DummyScheduler:
    pass


class _DummyOptimizerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModelChunk:
    role: str | None = None


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_environment():
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron", is_package=True)
        core_module = _stub_module("megatron.core", is_package=True)
        core_module.mpu = types.SimpleNamespace()
        core_module.tensor_parallel = _stub_module(
            "megatron.core.tensor_parallel",
            {"model_parallel_cuda_manual_seed": MagicMock()},
            is_package=True,
        )
        _stub_module(
            "megatron.core.tensor_parallel.random",
            {"_get_all_rng_states": MagicMock(), "_set_all_rng_states": MagicMock()},
        )
        _stub_module(
            "megatron.core.distributed",
            {
                "DistributedDataParallel": _DummyDDP,
                "finalize_model_grads": MagicMock(),
            },
        )
        _stub_module(
            "megatron.core.enums",
            {"ModelType": types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")},
        )
        _stub_module("megatron.core.models", is_package=True)
        _stub_module("megatron.core.models.gpt", {"GPTModel": _DummyModel})
        _stub_module(
            "megatron.core.optimizer",
            {
                "OptimizerConfig": _DummyOptimizerConfig,
                "get_megatron_optimizer": MagicMock(),
                "Adam": _DummyOptimizer,
                "CPUAdam": _DummyOptimizer,
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.emerging_optimizers", {"TensorParallelMuon": _DummyOptimizer})
        _stub_module("megatron.core.optimizer.muon", {"get_megatron_muon_optimizer": MagicMock()})
        _stub_module("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": _DummyDistributedOptimizer})
        _stub_module(
            "megatron.core.optimizer.optimizer",
            {
                "ChainedOptimizer": _DummyChainedOptimizer,
                "MegatronOptimizer": _DummyOptimizer,
            },
        )
        _stub_module("megatron.core.optimizer_param_scheduler", {"OptimizerParamScheduler": _DummyScheduler})
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": MagicMock()})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock(), "unwrap_model": MagicMock()})
        _stub_module("megatron.core.config", {"set_experimental_flag": MagicMock()})
        _stub_module("megatron.core.num_microbatches_calculator", {"init_num_microbatches_calculator": MagicMock()})
        _stub_module("megatron.training", is_package=True)
        _stub_module(
            "megatron.training.global_vars",
            {
                "get_args": MagicMock(),
                "_build_tokenizer": MagicMock(),
                "set_args": MagicMock(),
            },
        )
        _stub_module("megatron.training.training", {"get_model": MagicMock()})
        _stub_module(
            "megatron.training.checkpointing",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
            },
        )
        _stub_module("sglang.srt.debug_utils", is_package=True)
        _stub_module(
            "sglang.srt.debug_utils.dumper",
            {
                "DumperConfig": MagicMock(),
                "_get_rank": MagicMock(return_value=0),
                "dumper": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.lora.bridge",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.model_provider",
            {
                "get_model_provider_func": MagicMock(),
                "LinearForLastLayer": _DummyModel,
            },
        )
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=True, global_batch_size=8, finetune=False)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0, False, False))
        )
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (
        model,
        optimizer,
        opt_param_scheduler,
        LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101),
    )
    opt_param_scheduler.step.assert_not_called()


def test_initialize_steps_scheduler_when_checkpoint_did_not_restore_it():
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=8, finetune=False)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0, False, False))
        )
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (
        model,
        optimizer,
        opt_param_scheduler,
        LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101),
    )
    opt_param_scheduler.step.assert_called_once_with(increment=800)


def _load_model_state_with(
    *, tmp_path: Path, finetune: bool, iteration: int, lora_rank: int = 0
) -> "LoadCheckpointOutput":
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir()
    (load_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    with ExitStack() as stack:
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(iteration, 0, False, False))
        )
        _patch_initialize_side_effects(stack)
        return load_model_state(
            Namespace(
                use_checkpoint_opt_param_scheduler=True,
                global_batch_size=8,
                finetune=finetune,
                lora_rank=lora_rank,
                megatron_to_hf_mode="core",
                lora_adapter_path=None,
                load=str(load_dir),
            ),
            model=[_FakeModelChunk()],
            optimizer=None,
            opt_param_scheduler=None,
            role="actor",
            checkpointing_context=None,
        )


class TestWhereALoadSaysTheRunStarts:
    def test_a_finetune_load_starts_the_run_at_rollout_zero(self, tmp_path: Path):
        """--finetune means there is no run to continue, so rollout 0 is still ahead rather than behind."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=0).start_rollout_id == 0

    def test_a_resumed_load_starts_the_run_after_the_checkpoint_it_read(self, tmp_path: Path):
        """The checkpoint's own rollout is done, so the run continues at the next one."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=100).start_rollout_id == 101

    def test_a_run_that_restored_the_iteration_zero_checkpoint_it_wrote_starts_at_one(self, tmp_path: Path):
        """A real resume from the very first checkpoint must not be read as a finetune that starts over."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=0).start_rollout_id == 1

    def test_a_finetune_load_that_found_a_checkpoint_is_refused(self, tmp_path: Path):
        """--finetune promises iteration 0; anything else means the two disagree about where the run stands."""
        with pytest.raises(AssertionError, match="disagree about where this run stands"):
            _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=100)


class TestALoraAdapterThatCarriesItsOwnIteration:
    def test_a_lora_resume_under_finetune_continues_after_the_iteration_the_adapter_names(self, tmp_path: Path):
        """LoRA saves write no tracker, so a lora resume always arrives here with --finetune set."""
        output = _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=100, lora_rank=8)

        assert output.start_rollout_id == 101

    def test_a_lora_run_that_really_starts_from_scratch_still_starts_at_rollout_one(self, tmp_path: Path):
        """An adapter with no training state answers iteration 0, and the run continues from the next rollout."""
        assert _load_model_state_with(tmp_path=tmp_path, finetune=True, iteration=0, lora_rank=8).start_rollout_id == 1


_GLOBAL_BATCH = 64
_ADAPTER_PARAM = "decoder.layers.0.self_attention.linear_qkv.lora_A.weight"
_SCHEDULER_FLAGS = {
    "override": dict(override_opt_param_scheduler=True, use_checkpoint_opt_param_scheduler=False),
    "neither": dict(override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=False),
    "use_checkpoint": dict(override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=True),
}


class _MegatronScheduler:
    """OptimizerParamScheduler's step count: load_state_dict steps by the saved num_steps."""

    def __init__(self) -> None:
        self.num_steps = 0

    def step(self, increment: int) -> None:
        self.num_steps += increment

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.step(increment=state_dict["num_steps"])


class _AdapterChunk:
    def __init__(self) -> None:
        self.lora_A = torch.nn.Parameter(torch.zeros(2))

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(_ADAPTER_PARAM, self.lora_A)]


def _save_lora_checkpoint(directory: Path, *, iteration: int, scheduler_state: dict[str, int] | None) -> Path:
    """One rank's native LoRA checkpoint, saved with --no-save-optim."""
    directory.mkdir(parents=True)
    torch.save({_ADAPTER_PARAM: torch.ones(2)}, directory / "adapter_megatron_rank0.pt")
    torch.save(
        {"iteration": iteration, "optimizer": None, "opt_param_scheduler": scheduler_state},
        directory / "training_state_rank0.pt",
    )
    return directory


def _load_lora_run(
    tmp_path: Path, *, adapter: Path | None, scheduler: _MegatronScheduler, flags: dict[str, bool]
) -> "LoadCheckpointOutput":
    """Load the way a LoRA process does: a --finetune base load, then the native adapter, through the real loaders."""
    from miles.backends.megatron_utils.model import load_model_state

    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    (base / "latest_checkpointed_iteration.txt").write_text("0")
    args = Namespace(
        load=str(base),
        finetune=True,
        no_load_optim=False,
        no_load_rng=True,
        lora_rank=8,
        lora_adapter_path=None if adapter is None else str(adapter),
        megatron_to_hf_mode="raw",
        custom_model_provider_path=None,
        global_batch_size=_GLOBAL_BATCH,
        **flags,
    )
    rank0 = types.SimpleNamespace(rank=0)

    with ExitStack() as stack:
        stack.enter_context(patch("miles.backends.megatron_utils.checkpoint.get_args", return_value=args))
        # Under --finetune Megatron restores neither an iteration nor the scheduler from the base checkpoint.
        stack.enter_context(
            patch("miles.backends.megatron_utils.checkpoint._load_checkpoint_megatron", return_value=(0, 0))
        )
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.lora.utils.get_parallel_state",
                return_value=types.SimpleNamespace(tp=rank0, pp=rank0),
            )
        )
        _patch_initialize_side_effects(stack)
        return load_model_state(
            args,
            model=[_AdapterChunk()],
            optimizer=types.SimpleNamespace(reload_model_params=lambda: None),
            opt_param_scheduler=scheduler,
            role="actor",
            checkpointing_context=None,
        )


class TestALoraResumeKeepsTheScheduleItRestored:
    @pytest.mark.parametrize("flags", _SCHEDULER_FLAGS.values(), ids=_SCHEDULER_FLAGS.keys())
    def test_the_restored_count_is_not_advanced_by_the_iteration_again(self, tmp_path: Path, flags: dict[str, bool]):
        """The 2026-10-02 batch chain (global batch 64, one update per process) saved num_steps 64, 128, then 256:
        the third process restored 128 at iteration 1 and stepped by another 1 x 64 before its update."""
        adapter = _save_lora_checkpoint(tmp_path / "iter_0000001", iteration=1, scheduler_state={"num_steps": 128})
        scheduler = _MegatronScheduler()

        output = _load_lora_run(tmp_path, adapter=adapter, scheduler=scheduler, flags=flags)
        assert output.start_rollout_id == 2
        assert scheduler.num_steps == 128

        scheduler.step(increment=_GLOBAL_BATCH)  # the one update train_one_step applies
        assert scheduler.num_steps == 192

    def test_an_adapter_saved_without_scheduler_state_still_advances_by_the_iteration(self, tmp_path: Path):
        """With nothing restored, the iteration is all a fresh scheduler can go by."""
        adapter = _save_lora_checkpoint(tmp_path / "iter_0000001", iteration=1, scheduler_state=None)
        scheduler = _MegatronScheduler()

        _load_lora_run(tmp_path, adapter=adapter, scheduler=scheduler, flags=_SCHEDULER_FLAGS["override"])

        assert scheduler.num_steps == 1 * _GLOBAL_BATCH

    def test_a_run_without_an_adapter_starts_its_schedule_at_zero(self, tmp_path: Path):
        scheduler = _MegatronScheduler()

        _load_lora_run(tmp_path, adapter=None, scheduler=scheduler, flags=_SCHEDULER_FLAGS["override"])

        assert scheduler.num_steps == 0
