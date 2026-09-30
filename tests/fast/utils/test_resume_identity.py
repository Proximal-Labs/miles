"""A restarted LoRA run continues its step count and its W&B run."""

from argparse import Namespace

import pytest

from miles.utils.resume import resume_checkpoint_dir, resumes_lora_adapter


def test_only_an_adapter_with_training_state_is_a_resume(tmp_path):
    adapter = tmp_path / "iter_0000003" / "adapter"
    adapter.mkdir(parents=True)
    assert not resumes_lora_adapter(Namespace(lora_adapter_path=None))
    # A warm-start adapter (weights only) starts a fresh run at 0.
    assert not resumes_lora_adapter(Namespace(lora_adapter_path=str(adapter)))
    (adapter / "training_state_rank0.pt").write_bytes(b"")
    assert resumes_lora_adapter(Namespace(lora_adapter_path=str(adapter)))


def test_a_lora_resume_reads_step_state_from_the_adapters_checkpoint_root(tmp_path):
    """Run 013: --load is the HF base on a LoRA resume; the step state sits beside the adapter."""
    adapter = tmp_path / "checkpoints" / "iter_0000003" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "training_state_rank0.pt").write_bytes(b"")
    args = Namespace(load="/models/base", lora_adapter_path=str(adapter) + "/")
    assert resume_checkpoint_dir(args) == str(tmp_path / "checkpoints")


def test_any_other_start_reads_step_state_from_load(tmp_path):
    warm_start = tmp_path / "adapter"
    warm_start.mkdir()
    assert resume_checkpoint_dir(Namespace(load="/ckpt", lora_adapter_path=str(warm_start))) == "/ckpt"
    assert resume_checkpoint_dir(Namespace(load=None, lora_adapter_path=None)) is None


@pytest.fixture
def wandb_init(monkeypatch):
    wandb = pytest.importorskip("wandb")
    from miles.utils.tracking_utils import wandb_utils

    calls: list[dict] = []

    def init(**kwargs):
        calls.append(kwargs)
        monkeypatch.setattr(wandb, "run", Namespace(id=kwargs.get("id", "generated-id")), raising=False)

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda: None)
    return wandb_utils, calls


def _args(**overrides):
    base = dict(
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="run-008",
        wandb_team=None,
        wandb_project="p",
        wandb_dir=None,
        rank=0,
        wandb_run_id=None,
        env_report=None,
    )
    return Namespace(**(base | overrides))


def test_a_requested_wandb_run_id_is_resumed(wandb_init):
    wandb_utils, calls = wandb_init
    args = _args(wandb_run_id="qwen38-27b-overhead-008")
    wandb_utils.init_wandb_primary(args)
    assert calls[-1]["id"] == "qwen38-27b-overhead-008" and calls[-1]["resume"] == "allow"
    assert args.wandb_run_id == "qwen38-27b-overhead-008"


def test_without_a_requested_id_wandb_picks_one(wandb_init):
    wandb_utils, calls = wandb_init
    args = _args()
    wandb_utils.init_wandb_primary(args)
    assert "id" not in calls[-1] and "resume" not in calls[-1]
    assert args.wandb_run_id == "generated-id"
