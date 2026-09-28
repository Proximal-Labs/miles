"""Where a resumed run finds what it saved. Imports nothing from Miles, so any module can use it."""

import os


def resumes_lora_adapter(args) -> bool:
    """Whether --lora-adapter-path is a checkpoint this run saved: it carries training state."""
    adapter = getattr(args, "lora_adapter_path", None)
    return adapter is not None and os.path.exists(os.path.join(adapter, "training_state_rank0.pt"))


def resume_checkpoint_dir(args) -> str | None:
    """The checkpoint root a resume reads per-step state from (``rollout/`` data-source state).

    A LoRA resume loads the base model from --load and the step from --lora-adapter-path
    (``<root>/iter_<N>/adapter``), so its state sits two levels above the adapter, where
    --save wrote it; any other resume reads from --load.
    """
    if resumes_lora_adapter(args):
        return os.path.dirname(os.path.dirname(os.path.normpath(args.lora_adapter_path)))
    return args.load
