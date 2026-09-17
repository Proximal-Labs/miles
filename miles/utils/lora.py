from argparse import Namespace

LORA_ADAPTER_NAME = "miles_lora"


def is_lora_weight_name(name: str) -> bool:
    """Check if an HF weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled based on arguments."""
    return args.lora_rank > 0 or args.lora_adapter_path is not None


def lora_rollout_enabled(args: Namespace) -> bool:
    """LoRA enabled AND the rollout side participates; false under --lora-train-only.

    Gates everything rollout-facing: SGLang's ``enable_lora``, the per-request
    ``lora_path``, and the adapter weight sync. Training-side LoRA is unaffected.
    """
    return is_lora_enabled(args) and not args.lora_train_only


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and args.colocate and args.lora_base_cpu_backup
