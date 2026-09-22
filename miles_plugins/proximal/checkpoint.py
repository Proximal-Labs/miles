"""Local snapshot export through Miles's existing Megatron post-save hook."""

import argparse
import logging
from pathlib import Path

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.snapshot import BaseModelIdentity, Nonempty, SnapshotMetadata, prepare_snapshot

logger = logging.getLogger(__name__)


class SnapshotExportConfig(FrozenStrictBaseModel):
    run_id: Nonempty
    base_model: BaseModelIdentity
    output_root: Path


def read_export_config(path: str) -> SnapshotExportConfig:
    try:
        return SnapshotExportConfig.model_validate_json(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid snapshot export configuration: {exc}") from exc


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--proximal-snapshot-config",
        type=read_export_config,
        required=True,
        help="JSON run/base-model/output configuration for local immutable adapter snapshots (no remote publication).",
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not isinstance(getattr(args, "proximal_snapshot_config", None), SnapshotExportConfig):
        raise ValueError("--proximal-snapshot-config must be a validated JSON configuration")
    if getattr(args, "train_backend", None) != "megatron":
        raise ValueError("Snapshot post-save export requires the Megatron training backend")
    if getattr(args, "use_critic", False):
        raise ValueError("Snapshot post-save export currently supports actor-only training")
    if getattr(args, "lora_rank", 0) <= 0 or not getattr(args, "save", None):
        raise ValueError("Snapshot post-save export requires LoRA training and --save")


def proximal_snapshot_post_save(
    args: argparse.Namespace, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None
) -> None:
    """Export only the PEFT adapter from the completed native checkpoint.

    The hook is already called on rank zero after synchronous save completion.
    hf_checkpoint_dir may contain a merged full model; it is intentionally not
    used as a fallback when the adapter export is missing.
    """
    _validate_args(args)
    config = args.proximal_snapshot_config
    assert isinstance(config, SnapshotExportConfig)
    snapshot = prepare_snapshot(
        Path(checkpoint_dir) / "adapter",
        metadata=SnapshotMetadata(run_id=config.run_id, checkpoint_iteration=rollout_id, base_model=config.base_model),
        output_root=config.output_root,
    )
    logger.info("Exported adapter snapshot %s to %s", snapshot.reference.sha256, snapshot.directory)


# Miles discovers attributes on functions dynamically, as for rollout/generate hooks.
proximal_snapshot_post_save.add_arguments = _add_arguments  # type: ignore[attr-defined]
proximal_snapshot_post_save.validate_args = _validate_args  # type: ignore[attr-defined]
