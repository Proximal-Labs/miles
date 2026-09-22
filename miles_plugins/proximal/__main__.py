"""CPU-only adapter preparation and explicitly authorized publication/load commands."""

import argparse
import os
from pathlib import Path

import httpx

from miles_plugins.proximal.checkpoint import read_export_config
from miles_plugins.proximal.modal_volume import VolumeDestination, authorize_volume_publication, modal_publish_snapshot
from miles_plugins.proximal.replica import ReplicaConfig, ReplicaLoRALoader, authorize_replica_load
from miles_plugins.proximal.snapshot import SnapshotMetadata, SnapshotReference, prepare_snapshot, read_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare a local immutable bundle; no remote work")
    prepare.add_argument("--adapter-directory", type=Path, required=True)
    prepare.add_argument("--config", type=read_export_config, required=True)
    prepare.add_argument("--checkpoint-iteration", type=int, required=True)

    publish = commands.add_parser("publish-modal", help="Upload a bundle to an existing Modal Volume")
    publish.add_argument("--snapshot-directory", type=Path, required=True)
    publish.add_argument("--sha256", required=True)
    publish.add_argument("--volume-name", required=True)
    publish.add_argument("--environment-name", required=True)
    publish.add_argument("--yes-publish", action="store_true", required=True)

    load = commands.add_parser("load-replica", help="Run inside one replica, against its local SGLang process")
    load.add_argument("--config", type=Path, required=True, help="ReplicaConfig JSON")
    load.add_argument("--sha256", required=True)
    load.add_argument("--volume-name", required=True)
    load.add_argument("--environment-name", required=True)
    load.add_argument("--volume-mount", type=Path, required=True)
    load.add_argument("--local-cache", type=Path, required=True)
    load.add_argument("--api-key-env", default="SGLANG_API_KEY")
    load.add_argument("--timeout-seconds", type=float, default=120)
    load.add_argument("--yes-load", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        snapshot = prepare_snapshot(
            args.adapter_directory,
            metadata=SnapshotMetadata(
                run_id=args.config.run_id,
                checkpoint_iteration=args.checkpoint_iteration,
                base_model=args.config.base_model,
            ),
            output_root=args.config.output_root,
        )
        print(snapshot.reference.model_dump_json())
    elif args.command == "publish-modal":
        snapshot = read_snapshot(args.snapshot_directory, SnapshotReference(sha256=args.sha256))
        destination = VolumeDestination(volume_name=args.volume_name, environment_name=args.environment_name)
        authorization = authorize_volume_publication(destination, yes_publish=args.yes_publish)
        print(modal_publish_snapshot(authorization, snapshot).model_dump_json())
    elif args.command == "load-replica":
        _load_replica(args)


def _load_replica(args: argparse.Namespace) -> None:
    config = ReplicaConfig.model_validate_json(args.config.read_bytes())
    reference = SnapshotReference(sha256=args.sha256)
    destination = VolumeDestination(volume_name=args.volume_name, environment_name=args.environment_name)
    authorization = authorize_replica_load(config, yes_load=args.yes_load)
    if not 0 < args.timeout_seconds < float("inf"):
        raise ValueError("--timeout-seconds must be finite and positive")
    # Optional dependency is imported after free validation, only for the remote command.
    import modal

    volume = modal.Volume.from_name(
        destination.volume_name, environment_name=destination.environment_name, create_if_missing=False
    )
    key = os.environ.get(args.api_key_env)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(headers=headers, timeout=args.timeout_seconds, trust_env=False) as client:
        loader = ReplicaLoRALoader(
            authorization,
            volume_mount=args.volume_mount,
            local_cache=args.local_cache,
            reload_volume=volume.reload,
            client=client,
        )
        print(loader.ensure_loaded(reference).model_dump_json())


if __name__ == "__main__":
    main()
