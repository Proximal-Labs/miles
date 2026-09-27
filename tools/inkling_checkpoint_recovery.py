"""Durable checkpoint publication for the Modal Inkling launcher.

GPU/Ray/Modal imports are deferred so checkpoint selection can run on the CPU CLI.
"""

import json
from pathlib import Path

PROTOCOL_FILE = "checkpoint_protocol.json"
COMPLETE_FILE = "complete.json"
MOUNT_ROOT = Path("/mnt/inkling")


def checkpoint_files(iteration, world_size):
    adapter = Path(f"iter_{iteration:07d}") / "adapter"
    return [
        *(adapter / f"{prefix}{rank}.pt" for rank in range(world_size)
          for prefix in ("adapter_megatron_rank", "training_state_rank")),
        adapter / "adapter_config.json",
        Path(f"rollout/global_dataset_state_dict_{iteration}.pt"),
    ]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_complete(root, iteration, world_size):
    root = Path(root)
    required = checkpoint_files(iteration, world_size)
    if not all((root / name).is_file() and (root / name).stat().st_size > 0 for name in required):
        return False
    protocol = root / PROTOCOL_FILE
    if not protocol.exists():
        # Legacy runs predate durable publication. Validate their complete file set.
        return True
    marker = root / f"iter_{iteration:07d}" / COMPLETE_FILE
    if not marker.exists():
        return json.loads(protocol.read_text()).get("legacy_iteration") == iteration
    try:
        manifest = json.loads(marker.read_text())
        return manifest["iteration"] == iteration and manifest["world_size"] == world_size and all(
            manifest["files"].get(str(name)) == (root / name).stat().st_size for name in required
        )
    except (ValueError, KeyError, TypeError):
        return False


def _volume(environment):
    import modal

    return modal.Volume.from_name("inkling-small-rft", environment_name=environment)


def invalidate_checkpoint(root, iteration, environment):
    marker = Path(root) / f"iter_{iteration:07d}" / COMPLETE_FILE
    if marker.exists():
        marker.unlink()
        _volume(environment).commit()


def _commit_node(environment):
    _volume(environment).commit()


def publish_checkpoint(root, iteration, world_size, environment):
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Checkpoint writes and the dataset cursor have finished before this call.
    # Commit on each node: a commit on the head cannot flush a peer's mount.
    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0)]
    if sum(node["Resources"]["GPU"] for node in nodes) != world_size:
        raise RuntimeError("Cannot publish checkpoint: training nodes are missing")
    commit = ray.remote(num_cpus=0, max_retries=0)(_commit_node)
    ray.get([
        commit.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(environment)
        for node in nodes
    ])
    volume = _volume(environment)
    volume.commit()  # Include driver-side evaluation metadata as well.
    _publish_manifest(root, iteration, world_size, volume)


def _publish_manifest(root, iteration, world_size, volume):
    root = Path(root)
    relative = root.relative_to(MOUNT_ROOT)
    required = checkpoint_files(iteration, world_size)
    # Inspect persisted storage, not the head's potentially stale view of peer files.
    sizes = {}
    for directory in {name.parent for name in required}:
        for entry in volume.iterdir(str(relative / directory)):
            sizes[str(Path(entry.path).relative_to(relative))] = entry.size
    if not all(sizes.get(str(name), 0) > 0 for name in required):
        raise RuntimeError("Cannot publish checkpoint: persisted shards or dataset cursor are missing")
    write_json(root / f"iter_{iteration:07d}" / COMPLETE_FILE, {
        "iteration": iteration,
        "world_size": world_size,
        "files": {str(name): sizes[str(name)] for name in required},
    })
    volume.commit()
