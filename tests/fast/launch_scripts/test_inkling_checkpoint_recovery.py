import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import inkling_checkpoint_recovery as recovery


def _checkpoint(root, iteration, world_size=8):
    for name in recovery.checkpoint_files(iteration, world_size):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")


def _manifest(root, iteration, world_size=8):
    recovery.write_json(root / f"iter_{iteration:07d}" / recovery.COMPLETE_FILE, {
        "iteration": iteration,
        "world_size": world_size,
        "files": {str(name): (root / name).stat().st_size for name in recovery.checkpoint_files(iteration, world_size)},
    })


def test_publication_requires_persisted_peer_shards_and_cursor(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "MOUNT_ROOT", tmp_path)
    root = tmp_path / "run"
    _checkpoint(root, 5, 16)
    recovery.write_json(root / recovery.PROTOCOL_FILE, {"version": 1})
    committed = []
    missing = {"adapter_megatron_rank15.pt", "global_dataset_state_dict_5.pt"}

    def iterdir(directory):
        return [SimpleNamespace(path=str(p.relative_to(tmp_path)), size=p.stat().st_size)
                for p in (tmp_path / directory).iterdir() if p.name not in missing]

    volume = SimpleNamespace(iterdir=iterdir, commit=lambda: committed.append("commit"))
    marker = root / "iter_0000005" / recovery.COMPLETE_FILE
    for absent in ("adapter_megatron_rank15.pt", "global_dataset_state_dict_5.pt"):
        missing.clear()
        missing.add(absent)
        with pytest.raises(RuntimeError, match="persisted shards or dataset cursor"):
            recovery._publish_manifest(root, 5, 16, volume)
        assert not marker.exists()
        assert not committed
    missing.clear()
    recovery._publish_manifest(root, 5, 16, volume)
    assert committed == ["commit"]
    assert recovery.checkpoint_complete(root, 5, 16)
    (root / "iter_0000005/adapter/adapter_megatron_rank15.pt").write_bytes(b"truncated")
    assert not recovery.checkpoint_complete(root, 5, 16)


def test_auto_resume_skips_unpublished_checkpoint_and_rejects_changed_batch(tmp_path):
    pytest.importorskip("modal")
    from scripts.run_inkling_small_sft import ScriptArgs
    from tools.modal_inkling_sft import _recover_run

    args = ScriptArgs(run_id="run", output_dir=str(tmp_path), mode="train")
    root = Path(args.save_dir)
    _checkpoint(root, 5)
    _checkpoint(root, 6)
    recovery.write_json(root / "launch.json", asdict(args))
    recovery.write_json(root / recovery.PROTOCOL_FILE, {"version": 1})
    _manifest(root, 5)
    resumed = _recover_run(args)
    assert resumed.resume
    assert resumed.lora_adapter_path == str(root / "iter_0000005/adapter")
    assert not args.resume
    args.global_batch_size = 64
    with pytest.raises(ValueError, match="changed global_batch_size"):
        _recover_run(args)


def test_initial_restart_and_legacy_migration(tmp_path):
    pytest.importorskip("modal")
    from scripts.run_inkling_small_sft import ScriptArgs
    from tools.modal_inkling_sft import _recover_run

    args = ScriptArgs(run_id="run", output_dir=str(tmp_path), mode="train")
    root = Path(args.save_dir)
    root.mkdir()
    assert not _recover_run(args).resume  # Interrupted before launch.json existed.
    recovery.write_json(root / "launch.json", asdict(args))
    assert not _recover_run(args).resume  # Interrupted before protocol commit.
    recovery.write_json(root / recovery.PROTOCOL_FILE, {"version": 1})
    assert not _recover_run(args).resume
    _checkpoint(root, 5)
    recovery.write_json(root / recovery.PROTOCOL_FILE, {"version": 1, "legacy_iteration": 5})
    assert _recover_run(args).resume
    _checkpoint(root, 6)
    assert not recovery.checkpoint_complete(root, 6, 8)
    args.auto_resume = False
    with pytest.raises(FileExistsError):
        _recover_run(args)


def test_cluster_retry_ignores_old_stop_and_error():
    from tools.modal_inkling_sft_cluster import ClusterState

    backing = {}
    old = ClusterState(backing, "old-allocation")
    old["stop"] = True
    old["error"] = "preempted"
    new = ClusterState(backing, "new-allocation")
    peer = ClusterState(backing, "new-allocation")
    assert new.get("stop") is None and new.get("error") is None
    new["ready-0"] = True
    assert peer.get("ready-0") is True


@pytest.mark.parametrize("interrupted", [False, True])
def test_publication_commits_every_gpu_node_before_marker(monkeypatch, tmp_path, interrupted):
    import sys

    events = []

    class CommitTask:
        def options(self, *, scheduling_strategy):
            self.node = scheduling_strategy
            return self

        def remote(self, environment):
            events.append(("node-commit", self.node, environment))
            return self.node

    def wait(refs):
        events.append(("wait", refs))
        if interrupted:
            raise RuntimeError("worker preempted during commit")

    ray = SimpleNamespace(
        nodes=lambda: [{"Alive": True, "Resources": {"GPU": 8}, "NodeID": name} for name in ("head", "worker")],
        remote=lambda **kwargs: lambda fn: CommitTask(),
        get=wait,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", SimpleNamespace(
        NodeAffinitySchedulingStrategy=lambda node, soft: node,
    ))
    monkeypatch.setattr(recovery, "_volume", lambda env: SimpleNamespace(commit=lambda: events.append("driver-commit")))
    monkeypatch.setattr(recovery, "_publish_manifest", lambda *args: events.append("marker"))
    if interrupted:
        with pytest.raises(RuntimeError, match="preempted"):
            recovery.publish_checkpoint(tmp_path, 5, 16, "main")
        assert "marker" not in events
        return
    recovery.publish_checkpoint(tmp_path, 5, 16, "main")
    assert events == [("node-commit", "head", "main"), ("node-commit", "worker", "main"),
                      ("wait", ["head", "worker"]), "driver-commit", "marker"]


def test_same_modal_input_recovers_across_multiple_interruptions(monkeypatch, tmp_path):
    pytest.importorskip("modal")
    import miles.utils.external_utils.command_utils as U
    from scripts.run_inkling_small_sft import ScriptArgs
    from tools import modal_inkling_sft as launcher

    args = ScriptArgs(run_id="retry", output_dir=str(tmp_path), data_dir=str(tmp_path),
                      model_dir=str(tmp_path), mode="train")
    Path(args.dataset).touch()
    Path(args.torch_dist).mkdir()
    (Path(args.torch_dist) / "latest_checkpointed_iteration.txt").write_text("release")
    original_input = json.dumps(asdict(args))
    root = Path(args.save_dir)
    launches = []
    monkeypatch.setattr(launcher, "_gpu_preflight", lambda: None)
    monkeypatch.setattr(launcher, "volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))

    def worker(command):
        if command == "ray stop --force":
            return
        assert "MILES_INKLING_MODAL_ENVIRONMENT=main" in command
        config = json.loads((root / "launch.json").read_text())
        launches.append(config)
        iteration = len(launches) - 1
        if iteration < 2:
            _checkpoint(root, iteration)
            _manifest(root, iteration)
            _checkpoint(root, iteration + 1)  # Next write interrupted before publication.
            raise KeyboardInterrupt("simulated preemption")

    monkeypatch.setattr(U, "exec_command_cpu", worker)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            launcher.train.local(original_input)
    launcher.train.local(original_input)
    assert not launches[0]["resume"]
    assert launches[1]["lora_adapter_path"] == str(root / "iter_0000000/adapter")
    assert launches[2]["lora_adapter_path"] == str(root / "iter_0000001/adapter")


@pytest.mark.parametrize("nodes", [1, 2])
def test_volume_reload_precedes_training_imports(monkeypatch, nodes):
    pytest.importorskip("modal")
    import builtins
    from tools import modal_inkling_sft as launcher

    events = []
    original_import = builtins.__import__

    class StopBeforeTraining(Exception):
        pass

    def reload():
        assert not events, "GPU imports already opened compiler-cache files"
        events.append("reload")

    def guarded_import(name, *args, **kwargs):
        if name in {"scripts.run_inkling_small_sft", "tools.modal_inkling_sft_cluster"}:
            assert events == ["reload"]
            events.append("training-import")
            raise StopBeforeTraining
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(launcher, "volume", SimpleNamespace(reload=reload))
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(StopBeforeTraining):
        if nodes == 1:
            launcher.train.local(json.dumps({"num_nodes": 1}))
        else:
            launcher.train_two_nodes.local(json.dumps({"num_nodes": 2}), {})
    assert events == ["reload", "training-import"]


def test_cluster_head_does_not_reload_after_gpu_imports(monkeypatch):
    pytest.importorskip("modal")
    from tools import modal_inkling_sft as launcher

    class StopBeforeTraining(Exception):
        pass

    def unexpected_reload():
        pytest.fail("Cluster head must not reload the Volume again after GPU imports")

    def stop(config):
        raise StopBeforeTraining

    monkeypatch.setattr(launcher, "volume", SimpleNamespace(reload=unexpected_reload))
    monkeypatch.setattr(launcher, "_config", stop)
    with pytest.raises(StopBeforeTraining):
        launcher.train.local(json.dumps({"num_nodes": 2}))
