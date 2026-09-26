"""Exercise coordination with two real processes, without Modal, Ray or GPUs."""

import multiprocessing
import os
import signal
from types import SimpleNamespace

import pytest

from tools import modal_inkling_sft_cluster as cluster


def _node(state, rank, scenario):
    # Short deadlines keep lost-peer tests bounded. All process signals and the
    # actual coordination context run unchanged; only GPU/Ray work is replaced.
    cluster._POLL_SECONDS = 0.02
    cluster._STARTUP_SECONDS = 10
    cluster._PEER_TIMEOUT_SECONDS = 0.3
    cluster._CLEANUP_SECONDS = 1
    cluster._start_ray = lambda *args: None
    cluster.U.exec_command_cpu = lambda command: state.__setitem__(f"ray-stopped-{rank}", True)
    volume = SimpleNamespace(commit=lambda: state.__setitem__(f"volume-committed-{rank}", True))
    args = SimpleNamespace(num_nodes=2, num_gpus_per_node=8, timeout_hours=1)
    try:
        with cluster.training_cluster(args, state, rank, ["head", "worker"], volume, lambda: None):
            state[f"entered-{rank}"] = True
            cluster._wait(lambda: state.get("entered-0") and state.get("entered-1"), 10, "test nodes")
            if rank == 1 and scenario == "worker-error":
                raise ValueError("worker failed")
            if rank == 0 and scenario != "success":
                cluster._wait(lambda: False, 10, "test cancellation")
        state[f"success-{rank}"] = True
    except BaseException as exc:
        state[f"exception-{rank}"] = type(exc).__name__


@pytest.mark.parametrize(
    "scenario", ["success", "head-term", "worker-term", "worker-error", "head-kill", "worker-kill"]
)
def test_two_process_shutdown(scenario):
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        state = manager.dict()
        processes = [context.Process(target=_node, args=(state, rank, scenario)) for rank in range(2)]
        try:
            for process in processes:
                process.start()
            cluster._wait(lambda: state.get("entered-0") and state.get("entered-1"), 20, "test startup")
            if scenario in {"head-term", "worker-term", "head-kill", "worker-kill"}:
                rank = 0 if scenario.startswith("head-") else 1
                os.kill(processes[rank].pid, signal.SIGKILL if scenario.endswith("-kill") else signal.SIGTERM)
            for process in processes:
                process.join(timeout=15)
                assert not process.is_alive(), f"Stranded node in {scenario}"
            surviving_ranks = [0] if scenario == "worker-kill" else [1] if scenario == "head-kill" else [0, 1]
            for rank in surviving_ranks:
                assert state[f"ray-stopped-{rank}"]
                assert state[f"volume-committed-{rank}"]
            if scenario == "success":
                assert state["success-0"] and state["success-1"]
            else:
                assert "success-0" not in state
                assert state.get("error")
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                if process.pid is not None:
                    process.join(timeout=5)


def test_ray_head_waits_for_both_gpu_nodes(monkeypatch):
    import ray

    state, commands = {"ready-1": True}, []
    ips = ["10.0.0.1", "10.0.0.2"]
    responses = iter(
        [
            [{"Alive": True, "NodeManagerAddress": ips[0], "Resources": {"GPU": 8}}],
            [{"Alive": True, "NodeManagerAddress": ip, "Resources": {"GPU": 8}} for ip in ips],
        ]
    )
    monkeypatch.setattr(cluster.U, "exec_command_cpu", commands.append)
    monkeypatch.setattr(cluster, "_POLL_SECONDS", 0)
    monkeypatch.setattr(ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(ray, "nodes", lambda: next(responses))
    monkeypatch.setenv("RAY_ADDRESS", "unrelated")
    monkeypatch.setenv("MASTER_ADDR", "unrelated")
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "0")
    cluster._start_ray(state, 0, ips, 8)
    assert len(commands) == 1 and "--head" in commands[0]
    assert state["head-ready"] and state["ready-0"]
    assert os.environ["MILES_SCRIPT_EXTERNAL_RAY"] == "1"
    assert os.environ["MASTER_ADDR"] == ips[0]
    assert "RAY_ADDRESS" not in os.environ


def test_worker_joins_existing_head(monkeypatch):
    commands, state = [], {"head-ready": True, "ready-0": True}
    monkeypatch.setattr(cluster.U, "exec_command_cpu", commands.append)
    monkeypatch.setenv("MASTER_ADDR", "unrelated")
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "0")
    cluster._start_ray(state, 1, ["10.0.0.1", "10.0.0.2"], 8)
    assert len(commands) == 1
    assert "--address=10.0.0.1:6379" in commands[0]
    assert "--node-ip-address 10.0.0.2" in commands[0]
    assert state["ready-1"]
