"""Two-node Modal lifecycle around the existing Miles external-Ray launcher.

The per-call Modal Dict carries only readiness, heartbeats and shutdown status.
It is independent of Ray so a failed Ray job cannot strand the other container.
"""

import os
import shlex
import signal
import threading
import time
from contextlib import contextmanager

import miles.utils.external_utils.command_utils as U

_POLL_SECONDS = 2
_STARTUP_SECONDS = 600
_PEER_TIMEOUT_SECONDS = 120
_CLEANUP_SECONDS = 180


def _wait(predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {description}")
        time.sleep(_POLL_SECONDS)


def _cancel(signum, frame):
    # KeyboardInterrupt also interrupts subprocess.run without waiting forever
    # for the Ray CLI. The finally block explicitly stops the Ray daemons.
    raise KeyboardInterrupt(f"Stopping training node after signal {signum}")


def _monitor(state, rank, stopped, timeout):
    peer = 1 - rank
    deadline = time.monotonic() + timeout
    last_heartbeat = None
    last_seen = time.monotonic()
    try:
        while not stopped.is_set():
            state[f"heartbeat-{rank}"] = time.monotonic()
            heartbeat = state.get(f"heartbeat-{peer}")
            if heartbeat != last_heartbeat:
                last_seen, last_heartbeat = time.monotonic(), heartbeat
            grace = _STARTUP_SECONDS if heartbeat is None else _PEER_TIMEOUT_SECONDS
            if state.get("error"):
                raise RuntimeError(state["error"])
            if not state.get("stop") and time.monotonic() - last_seen > grace:
                raise TimeoutError(f"Training node {peer} stopped responding")
            if time.monotonic() > deadline:
                raise TimeoutError("Training exceeded its time limit")
            stopped.wait(_POLL_SECONDS)
    except Exception as exc:
        if not stopped.is_set():
            print(f"Cluster monitor: {exc}", flush=True)
            try:
                state["error"] = f"Node {rank}: {exc}"
            finally:
                if not stopped.is_set():
                    os.kill(os.getpid(), signal.SIGTERM)


def _start_ray(state, rank, ips, num_gpus):
    head, address = shlex.quote(ips[0]), shlex.quote(ips[rank])
    os.environ.update(MASTER_ADDR=ips[0], MILES_SCRIPT_EXTERNAL_RAY="1")
    # Job submission uses the local head dashboard; do not inherit a client address.
    os.environ.pop("RAY_ADDRESS", None)
    if rank == 0:
        U.exec_command_cpu(
            f"timeout 120 ray start --head --port=6379 --node-ip-address {address} "
            f"--num-gpus {num_gpus} --disable-usage-stats"
        )
        state["head-ready"] = True
    else:
        _wait(lambda: state.get("head-ready"), _STARTUP_SECONDS, "Ray head")
        U.exec_command_cpu(
            f"timeout 120 ray start --address={head}:6379 --node-ip-address {address} "
            f"--num-gpus {num_gpus} --disable-usage-stats"
        )
    state[f"ready-{rank}"] = True
    _wait(lambda: all(state.get(f"ready-{i}") for i in range(2)), _STARTUP_SECONDS, "both Ray nodes")
    if rank == 0:
        # Starting a raylet is not sufficient: wait until the head sees both GPU nodes.
        import ray

        ray.init(address=f"{ips[0]}:6379")
        try:
            _wait(
                lambda: {
                    n["NodeManagerAddress"]
                    for n in ray.nodes()
                    if n["Alive"] and n["Resources"].get("GPU") == num_gpus
                }
                == set(ips),
                _STARTUP_SECONDS,
                "all 16 GPUs registered with Ray",
            )
        finally:
            ray.shutdown()


@contextmanager
def training_cluster(args, state, rank, ips, volume, preflight):
    """Yield on both nodes; the caller runs training only on rank zero.

    The worker waits for the head on normal exit. Either node failing signals the
    other. Each node stops Ray and commits before rank zero can report success.
    Hard Modal app stops may bypass Python cleanup; persisted checkpoints survive.
    """
    if args.num_nodes != 2 or len(ips) != 2 or rank not in (0, 1):
        raise ValueError("Expected a two-node cluster")
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, _cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    monitor = threading.Thread(
        target=_monitor, args=(state, rank, stopped, args.timeout_hours * 3600 - 120), daemon=True
    )
    monitor.start()
    try:
        preflight()
        _start_ray(state, rank, ips, args.num_gpus_per_node)
        yield
        if rank != 0:
            _wait(lambda: state.get("stop"), args.timeout_hours * 3600, "head to finish")
    except BaseException as exc:
        state["error"] = f"Node {rank}: {type(exc).__name__}: {exc}"
        raise
    finally:
        stopped.set()
        monitor.join(timeout=10)
        try:
            try:
                state["stop"] = True
            finally:
                # Cleanup must still run if the coordination service is unavailable.
                try:
                    U.exec_command_cpu("timeout 60 ray stop --force")
                finally:
                    volume.commit()
            state[f"committed-{rank}"] = True
            if rank == 0:
                _wait(lambda: state.get("committed-1"), _CLEANUP_SECONDS, "worker checkpoint commit")
                if state.get("error"):
                    raise RuntimeError(state["error"])
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
