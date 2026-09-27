"""Step sizing: a model's trainer on N clustered nodes, fed fixed-length mock rollouts.

Answers "how large can a training step be" before paying for real rollouts. For each
phase (samples per optimizer step) it runs Miles's ``train.py`` on ``SIZING_NODES``
clustered 8-GPU nodes with RDMA and records per step:

- Miles's perf timers (log-prob recompute, actor train, step time, tokens/s);
- peak GPU memory per node (``nvidia-smi`` every 2 s), for fit headroom.

Every sample has the same length (``--length``, default 258,000 tokens), so step 1 of a
phase pays the one-time costs (kernel compiles, allocator warm-up) and later steps are
steady state: report those; step 1 only shows the warm-up cost.

Profiles (``SIZING_PROFILE``):

- ``qwen38``: exactly the production trainer (``trainer_replay``'s command: the
  deployment's image, model/train arguments, behavior correction and kernel cache).
  Needs PROXIMAL_RUN_CONFIG / PROXIMAL_SERVING_CONFIG / PROXIMAL_TRAINING_CONFIG.
- ``inkling-small``: the arguments of the validated Inkling-Small LoRA SFT recipe
  (``scripts/run_inkling_small_sft.py``: torch_dist base, all-linear + shared-outer
  expert adapters), with the SFT loss swapped for the RL loss and TIS recompute.
  Layout from ``--extra`` (e.g. TP8/PP2/EP8).

Nothing is written to shared Volumes: weights and caches mount read-only (the Qwen
kernel cache is copied to local disk) and checkpoint saving is disabled.

    SIZING_PROFILE=qwen38 SIZING_NODES=8 PROXIMAL_RUN_CONFIG=run.json \\
    PROXIMAL_SERVING_CONFIG=examples/proximal/qwen38/overhead/serving.json \\
    PROXIMAL_TRAINING_CONFIG=examples/proximal/qwen38/overhead/training.json \\
      modal run --env main -m miles_plugins.proximal.e2e.step_sizing --phases 16x2,64x2,128x2
"""

import ast
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal
import modal.experimental

from miles_plugins.proximal.modal_sources import add_fork_sources

PROFILE = os.environ.get("SIZING_PROFILE", "qwen38")
NODES = int(os.environ.get("SIZING_NODES", "2"))
GPUS_PER_NODE = 8
GROUP_SIZE = 8
MOCK = Path("/mock")
FORK = Path("/fork")
REPO = Path(__file__).resolve().parents[3]
PROMPT_TOKENS, MODEL_SPAN, TOOL_SPAN = 2_000, 330, 1_900  # As trainer_replay: agent-shaped masks.
LIBFABRIC, OFI_NCCL = "1.22.0", "1.19.0"
# The container imports this module too: it must select the same profile and cluster size.
_SIZING_ENV = {"SIZING_PROFILE": PROFILE, "SIZING_NODES": str(NODES)}

# Modal injects the RDMA NCCL environment in EFA regions only when the image carries the
# AWS OFI plugin at /opt/amazon (RoCE regions need nothing). Built only if absent.
_EFA_PACKAGES = (
    "curl", "ca-certificates", "bzip2", "build-essential", "autoconf", "automake", "libtool", "pkg-config",
    "libhwloc-dev", "libibverbs-dev", "librdmacm-dev", "rdma-core", "iproute2", "pciutils",
)  # fmt: skip
_EFA = (
    "test -e /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so || ("
    f"cd /tmp && curl -sSL -o lf.tbz https://github.com/ofiwg/libfabric/releases/download/v{LIBFABRIC}/libfabric-{LIBFABRIC}.tar.bz2"
    f" && tar xjf lf.tbz && cd libfabric-{LIBFABRIC} && ./configure --prefix=/opt/amazon/efa --enable-efa=yes --disable-verbs"
    " && make -j$(nproc) >/dev/null && make install >/dev/null"
    f" && cd /tmp && curl -sSL -o ofi.tgz https://github.com/aws/aws-ofi-nccl/releases/download/v{OFI_NCCL}/aws-ofi-nccl-{OFI_NCCL}.tar.gz"
    f" && tar xzf ofi.tgz && cd aws-ofi-nccl-{OFI_NCCL}"
    " && ./configure --prefix=/opt/amazon/ofi-nccl --with-libfabric=/opt/amazon/efa --with-cuda=/usr/local/cuda --enable-platform-aws"
    " && make -j$(nproc) >/dev/null && make install >/dev/null"
    " && cd /opt/amazon/ofi-nccl/lib && (test -e libnccl-net.so || ln -s libnccl-net-ofi.so libnccl-net.so)"
    " && rm -rf /tmp/libfabric-* /tmp/aws-ofi-nccl-* /tmp/*.tbz /tmp/*.tgz)"
)


def _set_flag(argv: list[str], flag: str, value: str | None) -> list[str]:
    """Drop every occurrence of ``flag`` (and its value); re-add it last when ``value`` is set."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2 if i + 1 < len(argv) and not argv[i + 1].startswith("--") else 1
            continue
        out.append(argv[i])
        i += 1
    return out + ([flag, value] if value is not None else [])


def _batch_flags(argv: list[str], *, directory: Path, nodes: int, samples: int, steps: int) -> list[str]:
    for flag, value in (
        ("--actor-num-nodes", str(nodes)),
        ("--actor-num-gpus-per-node", str(GPUS_PER_NODE)),
        ("--n-samples-per-prompt", str(GROUP_SIZE)),
        ("--rollout-batch-size", str(samples // GROUP_SIZE)),
        ("--global-batch-size", str(samples)),
        ("--num-rollout", str(steps)),
        ("--load-debug-rollout-data", str(directory / "{rollout_id}.pt")),
        ("--save", None),  # Sizing keeps no checkpoints.
        ("--save-interval", None),
    ):
        argv = _set_flag(argv, flag, value)
    return argv


# --- profiles ------------------------------------------------------------------------------
# Each profile provides: image, volumes, resources, a command builder and node preparation.

if PROFILE == "qwen38":
    from miles_plugins.proximal import modal_training as node
    from miles_plugins.proximal.e2e.trainer_replay import replay_command
    from miles_plugins.proximal.serving_app import DEPLOYMENT, RUN, with_configs

    assert RUN.research.group_size == GROUP_SIZE, "sizing assumes the production group size"
    KERNEL_SEED = Path("/kernels-seed")
    APP_NAME = f"{node.TRAINING.app_name}-sizing"
    # 1 TiB host memory: the first 8-node attempt had a container SIGKILLed (137) at
    # 768 GiB while eight ranks loaded weights; peak host RAM is recorded per node.
    RESOURCES = {"gpu": node.TRAINING.gpu, "cpu": float(node.TRAINING.cpu), "memory": 1024 * 1024}
    image = (
        add_fork_sources(
            with_configs(
                modal.Image.from_registry(DEPLOYMENT.image)
                .entrypoint([])
                .apt_install("postgresql", *_EFA_PACKAGES)
                .pip_install("psycopg[binary]")
                .run_commands(_EFA)
                .env({**node.MEGATRON_ENV, node._TRAINING_PATH: node._CONTAINER_TRAINING_CONFIG, **_SIZING_ENV})
            )
        )
        .add_local_file(os.environ[node._TRAINING_PATH], node._CONTAINER_TRAINING_CONFIG)
        .add_local_file(REPO / "train_async.py", str(FORK / "train_async.py"))
        .add_local_file(REPO / "train.py", str(FORK / "train.py"))
        .add_local_dir(REPO / "scripts/models", str(FORK / "scripts/models"))
        .add_local_file(REPO / node.TRAINING.train_args, str(FORK / "train_args.txt"))
    )
    VOLUMES: dict[str, modal.Volume] = {str(DEPLOYMENT.base_mount): node.base_volume.read_only()}
    if node.kernel_volume is not None:
        VOLUMES[str(KERNEL_SEED)] = node.kernel_volume.read_only()

    def build_command(directory: Path, *, nodes: int, samples: int, steps: int, extra: list[str]) -> list[str]:
        argv = _batch_flags(replay_command(steps), directory=directory, nodes=nodes, samples=samples, steps=steps)
        for token in extra:  # Layout overrides replace the production value rather than duplicate it.
            if token.startswith("--"):
                argv = _set_flag(argv, token, None)
        return argv + extra

    def prepare_node() -> None:
        cache = node.TRAINING.kernel_cache
        if node.kernel_volume is not None and cache is not None:
            # The whole production cache (Triton, Inductor, TileLang, FLA configs: ~21k files,
            # ~850 MB) so ranks start warm. A Volume serves small files one by one, so copy
            # with many threads (19 s measured; a serial copy did 47 MB in 5 min). Local, so
            # sizing never writes to the shared cache.
            started = time.monotonic()
            count = _parallel_copy(KERNEL_SEED, Path(str(cache.mount)))
            print(f"[sizing] kernel cache: {count} files copied in {time.monotonic() - started:.0f}s", flush=True)

elif PROFILE == "inkling-small":
    INKLING_IMAGE = os.environ.get(
        "INKLING_MODAL_IMAGE", "radixark/miles@sha256:8ee6528fa209dd3bc65ccb40556e6606e3e9e502cd521d994d3ee6da3a58b67d"
    )  # The image the Inkling-Small SFT runs used on B300.
    WEIGHTS = Path("/mnt/inkling")
    HF, TORCH_DIST = WEIGHTS / "models/Inkling-Small", WEIGHTS / "models/Inkling-Small_torch_dist"
    APP_NAME = "miles-inkling-small-sizing"
    RESOURCES = {"gpu": "B300:8", "cpu": 32.0, "memory": 1024 * 1024}
    ENV = {
        "PYTHONPATH": f"/root/Megatron-LM:{FORK}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONUNBUFFERED": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "MILES_INKLING_ATTN_BACKEND": "flex",
    }
    image = (
        add_fork_sources(
            modal.Image.from_registry(INKLING_IMAGE)
            .entrypoint([])
            .apt_install(*_EFA_PACKAGES)
            .run_commands(_EFA)
            .env({**ENV, **_SIZING_ENV})
        )
        .add_local_file(REPO / "train.py", str(FORK / "train.py"))
        .add_local_dir(REPO / "scripts/models", str(FORK / "scripts/models"))
    )
    VOLUMES = {str(WEIGHTS): modal.Volume.from_name("inkling-small-rft", environment_name="main").read_only()}

    def build_command(directory: Path, *, nodes: int, samples: int, steps: int, extra: list[str]) -> list[str]:
        from miles.utils.external_utils.model_args_utils import load_model_args

        from miles_plugins.proximal.contracts import TruncatedImportanceSampling, behavior_correction_argv

        tis = TruncatedImportanceSampling(kind="truncated_importance_sampling", clip=2.0, clip_low=0.0)  # As run 008.
        argv = [
            "python", str(FORK / "train.py"),
            *shlex.split(load_model_args("inkling-small", model_script_dir=FORK / "scripts/models")),
            # Checkpoint, adapters and memory settings of the validated SFT recipe.
            "--hf-checkpoint", str(HF), "--model-name", "inkling", "--megatron-to-hf-mode", "raw",
            "--load", str(TORCH_DIST), "--no-load-optim", "--no-load-rng", "--start-rollout-id", "0", "--finetune",
            "--lora-rank", "32", "--lora-alpha", "32", "--target-modules", "all-linear", "--experts-shared-outer-loras",
            "--train-backend", "megatron",
            "--expert-tensor-parallel-size", "1", "--context-parallel-size", "1", "--sequence-parallel",
            "--micro-batch-size", "1", "--recompute-granularity", "full", "--recompute-method", "uniform",
            "--recompute-num-layers", "1", "--seq-length", "262144",
            "--distributed-timeout-minutes", "60", "--bf16", "--moe-router-dtype", "fp32",
            "--transformer-impl", "transformer_engine", "--no-bias-dropout-fusion",
            "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
            "--num-gpus-per-node", str(GPUS_PER_NODE),
            # RL objective, as the Qwen production trainer (GRPO + TIS recompute).
            "--advantage-estimator", "grpo", "--calculate-per-token-loss",
            "--eps-clip", "0.2", "--eps-clip-high", "0.28",
            "--kl-loss-coef", "0", "--kl-coef", "0", "--entropy-coef", "0",
            *behavior_correction_argv(tis),
            "--log-probs-chunk-size", "4096", "--recompute-loss-function",
            "--optimizer", "adam", "--lr", "1e-5", "--lr-decay-style", "constant", "--weight-decay", "0.1",
            "--clip-grad", "1.0",
            "--rollout-max-response-len", "32768", "--rollout-max-context-len", "262144",
            "--disable-rollout-global-dataset", "--seed", "42",
        ]  # fmt: skip
        return _batch_flags(argv, directory=directory, nodes=nodes, samples=samples, steps=steps) + extra

    def prepare_node() -> None:
        for path in (HF / "config.json", TORCH_DIST / "latest_checkpointed_iteration.txt"):
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing from the inkling-small-rft Volume")

else:
    raise ValueError(f"SIZING_PROFILE must be qwen38 or inkling-small, got {PROFILE!r}")


app = modal.App(APP_NAME)


# --- mock rollouts -----------------------------------------------------------------------


def mock_step(rng: random.Random, *, rollout_id: int, groups: int, length: int) -> list[dict[str, Any]]:
    """One step's samples: ``groups`` groups of GROUP_SIZE, every sample exactly ``length``
    tokens, agent-shaped loss masks (trained replies, masked tool output), mixed rewards."""
    import numpy as np

    from miles.utils.types import Sample

    response = length - PROMPT_TOKENS
    mask: list[int] = []
    while len(mask) < response:
        mask += [1] * min(MODEL_SPAN, response - len(mask))
        mask += [0] * min(TOOL_SPAN, response - len(mask))
    if mask[-1] == 0:
        mask[-MODEL_SPAN:] = [1] * MODEL_SPAN
    gen = np.random.default_rng(rng.randrange(2**32))
    samples: list[dict[str, Any]] = []
    for g in range(groups):
        passes = rng.randint(1, GROUP_SIZE - 1)  # Never all-pass or all-fail: non-zero advantages.
        rewards = [1.0] * passes + [0.0] * (GROUP_SIZE - passes)
        rng.shuffle(rewards)
        for reward in rewards:
            sample = Sample(
                group_index=rollout_id * groups + g,
                index=len(samples),
                prompt="mock rollout",
                tokens=gen.integers(1_000, 150_000, size=length).tolist(),
                response_length=response,
                loss_mask=list(mask),
                rollout_log_probs=(-gen.uniform(0.01, 2.0, size=response)).tolist(),
                reward=reward,
                status=Sample.Status.COMPLETED,
            )
            samples.append(sample.to_dict())  # type: ignore[no-untyped-call]
    return samples


def write_phase(label: str, *, samples: int, steps: int, length: int) -> Path:
    import torch

    if samples % GROUP_SIZE:
        raise ValueError(f"samples per step ({samples}) must be a multiple of the group size ({GROUP_SIZE})")
    directory = MOCK / label
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for rollout_id in range(steps):
        batch = mock_step(rng, rollout_id=rollout_id, groups=samples // GROUP_SIZE, length=length)
        torch.save({"rollout_id": rollout_id, "metadata": {}, "samples": batch}, directory / f"{rollout_id}.pt")
    return directory


# --- node lifecycle ------------------------------------------------------------------------


def _sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def _parallel_copy(source: Path, destination: Path, threads: int = 64) -> int:
    from concurrent.futures import ThreadPoolExecutor

    files = [Path(root) / name for root, _, names in os.walk(source) for name in names]

    def copy(src: Path) -> None:
        dst = destination / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(copy, files))
    return len(files)


_PY_SPY = ("/opt/sglang/bin/py-spy", "py-spy")


def _stack_summary(dump: str, frames: int = 10) -> list[str]:
    """Each thread of a py-spy dump that is active (or the main thread): its name and top frames."""
    out: list[str] = []
    for block in re.split(r"\n(?=Thread )", dump):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines or not lines[0].startswith("Thread "):
            continue
        if "(idle)" in lines[0] and "MainThread" not in lines[0]:
            continue
        out.append(lines[0].strip()[:90])
        out += ["   " + line.strip()[:150] for line in lines[1 : frames + 1]]
    return out


def _dump_trainer_stacks(tag: str) -> str:
    """All threads (Python and native frames) of every trainer rank on this node, plus GPU
    utilization: what an idle rank is doing while its peers wait in a collective."""
    spy = next((p for p in _PY_SPY if shutil.which(p) or Path(p).exists()), None)
    util = _sh("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits").replace("\n", "; ")
    parts = [f"[dump {tag}] gpu idx,util%,MiB: {util}"]
    pids = _sh("pgrep -f '^ray::MegatronTrain'").split()
    full = []
    for pid in pids:
        cpu = _sh(f"ps -o pcpu= -p {pid}")
        dump = _sh(f"timeout 45 {spy} dump --native --pid {pid} 2>&1") if spy else "(no py-spy)"
        full.append(f"===== pid {pid} cpu {cpu}\n{dump}")
        parts.append(f"-- pid {pid} cpu {cpu}")
        parts += _stack_summary(dump)
    children = _sh("ps -eo pid,ppid,pcpu,etime,args --sort=-pcpu | head -15 | cut -c1-160")
    parts.append("-- top processes:\n" + children)
    Path(f"/tmp/dump-{tag}.txt").write_text("\n".join(full) + "\n" + children)
    return "\n".join(parts)


def _fabric() -> dict[str, Any]:
    """The RDMA fabric this node sees: RoCE netdevs, verbs devices, EFA PCI functions, and
    whether Modal injected the RDMA NCCL environment."""
    return {
        "region": os.environ.get("MODAL_REGION"),
        "cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "gpus": _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c"),
        "roce_netdevs": len([line for line in _sh("ip -brief addr").splitlines() if "rdma" in line]),
        "verbs_devices": len(_sh("ls /sys/class/infiniband 2>/dev/null").split()),
        "efa_pci": _sh("lspci 2>/dev/null").lower().count(" efa"),
        "nccl_env": {
            k: v for k, v in os.environ.items() if k.startswith(("NCCL_NET", "NCCL_IB_HCA", "FI_PROVIDER", "OFI_NCCL_FORCE"))
        },
    }


def _fabric_ok(f: dict[str, Any]) -> bool:
    return f["roce_netdevs"] >= GPUS_PER_NODE or f["verbs_devices"] >= GPUS_PER_NODE or bool(f["nccl_env"])


class _State:
    """The shared coordination Dict, scoped to one cluster attempt: Modal restarts a whole
    cluster after a container dies, and the new attempt must not read the old one's keys."""

    def __init__(self, store: modal.Dict, scope: str) -> None:
        self.store, self.scope = store, scope

    def get(self, key: str) -> Any:
        return self.store.get(f"{self.scope}/{key}")

    def __getitem__(self, key: str) -> Any:
        return self.store[f"{self.scope}/{key}"]

    def __setitem__(self, key: str, value: Any) -> None:
        self.store[f"{self.scope}/{key}"] = value


def _host_memory_gib() -> float:
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return round(int(Path(path).read_text()) / 2**30, 1)
        except (OSError, ValueError):
            continue
    return -1.0


class _MemorySampler:
    """Peak ``memory.used`` (MiB) per GPU, and peak host memory (GiB, key ``host``), on this
    node per phase label from the shared state."""

    def __init__(self, state: "_State", rank: int) -> None:
        self.state, self.rank, self.peaks, self.stop = state, rank, {}, threading.Event()
        self.dumped: set[str] = set()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(2):
            try:
                phase = self.state.get("phase") or "setup"
                tag = self.state.get("dump")
            except Exception:
                continue
            if tag and tag not in self.dumped:  # Rank 0 saw a stall: every node captures at once.
                self.dumped.add(tag)
                summary = _dump_trainer_stacks(f"{tag}-rank{self.rank}")
                print(f"[sizing] STACKS rank {self.rank} ({tag}):\n{summary}", flush=True)
                try:
                    self.state[f"dump-{tag}-{self.rank}"] = summary[-40000:]
                except Exception:
                    pass
            peaks = self.peaks.setdefault(phase, {})
            for line in _sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits").splitlines():
                idx, used = (x.strip() for x in line.split(","))
                peaks[idx] = max(peaks.get(idx, 0), int(used))
            peaks["host"] = max(peaks.get("host", 0), _host_memory_gib())


def _wait(predicate: Callable[[], Any], timeout: float, what: str, state: "_State") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if state.get("error"):
            raise RuntimeError(state["error"])
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        time.sleep(2)


def _start_ray(state: "_State", rank: int, ips: list[str]) -> None:
    head, own, nodes = ips[0], ips[rank], len(ips)
    os.environ.pop("RAY_ADDRESS", None)
    common = ["--node-ip-address", own, "--num-gpus", str(GPUS_PER_NODE), "--disable-usage-stats"]
    if rank == 0:
        subprocess.run(["ray", "start", "--head", "--port=6379", *common], check=True, timeout=180)
        state["head-ready"] = True
    else:
        _wait(lambda: state.get("head-ready"), 900, "Ray head", state)
        subprocess.run(["ray", "start", f"--address={head}:6379", *common], check=True, timeout=180)
    if rank == 0:
        import ray

        ray.init(address=f"{head}:6379")
        try:
            _wait(
                lambda: {n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU") == GPUS_PER_NODE}
                == set(ips),
                900,
                f"all {nodes * GPUS_PER_NODE} GPUs in Ray",
                state,
            )
        finally:
            ray.shutdown()
        os.environ["RAY_ADDRESS"] = f"{head}:6379"


_PERF = re.compile(r"perf (\d+): (\{.*\})")
# Stall handling (seconds without a Miles timer line). The first line follows model loading.
FIRST_PROGRESS_S, DUMP_AFTER_S, KILL_AFTER_S = 1800, 600, 1500
_NET = ("Using network", "NET/OFI Selected", "NET/OFI Initializing", "NET/IB : Using", "via NET/")


def _run_phase(label: str, nodes: int, samples: int, steps: int, length: int, extra: list[str], state: "_State") -> dict[str, Any]:
    started = time.monotonic()
    directory = write_phase(label, samples=samples, steps=steps, length=length)
    data_s = round(time.monotonic() - started)
    state["phase"] = label
    command = build_command(directory, nodes=nodes, samples=samples, steps=steps, extra=extra)
    env = {**os.environ, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}
    log = Path(f"/tmp/sizing-{label}.log")
    print(f"[sizing] phase {label}: {samples} x {length} tokens, {steps} steps\n  {shlex.join(command)}", flush=True)
    t0 = time.monotonic()
    progress = {"last": t0, "seen": False}
    stall: dict[str, Any] = {"dumps": [], "killed": False}
    done = threading.Event()

    def monitor(process: subprocess.Popen[str]) -> None:
        """No timer progress for DUMP_AFTER_S: every node dumps stacks. For KILL_AFTER_S:
        dump again, then end the phase rather than wait for the 60-min NCCL timeout."""
        episode = 0
        while not done.wait(15):
            quiet = time.monotonic() - progress["last"]
            grace = FIRST_PROGRESS_S if not progress["seen"] else 0
            if quiet > grace + DUMP_AFTER_S and episode == 0:
                episode = 1
                tag = f"{label}-stall-{round(time.monotonic() - t0)}s"
                stall["dumps"].append(tag)
                state["dump"] = tag
            if quiet > grace + KILL_AFTER_S:
                tag = f"{label}-kill-{round(time.monotonic() - t0)}s"
                stall["dumps"].append(tag)
                state["dump"] = tag
                time.sleep(90)  # Let every node finish its dump.
                stall["killed"] = True
                process.terminate()
                return

    live = re.compile(r"perf \d+:|Timer (log_probs|actor_train) (start|end)|OutOfMemory|out of memory|Traceback|Using network|NET/OFI Selected")
    shown: set[str] = set()
    with log.open("w") as out:
        process = subprocess.Popen(command, cwd=FORK, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        assert process.stdout is not None
        watcher = threading.Thread(target=monitor, args=(process,), daemon=True)
        watcher.start()
        for line in process.stdout:
            out.write(line)
            if "Timer " in line or "perf " in line:
                progress["last"], progress["seen"] = time.monotonic(), True
            if live.search(line):  # Progress as it happens, once per distinct message.
                key = re.sub(r"^.*?\] ", "", re.sub(r"\x1b\[[0-9;]*m", "", line))[:200]
                if key not in shown and len(shown) < 400:
                    shown.add(key)
                    print(f"[sizing:{label} +{round(time.monotonic() - t0)}s] {key.strip()}", flush=True)
        code = process.wait()
        done.set()
    wall = round(time.monotonic() - t0)
    dumps = {tag: {r: state.get(f"dump-{tag}-{r}") for r in range(nodes)} for tag in stall["dumps"]}
    text = log.read_text(errors="replace")
    perf = []
    for m in _PERF.finditer(text):
        try:
            perf.append({"rollout": int(m.group(1)), **ast.literal_eval(m.group(2))})
        except (ValueError, SyntaxError):
            pass
    net = list(dict.fromkeys(line.split("NCCL INFO ")[-1][:120] for line in text.splitlines() if any(k in line for k in _NET)))[:8]
    errors = [
        line[-300:] for line in text.splitlines() if re.search(r"OutOfMemory|out of memory|Traceback|NCCL WARN|Watchdog caught|Error:", line)
    ][:15]
    tail = "\n".join(line[:300] for line in text.splitlines()[-60:])
    print(f"[sizing] phase {label}: exit {code} in {wall}s; perf rows {len(perf)}; net {net}; errors {errors[:4]}\n{tail}", flush=True)
    return {
        "label": label, "samples_per_step": samples, "steps": steps, "length": length,
        "tokens_per_step": samples * length, "exit_code": code, "wall_s": wall, "mock_data_s": data_s,
        "perf": perf, "net": net, "errors": errors, "command": shlex.join(command),
        "stalled": stall["killed"], "stack_dumps": dumps,
    }  # fmt: skip


@app.function(image=image, volumes=VOLUMES, timeout=4 * 3600, **RESOURCES)
@modal.experimental.clustered(size=NODES, rdma=True)
def sizing(plan: dict[str, Any], store: modal.Dict) -> dict[str, Any]:
    info = modal.experimental.get_cluster_info()
    state = _State(store, info.cluster_id)
    print(f"[sizing] cluster attempt {info.cluster_id}", flush=True)
    rank, ips = info.rank, list(info.container_ipv4_ips)
    nodes = len(ips)
    sampler = _MemorySampler(state, rank)
    try:
        fabric = _fabric()
        state[f"fabric-{rank}"] = fabric
        print(f"[sizing] rank {rank} fabric {json.dumps(fabric)}", flush=True)
        prepare_node()
        _wait(lambda: all(state.get(f"fabric-{r}") for r in range(nodes)), 900, "every node's fabric report", state)
        fabrics = [state[f"fabric-{r}"] for r in range(nodes)]
        if not all(_fabric_ok(f) for f in fabrics) and not plan.get("allow_tcp"):
            raise RuntimeError(f"no RDMA fabric on some node: {[(r, f['region'], f['verbs_devices'], f['efa_pci']) for r, f in enumerate(fabrics)]}")
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
        _start_ray(state, rank, ips)
        sampler.thread.start()
        if rank != 0:
            _wait(lambda: state.get("stop"), 4 * 3600, "rank 0 to finish", state)
            state[f"memory-{rank}"] = sampler.peaks or {"none": {}}
            return {"rank": rank}
        phases = []
        for p in plan["phases"]:
            phases.append(_run_phase(f"s{p['samples']}", nodes, p["samples"], p["steps"], plan["length"], plan["extra_args"], state))
            if phases[-1]["exit_code"] != 0 and not plan.get("continue_on_failure"):
                break
        state["phase"] = "done"
        state["stop"] = True
        _wait(lambda: all(state.get(f"memory-{r}") for r in range(1, nodes)), 300, "workers' memory reports", state)
        memory = {"0": sampler.peaks, **{str(r): state[f"memory-{r}"] for r in range(1, nodes)}}
        return {"profile": PROFILE, "nodes": nodes, "cluster_id": info.cluster_id, "fabric": fabrics, "phases": phases, "peak_memory_mib": memory}
    except BaseException as exc:
        state["error"] = f"rank {rank}: {type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.stop.set()
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)


@app.local_entrypoint()
def main(phases: str = "8x2", length: int = 258_000, out: str = "step_sizing.json", extra: str = "", allow_tcp: bool = False) -> None:
    plan = {
        "length": length,
        "phases": [{"samples": int(s), "steps": int(n)} for s, n in (p.split("x") for p in phases.split(","))],
        "extra_args": shlex.split(extra),
        "allow_tcp": allow_tcp,
    }
    print(f"[sizing] {PROFILE} on {NODES} x {RESOURCES['gpu']}: {json.dumps(plan)}", flush=True)
    with modal.Dict.ephemeral() as state:
        result = sizing.remote(plan, state)
    Path(out).write_text(json.dumps(result, indent=2, default=str))
    keys = ("rollout", "perf/step_time", "perf/log_probs_time", "perf/actor_train_time", "perf/actor_train_tok_per_s")
    for ph in result.get("phases", []):
        rows = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items() if k in keys} for r in ph["perf"]]
        print(f"[sizing] {ph['label']} exit {ph['exit_code']} wall {ph['wall_s']}s: {rows}", flush=True)
    print(f"[sizing] details in {out}", flush=True)
