"""Trainer replay: the production training node on mock rollouts, no serving or platform.

Proves full training steps (log-probs, forward, backward, optimizer, checkpoint) on the
intended GPUs, parallelism and sequence lengths before paying for real rollouts. It runs
the same image, model arguments, train arguments and trainer environment as
``modal_training``; only the rollout source changes: Miles's --load-debug-rollout-data
feeds generated groups, so SGLang engines and weight pushes are skipped.

Samples look like agent rollouts: a prompt, then alternating model spans (trained) and
tool outputs (masked), with mixed rewards inside every group so advantages are non-zero.
Lengths default to the shape of a real batch (run 004 step 1 averaged ~147k tokens) and
a stress step near the context cap.

    PROXIMAL_RUN_CONFIG=run.json \\
    PROXIMAL_SERVING_CONFIG=examples/proximal/qwen38/overhead/serving.json \\
    PROXIMAL_TRAINING_CONFIG=examples/proximal/qwen38/overhead/training.json \\
      modal run --env main -m miles_plugins.proximal.e2e.trainer_replay
"""

import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

from miles_plugins.proximal import modal_training as node
from miles_plugins.proximal.contracts import behavior_correction_argv
from miles_plugins.proximal.serving_app import DEPLOYMENT, RUN, base_volume

MOCK = Path("/mock")
# Steps: per-sample total lengths. Step 1 is a realistic batch; step 2 packs near the cap.
DEFAULT_STEPS: tuple[tuple[int, int], ...] = ((90_000, 205_000), (200_000, 258_000))
PROMPT_TOKENS = 2_000
MODEL_SPAN, TOOL_SPAN = 330, 1_900  # Tokens per agent turn: generated reply, then a masked tool result.
TOKEN_RANGE = (1_000, 150_000)  # Ordinary vocabulary, no special tokens.


def mock_group(rng: random.Random, *, group_index: int, first_index: int, lengths: list[int]) -> list[dict[str, Any]]:
    """One group of Miles sample dicts with agent-shaped loss masks and mixed rewards."""
    from miles.utils.types import Sample

    passes = rng.randint(1, len(lengths) - 1)  # Never all-pass or all-fail: advantages stay non-zero.
    rewards = [1.0] * passes + [0.0] * (len(lengths) - passes)
    rng.shuffle(rewards)
    samples = []
    for offset, (total, reward) in enumerate(zip(lengths, rewards, strict=True)):
        response = total - PROMPT_TOKENS
        mask: list[int] = []
        while len(mask) < response:
            mask += [1] * min(MODEL_SPAN, response - len(mask))
            mask += [0] * min(TOOL_SPAN, response - len(mask))
        if mask[-1] == 0:  # A rollout ends on the model's own reply.
            mask[-MODEL_SPAN:] = [1] * MODEL_SPAN
        sample = Sample(
            group_index=group_index,
            index=first_index + offset,
            prompt="mock rollout",
            tokens=[rng.randrange(*TOKEN_RANGE) for _ in range(total)],
            response_length=response,
            loss_mask=mask,
            rollout_log_probs=[-rng.uniform(0.01, 2.0) for _ in range(response)],
            reward=reward,
            status=Sample.Status.COMPLETED,
        )
        samples.append(sample.to_dict())  # type: ignore[no-untyped-call]  # Miles's own codec.
    return samples


def write_mock_rollouts(steps: tuple[tuple[int, int], ...], *, groups: int, group_size: int, seed: int = 0) -> None:
    import torch

    rng = random.Random(seed)
    MOCK.mkdir(parents=True, exist_ok=True)
    for rollout_id, (low, high) in enumerate(steps):
        samples: list[dict[str, Any]] = []
        for g in range(groups):
            lengths = [rng.randint(low, high) for _ in range(group_size)]
            samples += mock_group(rng, group_index=rollout_id * groups + g, first_index=len(samples), lengths=lengths)
        torch.save({"rollout_id": rollout_id, "metadata": {}, "samples": samples}, MOCK / f"{rollout_id}.pt")
        print(
            f"[replay] step {rollout_id}: {len(samples)} samples, mean {sum(len(s['tokens']) for s in samples) // len(samples)} tokens",
            flush=True,
        )


def replay_command(num_steps: int) -> list[str]:
    from miles.utils.external_utils.model_args_utils import load_model_args

    run, training = RUN, node.TRAINING
    lines = (node.FORK / "train_args.txt").read_text().splitlines()
    args = [t for line in lines if line.strip() and not line.lstrip().startswith("#") for t in shlex.split(line)]
    for flag in ("--use-wandb", "--wandb-project", "--wandb-group"):  # No tracking for a smoke run.
        if flag in args:
            i = args.index(flag)
            del args[i : i + (1 if flag == "--use-wandb" else 2)]
    model_args = shlex.split(load_model_args(training.model_args, model_script_dir=node.FORK / "scripts/models"))
    research = run.research
    return [
        sys.executable,
        str(node.FORK / "train.py"),
        *args,
        *model_args,
        "--hf-checkpoint", str(run.tokenizer_path),
        "--train-backend", "megatron",
        "--megatron-to-hf-mode", "bridge",
        "--lora-rank", str(research.lora.rank),
        "--lora-alpha", str(research.lora.alpha),
        "--lora-dropout", "0",
        "--target-modules", ",".join(research.lora.target_modules),
        "--n-samples-per-prompt", str(research.group_size),
        *behavior_correction_argv(research.behavior_correction),
        "--rollout-max-response-len", str(research.sampling.max_tokens),
        "--rollout-max-context-len", str(research.sampling.max_sequence_tokens),
        "--disable-rollout-global-dataset",
        "--load-debug-rollout-data", str(MOCK / "{rollout_id}.pt"),
        "--num-rollout", str(num_steps),
    ]  # fmt: skip


app = modal.App(f"{node.TRAINING.app_name}-replay")


@app.function(
    image=node.image.add_local_file(node.REPO / "train.py", str(node.FORK / "train.py")),
    gpu=node.TRAINING.gpu,
    cpu=float(node.TRAINING.cpu),
    memory=node.TRAINING.memory_mib,
    volumes={str(DEPLOYMENT.base_mount): base_volume, **node.kernel_mounts},
    timeout=3 * 3600,
)
def replay(steps: list[list[int]], rollout_batch_size: int, tune: bool = False) -> dict[str, Any]:
    step_bounds = tuple((low, high) for low, high in steps)
    write_mock_rollouts(step_bounds, groups=rollout_batch_size, group_size=RUN.research.group_size)
    # py-spy, for stack dumps of every rank while a step hangs (ray stack / py-spy dump).
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "py-spy"], check=False)
    # NCCL flight recorder: on a collective timeout every rank dumps the collectives it
    # issued, so a desync shows which rank stopped where.
    os.environ.update(
        {
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "4000",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_FR_DUMP_TEMP_FILE": "/tmp/nccl_trace_rank_",
        }
    )
    if tune:
        # Tuning must autotune whatever the deployment's cache mode (kernel_tune).
        os.environ["FLA_CACHE_MODE"] = "disabled"
    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
    subprocess.run(
        ["ray", "start", "--head", "--num-gpus", str(node.TRAINING.num_gpus), "--disable-usage-stats"], check=True
    )
    started = time.monotonic()
    log = Path("/tmp/replay.log")
    with log.open("w") as out:
        code = subprocess.run(replay_command(len(step_bounds)), stdout=out, stderr=subprocess.STDOUT).returncode
    text = log.read_text(errors="replace")
    print(text[-20000:], flush=True)
    tuned = _write_fla_configs() if tune else []
    if node.kernel_volume is not None:
        node.kernel_volume.commit()
    wanted = (
        "train/",
        "grad_norm",
        "Timer train",
        "Timer log_probs",
        "compute_log_prob",
        "Watchdog",
        "tis",
        "Memory-Usage",
        "rollout 0",
        "rollout 1",
        "perf ",
        "OutOfMemory",
        "Error",
        "saved checkpoint",
        "successfully saved",
    )
    return {
        "exit_code": code,
        "seconds": round(time.monotonic() - started),
        "lines": [line[:400] for line in text.splitlines() if any(w in line for w in wanted)][-200:],
        "fla_configs": tuned,
    }


def _write_fla_configs() -> list[str]:
    """After a tuning replay: every kernel Triton tuned, as FLA configs on the kernel volume."""
    # Only in the training image.
    import tilelang  # type: ignore[import-not-found]
    import triton  # type: ignore[import-not-found]
    from fla.ops.utils.cache import AutotuneKey  # type: ignore[import-not-found]

    from miles_plugins.proximal.kernel_tune import fla_configs, fla_key_hash, write_fla_configs

    cache = node.TRAINING.kernel_cache
    assert cache is not None, "Tuning needs the deployment's kernel_cache"
    sample = [4096, 128, "torch.bfloat16"]
    if fla_key_hash(sample) != AutotuneKey.key_hash(tuple(sample)):
        raise RuntimeError("fla_key_hash no longer matches FLA's AutotuneKey.key_hash")
    print(f"[replay] TileLang cache dir: {getattr(tilelang.env, 'TILELANG_CACHE_DIR', '?')}", flush=True)
    files = fla_configs(Path(str(cache.triton_cache_dir)), triton.__version__)
    write_fla_configs(files, Path(str(cache.fla_config_dir)))
    return sorted(files)


@app.local_entrypoint()
def main(steps: str = "", out: str = "trainer_replay.json", tune: bool = False) -> None:
    bounds = [list(map(int, s.split("-"))) for s in steps.split(",")] if steps else [list(s) for s in DEFAULT_STEPS]
    lines = (node.REPO / node.TRAINING.train_args).read_text()
    batch = int(shlex.split(lines[lines.index("--rollout-batch-size") :])[1])
    result = replay.remote(bounds, batch, tune)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"[replay] exit {result['exit_code']} after {result['seconds']} s; details in {out}", flush=True)
