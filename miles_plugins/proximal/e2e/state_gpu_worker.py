"""In-container helpers for the bounded state GPU verification experiment."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles_plugins.proximal.offline_batch import Batch


def prepare_batch(root: Path, *, config_json: str, samples: int, resume: bool) -> dict[str, object]:
    from miles_plugins.proximal.contracts import RunConfig
    from miles_plugins.proximal.offline_batch import freeze_batch, load_group
    from miles_plugins.proximal.store import StoredGroup

    config = RunConfig.model_validate_json(config_json)
    source = Path("/source") / config.run_id / "artifacts" / config.run_id
    if root.exists() and not resume:
        raise FileExistsError(f"Use an unused verification namespace: {root}")
    if resume:
        if (root / "source-config.json").read_text() != config_json or (root / "batch/batch.json").exists():
            raise ValueError("Only an incomplete preparation with the identical config can be resumed")
        source = root / "batch"

    def read_header(path: Path) -> StoredGroup:
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "big")
            if not 0 < size <= 16 * 1024 * 1024:
                raise ValueError("Invalid stored group header")
            return StoredGroup.model_validate_json(stream.read(size))

    paths = sorted((source / "groups").glob("*.bin"))
    print(f"[state-check] inspecting {len(paths)} stored group headers", flush=True)
    selected: list[StoredGroup] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for offset in range(0, len(paths), 256):
            for header in pool.map(read_header, paths[offset : offset + 256]):
                if header.policy.version == 1 and len(selected) * config.research.group_size < samples:
                    selected.append(header)
            print(
                f"[state-check] headers {min(offset + 256, len(paths))}; selected {len(selected)} groups", flush=True
            )
            if len(selected) * config.research.group_size == samples:
                break
    if len(selected) * config.research.group_size != samples:
        raise ValueError(f"Need {samples} policy-v1 samples; found {len(selected) * config.research.group_size}")
    selected.sort(key=lambda header: header.identities[0].group_index)
    root.mkdir(parents=True, exist_ok=resume)
    (root / "source-config.json").write_text(config_json)
    batch = freeze_batch(
        config=config,
        source_root=source,
        group_ids=tuple(header.group_id for header in selected),
        policy=selected[0].policy,
        num_samples=samples,
        out=root / "batch",
    )
    tokens = loss_tokens = mixed_groups = 0
    for index in batch.groups:
        group = load_group(root / "batch", index, config)
        tokens += sum(len(sample.tokens) for sample in group)
        for sample in group:
            assert sample.loss_mask is not None
            loss_tokens += sum(sample.loss_mask)
        mixed_groups += len({sample.reward for sample in group}) > 1
    result = {
        "source_run": config.run_id,
        "samples": samples,
        "groups": len(selected),
        "policy": batch.policy.model_dump(mode="json"),
        "tokens": tokens,
        "loss_tokens": loss_tokens,
        "mixed_reward_groups": mixed_groups,
    }
    if mixed_groups == 0:
        raise ValueError("The verification batch needs a nonzero learning signal")
    (root / "preparation.json").write_text(json.dumps(result, indent=2))
    return result


def recipe(root: Path) -> list[str]:
    from miles.utils.external_utils.model_args_utils import load_model_args

    tokens = [
        token
        for line in Path("/fork/train_args.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for token in shlex.split(line)
    ]
    remove = {
        "--use-wandb",
        "--wandb-project",
        "--wandb-group",
        "--save",
        "--save-interval",
        "--num-rollout",
        "--global-batch-size",
        "--rollout-batch-size",
    }
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in remove:
            i += 1 if tokens[i] == "--use-wandb" else 2
        else:
            result.append(tokens[i])
            i += 1
    result += shlex.split(load_model_args("qwen3-0.6B", model_script_dir=Path("/fork/scripts/models")))
    return result + [
        "--save-debug-train-data",
        str(root / "train-data/{rollout_id}_{rank}.pt"),
    ]


def train_reference(root: Path, *, commit: Callable[[], None]) -> dict[str, object]:
    from miles_plugins.proximal.contracts import behavior_correction_argv
    from miles_plugins.proximal.offline_batch import training_groups, validate_batch

    batch = validate_batch(root / "batch")
    config, research = batch.source, batch.source.research
    working = Path("/work/reference")
    args = recipe(working) + [
        "--hf-checkpoint",
        str(config.tokenizer_path),
        "--load",
        str(config.tokenizer_path),
        "--train-backend",
        "megatron",
        "--megatron-to-hf-mode",
        "bridge",
        "--lora-rank",
        str(research.lora.rank),
        "--lora-alpha",
        str(research.lora.alpha),
        "--lora-dropout",
        "0",
        "--target-modules",
        ",".join(research.lora.target_modules),
        "--n-samples-per-prompt",
        str(research.group_size),
        "--rollout-batch-size",
        str(len(training_groups(batch))),
        "--global-batch-size",
        str(batch.num_samples),
        *behavior_correction_argv(research.behavior_correction),
        "--rollout-max-response-len",
        str(research.sampling.max_tokens),
        "--rollout-max-context-len",
        str(research.sampling.max_sequence_tokens),
        "--disable-rollout-global-dataset",
        "--debug-train-only",
        "--rollout-num-gpus",
        "0",
        "--rollout-function-path",
        "miles_plugins.proximal.e2e.state_gpu_replay.ReferenceReplay",
        "--verification-batch",
        str(root / "batch"),
        "--num-rollout",
        "2",
        "--start-rollout-id",
        "0",
        "--save",
        str(working / "checkpoints"),
        "--save-interval",
        "1",
    ]
    _run([sys.executable, "/fork/train.py", *args], root / "reference.log")
    shutil.copytree(working, root / "reference")
    checkpoint_id = _bundle_checkpoint(root, working, batch, args, commit=commit)
    result = {"checkpoint_id": checkpoint_id, "reference_step": 1, "samples": batch.num_samples}
    (root / "reference.json").write_text(json.dumps(result, indent=2))
    return result


def _run(command: list[str], log: Path) -> None:
    # These are checked-in Miles entry points, never model-generated code.
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[state-check] " + shlex.join(command), flush=True)
    subprocess.run(["ray", "start", "--head", "--node-ip-address", "127.0.0.1", "--num-gpus", "1"], check=True)
    try:
        with log.open("w") as stream:
            subprocess.run(command, cwd="/fork", stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1500)
    finally:
        subprocess.run(["ray", "stop", "--force"], check=False, stdout=subprocess.DEVNULL)
        if log.exists():
            print(log.read_text(errors="replace")[-12000:], flush=True)


def _bundle_checkpoint(
    root: Path, working: Path, batch: "Batch", args: list[str], *, commit: Callable[[], None]
) -> str:
    from miles_plugins.proximal.contracts import digest, training_contract
    from miles_plugins.proximal.e2e.local_postgres import local_postgres, server_binaries
    from miles_plugins.proximal.state_checkpoints import RecoveryContext, take

    recovery = root / "recovery"
    launch = recovery / "launches/reference"
    launch.mkdir(parents=True)
    (launch / "config.json").write_text(batch.source.model_dump_json())
    context = RecoveryContext(
        run_id=batch.source.run_id,
        contract_sha256=digest(training_contract(batch.source)),
        world_size=1,
        max_policy_lag=batch.source.research.max_policy_lag,
        model_args=(),
        train_args=tuple(args),
        image=os.environ["VERIFICATION_IMAGE"],
        code_sha256=os.environ["VERIFICATION_CODE_SHA256"],
    )
    # This experiment has no online producer/queue. The empty diagnostic database
    # and ReferenceReplay's explicit cursor certify only this fixed-batch experiment.
    with local_postgres() as dsn:
        return take(
            0,
            checkpoints=working / "checkpoints",
            dsn=dsn,
            snapshot_root=recovery,
            pg_bin=server_binaries(),
            context=context,
            launch_id="reference",
            parent=None,
            commit=commit,
        )


def train_resumed(root: Path) -> dict[str, object]:
    from miles_plugins.proximal.offline_batch import validate_batch

    batch = validate_batch(root / "batch")
    checkpoint = root / "recovery/checkpoints" / json.loads((root / "reference.json").read_text())["checkpoint_id"]
    working = Path("/work/resumed")
    command = [
        sys.executable,
        "-m",
        "miles_plugins.proximal.offline_batch",
        "train",
        "--bundle",
        str(root / "batch"),
        "--checkpoint",
        str(checkpoint),
        "--optimizer-state",
        "resume",
        "--yes-train",
        "--",
        *recipe(working),
        "--save",
        str(working / "checkpoints"),
    ]
    _run(command, root / "resumed.log")
    shutil.copytree(working, root / "resumed")
    return {"samples": batch.num_samples, "resumed_step": 1}
