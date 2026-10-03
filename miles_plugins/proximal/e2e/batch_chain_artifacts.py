"""Durable chain steps: one receipt per committed update, linked to the step it continued.

A receipt is written only after every node's native shards and the evaluation snapshot
are committed. The next step stages exactly the files its predecessor's receipt names.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import Field

from miles_plugins.proximal.authorization import AuthorizedRun
from miles_plugins.proximal.contracts import Contract, Policy, SafeId
from miles_plugins.proximal.e2e.batch_chain_inputs import ChainArm, ChainPlan
from miles_plugins.proximal.e2e.batch_sweep_artifacts import EXPORTS, certify_update, publish_receipt
from miles_plugins.proximal.snapshot import Digest, Nonempty, SnapshotReference
from miles_plugins.proximal.state_artifacts import RelativePath, StateFile, copy_verified, verify
from miles_plugins.proximal.state_checkpoints import NativeCompletion
from miles_plugins.proximal.storage import write_atomic

RECEIPT = "receipt.json"


class ChainStepReceipt(Contract):
    arm: SafeId
    step: Annotated[int, Field(ge=1)]  # Updates applied to the arm, this one included.
    batch_sha256: Digest
    behavior_policy: Policy
    policy_lag: Annotated[int, Field(ge=0)]
    targets: Annotated[tuple[Nonempty, ...], Field(min_length=1)]
    previous: StateFile | None
    native: NativeCompletion
    files: tuple[StateFile, ...]
    snapshot: SnapshotReference
    eval_path: RelativePath


def step_root(root: Path, arm: str, step: int, attempt: int) -> Path:
    return root / arm / f"step-{step + 1}" / f"attempt-{attempt}"


def checkpoint_dir(base: Path, step: int) -> Path:
    return base / "checkpoints" / f"iter_{step:07d}" / "adapter"


def finalize_chain_step(
    authorization: AuthorizedRun,
    *,
    plan: ChainPlan,
    arm: ChainArm,
    step: int,
    attempt: int,
    behavior_policy: Policy,
    previous: StateFile | None,
    native: NativeCompletion,
    receipts: Sequence[Sequence[StateFile]],
    root: Path,
    commit: Callable[[], None],
) -> ChainStepReceipt:
    if (step > 0) != (previous is not None):
        raise ValueError("Only the first chain step starts without a predecessor")
    base = step_root(root, arm.name, step, attempt)
    certified = certify_update(
        authorization,
        nodes=plan.nodes,
        targets=arm.target_modules,
        step=step,
        native=native,
        receipts=receipts,
        adapter=checkpoint_dir(base, step),
        eval_root=base / "eval",
        snapshot_run_id=f"{plan.experiment_id}-{arm.name}",
    )
    receipt = ChainStepReceipt(
        arm=arm.name,
        step=step + 1,
        batch_sha256=plan.batches[step].sha256,
        behavior_policy=behavior_policy,
        policy_lag=step,
        targets=arm.target_modules,
        previous=previous,
        native=native,
        files=certified.files,
        snapshot=certified.snapshot.reference,
        eval_path=str(certified.snapshot.directory.relative_to(root)),
    )
    publish_receipt(base / RECEIPT, receipt.model_dump(mode="json"), certified.snapshot, commit=commit)
    return receipt


def read_previous(
    mount: Path, root: Path, plan: ChainPlan, arm: ChainArm, step: int, receipt_file: StateFile
) -> tuple[Path, ChainStepReceipt]:
    """The verified native state that 0-based ``step`` (> 0) continues from."""
    path = mount / receipt_file.path
    verify(path, receipt_file)
    receipt = ChainStepReceipt.model_validate_json(path.read_bytes())
    previous = step - 1
    if (
        step <= 0
        or path.parent.parent.parent.parent != root
        or receipt.arm != arm.name
        or receipt.step != step
        or receipt.batch_sha256 != plan.batches[previous].sha256
        or receipt.targets != arm.target_modules
        or receipt.policy_lag != previous
        or receipt.native.iteration != previous
        or receipt.native.world_size != plan.nodes * 8
        or not (receipt.native.optimizer and receipt.native.scheduler and receipt.native.rng)
    ):
        raise ValueError("Previous receipt changes the arm, batch, targets, lag, topology, or update boundary")
    if ChainPlan.model_validate_json((root / "plan.json").read_bytes()) != plan:
        raise ValueError("Chain continuation changes the experiment plan")
    names = [file.path for file in receipt.files]
    required = {"adapter_config.json", *(name for name in EXPORTS if name in names)} | {
        f"{prefix}{rank}.pt"
        for rank in range(receipt.native.world_size)
        for prefix in ("adapter_megatron_rank", "training_state_rank")
    }
    if len(set(names)) != len(names) or set(names) != required or len(required & set(EXPORTS)) != 1:
        raise ValueError("Previous receipt must contain exactly every native, optimizer and serving file")
    adapter = checkpoint_dir(path.parent, previous)
    for file in receipt.files:
        verify(adapter / file.path, file)
    if NativeCompletion.model_validate_json((adapter / "native_checkpoint.json").read_bytes()) != receipt.native:
        raise ValueError("Previous native completion marker differs from its committed receipt")
    return adapter, receipt


def stage_previous(
    mount: Path, root: Path, plan: ChainPlan, arm: ChainArm, step: int, receipt_file: StateFile, target: Path
) -> Path:
    """Copy the predecessor's verified native state to local disk; training never reads the Volume."""
    adapter, receipt = read_previous(mount, root, plan, arm, step, receipt_file)
    for file in receipt.files:
        copy_verified(adapter / file.path, target / file.path, file)
    write_atomic(target / "native_checkpoint.json", receipt.native.model_dump_json().encode())
    return target
