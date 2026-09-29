"""Pinned platform tasks through Miles's DataSource, with a checkpointed cursor.

Dataset order is deterministic and cycles. Retry groups precede new tasks;
unfinished groups are regenerated on restart, never restored as trainable data.

The same checkpoint carries the consumption ledger: which stored groups this
training run has already trained on. Miles saves this state right after the
weights for the same step. Resuming a step restores the ledger saved with those
weights, so consumption by discarded steps is forgotten and those groups become
selectable again if still fresh. A step whose weights were saved but whose state
was not (an interrupted save) refuses to resume rather than pairing them wrongly.
"""

from argparse import Namespace
from collections import deque
from pathlib import Path

from miles.rollout.data_source import DataSource
from miles.utils.resume import resume_checkpoint_dir
from miles.utils.types import Sample
from miles_plugins.proximal.contracts import Contract, pinned_dataset, read_run_config
from miles_plugins.proximal.storage import write_atomic


class ConsumedGroup(Contract):
    group_id: str
    policy_version: int


class Cursor(Contract):
    dataset_sha256: str
    next_group: int
    pending_tasks: tuple[int, ...]
    consumed: tuple[ConsumedGroup, ...]


class ConsumptionLedger:
    """Group IDs consumed by this training run, keyed to their policy version."""

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}

    def add(self, group_id: str, policy_version: int) -> None:
        if group_id in self._versions:
            raise ValueError(f"Group {group_id} was already consumed by this training run")
        self._versions[group_id] = policy_version

    def ids(self) -> list[str]:
        return list(self._versions)

    def prune(self, *, below_version: int) -> None:
        # Staleness only grows, so groups below the window can never be selected again.
        self._versions = {g: v for g, v in self._versions.items() if v >= below_version}

    def snapshot(self) -> tuple[ConsumedGroup, ...]:
        return tuple(ConsumedGroup(group_id=g, policy_version=v) for g, v in sorted(self._versions.items()))

    def restore(self, entries: tuple[ConsumedGroup, ...]) -> None:
        self._versions = {entry.group_id: entry.policy_version for entry in entries}


class PlatformTaskSource(DataSource):
    def __init__(self, args: Namespace, *, num_groups: int | None = None) -> None:
        if num_groups is not None and num_groups <= 0:
            raise ValueError("Finite collection requires a positive group count")
        self.args = args
        self.config = read_run_config(args.proximal_config)
        if num_groups is not None:
            if self.config.research.unused_groups != "retry":
                raise ValueError("Finite collection requires retrying failed groups")
            if args.rollout_submission_granularity != "sample":
                raise ValueError("Finite collection requires the sample submission scheduler")
        self.dataset = self.config.dataset.tasks
        self.next_group = 0
        self._retry: deque[int] = deque()
        self.consumed = ConsumptionLedger()
        # Collection alone bounds new tasks. Retries do not consume this budget
        # or skip a task; next_group remains the unique execution-group counter.
        self._collection_groups = num_groups
        self._remaining_groups = num_groups

    @property
    def has_samples(self) -> bool:
        return bool(self._retry) or self._remaining_groups is None or self._remaining_groups > 0

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        result = []
        for _ in range(num_samples):
            if not self.has_samples:
                raise ValueError("Finite task source is exhausted; stop submitting new groups")
            if self._retry:
                task_index = self._retry.popleft()
            elif self._remaining_groups is not None:
                assert self._collection_groups is not None
                task_index = (self._collection_groups - self._remaining_groups) % len(self.dataset)
                self._remaining_groups -= 1
            else:
                task_index = self.next_group % len(self.dataset)
            group_index = self.next_group
            self.next_group += 1
            task = self.dataset[task_index]
            result.append(
                [
                    Sample(
                        group_index=group_index,
                        index=group_index * self.config.research.group_size + i,
                        prompt="",
                        metadata={"proximal_task_index": task_index, "proximal_task": task.model_dump()},
                    )
                    for i in range(self.config.research.group_size)
                ]
            )
        return result

    def add_samples(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self._retry.append(group[0].metadata["proximal_task_index"])

    def get_buffer_length(self) -> int:
        return len(self._retry)

    def save(self, rollout_id: int) -> None:
        if self.args.save is not None:
            state = Cursor(
                dataset_sha256=pinned_dataset(self.config.dataset).sha256,
                next_group=self.next_group,
                pending_tasks=tuple(self._retry),
                consumed=self.consumed.snapshot(),
            )
            # Overwritten, like Miles's own per-step data-source state: a run resumed
            # from an earlier step saves this step again with different contents.
            write_atomic(
                Path(self.args.save) / "rollout" / f"proximal_{rollout_id}.json", state.model_dump_json().encode()
            )

    def load(self, rollout_id: int | None = None) -> None:
        checkpoint_dir = resume_checkpoint_dir(self.args)
        restored = rollout_id if checkpoint_dir is not None and rollout_id is not None and rollout_id >= 0 else None
        state = (
            None if restored is None or checkpoint_dir is None else self._read_valid_state(checkpoint_dir, restored)
        )
        # Only after the requested checkpoint is known good: drop state saved for later
        # steps of the abandoned timeline, before training writes any new weights.
        # Otherwise a crash after re-saving step N's weights but before re-saving its
        # state would pair them with the old timeline's ledger. A failed restore
        # deletes nothing.
        self._invalidate_after(-1 if restored is None else restored)
        if state is None:
            return
        self.next_group = state.next_group
        self._retry = deque(state.pending_tasks)
        self.consumed.restore(state.consumed)

    def _read_valid_state(self, checkpoint_dir: str, step: int) -> Cursor:
        path = Path(checkpoint_dir) / "rollout" / f"proximal_{step}.json"
        if not path.exists():
            # Miles saves weights before this state. Weights without it means the save
            # was interrupted; resuming would pair those weights with a stale ledger.
            raise FileNotFoundError(
                f"Checkpoint step {step} has weights but no task/consumption state at {path}; "
                "resume from the previous complete checkpoint"
            )
        state = Cursor.model_validate_json(path.read_bytes())
        if state.dataset_sha256 != pinned_dataset(self.config.dataset).sha256:
            raise ValueError("Checkpoint task membership/source differs from this run")
        return state

    def _invalidate_after(self, step: int) -> None:
        if self.args.save is None:
            return
        for path in (Path(self.args.save) / "rollout").glob("proximal_*.json"):
            suffix = path.stem.removeprefix("proximal_")
            if suffix.isdigit() and int(suffix) > step:
                path.unlink()
