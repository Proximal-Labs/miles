"""Pinned platform tasks through Miles's DataSource, with a checkpointed cursor.

Dataset order is deterministic and cycles. Retry groups precede new tasks;
unfinished groups are regenerated on restart, never restored as trainable data.

The same checkpoint carries the consumption ledger: which stored groups this
training run has already trained on. It is saved with the weights, so a resume
forgets consumption from steps whose weights were discarded, and those groups
become selectable again (if still within the staleness bound).
"""

from argparse import Namespace
from collections import deque
from pathlib import Path

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample
from miles_plugins.proximal.contracts import Contract, digest, read_run_config
from miles_plugins.proximal.storage import write_immutable


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
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.config = read_run_config(args.proximal_config)
        self.dataset = self.config.dataset.tasks
        self.next_group = 0
        self._retry: deque[int] = deque()
        self.consumed = ConsumptionLedger()

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        result = []
        for _ in range(num_samples):
            task_index = self._retry.popleft() if self._retry else self.next_group % len(self.dataset)
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
                dataset_sha256=digest(self.config.dataset),
                next_group=self.next_group,
                pending_tasks=tuple(self._retry),
                consumed=self.consumed.snapshot(),
            )
            write_immutable(
                Path(self.args.save) / "rollout" / f"proximal_{rollout_id}.json", state.model_dump_json().encode()
            )

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None or rollout_id is None or rollout_id < 0:
            return
        path = Path(self.args.load) / "rollout" / f"proximal_{rollout_id}.json"
        state = Cursor.model_validate_json(path.read_bytes())
        if state.dataset_sha256 != digest(self.config.dataset):
            raise ValueError("Checkpoint task membership/source differs from this run")
        self.next_group = state.next_group
        self._retry = deque(state.pending_tasks)
        self.consumed.restore(state.consumed)
