"""Pinned platform tasks through Miles's DataSource, with a checkpointed cursor.

Dataset order is deterministic and cycles. Retry groups precede new tasks;
unfinished groups are regenerated on restart, never restored as trainable data.
"""

from argparse import Namespace
from collections import deque
from pathlib import Path

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample
from miles_plugins.proximal.contracts import Contract, digest, read_run_config
from miles_plugins.proximal.storage import write_immutable


class Cursor(Contract):
    dataset_sha256: str
    next_group: int
    pending_tasks: tuple[int, ...]


class PlatformTaskSource(DataSource):
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.config = read_run_config(args.proximal_config)
        self.dataset = self.config.dataset.tasks
        self.next_group = 0
        self._retry: deque[int] = deque()

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
