"""Run-owned, single-threaded Volume publication lifecycle.

CPU collectors use the same artifact publisher as online trainers. A trainer also
supplies its native checkpoint publication callback. Only this writer commits the
Volume while producers are running.
"""

import logging
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from miles_plugins.proximal import state_artifacts, state_checkpoints
from miles_plugins.proximal.state_checkpoints import RecoveryContext

logger = logging.getLogger(__name__)


def checkpoint_publisher(
    *,
    dsn: str,
    pg_bin: Path,
    checkpoints: Path,
    snapshot_root: Path,
    context: RecoveryContext,
    launch_id: str,
    parent: str | None,
    taken: int | None,
    commit: Callable[[], None],
) -> Callable[[], None]:
    """Compose native checkpoint publication into the run's existing writer."""

    def publish() -> None:
        nonlocal taken, parent
        steps = [step for step in state_checkpoints.complete_steps(checkpoints) if taken is None or step > taken]
        if not steps:
            return
        step = steps[-1]
        checkpoint_id = state_checkpoints.take(
            step,
            checkpoints=checkpoints,
            dsn=dsn,
            snapshot_root=snapshot_root,
            pg_bin=pg_bin,
            context=context,
            launch_id=launch_id,
            parent=parent,
            commit=commit,
        )
        taken, parent = step, checkpoint_id
        logger.info("Run state saved_step=%s committed_step=%s", step, step)
        state_checkpoints.prune(snapshot_root, latest=checkpoint_id, commit=commit)
        for old in checkpoints.glob("iter_*"):
            if int(old.name.removeprefix("iter_")) < step:
                shutil.rmtree(old)

    return publish


class StateWriter:
    def __init__(
        self,
        *,
        dsn: str,
        run_id: str,
        artifacts: Path,
        snapshot_root: Path,
        commit: Callable[[], None],
        publish_checkpoint: Callable[[], None] | None = None,
    ):
        self.dsn, self.run_id = dsn, run_id
        self.artifacts, self.root = artifacts, snapshot_root
        self.commit, self.publish_checkpoint = commit, publish_checkpoint
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="run-state-publisher")
        self.failures = 0
        self._reported_at = 0.0

    def start(self) -> None:
        self.thread.start()

    def tick(self) -> int:
        if time.monotonic() - self._reported_at > 15:
            count, size, age = state_artifacts.backlog(self.dsn, self.run_id)
            logger.info(
                "Run state pending_records=%d pending_bytes=%d oldest_seconds=%.1f failures=%d",
                count,
                size,
                age,
                self.failures,
            )
            self._reported_at = time.monotonic()
        published = state_artifacts.publish_pending(
            dsn=self.dsn,
            run_id=self.run_id,
            artifacts=self.artifacts,
            snapshot_root=self.root,
            commit=self.commit,
        )
        if self.publish_checkpoint is not None:
            self.publish_checkpoint()
        return published

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                if self.tick():
                    continue
            except Exception:
                self.failures += 1
                logger.exception("Run-state publication failed; retaining local results and retrying")
            self.stop.wait(1)

    def close(self) -> None:
        """Drain after producers stop, before Postgres/Volume close; failures propagate."""
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()
        while self.tick():
            pass
