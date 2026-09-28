"""Run-owned, single-threaded Volume publication lifecycle.

Shares one writer between the artifact outbox and native snapshots. No live
Postgres directory is copied, and no historical artifact tree is scanned.
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


class StateWriter:
    def __init__(
        self,
        *,
        dsn: str,
        pg_bin: Path,
        checkpoints: Path,
        artifacts: Path,
        snapshot_root: Path,
        context: RecoveryContext,
        launch_id: str,
        parent: str | None,
        taken: int | None,
        commit: Callable[[], None],
    ):
        self.dsn, self.pg_bin = dsn, pg_bin
        self.checkpoints, self.artifacts, self.root = checkpoints, artifacts, snapshot_root
        self.context, self.launch_id, self.parent, self.taken = context, launch_id, parent, taken
        self.commit = commit
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="run-state-publisher")
        self.failures = 0
        self._reported_at = 0.0

    def start(self) -> None:
        self.thread.start()

    def tick(self) -> int:
        if time.monotonic() - self._reported_at > 15:
            count, size, age = state_artifacts.backlog(self.dsn, self.context.run_id)
            logger.info(
                "Run state committed_step=%s pending_records=%d pending_bytes=%d oldest_seconds=%.1f failures=%d",
                self.taken,
                count,
                size,
                age,
                self.failures,
            )
            self._reported_at = time.monotonic()
        published = state_artifacts.publish_pending(
            dsn=self.dsn,
            run_id=self.context.run_id,
            artifacts=self.artifacts,
            snapshot_root=self.root,
            commit=self.commit,
        )
        steps = [
            step
            for step in state_checkpoints.complete_steps(self.checkpoints)
            if self.taken is None or step > self.taken
        ]
        if steps:
            step = steps[-1]
            checkpoint_id = state_checkpoints.take(
                step,
                checkpoints=self.checkpoints,
                dsn=self.dsn,
                snapshot_root=self.root,
                pg_bin=self.pg_bin,
                context=self.context,
                launch_id=self.launch_id,
                parent=self.parent,
                commit=self.commit,
            )
            self.taken, self.parent = step, checkpoint_id
            logger.info("Run state saved_step=%s committed_step=%s publication_failures=%s", step, step, self.failures)
            # Pruning only follows a successful committed checkpoint. Failure leaks
            # retained bytes, never makes the just-published recovery point disappear.
            state_checkpoints.prune(self.root, latest=checkpoint_id, commit=self.commit)
            for old in self.checkpoints.glob("iter_*"):
                if int(old.name.removeprefix("iter_")) < step:
                    shutil.rmtree(old)
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
        """Call after producers stop, while Postgres and the Volume are still available.

        Unlike periodic retries, failure of the final flush propagates to the run.
        """
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join()
        while self.tick():
            pass
