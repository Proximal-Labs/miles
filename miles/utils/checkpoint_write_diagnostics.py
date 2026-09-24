"""Preserve checkpoint I/O errors before Megatron replaces worker exceptions."""

import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _memory_snapshot():
    paths = (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.events",
        "/proc/meminfo",
    )
    result = {}
    for path in paths:
        try:
            result[path] = Path(path).read_text().strip()
        except OSError as error:
            result[path] = str(error)
    return result


def trace_checkpoint_writes(writer_module):
    """Log the original exception chain without changing serialization or retries."""
    original = writer_module._write_item

    @functools.wraps(original)
    def traced_write(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception:
            logger.exception("Checkpoint write failed before Megatron worker aggregation; memory=%s", _memory_snapshot())
            raise

    writer_module._write_item = traced_write
