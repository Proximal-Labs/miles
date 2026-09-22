"""Atomic local artifact writes, shared by the capture service, store and driver."""

import os
import tempfile
from pathlib import Path


def write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError(f"Immutable artifact conflict: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def write_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` atomically: readers see the old or new bytes, never a mix.

    For checkpoint state keyed by step, which a resumed run legitimately rewrites.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
