"""A throwaway local Postgres on a Unix socket: no network, no installation state.

Shared by the Stage A launcher and the CPU tests. Needs the server binaries from the
proximal_async test image; as root, the server runs as the ``postgres`` user.
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def local_postgres(root: Path | None = None) -> Iterator[str]:
    """Yield a DSN for a local cluster and stop it on exit.

    Without ``root``: a fresh throwaway cluster, deleted on exit. With ``root``: a
    persistent cluster kept there, so a later launch sees the same store.
    """
    binaries = sorted(Path("/usr/lib/postgresql").glob("*/bin"))
    if not binaries:
        raise RuntimeError("Postgres server binaries are required; use the proximal_async test image")
    bindir = binaries[-1]
    persistent = root is not None
    # Not under a private tmp root: the unprivileged server user must traverse it.
    root = root if root is not None else Path(tempfile.mkdtemp(prefix="proximal-pg-"))
    root.mkdir(parents=True, exist_ok=True)
    data, socket = root / "data", root / "socket"
    socket.mkdir(exist_ok=True)
    as_postgres: list[str] = []
    if os.geteuid() == 0:  # initdb refuses to run as root.
        shutil.chown(root, "postgres")
        shutil.chown(socket, "postgres")
        as_postgres = ["runuser", "-u", "postgres", "--"]
    try:
        if not (data / "PG_VERSION").exists():
            subprocess.run(
                [*as_postgres, str(bindir / "initdb"), "-D", str(data), "-U", "postgres", "--auth=trust"],
                check=True,
                capture_output=True,
            )
        subprocess.run(
            [
                *as_postgres,
                str(bindir / "pg_ctl"),
                "-D",
                str(data),
                "-l",
                str(root / "server.log"),
                "-o",
                f"-k {socket} -c listen_addresses=''",
                "-w",
                "start",
            ],
            check=True,
            capture_output=True,
        )
        yield f"host={socket} user=postgres dbname=postgres"
    finally:
        if data.exists():
            subprocess.run(
                [*as_postgres, str(bindir / "pg_ctl"), "-D", str(data), "-m", "fast", "stop"],
                check=False,
                capture_output=True,
            )
        if not persistent:
            shutil.rmtree(root, ignore_errors=True)
