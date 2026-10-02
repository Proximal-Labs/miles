"""Backfill results to Modal, optionally following already-running evaluations."""

import argparse
import fcntl
import time
from pathlib import Path

from miles_plugins.reward_hacking.result_store import result_destination, sync_results


def _running(output):
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path, nargs="+")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    pending = {path: result_destination(path) for path in args.outputs}
    while pending:
        for path, destination in list(pending.items()):
            running = _running(path)
            try:
                sync_results(path, destination)
            except Exception:
                if not args.watch:
                    raise
                print(f"Upload failed for {path}; retrying in 30 seconds", flush=True)
                continue
            print(f"Synced {path} -> modal://{destination['volume']}{destination['path']}", flush=True)
            if not args.watch or not running:
                del pending[path]
        if pending:
            time.sleep(30)


if __name__ == "__main__":
    main()
