"""Periodically copy durable local evaluation artifacts to a dedicated Modal volume."""

import io
import json
import threading
import uuid
from contextlib import contextmanager

from miles_plugins.reward_hacking.curate import write_json

RESULT_VOLUME = "reward-hacking"


def result_destination(output):
    path = output / "result-storage.json"
    if path.exists():
        return json.loads(path.read_text())
    identity = output.parent.name + "/" + output.name + "-" + uuid.uuid4().hex[:12]
    destination = {"volume": RESULT_VOLUME, "path": "/results/" + identity, "environment": "main"}
    write_json(path, destination)
    return destination


def snapshot_files(output):
    snapshots = {}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name.startswith(".") or path.suffix == ".partial":
            continue
        data = path.read_bytes()
        if path.suffix == ".jsonl" and data and not data.endswith(b"\n"):
            data = data[: data.rfind(b"\n") + 1]
        snapshots[path.relative_to(output).as_posix()] = data
    return snapshots


def sync_results(output, destination):
    # Modal is optional for dataset construction and inference dry runs.
    import modal

    volume = modal.Volume.from_name(
        destination["volume"], environment_name=destination["environment"], create_if_missing=True
    )
    snapshots = snapshot_files(output)
    with volume.batch_upload(force=True) as upload:
        for name, data in snapshots.items():
            upload.put_file(io.BytesIO(data), destination["path"] + "/" + name)


@contextmanager
def result_uploads(output, *, interval=30):
    destination = result_destination(output)
    sync_results(output, destination)
    print(f"Results: modal://{destination['volume']}{destination['path']}", flush=True)
    stopped = threading.Event()

    def periodically_upload():
        while not stopped.wait(interval):
            try:
                sync_results(output, destination)
            except Exception as error:
                print(f"Modal result upload failed ({type(error).__name__}); retrying next interval", flush=True)

    worker = threading.Thread(target=periodically_upload, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()
        # Final upload failure is surfaced, not mistaken for durable completion.
        sync_results(output, destination)
