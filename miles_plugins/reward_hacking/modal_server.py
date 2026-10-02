"""Own one temporary Inkling endpoint using the checkpoint evaluator's serving code."""

import os
import signal
import time
import uuid
from contextlib import contextmanager

from miles_plugins.inkling_eval import serving
from miles_plugins.reward_hacking.curate import write_json


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


@contextmanager
def cancellation_handlers():
    """Let normal stack unwinding stop owned inference on SIGTERM as on Ctrl-C."""
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _stop_owned(identity, environment):
    # A second Ctrl-C must not interrupt the cleanup caused by the first one.
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for attempt in range(3):
            try:
                serving.stop(identity, environment)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def inference_endpoint(config, output):
    """Persist identity before deployment; clean up by unique name if deploy is interrupted."""
    if not os.environ.get("MODAL_INFERENCE_API_KEY"):
        raise ValueError("Set MODAL_INFERENCE_API_KEY before deploying Inkling inference")
    environment = config.get("environment", "main")
    name = "reward-hacking-monitor-" + uuid.uuid4().hex[:16]
    state = {"name": name, "environment": environment, "status": "deploying", "settings": config}
    path = output / "modal-deployment.json"
    write_json(path, state)
    identity = name
    try:
        deployment = serving.deploy(
            {
                "base": config["base"],
                "adapter": config.get("adapter"),
                "rank": config.get("rank", 32),
                "tp": config.get("tp", 8),
                "replicas": config.get("replicas", 1),
                "max_replicas": config.get("replicas", 1) + 1,
                "context_length": config.get("context_length", 1048576),
                "concurrency": config.get("concurrency", 4),
                "tokenizer_workers": 1,
                "cpu": 32,
            },
            name=name,
            image=config["image"],
            environment=environment,
            gpu=config.get("gpu", "B300:8"),
        )
        identity = deployment["app_id"]
        state.update(deployment, status="starting")
        write_json(path, state)
        model = serving.WIRE_MODEL if config.get("adapter") else serving.BASE_MODEL
        serving.wait_ready(
            deployment["url"],
            timeout=config.get("startup_timeout", 7200),
            model_id="snapshot" if config.get("adapter") else model,
        )
        state["status"] = "ready"
        write_json(path, state)
        yield model, (deployment["url"] + "/v1", {"Authorization": "Bearer " + os.environ["MODAL_INFERENCE_API_KEY"]})
    finally:
        state["status"] = "stopping"
        write_json(path, state)
        try:
            _stop_owned(identity, environment)
        except Exception:
            state["status"] = "cleanup_failed"
            write_json(path, state)
            raise
        state["status"] = "stopped"
        write_json(path, state)
