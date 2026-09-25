"""One immutable, authenticated Modal SGLang deployment per evaluation point."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

WIRE_MODEL = "thinkingmachines/Inkling-Small:snapshot"


def server_command(settings):
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--model-path",
        settings["base"],
        "--served-model-name",
        "thinkingmachines/Inkling-Small",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--tp",
        str(settings["tp"]),
        "--context-length",
        str(settings["context_length"]),
        "--enable-lora",
        "--lora-backend",
        "triton",
        "--lora-use-virtual-experts",
        "--experts-shared-outer-loras",
        "--lora-strict-loading",
        "--max-loras-per-batch",
        "1",
        "--attention-backend",
        "fa4",
        "--moe-runner-backend",
        "triton",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--disable-custom-all-reduce",
        "--max-lora-rank",
        str(settings["rank"]),
        "--lora-target-modules",
        "all",
        "--lora-paths",
        f"snapshot={settings['adapter']}",
        "--max-loaded-loras",
        "1",
        "--max-running-requests",
        str(settings["concurrency"]),
        "--mem-fraction-static",
        "0.8",
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-cache-report",
    ]


def _verify_snapshot(path):
    root = Path(path)
    hashes = json.loads((root / ".complete").read_text())
    if set(hashes) != {"adapter_model.safetensors", "adapter_config.json"}:
        raise ValueError("Incomplete adapter manifest")
    for name, expected in hashes.items():
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Corrupt evaluation snapshot: {name}")


def _start_server(settings):
    _verify_snapshot(settings["adapter"])
    process = subprocess.Popen(server_command(settings))
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Evaluation inference exited with {process.returncode}")
        try:
            response = httpx.get("http://127.0.0.1:8000/v1/models", timeout=5)
            response.raise_for_status()
            if "snapshot" in {m["id"] for m in response.json()["data"]}:
                return process
        except (httpx.HTTPError, KeyError):
            pass
        time.sleep(5)
    process.terminate()
    raise TimeoutError("Evaluation inference did not load the snapshot within 30 minutes")


def deploy(settings, *, name, image, environment, gpu):
    # Modal is optional for all non-evaluation training paths.
    import modal

    root = Path(__file__).resolve().parents[2]
    container_image = modal.Image.from_registry(image).entrypoint([]).pip_install("httpx==0.28.1")
    container_image = container_image.env({"PYTHONPATH": "/opt/inkling-miles:/root/Megatron-LM"})
    for directory in ("miles", "miles_plugins"):
        container_image = container_image.add_local_dir(root / directory, f"/opt/inkling-miles/{directory}")
    volume = modal.Volume.from_name("inkling-small-rft", environment_name=environment)
    app = modal.App(name)

    @app.server(
        image=container_image,
        gpu=gpu,
        volumes={"/mnt/inkling": volume},
        serialized=True,
        min_containers=0,
        max_containers=1,
        scaledown_window=300,
        startup_timeout=1800,
        port=8000,
        routing_region="us-west",
        unauthenticated=False,
        memory=262144,
    )
    class Inference:
        @modal.enter()
        def start(self):
            self.process = _start_server(settings)

        @modal.exit()
        def stop(self):
            self.process.terminate()

    app.deploy(environment_name=environment)
    url = Inference.get_url()
    if not url:
        raise RuntimeError(f"Modal deployment {name} returned no serving URL")
    return {"app_id": app.app_id, "url": url.rstrip("/")}


def wait_ready(url, timeout=2100):
    key = os.environ["MODAL_INFERENCE_API_KEY"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url + "/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=60)
            response.raise_for_status()
            if "snapshot" in {model["id"] for model in response.json()["data"]}:
                return
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403}:
                raise
        except httpx.TransportError:
            pass
        time.sleep(5)
    raise TimeoutError("Modal evaluation endpoint did not become ready")


def stop(app_id, environment):
    subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id, "--env", environment, "--yes"], check=True)


def commit_volume(environment):
    import modal

    modal.Volume.from_name("inkling-small-rft", environment_name=environment).commit()
