"""Modal deployment of the Miles-owned serving pool (SGLang + replica gateway).

Deploy from the repository root, in the Miles environment (SGLang importable),
with both configs as local files:

    PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=serving.json \\
        modal deploy --env <environment> -m miles_plugins.proximal.serving_app

Engine arguments are rendered and validated here, at deploy time, and baked
into the image: a bad flag fails the deploy, not a GPU replica's startup.
This creates or updates only this app. The base-weight and adapter Volumes and
the gateway secret must already exist; nothing here creates or deletes them.
The deployed URL (``https://<workspace>--<app>-replica.<region>.modal.direct``)
is what the platform's endpoint registry records.

Each replica runs SGLang on loopback with Miles-derived arguments, then the
gateway in front of it. If either process exits, the replica exits so Modal
replaces it: adapter state is never repaired in place.
"""

import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

from miles_plugins.proximal.contracts import RunConfig
from miles_plugins.proximal.serving import ENGINE_PORT, GATEWAY_PORT, ServingDeployment, engine_argv, gateway_config

_RUN_PATH = "PROXIMAL_RUN_CONFIG"
_SERVING_PATH = "PROXIMAL_SERVING_CONFIG"
_ENGINE_ARGV_PATH = "PROXIMAL_ENGINE_ARGV"
_ENGINE_KEY_ENV = "MILES_ENGINE_API_KEY"
_CONTAINER_CONFIG_DIR = "/proximal-config"


def _read(path_env: str) -> str:
    # Deploy time: the local file. In the container: the copy baked into the image.
    path = os.environ.get(path_env)
    if not path:
        raise RuntimeError(f"Set {path_env} to the config file to deploy")
    return Path(path).read_text()


RUN_JSON = _read(_RUN_PATH)
SERVING_JSON = _read(_SERVING_PATH)
RUN = RunConfig.model_validate_json(RUN_JSON)
DEPLOYMENT = ServingDeployment.model_validate_json(SERVING_JSON)
# Resolved at deploy time, so the container need not import SGLang to read it.
_engine_argv_path = os.environ.get(_ENGINE_ARGV_PATH)
ENGINE_ARGV_JSON = (
    Path(_engine_argv_path).read_text() if _engine_argv_path else json.dumps(engine_argv(RUN, DEPLOYMENT))
)


def with_configs(image: modal.Image) -> modal.Image:
    """Bake the deploy-time configs into an image as files, for every container that imports this module.

    Files, not environment variables: a run config pinning thousands of tasks exceeds
    the kernel's 128 KiB limit on one environment string, and the container cannot exec.
    """
    local = Path(tempfile.mkdtemp(prefix="proximal-config-"))
    files = {
        _RUN_PATH: ("run.json", RUN_JSON),
        _SERVING_PATH: ("serving.json", SERVING_JSON),
        _ENGINE_ARGV_PATH: ("engine-argv.json", ENGINE_ARGV_JSON),
    }
    image = image.env({env: f"{_CONTAINER_CONFIG_DIR}/{name}" for env, (name, _) in files.items()})
    for name, text in files.values():
        (local / name).write_text(text)
        image = image.add_local_file(local / name, f"{_CONTAINER_CONFIG_DIR}/{name}")
    return image


base_volume = modal.Volume.from_name(
    DEPLOYMENT.base_volume.volume_name,
    environment_name=DEPLOYMENT.base_volume.environment_name,
    create_if_missing=False,
)
adapter_volume = modal.Volume.from_name(
    RUN.volume.volume_name, environment_name=RUN.volume.environment_name, create_if_missing=False
)

image = (
    with_configs(modal.Image.from_registry(DEPLOYMENT.image).entrypoint([]))
    # This fork's plugin and Miles sources, over the Miles image's installed copy.
    .add_local_python_source("miles", "miles_plugins")
)

app = modal.App(DEPLOYMENT.app_name)


def _wait_healthy(url: str, process: subprocess.Popen[bytes], timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{url} process exited with code {process.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"{url} did not become healthy")


def _exit_when_any_dies(processes: list[subprocess.Popen[bytes]]) -> None:
    while all(process.poll() is None for process in processes):
        time.sleep(5)
    # Engine and gateway share one lifetime; a replacement replica starts clean.
    os._exit(1)


@app.server(
    image=image,
    gpu=DEPLOYMENT.gpu,
    volumes={
        str(DEPLOYMENT.base_mount): base_volume,
        str(DEPLOYMENT.adapter_mount): adapter_volume,
    },
    secrets=[modal.Secret.from_name(DEPLOYMENT.gateway_secret, environment_name=RUN.volume.environment_name)],
    min_containers=DEPLOYMENT.min_replicas,
    max_containers=DEPLOYMENT.max_replicas,
    target_concurrency=DEPLOYMENT.target_concurrency,
    scaledown_window=DEPLOYMENT.scaledown_window_seconds,
    startup_timeout=DEPLOYMENT.startup_timeout_seconds,
    port=GATEWAY_PORT,
    routing_region=DEPLOYMENT.routing_region,
    unauthenticated=False,  # Modal proxy auth, plus the gateway's own credential.
    exit_grace_period=25,
)
class Replica:
    @modal.enter()
    def start(self) -> None:
        os.environ[_ENGINE_KEY_ENV] = secrets.token_urlsafe(32)  # Loopback-only engine credential.
        engine_cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            *json.loads(ENGINE_ARGV_JSON),
            "--api-key",
            os.environ[_ENGINE_KEY_ENV],
        ]
        self.engine = subprocess.Popen(engine_cmd, start_new_session=True)
        _wait_healthy(f"http://127.0.0.1:{ENGINE_PORT}/health", self.engine, DEPLOYMENT.startup_timeout_seconds)

        Path(DEPLOYMENT.local_cache).mkdir(parents=True, exist_ok=True)
        config_path = Path(DEPLOYMENT.local_cache) / "gateway.json"
        config_path.write_text(gateway_config(RUN, DEPLOYMENT).model_dump_json())
        self.gateway = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "miles_plugins.proximal.serve_replica",
                "--config",
                str(config_path),
                "--volume-name",
                RUN.volume.volume_name,
                "--environment-name",
                RUN.volume.environment_name,
                "--volume-mount",
                str(DEPLOYMENT.adapter_mount),
                "--local-cache",
                str(DEPLOYMENT.local_cache / "adapters"),
                "--engine-api-key-env",
                _ENGINE_KEY_ENV,
                "--host",
                "0.0.0.0",
                "--port",
                str(GATEWAY_PORT),
                "--yes-load",
            ],
            start_new_session=True,
        )
        _wait_healthy(f"http://127.0.0.1:{GATEWAY_PORT}/health", self.gateway, 300)
        threading.Thread(target=_exit_when_any_dies, args=([self.engine, self.gateway],), daemon=True).start()

    @modal.exit()
    def stop(self) -> None:
        for name in ("gateway", "engine"):
            process = getattr(self, name, None)
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
