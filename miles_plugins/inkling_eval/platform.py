"""Proximal Connect JSON client; request fields follow proximal.v1 protobufs."""

import concurrent.futures
import hashlib
import json
import os
import time

import httpx

from miles_plugins.inkling_eval.config import EvalConfig

_MODEL = "modal/inkling-small"
_REGISTRY = "modal.inference.endpoints"
_TERMINAL = {3, 4, 5, 7, 8, 9}
_STATUS = {"SUCCESS": 3, "FAILED": 4, "STOPPED": 5, "ERROR": 7, "TIMEOUT": 8, "COMPLETED": 9}


class Platform:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.http = httpx.Client(
            base_url=config.platform_url.rstrip("/"),
            timeout=60,
            headers={"x-api-key": os.environ[config.api_key_env], "Connect-Protocol-Version": "1"},
        )

    def close(self):
        self.http.close()

    def rpc(self, service, method, payload):
        for attempt in range(5):
            try:
                response = self.http.post(f"/proximal.v1.{service}/{method}", json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 502, 503, 504} or attempt == 4:
                    raise
            except httpx.TransportError:
                if attempt == 4:
                    raise
            time.sleep(min(2**attempt, 15))

    def resolve_suite(self):
        ids = sorted({i for values in self.config.sets.values() for i in values})

        def resolve(environment_id):
            images = self.rpc("EnvironmentService", "ListImages", {"environmentId": environment_id}).get("images", [])
            images = [i for i in images if i.get("digest") and i.get("commitHash") and int(i.get("pushedAt", 0)) > 0]
            if not images:
                raise ValueError(f"Environment {environment_id} has no pushed image with a pinned commit")
            image = max(images, key=lambda i: (int(i["pushedAt"]), i["id"]))
            return str(environment_id), {
                "environmentId": environment_id,
                "imageId": image["id"],
                "sourceCommitSha": image["commitHash"],
                "imageDigest": image["digest"],
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            return dict(pool.map(resolve, ids))

    def endpoint(self, name, value):
        for attempt in range(8):
            record = self.rpc("LiveConfigService", "GetLiveConfig", {"key": _REGISTRY})["config"]
            registry = json.loads(record["valueJson"])
            model = registry["models"].setdefault(_MODEL, {"defaultEndpoint": name, "endpoints": {}})
            existing = model["endpoints"].get(name)
            if value is not None:
                if existing is not None and existing != value:
                    raise ValueError(f"Refusing to replace immutable endpoint {name}")
                model["endpoints"][name] = value
            else:
                model["endpoints"].pop(name, None)
                if model["defaultEndpoint"] == name:
                    if model["endpoints"]:
                        model["defaultEndpoint"] = sorted(model["endpoints"])[0]
                    else:
                        del registry["models"][_MODEL]
            try:
                self.rpc(
                    "LiveConfigService",
                    "SetLiveConfig",
                    {
                        "key": _REGISTRY,
                        "valueJson": json.dumps(registry),
                        "expectedRevision": record["revision"],
                    },
                )
                if value is not None:
                    # Platform workers refresh the endpoint LiveConfig cache every five seconds.
                    time.sleep(6)
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 409 or attempt == 7:
                    raise

    def rollout(self, pin, *, identity, endpoint):
        run_id = "inkling-eval-" + hashlib.sha256(identity.encode()).hexdigest()[:32]
        payload = {k: v for k, v in pin.items() if k != "imageDigest"}
        payload.update(
            runId=run_id,
            instances=1,
            ensureRolloutLaunchWorkflows=True,
            deploymentConfig=self.config.deployment_config,
            autoTriggerAnalysis=False,
            # Explicit false overrides environment-level QA defaults on the platform.
            autoTriggerPostQa=False,
            config={
                "agents": [
                    {
                        "agentType": self.config.agent_type,
                        "agentModel": _MODEL,
                        "endpointName": endpoint,
                        "reasoningEffort": "AGENT_REASONING_EFFORT_" + self.config.reasoning_effort,
                        "agentTimeoutSec": self.config.timeout_seconds,
                    }
                ]
            },
        )
        self.rpc("EnvironmentRunService", "CreateEnvironmentRun", payload)
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            response = self.rpc("EnvironmentRunService", "GetEnvironmentRunContainers", {"runId": run_id})
            containers = response.get("containers", [])
            if containers:
                if len(containers) != 1:
                    raise ValueError(f"Expected one rollout for {run_id}, got {len(containers)}")
                item = containers[0]
                status = item.get("status", 0)
                if isinstance(status, str):
                    status = _STATUS.get(status.removeprefix("ROLLOUT_CONTAINER_STATUS_"), 0)
                if status in _TERMINAL:
                    scored = status in {3, 9} and item.get("rewardScored", False)
                    return {"run_id": run_id, "status": status, "reward": item.get("reward", 0.0) if scored else None}
            time.sleep(self.config.poll_seconds)
        self.rpc("EnvironmentRunService", "StopEnvironmentRun", {"runId": run_id})
        return {"run_id": run_id, "status": "timeout", "reward": None}
