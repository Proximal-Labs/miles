"""The front process of a serving replica: its gateway and its capture, alongside SGLang.

The platform owns the container and mounts the existing adapter Volume. This
entrypoint neither launches SGLang nor creates/deletes any Modal resource.

One HTTP port serves both apps. Capture's routes (``/sessions``, ``/rollouts``) go to
capture; everything else goes to the gateway. Capture sends each rendered turn to the
gateway in-process, so a rollout's calls never leave the replica between capture and
SGLang. Sticky routing (``contracts.AFFINITY_HEADER``) sends every call for a rollout to
the replica that holds its session.
"""

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from miles_plugins.proximal.authorization import authorize_run, secret_env
from miles_plugins.proximal.contracts import Policy, read_run_config
from miles_plugins.proximal.gateway import GatewayConfig, PreparePolicy
from miles_plugins.proximal.replica import authorize_replica_load

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

    from miles_plugins.proximal.capture_server import PolicyCheck
    from miles_plugins.proximal.gateway import ReplicaGateway

CAPTURE_PREFIXES = ("/sessions", "/rollouts/", "/capture/")
# The gateway, seen from capture over an in-process transport; the host is never resolved.
IN_PROCESS_GATEWAY = "http://replica-gateway"


def front(gateway: "ASGIApp", capture: "ASGIApp") -> "ASGIApp":
    """Dispatch capture's routes to capture and all others to the gateway."""

    async def app(scope: "Scope", receive: "Receive", send: "Send") -> None:
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if path == "/sessions" or path.startswith(CAPTURE_PREFIXES):
            await capture(scope, receive, send)
        else:
            await gateway(scope, receive, send)

    return app


def admitted_by(gateway: "ReplicaGateway") -> "PolicyCheck":
    """On a replica, a session's policy is valid when this replica admits its adapter."""

    async def check(policy: Policy) -> bool:
        evidence = await gateway.prepare(PreparePolicy(snapshot=policy.snapshot, base_model=policy.base_model))
        return evidence.snapshot == policy.snapshot and evidence.base_model == policy.base_model

    return check


async def serve(args: argparse.Namespace) -> None:
    config = GatewayConfig.model_validate_json(Path(args.config).read_bytes())
    authorization = authorize_replica_load(config.replica, yes_load=args.yes_load)
    run = read_run_config(args.run_config)
    capture_authorization = authorize_run(run, yes_rollouts=args.yes_capture, yes_publish=args.yes_capture)
    key = secret_env(args.engine_api_key_env)
    # Optional SDK/runtime dependencies after free validation and authorization.
    import httpx
    import modal
    import uvicorn

    from miles_plugins.proximal.capture_server import CaptureServer, EngineEndpoint, capture_tokenizer
    from miles_plugins.proximal.gateway import ReplicaGateway
    from miles_plugins.proximal.replica import ReplicaLoRALoader

    volume = modal.Volume.from_name(args.volume_name, environment_name=args.environment_name, create_if_missing=False)
    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(headers=headers, timeout=args.timeout_seconds, trust_env=False) as loader_client:
        loader = ReplicaLoRALoader(
            authorization,
            volume_mount=Path(args.volume_mount),
            local_cache=Path(args.local_cache),
            reload_volume=volume.reload,
            client=loader_client,
        )
        async with httpx.AsyncClient(
            headers=headers, timeout=args.timeout_seconds, trust_env=False
        ) as inference_client:
            gateway = ReplicaGateway(config, loader=loader, client=inference_client)
            await gateway.validate_engine()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway.app), timeout=args.timeout_seconds
            ) as gateway_client:
                capture = CaptureServer(
                    capture_authorization,
                    tokenizer=capture_tokenizer(run.tokenizer_path, run.tito_model),
                    engine=EngineEndpoint(
                        client=gateway_client,
                        url=IN_PROCESS_GATEWAY,
                        headers={"Authorization": f"Bearer {secret_env(config.api_key_env)}"},
                    ),
                    policy_known=admitted_by(gateway),
                    root=Path(args.capture_root),
                )
                await uvicorn.Server(
                    uvicorn.Config(
                        front(gateway.app, capture.app), host=args.host, port=args.port, workers=1, access_log=False
                    )
                ).serve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Gateway config")
    parser.add_argument("--run-config", required=True, help="The run config capture enforces")
    parser.add_argument("--capture-root", required=True, help="Capture's state directory on this replica")
    parser.add_argument("--volume-name", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--volume-mount", required=True)
    parser.add_argument("--local-cache", required=True)
    parser.add_argument("--engine-api-key-env", default="SGLANG_API_KEY")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--yes-load", action="store_true", required=True)
    parser.add_argument("--yes-capture", action="store_true", required=True, help="Record rollouts on this replica")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or not 0 < args.port < 65536:
        parser.error("Invalid timeout or port")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
