"""Run inside an existing Modal replica, alongside its already-running SGLang.

The platform owns the container and mounts the existing adapter Volume. This
entrypoint neither launches SGLang nor creates/deletes any Modal resource.
"""

import argparse
import asyncio
from pathlib import Path

from miles_plugins.proximal.authorization import secret_env
from miles_plugins.proximal.gateway import GatewayConfig
from miles_plugins.proximal.replica import authorize_replica_load


async def serve(args: argparse.Namespace) -> None:
    config = GatewayConfig.model_validate_json(Path(args.config).read_bytes())
    authorization = authorize_replica_load(config.replica, yes_load=args.yes_load)
    key = secret_env(args.engine_api_key_env)
    # Optional SDK/runtime dependencies after free validation and authorization.
    import httpx
    import modal
    import uvicorn

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
            await uvicorn.Server(
                uvicorn.Config(gateway.app, host=args.host, port=args.port, workers=1, access_log=False)
            ).serve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--volume-name", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--volume-mount", required=True)
    parser.add_argument("--local-cache", required=True)
    parser.add_argument("--engine-api-key-env", default="SGLANG_API_KEY")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--yes-load", action="store_true", required=True)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or not 0 < args.port < 65536:
        parser.error("Invalid timeout or port")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
