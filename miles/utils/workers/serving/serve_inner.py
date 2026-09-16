from __future__ import annotations

import os
import sys
from typing import Any

import uvicorn

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.function_registry import load_function
from miles.utils.workers.backend_capability.base import BackendCapability, DeferredBackendCapability
from miles.utils.workers.connection_config import StaticConnConfig
from miles.utils.workers.rpc.server.app import create_rpc_app
from miles.utils.workers.serving.utils import create_server_socket, parse_own_args, parse_runtime_config
from miles.utils.workers.serving.worker_identity import (
    read_worker_identity_from_metadata,
    read_worker_in_pod_index,
    read_worker_metadata,
)
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_spec import RPC_PORT_NAME, BaseServeSpec, PortInfo


def main() -> None:
    own_args = parse_own_args(sys.argv[1:])
    runtime = parse_runtime_config(own_args.config)
    args = runtime.worker.args
    [spec] = compute_specs(args, worker_type=runtime.worker.kind)
    worker = create_worker(
        spec,
        args=args,
        static_connections=runtime.static_connections,
    )
    _log(f"pool_id={spec.name} worker_class={spec.worker_class}")

    port = _rpc_port_of(spec).effective_static_port(worker_in_pod_index=read_worker_in_pod_index(os.environ))
    app = create_rpc_app(worker)
    with create_server_socket(port=port) as server_socket:
        _log(f"serve address={server_socket.getsockname()}")
        uvicorn.Server(uvicorn.Config(app)).run(sockets=[server_socket])


def create_worker(spec: BaseServeSpec, *, args: Any, static_connections: StaticConnConfig) -> Any:
    identity = read_worker_identity_from_metadata(os.environ)
    _log(f"identity={identity}")
    capability = DeferredBackendCapability(create=lambda: _backend_capability(static_connections))
    context = identity.ctor_context(capability=capability).model_copy(update={"args": args})
    return load_function(spec.worker_class)(**spec.ctor_kwargs(context))


def _backend_capability(static_connections: StaticConnConfig) -> BackendCapability:
    return compute_helm_backend_capability(args=static_connections)


def _rpc_port_of(spec: BaseServeSpec) -> PortInfo:
    ports = [port_info for port_info in read_worker_metadata(os.environ).port_infos if port_info.name == RPC_PORT_NAME]
    assert len(ports) == 1, f"spec '{spec.name}' declares {len(ports)} rpc ports, so this process cannot pick one"
    return ports[0]


def _log(message: str) -> None:
    print(f"[serve_inner] {message}", flush=True)


if __name__ == "__main__":
    main()
