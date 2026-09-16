from __future__ import annotations

import os
import sys

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.argv_utils import python_argv_prefix
from miles.utils.workers.env_vars import PLATFORM_IDENTITY_ENV_VARS
from miles.utils.workers.serving.utils import parse_own_args, parse_runtime_config, split_worker_argv
from miles.utils.workers.serving.worker_identity import read_worker_identity
from miles.utils.workers.worker_spec import WorkerLaunchContext

SERVE_INNER_MODULE = "miles.utils.workers.serving.serve_inner"


def main() -> None:
    own_argv, worker_argv = split_worker_argv(sys.argv[1:])
    own_args = parse_own_args(own_argv)
    _log(f"start own_argv={own_argv} worker_argv={worker_argv}")

    runtime = parse_runtime_config(own_args.config)
    args = runtime.worker.args
    [spec] = compute_specs(args, worker_type=runtime.worker.kind)
    identity = read_worker_identity(scheduling=spec.scheduling(), environ=os.environ)
    env_vars = spec.env_var(
        WorkerLaunchContext(
            args=args,
            cell_index=identity.cell_index,
            worker_in_cell_index=identity.worker_in_cell_index,
            gpu_ids=identity.gpu_ids,
        )
    )
    overridden = sorted(name for name in PLATFORM_IDENTITY_ENV_VARS if name in env_vars)
    assert not overridden, (
        f"spec {own_args.pool_id} sets {overridden}, which the platform owns; a worker that read the spec's value "
        f"would report the identity of another worker and bind that worker's ports"
    )
    _log(f"pool_id={own_args.pool_id} env_vars={env_vars}")

    inner_argv = [*python_argv_prefix(), "-m", SERVE_INNER_MODULE, *own_argv, "--", *worker_argv]
    _log(f"exec {inner_argv}")
    os.execve(sys.executable, inner_argv, dict(os.environ) | env_vars)


def _log(message: str) -> None:
    print(f"[serve] {message}", flush=True)


if __name__ == "__main__":
    main()
