import os
from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


# prometheus
class PrometheusConfig(BaseConfig):
    use_prometheus: Annotated[bool, A("--use-prometheus", action="store_true", default=False)]
    prometheus_port: Annotated[
        int,
        A(
            "--prometheus-port",
            type=int,
            default_factory=lambda: int(os.environ.get("PROMETHEUS_PORT", "9090")),
            help="Port for the Prometheus metrics HTTP server. "
            "Prometheus scrapes /metrics on this port. "
            "Defaults to PROMETHEUS_PORT env var or 9090.",
        ),
    ]
    prometheus_run_name: Annotated[
        str | None,
        A(
            "--prometheus-run-name",
            type=str,
            default=None,
            help="Human-readable run name attached as a 'run_name' label to all "
            "Prometheus metrics. Used to distinguish runs in Grafana. "
            "Defaults to --wandb-group if set.",
        ),
    ]
