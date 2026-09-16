import argparse
from typing import Annotated

from sglang_router.launch_router import RouterArgs

from miles.utils.args.schema import A, BaseConfig


class RouterConfig(BaseConfig):
    use_miles_router: Annotated[
        bool,
        A(
            "--use-miles-router",
            action="store_true",
            default=False,
            help="Whether to use MilesRouter for text-based routing instead of SGLang token-based routing",
        ),
    ]
    miles_router_timeout: Annotated[
        float | None,
        A("--miles-router-timeout", type=float, default=None, help="Timeout for MilesRouter HTTP requests in seconds."),
    ]
    miles_router_max_connections: Annotated[
        int | None,
        A("--miles-router-max-connections", type=int, default=None, help="Max connections for MilesRouter HTTP client."),
    ]
    miles_router_health_check_failure_threshold: Annotated[
        int,
        A(
            "--miles-router-health-check-failure-threshold",
            type=int,
            default=3,
            help="Number of consecutive failures before marking a worker as unhealthy.",
        ),
    ]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
