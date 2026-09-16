import argparse
from typing import Annotated

from miles.utils.args.schema import A, BaseConfig
from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig

_FT_CHOICES = ["rollout", "train"]
_DEFAULT_FT_API_SERVER_PORT = 18080


class FaultToleranceConfig(BaseConfig):
    use_fault_tolerance: Annotated[
        bool,
        A(
            "--use-fault-tolerance",
            action="store_true",
            default=False,
            help="Enable fault tolerance. Use --ft-components to select which components.",
        ),
    ]
    ft_components: Annotated[
        list[str] | None,
        A(
            "--ft-components",
            nargs="+",
            default=None,
            choices=_FT_CHOICES,
            help="FT components to enable (requires --use-fault-tolerance). "
            "Choices: rollout, train. Default when omitted: rollout.",
        ),
    ]
    api_server_host: Annotated[
        str,
        A(
            "--api-server-host",
            type=str,
            default="127.0.0.1",
            help="Host the HTTP api server binds to. The default only serves the local mini "
            "fault-tolerance controller; set 0.0.0.0 to accept remote controllers.",
        ),
    ]
    api_server_port: Annotated[
        int | None,
        A(
            "--api-server-port",
            type=int,
            default=None,
            help=f"Port for HTTP api server. 0 = disabled. Left unset it is "
            f"{_DEFAULT_FT_API_SERVER_PORT} under --use-fault-tolerance and 0 otherwise, "
            f"because the mini fault-tolerance controller drives cells over this port.",
        ),
    ]
    mini_ft_controller_enable: Annotated[
        bool | None,
        A(
            "--mini-ft-controller-enable",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable the mini fault-tolerance controller that auto-heals Fatal cells. "
            "Left unset it follows --ft-components and --api-server-port, which is what makes "
            "--use-fault-tolerance heal on its own.",
        ),
    ]
    mini_ft_controller_poll_interval: Annotated[
        float,
        A(
            "--mini-ft-controller-poll-interval",
            type=float,
            default=10.0,
            help="Interval in seconds between cell health polls.",
        ),
    ]
    mini_ft_controller_resume_delay: Annotated[
        float,
        A(
            "--mini-ft-controller-resume-delay",
            type=float,
            default=10.0,
            help="Delay in seconds between suspending and resuming a cell during heal.",
        ),
    ]

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser=parser)
        SimpleHealthCheckerConfig.add_arguments(parser, prefix="trainer-heartbeat-checker")

    @classmethod
    def _before_argument(cls, *, parser: argparse.ArgumentParser, name: str) -> None:
        if name == "api_server_host":
            SimpleHealthCheckerConfig.add_arguments(
                parser,
                prefix="rollout-health-check",
                interval_default=30.0,
                timeout_default=30.0,
                first_wait_default=0.0,
                failure_threshold_default=1,
            )
