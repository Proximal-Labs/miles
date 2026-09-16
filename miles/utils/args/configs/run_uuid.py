from typing import Annotated

from miles.utils.args.schema import A, BaseConfig
from miles.utils.run_uuid import RUN_UUID_LENGTH


class RunUuidConfig(BaseConfig):
    run_uuid: Annotated[
        str | None,
        A(
            "--run-uuid",
            type=str,
            default=None,
            help=(
                f"Machine-readable identifier for this launch: exactly {RUN_UUID_LENGTH} lowercase "
                "hex characters, auto-generated when unset. Unlike the human-readable run "
                "names, two runs never share one, so anything stamped with it can be "
                "traced back to the launch that produced it."
            ),
        ),
    ]
