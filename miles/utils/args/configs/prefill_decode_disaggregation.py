from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class PrefillDecodeDisaggregationConfig(BaseConfig):
    prefill_num_servers: Annotated[
        int | None,
        A("--prefill-num-servers", type=int, default=None, help="Number of prefill servers for disaggregation."),
    ]
