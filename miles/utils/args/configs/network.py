from typing import Annotated

from miles.utils.args.schema import A, BaseConfig


class NetworkConfig(BaseConfig):
    http_proxy: Annotated[str | None, A("--http-proxy", type=str, default=None)]
    use_distributed_post: Annotated[bool, A("--use-distributed-post", action="store_true", default=False)]
