from argparse import Namespace
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, TypeAlias

from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.schema import BaseConfig


ConfigSource: TypeAlias = BaseConfig | Namespace | Mapping[str, Any]


class ImmutableNamespace:
    __slots__ = ("_values",)

    def __init__(self, **values: Any) -> None:
        object.__setattr__(self, "_values", MappingProxyType(values))

    @property
    def __dict__(self) -> Mapping[str, Any]:
        return self._values

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"Configuration field {name!r} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Configuration field {name!r} is immutable")

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        return _compute_namespace_from_sources, (dict(self._values),)


def custom_config_view(
    args: BaseConfig,
    custom_config: BaseConfig,
    *runtime_sources: ConfigSource,
) -> ImmutableNamespace:
    sources: list[ConfigSource] = [args]
    if isinstance(args, TrainerConfig):
        sources.append(args.backend)
    sources.extend((custom_config, *runtime_sources))
    return _compute_namespace_from_sources(*sources)


def _compute_namespace_from_sources(*sources: ConfigSource) -> ImmutableNamespace:
    values: dict[str, Any] = {}
    for source in sources:
        source_values = dict(source) if isinstance(source, (BaseConfig, Mapping)) else vars(source)
        for name, value in source_values.items():
            if name in values and values[name] != value:
                raise ValueError(
                    f"Custom configuration field {name!r} has conflicting values " f"{values[name]!r} and {value!r}"
                )
            values[name] = value
    return ImmutableNamespace(**values)
