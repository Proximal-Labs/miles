from typing import Any, ClassVar

from pydantic import ConfigDict, Field, SerializeAsAny, field_validator

from miles.utils.args.schema import BaseConfig
from miles.utils.function_registry import load_function
from miles.utils.pydantic_utils import StrictBaseModel


class LegacyCustomArgsConfig(BaseConfig):
    model_config = ConfigDict(extra="allow")


class BaseLeafConfig(StrictBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    _mutable_fields: ClassVar[frozenset[str]] = frozenset()
    custom_function_configs: dict[str, dict[str, SerializeAsAny[BaseConfig]]] = Field(default_factory=dict)
    legacy_custom_configs: dict[str, LegacyCustomArgsConfig] = Field(default_factory=dict)

    @field_validator("custom_function_configs", mode="before")
    @classmethod
    def _validate_custom_function_configs(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        configs: dict[str, dict[str, BaseConfig]] = {}
        for owner, values_by_path in values.items():
            configs[owner] = {}
            for path, config_values in values_by_path.items():
                function = load_function(path)
                config_class = getattr(
                    function, "config_class", None
                )  # config-access-exempt: custom hook protocol discovery
                if config_class is None:
                    config_class = LegacyCustomArgsConfig
                if not isinstance(config_class, type) or not issubclass(config_class, BaseConfig):
                    raise TypeError(f"{path}.config_class must inherit BaseConfig")
                configs[owner][path] = (
                    config_values
                    if isinstance(config_values, config_class)
                    else config_class.model_validate(config_values)
                )
        return configs

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls._mutable_fields = frozenset(name for base in cls.__mro__ for name in vars(base).get("_mutable_fields", ()))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).model_fields and name not in self._mutable_fields:
            raise TypeError(f"{type(self).__name__}.{name} is immutable")
        super().__setattr__(name, value)
