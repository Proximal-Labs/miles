import argparse
from collections.abc import Callable
from copy import deepcopy
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import ConfigDict

from miles.utils.pydantic_utils import StrictBaseModel


class A:
    def __init__(
        self,
        *flags: str,
        reset: bool = False,
        default_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not flags:
            raise ValueError("Argument metadata requires at least one argument name")
        if reset and len(flags) != 1:
            raise ValueError("Reset arguments require exactly one argument name")
        if default_factory is not None and "default" in kwargs:
            raise ValueError("Specify either default or default_factory")

        self._flags = flags
        self._reset = reset
        self._default_factory = default_factory
        self._kwargs = kwargs

    def _add_argument(self, *, parser: argparse.ArgumentParser) -> None:
        kwargs = self._kwargs.copy()
        if self._default_factory is not None:
            kwargs["default"] = self._default_factory()
        elif "default" in kwargs:
            kwargs["default"] = deepcopy(kwargs["default"])

        if self._reset:
            reset_arg(parser=parser, name=self._flags[0], **kwargs)
        else:
            parser.add_argument(*self._flags, **kwargs)


class BaseConfig(StrictBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        for name, annotation in get_type_hints(cls, include_extras=True).items():
            if get_origin(annotation) is not Annotated:
                continue
            arguments = [metadata for metadata in get_args(annotation)[1:] if isinstance(metadata, A)]
            if len(arguments) > 1:
                raise ValueError(f"Multiple argument declarations for {cls.__name__}.{name}")
            if arguments:
                cls._before_argument(parser=parser, name=name)
                arguments[0]._add_argument(parser=parser)

    @classmethod
    def _before_argument(cls, *, parser: argparse.ArgumentParser, name: str) -> None:
        pass


def reset_arg(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)
