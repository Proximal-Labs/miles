"""A run-scoped capability required by remote rollout/publication adapters."""

import os
from dataclasses import dataclass

from miles_plugins.proximal.contracts import RunConfig

_AUTHORITY = object()


@dataclass(frozen=True, init=False)
class AuthorizedRun:
    config: RunConfig

    def __init__(self, config: RunConfig, *, _authority: object):
        if _authority is not _AUTHORITY:
            raise PermissionError("Use authorize_run with explicit rollout and publication consent")
        object.__setattr__(self, "config", config)


def authorize_run(config: RunConfig, *, yes_rollouts: bool, yes_publish: bool) -> AuthorizedRun:
    if yes_rollouts is not True or yes_publish is not True:
        raise PermissionError("Requires explicit rollout and publication consent (--yes-rollouts and --yes-publish)")
    return AuthorizedRun(config, _authority=_AUTHORITY)


def require_authorization(value: AuthorizedRun) -> RunConfig:
    if not isinstance(value, AuthorizedRun):
        raise PermissionError("Expected an authorized training run")
    return value.config


def secret_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required credential environment variable is unset: {name}")
    return value
