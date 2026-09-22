"""Replica-local snapshot verification and SGLang adapter registration.

One loader is scoped to one SGLang process lifetime. Construct a new loader if
that process restarts. This helper does not provision replicas, evict adapters,
perform inference, or declare a policy globally ready.
"""

import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles_plugins.proximal.snapshot import (
    BaseModelIdentity,
    Nonempty,
    SnapshotReference,
    read_snapshot,
    snapshot_relative_path,
)

_AUTHORITY = object()


class ReplicaConfig(FrozenStrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    base_model: BaseModelIdentity
    served_model_name: Nonempty
    backend_url: str

    @field_validator("backend_url")
    @classmethod
    def _local_engine(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.scheme not in ("http", "https")
            or url.hostname not in ("127.0.0.1", "::1", "localhost")
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("Use a replica-local SGLang root URL, not a fleet/proxy endpoint")
        return value.rstrip("/")


@dataclass(frozen=True, init=False)
class AuthorizedReplicaLoad:
    config: ReplicaConfig

    def __init__(self, config: ReplicaConfig, *, _authority: object):
        if _authority is not _AUTHORITY:
            raise PermissionError("Use authorize_replica_load with explicit load consent")
        object.__setattr__(self, "config", config)


def authorize_replica_load(config: ReplicaConfig, *, yes_load: bool) -> AuthorizedReplicaLoad:
    if yes_load is not True:
        raise PermissionError("Loading requires --yes-load")
    return AuthorizedReplicaLoad(config, _authority=_AUTHORITY)


class RegisteredAdapter(FrozenStrictBaseModel):
    snapshot: SnapshotReference
    adapter_name: str
    request_model: str


class _LoadReply(BaseModel):
    # Narrow SGLang's response at I/O; its other response fields stay at this boundary.
    model_config = ConfigDict(extra="ignore", strict=True)
    success: bool
    loaded_adapters: dict[str, str]


class ReplicaLoRALoader:
    def __init__(
        self,
        authorization: AuthorizedReplicaLoad,
        *,
        volume_mount: Path,
        local_cache: Path,
        reload_volume: Callable[[], None],
        client: httpx.Client,
    ):
        if not isinstance(authorization, AuthorizedReplicaLoad):
            raise PermissionError("Expected an authorized replica load")
        self._config = authorization.config
        self._mount = volume_mount.resolve()
        self._cache = local_cache.resolve()
        if (
            self._cache == self._mount
            or self._cache.is_relative_to(self._mount)
            or self._mount.is_relative_to(self._cache)
        ):
            raise ValueError("Replica cache and Volume mount must be separate directory trees")
        self._reload_volume = reload_volume
        self._client = client  # Borrowed: the replica owns its HTTP client's lifetime.
        self._lock = threading.Lock()
        self._loaded: dict[str, RegisteredAdapter] = {}

    def ensure_loaded(self, reference: SnapshotReference) -> RegisteredAdapter:
        """Serialize refresh/copy/register within this replica; never change an existing adapter name."""
        with self._lock:
            if reference.sha256 in self._loaded:
                return self._loaded[reference.sha256]
            snapshot_directory = self._prepare_cache(reference)
            adapter_name = f"miles-{reference.sha256}"
            response = self._client.post(
                f"{self._config.backend_url}/load_lora_adapter",
                json={"lora_name": adapter_name, "lora_path": str(snapshot_directory)},
                follow_redirects=False,
            )
            response.raise_for_status()
            reply = _LoadReply.model_validate_json(response.content)
            if not reply.success or reply.loaded_adapters.get(adapter_name) != str(snapshot_directory):
                raise ValueError("SGLang did not acknowledge the requested adapter name and path")
            registered = RegisteredAdapter(
                snapshot=reference,
                adapter_name=adapter_name,
                request_model=f"{self._config.served_model_name}:{adapter_name}",
            )
            self._loaded[reference.sha256] = registered
            return registered

    def _prepare_cache(self, reference: SnapshotReference) -> Path:
        destination = self._cache / snapshot_relative_path(reference)
        if not destination.exists():
            self._reload_volume()  # No Volume file is held open across this call.
            source = read_snapshot(self._mount / snapshot_relative_path(reference), reference)
            if source.manifest.metadata.base_model != self._config.base_model:
                raise ValueError("Snapshot base model does not match this replica")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Copy into a temporary sibling so a failed copy is never a cache hit.
            with tempfile.TemporaryDirectory(prefix=".load-", dir=destination.parent) as temporary:
                staged = Path(temporary) / "bundle"
                shutil.copytree(source.directory, staged)
                read_snapshot(staged, reference)
                staged.rename(destination)
        cached = read_snapshot(destination, reference)
        if cached.manifest.metadata.base_model != self._config.base_model:
            raise ValueError("Cached snapshot base model does not match this replica")
        return destination

    def unload_idle(self, reference: SnapshotReference) -> None:
        """Caller must hold its request-admission lock and prove no active users.

        Authority is the same replica-scoped capability used to register adapters;
        this does not delete a Volume artifact or any platform resource.
        """
        with self._lock:
            loaded = self._loaded.get(reference.sha256)
            if loaded is None:
                return
            response = self._client.post(
                f"{self._config.backend_url}/unload_lora_adapter",
                json={"lora_name": loaded.adapter_name},
                follow_redirects=False,
            )
            response.raise_for_status()
            reply = _LoadReply.model_validate_json(response.content)
            if not reply.success or loaded.adapter_name in reply.loaded_adapters:
                raise ValueError("SGLang did not acknowledge adapter eviction")
            del self._loaded[reference.sha256]
