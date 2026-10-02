"""Reuse FA4 outputs during bf16 full activation recompute.

Enable with --custom-megatron-init-path
miles_plugins.proximal.attention_recompute.install.
The original FA4 autograd function still owns backward and the recomputed Q/K/V.
"""

from argparse import Namespace
from collections import deque
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import Any

import torch

_ACTIVE: ContextVar[tuple[deque[tuple[Any, tuple[Any, ...]]], bool] | None] = ContextVar(
    "attention_checkpoint", default=None
)
_INSTALLED = False


def _metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    def signature(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return (value.shape, value.dtype, value.device, value.stride())
        return value

    return tuple(map(signature, args)), {k: signature(v) for k, v in kwargs.items()}


def _cached_forward(original: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    active = _ACTIVE.get()
    if active is None:
        return original(*args, **kwargs)
    cache, replay = active
    q = args[0]
    if q.ndim != 3 or q.dtype != torch.bfloat16 or not kwargs.get("causal"):
        raise ValueError("Attention cache requires packed causal bf16 attention")
    unsupported = (
        "qv",
        "page_table",
        "learnable_sink",
        "score_mod",
        "mask_mod",
        "block_sparse_tensors",
        "aux_tensors",
        "aux_scalars",
        "gather_kv_indices",
        "out",
        "lse",
    )
    if any(kwargs.get(name) is not None for name in unsupported):
        raise ValueError("Unsupported FA4 option in attention cache")
    metadata = _metadata(args, kwargs)
    if replay:
        if not cache:
            raise RuntimeError("Attention recompute has more calls than the original forward")
        expected, result = cache.popleft()
        if metadata != expected:
            raise RuntimeError("Attention arguments changed during recompute")
        return result
    # Q/K/V made under no_grad do not request LSE, but backward will need it.
    result = original(*args, **{**kwargs, "return_lse": True})
    saved = tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in result)
    cache.append((metadata, saved))
    return result


def _checkpoint_forward(
    original: Callable[..., Any],
    ctx: Any,
    function: Callable[..., Any],
    distribute_saved_activations: bool,
    *args: Any,
) -> Any:
    cache: deque[tuple[Any, tuple[Any, ...]]] = deque()

    @wraps(function)
    def wrapped(*inputs: Any) -> Any:
        replay = torch.is_grad_enabled()
        token = _ACTIVE.set((cache, replay))
        try:
            result = function(*inputs)
            if replay and cache:
                raise RuntimeError("Attention recompute has fewer calls than the original forward")
            return result
        except BaseException:
            cache.clear()
            raise
        finally:
            _ACTIVE.reset(token)

    return original(ctx, wrapped, distribute_saved_activations, *args)


def enable() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Install once in this process; return originals for the standalone benchmark."""
    # Only available inside the pinned GPU training image.
    from flash_attn.cute import interface  # type: ignore[import-not-found]
    from megatron.core.tensor_parallel.random import CheckpointFunction  # type: ignore[import-not-found]

    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("Attention cache is already installed")
    original_fwd = interface._flash_attn_fwd
    original_checkpoint = CheckpointFunction.forward
    interface._flash_attn_fwd = wraps(original_fwd)(lambda *a, **kw: _cached_forward(original_fwd, *a, **kw))
    CheckpointFunction.forward = staticmethod(
        wraps(original_checkpoint)(lambda *a, **kw: _checkpoint_forward(original_checkpoint, *a, **kw))
    )
    _INSTALLED = True
    return original_fwd, original_checkpoint


def disable(originals: tuple[Callable[..., Any], Callable[..., Any]]) -> None:
    from flash_attn.cute import interface
    from megatron.core.tensor_parallel.random import CheckpointFunction

    global _INSTALLED
    interface._flash_attn_fwd, checkpoint = originals
    CheckpointFunction.forward = staticmethod(checkpoint)
    _INSTALLED = False


def install(args: Namespace) -> None:
    """Install at actor initialization without a policy-mutating train-step hook."""
    if args.recompute_granularity != "full" or args.attention_dropout != 0:
        raise ValueError("Attention cache requires full recompute and zero attention dropout")
    if not args.bf16 or args.context_parallel_size != 1 or args.fp8 or getattr(args, "fp4", None):
        raise ValueError("Attention cache requires CP=1 and bf16")
    if getattr(args, "cuda_graph_impl", "none") != "none":
        raise ValueError("Attention cache does not support CUDA graphs")
    enable()
