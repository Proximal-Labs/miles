"""Turn Triton's autotune results into flash-linear-attention's config files.

FLA autotunes its kernels with Triton; ``FLA_CACHE_RESULTS`` (on by default) makes Triton
write every tuning to ``{kernel}.autotune.json`` in its cache. Those files are keyed by
exact shape, so a new sequence length still re-tunes. FLA can instead load configs from
``FLA_CONFIG_DIR/{kernel}.json``, and in ``fuzzy`` mode a config matches any key of the
same structure whatever its numbers, i.e. any sequence length. Writing one tuned config
per kernel there, once, removes runtime autotuning (see ``training.KernelCache``).
"""

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The Triton ``Config`` fields FLA's CachedAutotuner reads back (fla/ops/utils/cache.py).
CONFIG_FIELDS = ("kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg", "ir_override")


def fla_key_hash(key: Any) -> str:
    """FLA's ``AutotuneKey.key_hash``: MD5 of the key as compact, sorted JSON (tuples as lists)."""
    return hashlib.md5(json.dumps(key, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _median(timing: Any) -> float:
    """Triton benchmarks each config at quantiles (0.5, 0.2, 0.8); the first is the median."""
    value = timing[0] if isinstance(timing, list) else timing
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else math.inf


def _size(key: list[Any]) -> float:
    return sum(abs(v) for v in key if isinstance(v, (int, float)) and not isinstance(v, bool))


def fla_configs(
    triton_cache: Path, triton_version: str, key_hash: Callable[[Any], str] = fla_key_hash
) -> dict[str, dict[str, Any]]:
    """One FLA config file per tuned kernel, from every ``*.autotune.json`` under ``triton_cache``.

    Each tuned key keeps its fastest config. FLA's fuzzy lookup takes the first entry of
    matching structure, so entries run from the largest tuned shape down: our sequences
    are long, and a config tuned on a long one is the better default.
    """
    tuned: dict[str, list[tuple[list[Any], dict[str, Any]]]] = {}
    for path in sorted(triton_cache.rglob("*.autotune.json")):
        record = json.loads(path.read_text())
        timings = [(config, _median(timing)) for config, timing in record["configs_timings"]]
        best, seconds = min(timings, key=lambda pair: pair[1])
        if not math.isfinite(seconds):
            continue
        kernel = path.name.removesuffix(".autotune.json")
        config = {field: best.get(field) for field in CONFIG_FIELDS}
        tuned.setdefault(kernel, []).append((list(record["key"]), config))
    files = {}
    for kernel, entries in tuned.items():
        entries.sort(key=lambda entry: _size(entry[0]), reverse=True)
        files[kernel] = {
            "kernel_name": kernel,
            "triton_version": triton_version,
            "autotune_entries": {key_hash(key): {"autotune_key": key, "config": config} for key, config in entries},
            "default_config": entries[0][1],
        }
    return files


def write_fla_configs(files: dict[str, dict[str, Any]], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for kernel, content in files.items():
        (directory / f"{kernel}.json").write_text(json.dumps(content, indent=1, sort_keys=True))
