"""Triton's autotune results become flash-linear-attention config files (kernel_tune)."""

import hashlib
import json
from pathlib import PurePosixPath

import pytest
from pydantic import ValidationError

from miles_plugins.proximal.kernel_tune import fla_configs, fla_key_hash, write_fla_configs
from miles_plugins.proximal.training import KernelCache


def _tuned(cache, kernel, key, timings, subdir):
    """One Triton ``{kernel}.autotune.json``, as Autotuner.check_disk_cache writes it."""
    directory = cache / subdir
    directory.mkdir(parents=True)
    configs = [
        (
            {
                "kwargs": {"BT": bt},
                "num_warps": warps,
                "num_stages": 2,
                "num_ctas": 1,
                "maxnreg": None,
                "ir_override": None,
                "pre_hook": None,
            },
            timing,
        )
        for (bt, warps), timing in timings
    ]
    (directory / f"{kernel}.autotune.json").write_text(json.dumps({"key": key, "configs_timings": configs}))


def test_each_tuned_key_keeps_its_fastest_config_largest_shape_first(tmp_path):
    cache = tmp_path / "triton"
    _tuned(
        cache,
        "l2norm_bwd_kernel",
        [128, 64, "torch.bfloat16"],
        [((32, 4), [0.9, 0.8, 1.0]), ((64, 8), [0.5, 0.4, 0.6])],
        "a",
    )
    _tuned(
        cache,
        "l2norm_bwd_kernel",
        [128, 4096, "torch.bfloat16"],
        [((32, 4), [0.3, 0.2, 0.4]), ((64, 8), float("inf"))],
        "b",
    )
    _tuned(cache, "causal_conv1d_fwd_kernel", [4, "torch.bfloat16"], [((16, 2), 0.1)], "c")

    files = fla_configs(cache, "3.7.1")

    assert sorted(files) == ["causal_conv1d_fwd_kernel", "l2norm_bwd_kernel"]
    l2norm = files["l2norm_bwd_kernel"]
    entries = list(l2norm["autotune_entries"].values())
    # FLA's fuzzy lookup takes the first entry of matching structure: the largest shape.
    assert [e["autotune_key"] for e in entries] == [[128, 4096, "torch.bfloat16"], [128, 64, "torch.bfloat16"]]
    assert [e["config"]["kwargs"]["BT"] for e in entries] == [32, 64]
    assert l2norm["default_config"] == entries[0]["config"]
    assert "pre_hook" not in entries[0]["config"] and l2norm["triton_version"] == "3.7.1"
    key = [128, 4096, "torch.bfloat16"]
    assert l2norm["autotune_entries"][fla_key_hash(key)]["autotune_key"] == key

    write_fla_configs(files, tmp_path / "fla")
    assert json.loads((tmp_path / "fla" / "l2norm_bwd_kernel.json").read_text()) == l2norm


def test_fla_key_hash_is_md5_of_compact_sorted_json():
    key = [128, 4096, "torch.bfloat16"]
    expected = hashlib.md5(json.dumps(key, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    assert fla_key_hash(key) == expected


def test_kernel_cache_env_and_mount():
    volume = {"volume_name": "k", "environment_name": "main"}
    cache = KernelCache.model_validate_json(
        json.dumps({"volume": volume, "mount": "/kernels", "fla_cache_mode": "fuzzy"})
    )
    assert cache.env() == {
        "TRITON_CACHE_DIR": "/kernels/triton",
        "TILELANG_CACHE_DIR": "/kernels/tilelang",
        "TORCHINDUCTOR_CACHE_DIR": "/kernels/inductor",
        "FLA_CONFIG_DIR": "/kernels/fla-configs",
        "FLA_CACHE_MODE": "fuzzy",
    }
    assert cache.fla_config_dir == PurePosixPath("/kernels/fla-configs")
    with pytest.raises(ValidationError):
        KernelCache.model_validate_json(json.dumps({"volume": volume, "mount": "kernels", "fla_cache_mode": "fuzzy"}))
