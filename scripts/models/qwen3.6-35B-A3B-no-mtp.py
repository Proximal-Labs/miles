"""Qwen3.6 target-only preset for policy training without auxiliary MTP."""

import shlex

from miles.utils.external_utils.model_args_utils import load_sibling_model_args


def model_args() -> str:
    tokens = shlex.split(load_sibling_model_args(__file__, "qwen3.6-35B-A3B"))
    index = tokens.index("--mtp-num-layers")
    del tokens[index : index + 2]
    assert "--mtp-num-layers" not in tokens
    return " ".join(tokens)
