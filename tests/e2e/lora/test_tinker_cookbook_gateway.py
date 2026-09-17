"""Run the official cookbook SFT and RL recipes against a real Tinker gateway."""

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.tinker_gateway import BASE_MODEL, prepare_gateway, running_gateway

import miles.utils.external_utils.command_utils as U

register_cuda_ci(
    est_time=2400,
    suite="stage-c-8-gpu-h200",
    labels=["lora", "weight-update", "multi-lora"],
    hardware=["hopper"],
)

COOKBOOK_PIN = "git+https://github.com/thinking-machines-lab/tinker-cookbook@1f962eda3a2c"
# The cookbook's own deps that the image does not already carry. Everything else it lists (torch, transformers,
# datasets, ...) is preinstalled, and its `transformers<=5.5.4` pin must not be resolved: pip would downgrade the
# image's transformers 5.12.1 to 5.5.4, which breaks `import megatron.bridge` (Exaone 4.5 bridge needs >=5.10)
# for every later test in the same container.
COOKBOOK_DEPS = "chz>=0.4.0 termcolor>=2.0.0 tml-renderers>=0.0.1"


def prepare():
    prepare_gateway()
    U.exec_command_cpu(f"pip install tinker==0.26.2 {COOKBOOK_DEPS}")
    U.exec_command_cpu(f"pip install --no-deps {COOKBOOK_PIN}")


def execute():
    with running_gateway() as base_url:
        U.exec_command_cpu(
            "python examples/multi_lora/run_client_recipes.py "
            f"--base-url {base_url} --base-model {BASE_MODEL} --mode both --steps 2"
        )


if __name__ == "__main__":
    prepare()
    execute()
