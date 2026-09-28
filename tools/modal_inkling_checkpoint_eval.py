"""Standalone checkpoint evaluations; preparation never allocates GPUs.

Prepare: MODAL_PROFILE=proximal modal run --env main tools/modal_inkling_checkpoint_eval.py --plan-file run-configs/inkling-checkpoint-eval-010-8rollouts.json
Launch:  MODAL_PROFILE=proximal modal run --detach --env main tools/modal_inkling_checkpoint_eval.py --plan-file run-configs/inkling-checkpoint-eval-010-8rollouts.json --no-prepare-only
"""

import json
from pathlib import Path

import modal

from tools.modal_inkling_sft import image, volume

app = modal.App('inkling-checkpoint-evaluation')


@app.function(image=image, volumes={'/mnt/inkling': volume}, cpu=8, memory=32768,
              timeout=1800, retries=0)
def prepare(plan_json: str):
    volume.reload()
    from miles_plugins.inkling_eval.standalone import prepare as prepare_checkpoints

    return prepare_checkpoints(json.loads(plan_json))


@app.function(image=image, volumes={'/mnt/inkling': volume}, cpu=2, memory=8192,
              timeout=86400, retries=0, nonpreemptible=True,
              secrets=[modal.Secret.from_name('inkling-eval'), modal.Secret.from_name('rft_wandb_api_key')])
def evaluate(plan_json: str):
    volume.reload()
    from miles_plugins.inkling_eval.standalone import evaluate as evaluate_checkpoints

    evaluate_checkpoints(json.loads(plan_json))


@app.local_entrypoint()
def main(plan_file: str, prepare_only: bool = True):
    plan = json.loads(Path(plan_file).read_text())
    encoded = json.dumps(plan)
    if prepare_only:
        print(prepare.remote(encoded))
    else:
        call = evaluate.spawn(encoded)
        print(f'Submitted checkpoint evaluation: {call.object_id}', flush=True)
