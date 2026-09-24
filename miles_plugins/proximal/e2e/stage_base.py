"""Stage the pinned base model into the existing base-weights Volume. PAID: a small CPU run.

Downloads ``base_model.name`` at ``base_model.revision`` from Hugging Face into
``<base_mount>/<basename of tokenizer_path>`` on the serving config's base Volume,
exactly where serving_app.py's replicas look for it. The Volume must already exist;
this neither creates nor deletes it.

    PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=serving.json \\
        modal run --env main -m miles_plugins.proximal.e2e.stage_base
"""

import modal

from miles_plugins.proximal.serving import engine_model_path
from miles_plugins.proximal.serving_app import DEPLOYMENT, RUN, base_volume, image

app = modal.App(f"{DEPLOYMENT.app_name}-stage-base")
# The serving image: importing this module in the container needs the plugin, its
# dependencies and the deploy-time configs, which only that image carries.


@app.function(image=image, volumes={str(DEPLOYMENT.base_mount): base_volume}, timeout=3600)
def stage(repo_id: str, revision: str, target: str) -> list[str]:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, local_dir=target)
    base_volume.commit()
    return sorted(path.name for path in Path(target).iterdir())


@app.local_entrypoint()
def main() -> None:
    files = stage.remote(RUN.base_model.name, RUN.base_model.revision, engine_model_path(RUN, DEPLOYMENT))
    print(f"Staged {RUN.base_model.name}@{RUN.base_model.revision}: {files}")
