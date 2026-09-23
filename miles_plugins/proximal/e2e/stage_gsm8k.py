"""Stage the gsm8k training split into the base-weights Volume. PAID: a small CPU run.

Downloads ``zhuzilin/gsm8k`` (the dataset Miles's gsm8k LoRA recipes use) at a pinned
revision into ``<base_mount>/gsm8k/train.parquet``, where the training app reads it.
Stage the base model separately with ``stage_base``. The Volume must already exist.

    PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=serving.json \\
        modal run --env main -m miles_plugins.proximal.e2e.stage_gsm8k --revision <commit>
"""

import modal

from miles_plugins.proximal.serving_app import DEPLOYMENT, base_volume

app = modal.App(f"{DEPLOYMENT.app_name}-stage-gsm8k")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.35.3")


@app.function(image=image, volumes={str(DEPLOYMENT.base_mount): base_volume}, timeout=1800)
def stage(revision: str, target: str) -> list[str]:
    from pathlib import Path

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="zhuzilin/gsm8k",
        repo_type="dataset",
        revision=revision,
        allow_patterns=["train.parquet"],
        local_dir=target,
    )
    base_volume.commit()
    return sorted(path.name for path in Path(target).iterdir())


@app.local_entrypoint()
def main(revision: str) -> None:
    files = stage.remote(revision, f"{DEPLOYMENT.base_mount}/gsm8k")
    print(f"Staged zhuzilin/gsm8k@{revision}: {files}")
