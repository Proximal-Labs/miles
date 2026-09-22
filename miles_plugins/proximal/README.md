# Experimental Proximal adapter publication

First executable slice of the [platform rollout architecture](../../docs/proximal/architecture.md). It prepares and distributes immutable LoRA artifacts; it is **not yet an end-to-end platform rollout backend**.

CPU requirements: Python 3.10+, `pydantic>=2.10,<3` and `httpx>=0.28,<1`. Remote Volume operations additionally need `modal>=1.5.5`. Importing the package or preparing an export does not import Torch, Ray, SGLang or Modal.

## Prepare an existing adapter

Create `snapshot-config.json`, replacing the revision placeholder with the exact immutable checkpoint commit/digest:

```json
{
  "run_id": "experiment-1",
  "base_model": {
    "name": "your/base-model",
    "revision": "<40-character commit or 64-character digest>"
  },
  "output_root": "./published-adapters"
}
```

Run the local-only command:

```console
python -m miles_plugins.proximal prepare \
  --adapter-directory ./checkpoint/adapter \
  --config snapshot-config.json \
  --checkpoint-iteration 3
```

It prints `{"sha256":"..."}` and writes `published-adapters/snapshots/<sha256>/`. The bundle contains a canonical manifest, `adapter_config.json`, and exactly one unsharded `adapter_model.bin` or `adapter_model.safetensors`. It excludes native checkpoint/optimizer shards. Missing, ambiguous, empty or symlinked serving files fail. Tensor files are copied and hashed, never deserialized here. Repeating the same export is idempotent; changed bytes or metadata create another identity.

The caller must finish checkpoint export before preparing it. This does not certify tensor layout or numerical compatibility with the base model.

## Optional Miles post-save hook

Add these flags to an existing Megatron actor-only LoRA training command:

```console
--custom-megatron-post-save-hook-path miles_plugins.proximal.checkpoint.proximal_snapshot_post_save \
--proximal-snapshot-config snapshot-config.json
```

The normal Miles hook invokes this on rank zero after checkpoint saving completes. It takes the PEFT export from `<checkpoint_dir>/adapter`; a failed optional HF adapter export fails this hook instead of falling back to native shards or a merged full-model directory. Configuration, Megatron backend, LoRA enablement, `--save` and actor-only support are validated before training resources are created.

This hook performs **local export only**. It neither contacts Modal nor advances Miles's serving version. Its `checkpoint_iteration` records the supplied Miles rollout ID, not an assumed optimizer step count. Export at save cadence is scaffolding; initial publication and serve cadence still need integration with the weight-update boundary.

## Publish to an existing shared Volume

After replacing the snapshot ID and confirming the intended destination:

```console
python -m miles_plugins.proximal publish-modal \
  --snapshot-directory ./published-adapters/snapshots/<sha256> \
  --sha256 <sha256> \
  --volume-name training-adapters \
  --environment-name dev \
  --yes-publish
```

The command uses existing Modal credentials. It looks up an existing Volume without creating one, uploads only missing serving files, verifies their remote bytes, then uploads `manifest.json`. Existing conflicting files are never overwritten. Retries after partial upload verify and reuse matching files. Readback verification adds transfer cost; measure before optimizing it.

The output is a `PublishedSnapshot` artifact reference, **not** fleet readiness. The single-writer publication path is idempotent; concurrent identical publishers may fail on file creation and can retry. There is no deletion or retention implementation in this slice.

## Register on each replica

The platform must mount that shared Volume on each replica and start a compatible SGLang engine with LoRA support, sufficient rank/adapter capacity and matching base weights. Put the local cache outside the Volume, separate from base-model/compiler caches.

Replica configuration:

```json
{
  "base_model": {
    "name": "your/base-model",
    "revision": "<same immutable commit or digest>"
  },
  "served_model_name": "your-served-model-name",
  "backend_url": "http://127.0.0.1:30000"
}
```

Run **inside each replica container**, substituting its actual mount/cache locations:

```console
python -m miles_plugins.proximal load-replica \
  --config replica-config.json \
  --sha256 <sha256> \
  --volume-name training-adapters \
  --environment-name dev \
  --volume-mount /adapter-volume \
  --local-cache /tmp/adapter-cache \
  --yes-load
```

An optional SGLang key is read from `SGLANG_API_KEY` (`--api-key-env` changes that variable name). The backend URL must be a loopback root URL, so a load-balanced fleet address cannot be mistaken for one replica. The platform owns authentication and lifecycle of this replica-local control path.

The loader refreshes the mount, checks the complete bundle and declared base identity, copies to local cache, then POSTs `/load_lora_adapter`. It requires SGLang's `success` and exact `loaded_adapters[name]` path acknowledgement, matching the reviewed v0.5.20 response. It returns the immutable adapter name and the `base-model:adapter-name` request selector. It does not run inference or prove the resident base weights match the declaration.

For a long-lived serving wrapper, construct one `ReplicaLoRALoader` with a scoped `AuthorizedReplicaLoad`, the mounted Volume's `reload` callback and a borrowed `httpx.Client` with an explicit timeout. Calls on that loader serialize refresh/copy/load and reuse successful registrations. Recreate the loader whenever SGLang restarts; the wrapper must exclusively manage these names. An async host should call the synchronous helper on its worker thread, not block its event loop.

The one-shot CLI creates a fresh loader. Re-running it against an already-registered adapter may fail; failed/ambiguous loads propagate and never trigger automatic unload/reload. Live registration/reconciliation and inference smoke tests are follow-up work. Old adapters remain available until the platform's future retention policy releases them.

## CPU checks

```console
python -m pip install 'pydantic>=2.10,<3' 'httpx>=0.28,<1' 'modal>=1.5.5' pytest pytest-asyncio mypy
python -m pytest --confcutdir=tests/fast/proximal_publication tests/fast/proximal_publication
python -m mypy --follow-imports=silent --strict miles_plugins/proximal
```

`--confcutdir` isolates these CPU tests from the repository-wide fixtures that import the GPU/Ray stack. Tests exercise actual filesystem hashing/copying, a fresh CLI process, the real Modal adapter against an SDK-boundary fixture, and independent loopback HTTP engine fixtures. They do not launch Modal resources, load real GPU adapters, execute training, or prove TITO/platform rollout compatibility.
