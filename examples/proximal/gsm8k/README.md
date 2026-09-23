# gsm8k on the real topology: one training node, two replicas, a stand-in platform

Qwen3-0.6B learns gsm8k with Miles's LoRA GRPO recipe
(`examples/lora/run-qwen2.5-3B-megatron-lora-disaggregated.sh`: rank 32, all dense linear
modules, lr 1e-5, 32 prompts × 8 samples), on the same topology as platform training:

| Part | Here |
| --- | --- |
| Training node | Modal 1× H100 (`training_app`): Megatron LoRA trainer, capture service, rollout store, gsm8k platform |
| Serving | Modal 2× L4 (`serving_app`), loading each published version from the adapter Volume |
| Platform | `math_platform`: the platform's run API; each run is one gsm8k problem, one Chat Completions call through capture, graded with Miles's `math` reward |

Swapping `platform.url` for the real platform (and the dataset for real tasks) is the platform run.
Thinking is off: the recipe's 1,024-token answers would cut off Qwen3's thinking.

Both apps use `radixark/miles@sha256:e4fcc8c4…` (2026-09-23). The earlier `796a29b2…` cannot
import Transformer Engine (built against a different torch), so Megatron fails there.

## Checked without paid resources

- Offline, through capture, the fake pool and the fake trainer: gsm8k tasks → groups of 8 →
  one captured call each → graded → trained. Grader: `\boxed{72}` vs `72` true, `71` false.
- In the image: Megatron and Transformer Engine import; Miles's parser accepts the full
  training command; SGLang's `ServerArgs` accepts the replica's engine arguments.
- Crash recovery (`snapshots`): snapshot a step, lose all local state, restore, resume:
  continues at the next step and never retrains a consumed group.

Not checked until it runs on GPUs: Megatron LoRA training and export, SGLang serving a real
LoRA with the exact token IDs and logprobs capture requires, and LoRA resume in Miles.

## Paid steps, in order (workspace `proximal`, environment `main`)

Use `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` for the **proximal** workspace. Run from the CPU
test image so SGLang and this fork are importable.

1. **Volumes and secrets (one time).**
   ```bash
   modal volume create miles-gsm8k-base --env main
   modal volume create miles-gsm8k-adapters --env main
   modal volume create miles-gsm8k-state --env main
   modal secret create miles-gsm8k-gateway MILES_GATEWAY_KEY=$(openssl rand -hex 32) --env main
   modal secret create miles-gsm8k-proxy MODAL_PROXY_KEY=wk-... MODAL_PROXY_SECRET=ws-... --env main
   modal secret create miles-gsm8k-wandb WANDB_API_KEY=... --env main
   ```
2. **Stage the base model and the data** (small CPU jobs).
   ```bash
   PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=examples/proximal/gsm8k/serving.json \
     modal run --env main -m miles_plugins.proximal.e2e.stage_base
   PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=examples/proximal/gsm8k/serving.json \
     modal run --env main -m miles_plugins.proximal.e2e.stage_gsm8k --revision 0cbd9f31d91ac21a7613dcbc7fef992adac459ae
   ```
3. **Deploy the two replicas** and note the printed URL.
   ```bash
   PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=examples/proximal/gsm8k/serving.json \
     modal deploy --env main -m miles_plugins.proximal.serving_app
   ```
4. **Write the run config** with the pool URL (tasks are pinned to the dataset file's hash).
   ```bash
   python -m miles_plugins.proximal.e2e.math_platform prepare --data train.parquet \
     --template examples/proximal/gsm8k/run.template.json --inference-url <pool URL> --out run.json
   ```
5. **Real-SGLang check** before the trainer: the Stage A launcher with the fake trainer
   against the pool (`--publish modal`), a couple of steps.
6. **Start training** (runs until stopped; restarts from the latest snapshot on a crash).
   ```bash
   PROXIMAL_RUN_CONFIG=run.json PROXIMAL_SERVING_CONFIG=examples/proximal/gsm8k/serving.json \
     modal run --detach --env main -m miles_plugins.proximal.e2e.training_app
   ```
7. **Tear down**: `modal app stop miles-gsm8k-training --env main` and
   `modal app stop miles-gsm8k-serving --env main`.
