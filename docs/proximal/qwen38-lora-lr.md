# Qwen3.8-27B LoRA RL learning rate

Calibration of 2026-10-02 for production async RL on Qwen3.8-27B with LoRA rank 32 and
alpha 32, at about 1,024 samples per optimizer update. The recipe that applies it is
[`overhead/train_args.txt`](../../examples/proximal/qwen38/overhead/train_args.txt).

## Decision

- **Start at 2e-5** (previously 4e-5), after a **linear warmup over the first 8 optimizer
  updates**, then constant.
- **Ceiling 4e-5.** Fall back to **1e-5** on any drift alarm below.
- **Both LoRA arms use the same LR** (MLP-only and all-linear). Exception: if the all-linear
  arm's measured per-step drift exceeds 1.5× the MLP arm's, run it at about 0.7× the LR.
- **Confidence is moderate.** The optimum could be 2× higher or 2× lower.

## Why our nominal LR runs hot

- **Larger A at init.** Megatron-Bridge initializes LoRA A with "xavier". Measured rms is
  0.0197 for `fc1` and 0.0107 for `fc2`. That is about 2.44× the kaiming
  U(±1/√d_in) used by PEFT, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
  and SkyRL. Since ΔW = (α/r)·B·A, every step of B moves the weights 2.44× further.
- **α/r = 1** (α 32, r 32), the convention of LoRA Without Regret and Tinker.
- **Damped B steps.** The per-token-mean loss together with Adam's eps of 1e-8 shrinks B's
  steps: mean |ΔB|/lr measured 0.81–0.84 at 1,024 samples. A is effectively frozen early,
  because its gradient scales with B, which starts at zero.
- **Net:** our 4e-5 ≈ Tinker's 8e-5. The factor lies between 1.2× and 2.4×.

## Anchors in our units

| Source | Published LR | Setting | ≈ Ours at 1,024 samples/update |
| --- | --- | --- | --- |
| Tinker cookbook RL recipes ([RL hyperparameters](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)) | 1e-5–4e-5 | `rl_numerics_check` on Qwen3.6-27B, multi-turn, 62–80k tokens: 4e-5 | 0.5–2e-5 at equal batch (init factor) |
| Our Tinker hill-climb | 1e-5 | Qwen3-32B r16, 64 samples/step, 18 steps, +8.3 pp paired | ≈ 2e-5 (init ×~½, √batch ×4) |
| SkyRL | 1e-5 | Qwen3.6-27B Megatron LoRA, 4 steps over 2,048 samples | ≈ 0.6e-5, reading it as 512 samples/update (kaiming init ÷2.4, √batch ×1.4) |
| Miles LoRA examples | 1e-5–2e-5 | Bridge init, same as ours | 1e-5–2e-5 before batch scaling |
| Full-FT RL consensus: DeepSWE, DAPO, Skywork, Polaris | 1e-6 | ×10 for LoRA (LoRA Without Regret) | ≈ 4–6e-6 |

## Measurements

- **One update barely moves per-token metrics.** A second update at 4e-5 on the same
  1,024-sample batch changed per-token metrics by about 1e-5 to 1e-6. The drift KL was at
  most about 5e-6, at least 250× below the ~0.0013 floor of the serving-vs-trainer mismatch.
- **Per trajectory it is large.** In-sample, the same update moved about 3–6 nats per
  trajectory per unit of advantage on ~100k-token trajectories.
- **Lag barely shows.** In a 64-sample chain smoke at lag 0/1/2, `train_rollout_kl` was
  0.00135/0.00118/0.00123 and TIS stayed at 1 ± 3e-5.
- **No persistent direction.** The cosine between consecutive steps' `exp_avg` was 0.67,
  what orthogonal gradients predict.

## Consequences

- **TIS cannot bite at a sane LR.** With clip 2 it would take an LR of about 6.5e-4, which
  is the collapse regime.
- **Per-token TIS metrics are a weak, lagging alarm.** Run 013's collapse moved
  `train_rollout_kl` only from 0.0014 to about 0.0026.
- **At lag ≤ 3 the LR is the staleness control that matters.**

## What to watch

Signs the LR is too high (grad norm is not a reliable alarm):

- **Rollout log-prob falls.** Mean on-policy rollout log-prob drops by more than 0.03
  nats/token within 3 steps, or more than ~8% of tokens fall below −3 nats (baseline 6.4%).
- **Lagged batches diverge.** At lag ≥ 1, `train_rollout_kl` exceeds 1.3× the lag-0 value
  from the same window, or `tis_clipfrac` exceeds 1e-4.
- **Updates share a direction.** The cosine between consecutive updates stays above 0.3: a
  persistent bias, like run 013's MTP leak.
- **Behavior shifts.** Turns, length or debug loops rise by more than 20%, or the submit
  rate falls.
- **Entropy moves** by more than about 5% per step. The recipe logs it as
  `train/entropy_loss` (below).

## Cheap held-out learning test

A 4e-5 chain trains on batch A, then B, then C. Its step on B reports the loss at θ1,
which has seen only A, and its step on C reports the loss at θ2. Score B and C at the base
policy θ0 with `--lr 1e-30` measurement steps, which move no weight. Each Δ below is the
chain's value minus the θ0 value on the same batch:

| Result | Reading |
| --- | --- |
| Δloss_B ≤ −5e-6, Δloss_C more negative, and ΔKL ≤ 2e-5 | 4e-5 learns safely; production may start at 3–4e-5 |
| \|Δloss\| < 2e-6 and ΔKL < 1e-6 | Steps are too weak; about 2× is defensible with bias monitoring |
| ΔKL > 1e-4 or Δ`train_rollout_logprob_abs_diff` > 5% | Use ≤ 2e-5 |

## How the recipe applies it

```
--lr 2e-5
--lr-warmup-iters 8
--lr-decay-style constant
--override-opt_param-scheduler
```

- **Warmup is counted in samples.** Miles builds the scheduler in
  [`get_optimizer_param_scheduler`](../../miles/backends/megatron_utils/model.py) with
  `lr_warmup_steps = lr_warmup_iters × global_batch_size`. Each update then advances it by
  that update's sample count (`opt_param_scheduler.step(increment=num_rollouts)`, which is
  `global_batch_size` for a full step). So `--lr-warmup-iters 8` is 8 updates at any batch
  size: 768 samples at the recipe's 96, 8,192 at 1,024. Miles ignores
  `--lr-warmup-samples`.
- **Each update's LR is computed before it runs.** Megatron's `OptimizerParamScheduler`
  starts at `--lr-warmup-init` (default 0), so update k runs at (k−1)/8 of `--lr`:

  | Update | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9+ |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | LR | 0 | 2.5e-6 | 5e-6 | 7.5e-6 | 1e-5 | 1.25e-5 | 1.5e-5 | 1.75e-5 | 2e-5 |

  Update 1 only fills Adam's moments, and about half of its gradient reaches update 2
  through the first moment. These values come from running Megatron's scheduler (current
  `miles-main`) with Miles's arguments at both batch sizes. In W&B, `train/lr-pg_*` is
  logged after each update and shows the LR the next update will use.
  - Setting `--lr-warmup-init 2.5e-6 --lr-warmup-iters 7` would give exactly k/8 with no
    zero-LR update. That ties a second flag to `--lr`, and Megatron refuses
    `init > lr`, which breaks `--lr 1e-30` measurement overrides.
- **Warmup then constant.** With `constant` decay, the LR is `--lr` once warmup ends.
- **Warmup must end within the run.** Megatron asserts `lr_warmup_steps < lr_decay_steps`,
  and the decay length defaults to the run's update count, so this recipe needs
  `--num-rollout` above 8. `trainer_replay`, and through it `step_sizing` and
  `grad_attribution`, run fewer updates and drop `--lr-warmup-iters`.
- **`--override-opt_param-scheduler` makes the arguments win over a checkpoint.** On resume,
  the LR, warmup and decay come from the arguments, so a changed `--lr` (the 1e-5 fallback)
  takes effect. The checkpoint supplies only `num_steps`, the samples trained so far. A
  resume at another global batch size therefore moves the warmup position proportionally.
- **A LoRA resume currently skips warmup.** The LoRA adapter load restores `num_steps`,
  and [`load_model_state`](../../miles/backends/megatron_utils/model.py) then adds
  `iteration × global_batch_size` again, where `iteration` is the last saved rollout. A run
  resumed after 3 updates continues at 5/8 of the LR instead of 3/8. After 5 or more
  updates it resumes at the full LR. The schedule never exceeds `--lr`, but warmup is
  lost. [#52](https://github.com/Proximal-Labs/miles/pull/52)
  (`proximal/lora-scheduler-resume`) fixes this; land this recipe after it or with it.

### Frozen-batch runs

The chain, the sweep and `offline_batch train` measure fixed-LR updates, and they run fewer
updates than the warmup. Remove `--lr-warmup-iters 8` from any recipe built from this file,
including a collection's saved `training-args.json`.

- **The batch chain refuses it** before launch (`check_chain_recipe`, in
  [#49](https://github.com/Proximal-Labs/miles/pull/49)).
- **The sweep and `offline_batch train` don't check for it.** Megatron then fails when it
  builds the scheduler, on the allocated GPUs.

### Batch size

The calibration is per update at about 1,024 samples. The checked-in overhead recipe trains
96 samples per update on one 8-GPU node. Using the √batch conversion behind the anchors
above, 2e-5 at 96 samples corresponds to about 6.5e-5 at 1,024, above the ceiling. The
equivalent of 2e-5 at 1,024 is about 6e-6 at 96. Size the batch toward 1,024, or scale the
LR, before relying on this calibration at a much smaller batch.

### Entropy logging

`--observe-training-entropy` logs entropy without changing the loss. With `--entropy-coef 0`
it is computed under `no_grad`
([`calculate_log_probs_and_entropy`](../../miles/backends/training_utils/loss_hub/math_utils.py)).

- **No extra model pass.** It reuses the response logits that the log-probs use, in the same
  `--log-probs-chunk-size` 4,096-token chunks.
- **It recomputes the softmax.** The log-softmax is not shared, so each chunk gets one more
  elementwise pass: an fp32 copy (4,096 × 62,080 at TP 4, about 1 GiB) and a few temporaries
  of the same size, freed chunk by chunk. Nothing is kept for backward.
- **It runs twice per micro-batch,** because `--recompute-loss-function` reruns the loss in
  backward. That adds 3 small all-reduces per chunk and a one-time `torch.compile` of the
  reduction.
- **Estimated cost:** about 1 TB of HBM traffic per 180k-token sample per GPU, counting the
  recompute. That is roughly 0.15 s, well under 1% of a step, plus about 3 GiB of transient
  memory. These are estimates, not measurements; `trainer_replay` keeps the flag and
  reports both step time and peak memory.

## Not verified

- Tinker's server-side LoRA A init. The factor could be 1.4× rather than 2.4×.
- The exact optimal RL LRs in LoRA Without Regret.
- Some per-paper numbers above, which come from abstracts.
