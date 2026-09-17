# Terminal Universe training

This recipe trains Qwen3.6 on a Harbor task release whose environments have
already been built on E2B. Each JSONL row carries a task ID and the exact E2B
template ID. The Harbor adapter passes that ID to the E2B backend, so training
creates sandboxes from the published image instead of rebuilding it.

The default topology is one eight-GPU trainer node and two eight-GPU rollout
nodes. It uses fully asynchronous GRPO, Qwen3.6 TITO v2, R3 routing replay, and
`retract` pause generation.

The task archive must be unpacked with its executable modes preserved. Set
`E2B_API_KEY`, `WANDB_API_KEY`, and `MILES_ROUTER_EXTERNAL_HOST` in the launch
environment. The launcher writes a redacted reproducibility manifest beside
the checkpoints and traces.
