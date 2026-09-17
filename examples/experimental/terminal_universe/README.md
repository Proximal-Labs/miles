# Terminal Universe training

This recipe trains Qwen3.6 on a Harbor task release whose environments have
already been built on E2B. Each JSONL row carries a task ID and the exact E2B
template ID. The Harbor adapter passes that ID to the E2B backend, so training
creates sandboxes from the published image instead of rebuilding it.

The default topology is one eight-GPU trainer node and two eight-GPU rollout
nodes. It uses fully asynchronous GRPO, Qwen3.6 TITO v2, R3 routing replay, and
`retract` pause generation.

The rollout coordinator is pinned to the Ray head node. Start the Ray head on
the node that can reach E2B's control plane and sandbox endpoints; connectivity
from another allocation member is not sufficient. Verify sandbox creation,
command execution, and cleanup from the head before submitting training.

Terminus 2's model client runs in the rollout process, outside the sandbox.
The recipe therefore preserves TITO's internal session URLs instead of
rewriting them to an external address. An external address rewrite is only
appropriate for agents that call the model from inside a remote sandbox.

The task archive must be unpacked with its executable modes preserved. Set
`E2B_API_KEY`, `WANDB_API_KEY`, and `MILES_ROUTER_EXTERNAL_HOST` in the launch
environment. The launcher writes a redacted reproducibility manifest beside
the checkpoints and traces.
