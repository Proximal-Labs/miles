# Terminal Universe training

This recipe trains Qwen3.6 on a Harbor task release whose environments have
already been built on E2B. Each JSONL row carries a task ID and the exact E2B
template ID. The Harbor adapter passes that ID to the E2B backend, so training
creates sandboxes from the published image instead of rebuilding it.

The default topology is one eight-GPU trainer node and two eight-GPU rollout
nodes. It uses fully asynchronous GRPO, Qwen3.6 TITO v2, R3 routing replay, and
`retract` pause generation.

Task concurrency is independent of the training batch size. By default, Miles
keeps up to 64 episodes active across the whole rollout fleet (eight prompts
times eight samples). Pass `--async-max-concurrent-samples 128` to raise that
limit to 128 while retaining 64 episodes per training batch. This allows more
tasks to make progress while others wait for shell commands or tests; provision
enough sandbox capacity for the higher number of simultaneous episodes.

Pass `--pause-generation-mode in_place` to retain in-flight requests during
weight updates. This also enables incremental R3 payloads in the session server,
reducing repeated prefix storage. It preserves cached inference state across
updates and is not equivalent to recomputing that state with the new weights.
The default remains `retract`; neither mode guarantees immediate reclamation
of completed session trees.

Terminus 2 summarization and linear-history recording are explicitly enabled
in the Ray runtime environment. With the default budgets, its configured input
budget is 49,152 tokens and each response is limited to 16,384 tokens. The
65,536-token Miles sample cap is not itself a live-context compaction switch.
Compacted histories must be retained as separate TITO v2 samples so later
actions remain available to the trainer. The run manifest records these harness
settings; check actual trial configurations and collected samples at startup.

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
