# Instrumented multi-agent session

> **Read the docs:** [Agentic rollout](../../../docs/user-guide/agentic-rollout.md).

`agent.py` drafts an answer, runs two independent reviewer contexts, and asks the
parent to revise it. Use it with an existing training recipe and these options:

```bash
--use-session-server v2 \
--custom-agent-function-path examples.experimental.session_multi_agent.agent.run
```

Keep the recipe's reward function: this example returns the final answer as
metadata and does not invent a reward. All three agents contribute to one episode.
The task group joins all registered work before the result permits finalization.
The example requires a TITO family that supports appending user messages.

This is an instrumented Python harness. Opaque CLI harnesses must expose their
own context identities and child-completion boundary to provide the same contract.
