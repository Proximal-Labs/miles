# Single-turn Sokoban RL

Miles generates a solution using its standard SGLang rollout path, including
routing replay for MoE models. The reward hook posts only the final move sequence to the
actual NeMo Gym Reasoning Gym resource server. That server replays the moves on
the initial board and returns a binary solved reward. Network failures raise an
error instead of becoming unsuccessful puzzle attempts.

Use a separate Python 3.13 environment for the CPU verifier. Install
`requirements-sokoban-server.txt` with `uv pip install`. Check out NeMo Gym at the
revision pinned in that file and put its root on the verifier's `PYTHONPATH`.
Serve `sokoban_server:create_app` using Uvicorn's `--factory` option, with this
example directory as `--app-dir`, listening on port 8210. The factory installs
the upstream resource server's HTTP routes without starting a second Ray
cluster. This is sufficient for the stateless Sokoban verifier.

Each Miles JSONL row should contain:

- `prompt`: a user message containing the original Reasoning Gym question;
- `label`: the reference move sequence (only sent to the verifier);
- `metadata`: the original board metadata, plus `sokoban_question` containing
  the original question and `source_dataset: "sokoban"`.

Keep reference solutions out of the prompt. The current reward hook expects a
reasoning model that generates `</think>` before its final answer; the opening
`<think>` may already be in the prompt. Require exactly one
`<answer>...</answer>` block after that boundary. Answers may contain uppercase
`U`, `D`, `L`, `R` and whitespace, so both `<answer>UDLR</answer>` and
`<answer>U D L R</answer>` are accepted. An optional trailing `<|im_end|>` is
supported. Boxed answers, raw text, nested or multiple answer blocks, and
answers without a reasoning boundary are rejected. A different reasoning
format needs an explicit adapter change; do not fall back to scanning all text.

Preserve the complete generation, including reasoning, for training and traces.
Only the validated final moves are sent to NeMo Gym, wrapped in a fresh answer
block. Sending reasoning as `output_text` lets upstream answer extraction consume
tags mentioned in the reasoning; the permissive Sokoban scorer can then replay
movement letters from that prose.

The adapter assigns three rewards:

- **1.0**: a valid final answer solves the board;
- **0.0**: a valid final answer does not solve the board;
- **-0.5**: no unambiguous, correctly formatted final answer is available.

Missing or malformed final answers receive the penalty without contacting the
verifier. This includes reasoning that reaches the token limit before producing
a final answer. A truncated response with a complete valid final answer is still
verified normally. The full response remains available for training.
`sokoban_grading_status` records the rejection reason, and
`sokoban_grader_version` identifies this policy as `final-answer-v2-format-penalty`.
Mean raw reward includes format penalties and is not the solve rate; compute
solve rate as the fraction of samples whose reward is 1.0.
Malformed task metadata, unexpected sample states, HTTP failures, invalid
verifier responses, and disagreement between the submitted and extracted moves
raise errors instead of becoming penalized training examples. Valid replies
must have matching numeric binary `score` and `reward` values.

In the Miles worker environment, put this example directory on `PYTHONPATH`
and set `NEMO_GYM_SOKOBAN_URL=http://127.0.0.1:8210`. Add
`--custom-rm-path sokoban_reward.reward_func` to a compatible model recipe, and
override its prompt data, checkpoint, batch size, and response budget as needed.
The localhost URL assumes the verifier and rollout worker share a node.

Before training, send known solutions and broken paths through the HTTP endpoint
and the reward hook, using completed samples with an explicit reasoning boundary
and final answer. Expect rewards 1 and 0, respectively, and -0.5 for missing or
malformed final answers in both completed and truncated samples. Include a correct final
answer following an unclosed answer tag in reasoning, and an incorrect final
answer following a correct reasoning-only plan. Run the offline regression suite
with `pytest tests/fast/examples/experimental/nemo_gym/test_sokoban_reward.py`.
During training,
inspect solve-rate variation, nonzero gradients, routing-replay diagnostics, and
sample traces. Training reward on the training puzzles is not a held-out score.

For policy-only Nemotron-H training, leave `MILES_NEMOTRONH_KEEP_MTP` unset or
set it to an empty string (the string `0` is truthy). A constructed MTP head can add a next-token loss independently
of task rewards. For a fresh HF run, add
`--custom-megatron-before-train-step-hook-path sokoban_training_checks.before_train_step`
to fail before an optimizer update if the actual model contains an MTP head
or the initialization flags allow resumed training state.

## Two-node asynchronous Nemotron 3.5 Lightning

The `run_nemotron35_sokoban.py` launcher assigns eight GPUs to training and eight
to rollout. Join the two nodes to one Ray cluster first, using the same source,
model, and dataset paths on both nodes; set `MILES_SCRIPT_EXTERNAL_RAY=1` before
invoking the launcher. Install `requirements-sokoban-launcher.txt` in the training
environment using `uv pip install`. Supply the `ScriptArgs` fields in a JSON file
and pass its absolute path with `--config`.

The default batch contains eight puzzles and sixteen samples per puzzle, giving
128 responses per optimizer update, with checkpoints every 100 updates.
The async train loop overlaps generation of the next batch with training of the
current batch. It finishes that generation before publishing updated weights.
The serving pause mode is `abort`; this recipe deliberately does not enable
`--fully-async`, whose continuously running producer rejects `abort`.
Recorded rollout log probabilities anchor the policy ratio to the behavior policy.

Set `verifier_url` to a cluster-reachable address if the verifier and rollout
executor run on different nodes. W&B credentials must already be installed in
the credential store on both nodes; they are omitted from the submitted argv.
This recipe preserves the existing final-answer-v2-format-penalty grader.
