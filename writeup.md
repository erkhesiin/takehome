# DTR Harbor Task Writeup

## Paper And Capability

The task is based on "Think Deep, Not Just Long: Measuring LLM Reasoning Effort
via Deep-Thinking Tokens." The paper proposes Deep-Thinking Ratio (DTR), a
measure that identifies token positions whose prediction distributions only
stabilize near the final layers. The evaluated capability is whether a coding
agent can translate a paper algorithm into a correct, testable PyTorch
implementation.

## Agent Task

The agent receives `tasks/deep-thinking-ratio/instruction.md` and the included
paper PDF. It must create `/solution/dtr.py` with an importable
`compute_dtr(hidden_states, unembedding_matrix, threshold=0.01,
depth_fraction=0.25) -> float` function. The implementation must use only
PyTorch and the Python standard library, run on CPU, avoid loading real model
weights, and return the fraction of sequence positions whose exit layer falls
in the final `depth_fraction` portion of the network.

## Environment

The Docker image is `python:3.11-slim` with `torch==2.3.1` and `pytest==8.4.1`.
The task is CPU-only, pre-creates writable `/solution`, and enables internet so
installed agent CLIs can install and call model APIs. Torch is configured with
conservative single-threaded CPU dispatch to avoid host-specific illegal
instruction failures.

## Verifier And Reward

The verifier runs deterministic pytest cases against the submitted
`compute_dtr`. It compares the output with an independent reference
implementation over seven synthetic test cases. The reward is partial credit:

```text
reward = passed_cases / 7
```

This gives agents useful signal for partially correct implementations while
keeping the verifier fully deterministic.

## Model And Reasoning Tiers

The rollout matrix follows `Hillclimb_Take-Home.pdf`:

| Agent | Model | Reasoning effort | Rollouts |
| --- | --- | --- | --- |
| Claude Code | `anthropic/claude-opus-4-7` | `high` | 10 |
| Codex | `gpt-5.5` | `xhigh` | 10 |

These settings are implemented in:

```text
configs/rollouts/claude-opus-4-7-high.yaml
configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml
```

## Design Decisions And Tradeoffs

Synthetic hidden states avoid downloading model weights and keep rollouts
reproducible. Runtime reference computation avoids brittle hardcoded expected
floats. A seven-case graded reward catches common implementation mistakes such
as comparing adjacent layers instead of comparing each layer to the final layer,
averaging across tokens before classification, missing the cumulative-min exit
rule, or using the wrong deep-region boundary.

The verifier uses programmatic checks rather than an LLM judge because the
expected behavior is mathematical and deterministic. This makes the reward
cheaper, faster, and easier to reproduce.

## Sample Rollouts

Run the required jobs with:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

Attach the resulting `jobs/dtr-claude-opus-4-7-high/` and
`jobs/dtr-codex-gpt-5-5-xhigh/` directories as the sample rollout artifacts.
Those directories contain agent output, verifier output, rewards, and available
trajectory files.

## Possible Improvements

The task could be extended with additional hidden cases around invalid inputs,
varying thresholds near the boundary, and larger tensor shapes. A future version
could also add performance-oriented tests, but the current task intentionally
prioritizes correctness and clarity over optimization.
