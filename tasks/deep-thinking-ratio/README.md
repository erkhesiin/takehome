# takehome/deep-thinking-ratio

This Harbor task asks an agent to implement the Deep-Thinking Ratio algorithm
from the included DTR paper. The agent prompt is in `instruction.md`; the
expected deliverable is a single importable file at `/solution/dtr.py` defining
`compute_dtr`.

## Environment

The task uses `python:3.11-slim` with `torch==2.3.1` and `pytest==8.4.1`
installed at image build time. Runtime internet is enabled so installed CLI
agents such as Codex can set themselves up and call model APIs. The task is CPU
only with 2 CPUs, 2048 MB RAM, 10240 MB storage, a 600 second agent timeout,
and a 120 second verifier timeout. Torch is configured for conservative
single-threaded CPU dispatch to avoid host-specific illegal-instruction
failures from optimized native kernels. The image pre-creates writable
`/solution` because agents are expected to place `/solution/dtr.py` there.

## Verifier

The verifier is deterministic pytest. It generates synthetic hidden states and
unembedding matrices from fixed Torch seeds, calls the submitted implementation,
and compares it with a reference DTR implementation.

| Dimension | Type | What it measures |
| --- | --- | --- |
| Accuracy | Programmatic | Fraction of seven DTR cases matching the reference within 0.01 |

Reward is written to `/logs/verifier/reward.txt` as `passed_cases / 7`.

## Layout

```text
tasks/deep-thinking-ratio/
├── DTR_Paper.pdf
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   └── DTR_Paper.pdf
├── solution/
│   ├── dtr.py
│   └── solve.sh
└── tests/
    ├── test.sh
    ├── run_verifier.py
    ├── test_dtr.py
    ├── test_cases.py
    └── dtr_reference.py
```

## Running

From the repo root, run the oracle/reference solution:

```bash
harbor run -p tasks/deep-thinking-ratio --agent oracle --force-build
```

Expected reward: `1.0`.

Run Codex with an OpenRouter key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" --agent-env OPENAI_BASE_URL="$OPENAI_BASE_URL"
```

Run repeated attempts:

```bash
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" --agent-env OPENAI_BASE_URL="$OPENAI_BASE_URL" --n-attempts 10 --n-concurrent 2
```

The required take-home rollout matrix is also available as Harbor job configs:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

Those configs run 10 attempts each with Claude Code/Opus 4.7 at `high`
reasoning effort and Codex/GPT-5.5 at `xhigh` reasoning effort.

Results are written under `jobs/<job-name>/<trial-id>/`. The most useful files
are `verifier/reward.txt`, `verifier/pytest.log`, `verifier/test-stdout.txt`,
and `agent/oracle.txt`.
