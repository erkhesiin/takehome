# Forward Projection Harbor Task

This repository contains a Harbor task based on `Forward_Projection_Paper.pdf`, the paper "Closed-form feedback-free learning with forward projection." The agent must create `/solution/forward_projection.py` and implement the core Forward Projection fitting routine in PyTorch.

The task focuses on whether an agent can translate a recent ML paper into numerically stable, deterministic code. It asks for layerwise target membrane construction, closed-form ridge fitting, and replay of the fitted stack on new inputs.

## Layout

```text
.
|-- README.md
|-- SPEC.md
|-- writeup.md
|-- Forward_Projection_Paper.pdf
|-- configs/
|   `-- rollouts/
|       |-- README.md
|       |-- claude-opus-4-7-high.yaml
|       `-- codex-gpt-5-5-xhigh-openrouter.yaml
`-- tasks/
    `-- forward-projection/
        |-- instruction.md
        |-- task.toml
        |-- environment/
        |   |-- Dockerfile
        |   `-- Forward_Projection_Paper.pdf
        |-- solution/
        |   |-- forward_projection.py
        |   `-- solve.sh
        `-- tests/
            |-- test.sh
            |-- run_verifier.py
            |-- test_forward_projection.py
            |-- test_cases.py
            `-- fp_reference.py
```

## Task Summary

The submitted file must define two top-level functions:

```python
def fit_forward_projection(
    X,
    Y,
    Qs,
    Us,
    ridge=1e-3,
    activation="tanh",
    target_nonlinearity="tanh",
):
    ...


def forward_projection_predict(X, weights, activation="tanh"):
    ...
```

`fit_forward_projection` sequentially generates target membrane potentials using fixed random projections and fits each layer by closed-form ridge regression. `forward_projection_predict` replays the learned weights on new input and returns both membrane potentials and activations.

The verifier gives partial credit:

```text
reward = passed_cases / 12
```

## Prerequisites

Install Harbor, make sure Docker is running, and install the agent CLIs you plan to evaluate:

```bash
pip install harbor
harbor --version
docker info
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex
```

## Run The Oracle

Run the checked-in reference solution from the repo root:

```bash
harbor run -p tasks/forward-projection --agent oracle --force-build
```

Expected reward: `1.0`.

Use `--force-build` after changing files in `tasks/forward-projection/environment`. For normal reruns, this is enough:

```bash
harbor run -p tasks/forward-projection --agent oracle
```

## Run The Verifier Locally

This command runs the pytest verifier against the checked-in solution without starting a Harbor job:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --python 3.11 --with torch==2.3.1 --with pytest==8.4.1 python -m pytest -q -p no:cacheprovider tasks/forward-projection/tests/test_forward_projection.py
```

Expected local result: all 12 tests pass.

## Run Model Rollouts

The required rollout configs live under `configs/rollouts`.

| Config | Agent | Model | Reasoning effort | Attempts |
| --- | --- | --- | --- | --- |
| `claude-opus-4-7-high.yaml` | Claude Code | `anthropic/claude-opus-4-7` | `high` | 10 |
| `codex-gpt-5-5-xhigh-openrouter.yaml` | Codex | `gpt-5.5` | `xhigh` | 10 |

Set OpenRouter credentials before running the matrix.

Claude Code:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""
```

Codex:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

Run the jobs:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

Use lower concurrency if provider limits or local resources are tight:

```bash
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --n-concurrent 1 --yes
```

The Codex config pins `@openai/codex@0.118.0` for OpenRouter compatibility. Newer CLI releases can fail against OpenRouter before writing `/solution/forward_projection.py`.

## Inspect Results

Harbor writes job output under `jobs/`. Useful files inside each trial are:

```text
jobs/<job-name>/<trial-id>/result.json
jobs/<job-name>/<trial-id>/trial.log
jobs/<job-name>/<trial-id>/verifier/reward.txt
jobs/<job-name>/<trial-id>/verifier/test-stdout.txt
jobs/<job-name>/<trial-id>/artifacts/solution/forward_projection.py
jobs/<job-name>/<trial-id>/artifacts/manifest.json
jobs/<job-name>/<trial-id>/agent/<agent-name>.txt
jobs/<job-name>/<trial-id>/agent/trajectory.json
```

`verifier/test-stdout.txt` explains pytest failures. The submitted implementation is copied to `artifacts/solution/forward_projection.py` when the agent writes the expected file.

You can also open Harbor's viewer:

```bash
harbor view jobs
```

## Check Task Quality

`harbor check` uses an LLM judge and requires an Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
PYTHONIOENCODING=utf-8 harbor check tasks/forward-projection
```

## Troubleshooting

If Docker reuses an old image after task changes, rerun with `--force-build`.

If a rollout exits before verification, inspect `jobs/<job-name>/<trial-id>/agent/` and `trial.log` for CLI, credential, provider, network, or timeout errors.

If the reward is `0.0`, inspect `verifier/test-stdout.txt` first. Common causes are a missing `/solution/forward_projection.py`, missing top-level functions, returning only weights, using gradient descent instead of the closed-form ridge fit, mixing up the activation and target nonlinearity, or failing to include the original input as `activations[0]`.

A direct LAPACK-backed `torch.linalg.solve` may crash with `Illegal instruction` in some Docker CPU environments. The oracle and reference use a small closed-form PyTorch linear-system solve to avoid that host-specific failure mode.
