# Deep-Thinking Ratio Harbor Task

This repo contains a Harbor RL evaluation task for implementing the
Deep-Thinking Ratio (DTR) algorithm from `DTR_Paper.pdf`. The task focuses on
the exit-depth formulation: compare each layer's logit-lens distribution with
the final layer, find the earliest layer where the token distribution has
stabilized, and count tokens whose exit layer is in the final portion of the
network.

The task lives at `tasks/deep-thinking-ratio`. Agents receive
`instruction.md` and must create `/solution/dtr.py` with an importable
`compute_dtr` function. The verifier runs seven deterministic cases and writes a
graded reward in `[0, 1]`.

## Layout

```text
.
├── README.md
├── SPEC.md
├── writeup.md
├── DTR_Paper.pdf
├── configs/
│   └── rollouts/
│       ├── README.md
│       ├── claude-opus-4-7-high.yaml
│       └── codex-gpt-5-5-xhigh-openrouter.yaml
└── tasks/
    └── deep-thinking-ratio/
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

## Prerequisites

Install Harbor and make sure Docker is running:

```bash
pip install harbor
harbor --version
docker info
```

For real-agent rollouts, install the agent CLI you want to use, for example:

```bash
npm install -g @openai/codex
npm install -g @anthropic-ai/claude-code
```

## Run The Oracle

From the repo root, run the reference solution:

```bash
harbor run -p tasks/deep-thinking-ratio --agent oracle --force-build
```

Expected reward: `1.0`.

Use `--force-build` after changing `environment/Dockerfile`; omit it once the
image is current:

```bash
harbor run -p tasks/deep-thinking-ratio --agent oracle
```

## Run A Real Agent

The take-home PDF asks for these model/reasoning tiers:

| Agent | Model | Reasoning effort | Attempts |
| --- | --- | --- | --- |
| Claude Code | `anthropic/claude-opus-4-7` | `high` | 10 |
| Codex | `gpt-5.5` | `xhigh` | 10 |

Codex with an OpenRouter key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" --agent-env OPENAI_BASE_URL="$OPENAI_BASE_URL"
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:OPENAI_API_KEY = $env:OPENROUTER_API_KEY
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY=$env:OPENAI_API_KEY --agent-env OPENAI_BASE_URL=$env:OPENAI_BASE_URL
```

Claude Code with OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""
harbor run -p tasks/deep-thinking-ratio --agent claude-code --model anthropic/claude-opus-4-7 --agent-kwarg reasoning_effort=high
```

Run repeated attempts with `--n-attempts`. `--n-concurrent` controls how many
trials run at once.

```bash
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" --agent-env OPENAI_BASE_URL="$OPENAI_BASE_URL" --n-attempts 10 --n-concurrent 2
```

## Run Required Rollout Matrix

The same settings are checked into reproducible Harbor job configs:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

PowerShell uses the same Harbor commands after setting the environment variables
shown above. Codex credentials are read from the config file's
`${OPENAI_API_KEY}` and `${OPENAI_BASE_URL}` templates. Claude Code reads
Anthropic/OpenRouter credentials from the Harbor process environment, so export
the `ANTHROPIC_*` variables first or pass them with `--env-file`.

To override concurrency without editing the files:

```bash
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --n-concurrent 1 --yes
```

## Run All Local Tasks

This repo currently has one task, but `tasks/` can be run as a local task
collection:

```bash
harbor run -p tasks --agent oracle --force-build
```

## Inspect Results

Harbor writes job outputs under `jobs/` by default. Useful files are:

```text
jobs/<job-name>/<trial-id>/result.json
jobs/<job-name>/<trial-id>/verifier/reward.txt
jobs/<job-name>/<trial-id>/verifier/pytest.log
jobs/<job-name>/<trial-id>/verifier/test-stdout.txt
jobs/<job-name>/<trial-id>/agent/oracle.txt
```

You can also start Harbor's viewer:

```bash
harbor view jobs
```

## Check Task Quality

`harbor check` uses an LLM judge, so it needs `ANTHROPIC_API_KEY`.

macOS/Linux:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
PYTHONIOENCODING=utf-8 harbor check tasks/deep-thinking-ratio
```

PowerShell:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:PYTHONIOENCODING = "utf-8"
harbor check tasks/deep-thinking-ratio
```

## Codex Credentials

Harbor runs Codex inside the task container with `CODEX_HOME=/logs/agent`, so
your local `~/.codex/config.toml` is not automatically used. Pass credentials
with `--agent-env`.

OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" --agent-env OPENAI_BASE_URL="$OPENAI_BASE_URL"
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:OPENAI_API_KEY = $env:OPENROUTER_API_KEY
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY=$env:OPENAI_API_KEY --agent-env OPENAI_BASE_URL=$env:OPENAI_BASE_URL
```

Direct OpenAI, if you have an OpenAI key instead:

```bash
export OPENAI_API_KEY="sk-..."
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY="$OPENAI_API_KEY"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-kwarg reasoning_effort=xhigh --agent-env OPENAI_API_KEY=$env:OPENAI_API_KEY
```

## Claude Code Credentials

Claude Code with OpenRouter uses OpenRouter's Anthropic skin without `/v1`:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:ANTHROPIC_BASE_URL = "https://openrouter.ai/api"
$env:ANTHROPIC_AUTH_TOKEN = $env:OPENROUTER_API_KEY
$env:ANTHROPIC_API_KEY = ""
```

## Troubleshooting

If Docker reuses an old image after task changes, rerun with `--force-build`.

If a real agent exits with `NonZeroAgentExitCodeError`, inspect its command
logs under `jobs/<job-name>/<trial-id>/agent/`. This usually means the agent
CLI failed before verification, for example from missing API credentials,
model/provider configuration, lack of network access, or a command/runtime
error. The task leaves internet enabled because installed agents such as Codex
need it for setup and model API calls.

If the reward is `0.0`, inspect:

```text
jobs/<job-name>/<trial-id>/verifier/pytest.log
jobs/<job-name>/<trial-id>/verifier/test-stdout.txt
jobs/<job-name>/<trial-id>/agent/oracle.txt
```

If Torch reports `Illegal instruction`, make sure you rebuilt the image after
the conservative CPU-dispatch environment variables were added.
