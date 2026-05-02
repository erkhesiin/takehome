# Deep-Thinking Ratio Harbor Task

This repo contains a Harbor RL evaluation task for implementing the
Deep-Thinking Ratio (DTR) algorithm from `DTR_Paper.pdf`.

The task lives at `tasks/deep-thinking-ratio`. Agents receive
`instruction.md` and must create `/solution/dtr.py` with an importable
`compute_dtr` function. The verifier runs seven deterministic cases and writes
a graded reward in `[0, 1]`.

## Layout

```text
.
├── README.md
├── SPEC.md
├── DTR_Paper.pdf
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

Codex example:

```bash
export OPENAI_API_KEY="sk-..."
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-env OPENAI_API_KEY="$OPENAI_API_KEY"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-env OPENAI_API_KEY=$env:OPENAI_API_KEY
```

Claude Code example:

```bash
harbor run -p tasks/deep-thinking-ratio --agent claude-code --model anthropic/claude-opus-4-7
```

Run repeated attempts with `--n-attempts`. `--n-concurrent` controls how many
trials run at once.

```bash
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" --n-attempts 10 --n-concurrent 2
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

Direct OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-env OPENAI_API_KEY="$OPENAI_API_KEY"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-env OPENAI_API_KEY=$env:OPENAI_API_KEY
```

OpenRouter through Codex's OpenAI-compatible path:

```bash
export OPENAI_API_KEY="sk-or-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" --agent-env OPENAI_BASE_URL="$OPENAI_BASE_URL"
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-or-..."
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
harbor run -p tasks/deep-thinking-ratio --agent codex --model gpt-5.5 --agent-env OPENAI_API_KEY=$env:OPENAI_API_KEY --agent-env OPENAI_BASE_URL=$env:OPENAI_BASE_URL
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
