# Rollout Configs

These Harbor job configs implement the reasoning-effort matrix requested in
`Hillclimb_Take-Home.pdf`.

| Config | Agent | Model | Reasoning effort | Attempts |
| --- | --- | --- | --- | --- |
| `claude-opus-4-7-high.yaml` | Claude Code | `anthropic/claude-opus-4-7` | `high` | 10 |
| `codex-gpt-5-5-xhigh-openrouter.yaml` | Codex | `gpt-5.5` | `xhigh` | 10 |

## OpenRouter Environment

Claude Code uses OpenRouter's Anthropic-compatible endpoint. Set these in the
shell that runs Harbor:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""
```

Codex uses OpenRouter's OpenAI-compatible endpoint:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
$env:ANTHROPIC_BASE_URL = "https://openrouter.ai/api"
$env:ANTHROPIC_AUTH_TOKEN = $env:OPENROUTER_API_KEY
$env:ANTHROPIC_API_KEY = ""
$env:OPENAI_API_KEY = $env:OPENROUTER_API_KEY
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
```

## Run

From the repo root:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

Results are written under `jobs/`. Each job produces 10 attempts, including
agent logs, verifier logs, rewards, and trajectories when the adapter emits
them.
