# Rollout Configs

These Harbor configs run the required model and reasoning-effort matrix for the Forward Projection task at `tasks/forward-projection`.

| Config | Agent | Model | Reasoning effort | Attempts | Notes |
| --- | --- | --- | --- | --- | --- |
| `claude-opus-4-7-high.yaml` | Claude Code | `anthropic/claude-opus-4-7` | `high` | 10 | Uses the installed Claude Code CLI |
| `codex-gpt-5-5-xhigh-openrouter.yaml` | Codex | `gpt-5.5` | `xhigh` | 10 | Pins `@openai/codex@0.118.0` |

## Credentials

Set OpenRouter credentials in the same shell before running a config. In zsh and bash, keep the variable name, equals sign, and value adjacent, and quote values that may contain punctuation.

Claude Code via OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""
```

Codex via OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

The Codex config pins `@openai/codex@0.118.0`. Newer Codex CLI releases can attempt an OpenRouter `/responses` WebSocket path that returns `404` before the agent writes `/solution/forward_projection.py`.

## Run

From the repo root:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

For slower but easier-to-debug runs:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --n-concurrent 1 --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --n-concurrent 1 --yes
```

Generated jobs are written under `jobs/`. Check each trial's `verifier/reward.txt`, `verifier/test-stdout.txt`, `result.json`, and `artifacts/solution/forward_projection.py`.
