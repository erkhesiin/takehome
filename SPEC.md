# SPEC.md — Deep-Thinking Ratio (DTR) Harbor Task

## Overview

This is a [Harbor](https://harborframework.com) RL evaluation task based on the paper:

> **"Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"**
> University of Virginia & Google, February 2026. arXiv:2602.13517

A coding agent is asked to implement the **Deep-Thinking Ratio (DTR)** computation
algorithm from scratch in a sandboxed Docker container. A verifier scores the
implementation against 7 deterministic test cases, producing a graded reward in [0, 1].

---

## Repo Structure

```
.
├── README.md
├── SPEC.md                          ← this file
├── writeup.md                       ← design decisions and rollout analysis
└── tasks/
    └── deep-thinking-ratio/
        ├── instruction.md           ← agent-facing task description
        ├── task.toml                ← Harbor config (version, timeouts, resources)
        ├── environment/
        │   └── Dockerfile           ← python:3.11-slim + torch 2.3.1 CPU
        ├── solution/
        │   ├── dtr.py               ← reference implementation
        │   └── solve.sh             ← oracle solve script
        └── tests/
            ├── test.sh              ← Harbor verifier entrypoint
            ├── test_dtr.py          ← pytest file (one function per test case)
            ├── test_cases.py        ← deterministic test case generator
            └── dtr_reference.py     ← copy of reference solution for verifier use
```

---

## Task Spec

### What the agent is asked to do

Implement a single Python function `compute_dtr` in `/solution/dtr.py`:

```python
def compute_dtr(
    hidden_states: list[torch.Tensor],   # L tensors of shape (seq_len, hidden_dim)
    unembedding_matrix: torch.Tensor,    # shape (vocab_size, hidden_dim)
    threshold: float = 0.01,
) -> float:
    ...
```

The function takes transformer hidden states from L layers and returns a float in
[0.0, 1.0] representing the fraction of tokens that are "deep-thinking."

### Algorithm (ground truth)

The DTR algorithm has five steps:

**Step 1 — Logit lens projection**
For each layer `i`, project its hidden states through the unembedding matrix and apply softmax:
```
logits_i = hidden_states[i] @ unembedding_matrix.T   # (seq_len, vocab_size)
probs_i  = softmax(logits_i, dim=-1)
```

**Step 2 — Jensen-Shannon Divergence between consecutive layers**
For each pair of adjacent layers `(i, i+1)` and each token position `t`:
```
M = 0.5 * (P + Q)
JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```
Use natural log. JSD is bounded in [0, ln(2)] ≈ [0, 0.693].
Add epsilon (1e-10) inside logs for numerical stability.

This produces a matrix of shape `(L-1, seq_len)` — one JSD value per layer
transition per token.

**Step 3 — Late regime identification**
Define the late regime as the last 25% of layer transitions, with a minimum of 1:
```
n_transitions = L - 1
n_late = max(1, floor(n_transitions * 0.25))
late_jsd = jsd_matrix[-n_late:]   # shape (n_late, seq_len)
```

**Step 4 — Deep-thinking token classification**
A token at position `t` is a deep-thinking token if its maximum JSD across the
late regime exceeds the threshold:
```
max_late_jsd = late_jsd.max(dim=0)        # shape (seq_len,)
is_deep_thinking = max_late_jsd > threshold
```

**Step 5 — Ratio**
```
DTR = count(is_deep_thinking) / seq_len
```

### Constraints on the agent's implementation

- PyTorch and Python standard library only — no scipy, no sklearn, no other ML libraries
- Must run on CPU (no CUDA)
- Must not import or load any real language model or weights
- Must be a single file at `/solution/dtr.py`
- Must be importable — `from dtr import compute_dtr` must work

---

## Verifier Spec

### Entrypoint

`tests/test.sh` is the Harbor verifier script. It:

1. Installs `uv` and runs `pytest` via `uvx`
2. Calls `tests/test_dtr.py`
3. Counts `PASSED` vs total `test_case_*` functions
4. Writes `reward = PASSED / TOTAL` to `/logs/verifier/reward.txt`

### Reward

```
reward = cases_passed / 7       # float in [0.0, 1.0]
```

A case passes if:
```
abs(agent_dtr - reference_dtr) <= 0.01
```

The tolerance of 0.01 is strict relative to the [0, 1] range but robust to minor
floating-point differences across torch versions.

### Test cases

All cases use `torch.manual_seed(N)` for full reproducibility. Expected values are
computed at runtime from `dtr_reference.py` (the reference implementation), not
hardcoded, so they remain correct if torch's RNG output ever changes.

| # | Name | Seed | L | seq_len | hidden_dim | vocab_size | threshold | What it tests |
|---|------|------|---|---------|------------|------------|-----------|---------------|
| 1 | `all_deep_thinking` | 0 | 8 | 10 | 32 | 64 | 0.01 | Late-layer divergence → DTR ≈ 1.0 |
| 2 | `no_deep_thinking` | 1 | 8 | 10 | 32 | 64 | 0.01 | Near-zero JSD everywhere → DTR ≈ 0.0 |
| 3 | `half_deep_thinking` | 2 | 8 | 10 | 32 | 64 | 0.01 | Token-level selectivity → DTR ≈ 0.5 |
| 4 | `multilayer_late_divergence` | 3 | 12 | 20 | 64 | 128 | 0.01 | 25% boundary with 12 layers |
| 5 | `minimal_edge_case` | 4 | 2 | 1 | 16 | 32 | 0.01 | `max(1, floor(...))` floor, single token |
| 6 | `high_threshold` | 5 | 8 | 15 | 32 | 64 | 100.0 | Threshold above max JSD → DTR = 0.0 |
| 7 | `realistic_partial` | 6 | 16 | 30 | 128 | 256 | 0.01 | Larger dims, partial divergence → DTR ≈ 0.333 |

### Common failure modes each case is designed to catch

- **Case 1** — basic late-layer logic, handles large magnitude divergence
- **Case 2** — numerical stability; tiny JSD must not exceed threshold due to floating-point noise
- **Case 3** — per-token selectivity; must not average across tokens before classifying
- **Case 4** — correct `floor(n_transitions * 0.25)` with 12 layers (11 transitions → n_late=2)
- **Case 5** — the `max(1, ...)` floor; also tests 1-token sequences
- **Case 6** — threshold comparison must be strict `>`, not `>=`; impossibly high threshold must yield 0.0 without crashing
- **Case 7** — correct handling of larger hidden/vocab dims and multi-layer late regime

---

## Environment Spec

### Docker image

Base: `python:3.11-slim`
Packages: `torch==2.3.1` (CPU wheel), `pytest==8.4.1`
CPU dispatch is pinned to conservative single-threaded settings to avoid
architecture-specific illegal-instruction failures in optimized Torch kernels.
No GPU, no internet access at runtime.

### Directory layout inside the container at verifier time

```
/solution/
    dtr.py              ← agent's implementation (written during agent turn)
/tests/
    test_dtr.py
    test_cases.py
    dtr_reference.py
/logs/verifier/
    reward.txt          ← written by test.sh
    pytest.log          ← full pytest stdout
```

### Resource limits (from task.toml)

| Resource | Limit |
|---|---|
| CPUs | 2 |
| Memory | 2048 MB |
| Storage | 10240 MB |
| GPUs | 0 |
| Internet | disabled |
| Agent timeout | 600s |
| Verifier timeout | 120s |

---

## Running Locally

### Prerequisites

```bash
pip install harbor
npm install -g @anthropic-ai/claude-code   # Claude Code CLI
npm install -g @openai/codex               # Codex CLI
docker --version                           # Docker must be running
```

### Sanity check with oracle solution

```bash
harbor run -p tasks/deep-thinking-ratio --agent oracle
```

Expected: `reward = 1.0`

---

### OpenRouter setup — Claude Code

Claude Code speaks the native Anthropic protocol. OpenRouter exposes this via its
"Anthropic Skin" at `https://openrouter.ai/api` (**no `/v1` suffix** — using `/v1`
causes model-not-found errors).

```bash
export OPENROUTER_API_KEY="sk-or-..."
export ANTHROPIC_BASE_URL="https://openrouter.ai/api"   # no /v1
export ANTHROPIC_AUTH_TOKEN="$OPENROUTER_API_KEY"
export ANTHROPIC_API_KEY=""                              # must be blank
```

Verify the connection by running `/status` inside a Claude Code session — it should
show `ANTHROPIC_AUTH_TOKEN` and `https://openrouter.ai/api`.

### OpenRouter setup — Codex CLI

Codex is OpenAI-compatible and uses the standard `/v1` endpoint (opposite of Claude Code).
Add an OpenRouter provider profile to `~/.codex/config.toml`:

```toml
model = "openai/gpt-5.5"
model_provider = "openrouter"

[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"   # /v1 required here
env_key = "OPENROUTER_API_KEY"
wire_api = "chat"
```

Then set your key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

---

### Running rollouts

With both CLIs configured, run Harbor against each agent. Rollout outputs are written
to `jobs/` automatically.

```bash
# 10 rollouts — Claude Code, Opus 4.7, high reasoning
harbor run \
  -p tasks/deep-thinking-ratio \
  --agent claude-code \
  --model anthropic/claude-opus-4-7 \
  --n 10 \
  --agent-kwarg reasoning_effort=high

# 10 rollouts — Codex, GPT-5.5, xhigh reasoning
harbor run \
  -p tasks/deep-thinking-ratio \
  --agent codex \
  --model openai/gpt-5.5 \
  --n 10 \
  --agent-kwarg reasoning_effort=xhigh
```

> **Note on base URLs:** Claude Code uses `https://openrouter.ai/api` (no `/v1`);
> Codex uses `https://openrouter.ai/api/v1` (with `/v1`). They use different protocol
> skins — mixing them up is the most common setup mistake.

---

## Key Design Decisions

**Synthetic inputs, no model weights** — the verifier generates all hidden states
from random seeds at runtime. This keeps the Docker image small (~1GB for torch),
eliminates download dependencies, and makes every test case fully deterministic.

**Graded reward over 7 cases** — a binary pass/fail would give no gradient signal
for partially correct implementations. Seven distinct cases each targeting a different
failure mode gives the RL loop more information about where the agent went wrong.

**Reference computed at runtime, not hardcoded** — `test_dtr.py` computes expected
DTR values by calling `dtr_reference.py` at test time rather than storing floats in
the test file. This means the expected values are always consistent with the reference
implementation regardless of minor torch version differences.

**Tolerance of 0.01** — JSD is bounded by ln(2) ≈ 0.693. A DTR tolerance of 0.01
is strict enough to catch most implementation bugs (wrong layer indexing, wrong KL
direction, missing the floor) while being robust to floating-point rounding across
platforms.

**`dtr_reference.py` is a copy of `solution/dtr.py`** — Harbor mounts `tests/` and
`solution/` as separate directories. Keeping the reference self-contained in `tests/`
means the verifier never depends on what the agent wrote in `/solution/`.
