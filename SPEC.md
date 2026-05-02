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
├── configs/
│   └── rollouts/
│       ├── claude-opus-4-7-high.yaml
│       └── codex-gpt-5-5-xhigh-openrouter.yaml
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
            └── dtr_reference.py     ← verifier reference implementation
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
    depth_fraction: float = 0.25,
) -> float:
    ...
```

The function takes transformer hidden states from L layers, a logit-lens
unembedding matrix, a JSD threshold, and a `depth_fraction`. It returns a float
in [0.0, 1.0] representing the fraction of tokens whose predicted distribution
only stabilizes near the end of the network.

### Algorithm (ground truth)

The DTR algorithm has five steps:

**Step 1 — Logit lens projection**
For each layer `l`, project hidden states through the unembedding matrix and
apply softmax:
```
logits_l = hidden_states[l] @ unembedding_matrix.T   # (seq_len, vocab_size)
probs_l  = softmax(logits_l, dim=-1)
```

The final layer distribution is the target distribution for each token:
```
target_t = probs_L[t]
```

**Step 2 — Jensen-Shannon Divergence against the final layer**
For each layer `l` and token position `t`, compute JSD between the final-layer
distribution and the current layer distribution:
```
M = 0.5 * (P + Q)
JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```
Use natural log. JSD is bounded in [0, ln(2)] ≈ [0, 0.693].
Add epsilon (1e-10) inside logs for numerical stability.

This produces a matrix of shape `(L, seq_len)` — one JSD value per layer per
token. The final layer has JSD 0 against itself.

**Step 3 — Exit layer identification**
For each token, compute the earliest 1-based layer index whose cumulative
minimum JSD is less than or equal to `threshold`:
```
cummin_jsd[l, t] = min(jsd_matrix[0:l+1, t])
c_t = earliest l where cummin_jsd[l, t] <= threshold
```
If the threshold is never met, default the exit layer to `L`. Because the final
layer is compared with itself, well-formed inputs normally meet the threshold by
the final layer.

**Step 4 — Deep-thinking token classification**
A token is deep-thinking if its exit layer occurs in the final
`depth_fraction` portion of the network:
```
depth_start = ceil((1 - depth_fraction) * L)
is_deep_thinking_t = c_t >= depth_start
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

1. Creates `/logs/verifier/reward.txt` with a default reward of `0`
2. Runs `python /tests/run_verifier.py`
3. Calls `pytest` on `/tests/test_dtr.py`
4. Counts passed test calls out of the seven deterministic cases
5. Writes `reward = PASSED / 7` to `/logs/verifier/reward.txt`

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
| 1 | `all_deep_thinking` | 0 | 8 | 10 | 32 | 64 | 0.01 | Final distribution reached only at the final layer → DTR ≈ 1.0 |
| 2 | `no_deep_thinking` | 1 | 8 | 10 | 32 | 64 | 0.01 | Final distribution already matched from the first layer → DTR ≈ 0.0 |
| 3 | `half_deep_thinking` | 2 | 8 | 10 | 32 | 64 | 0.01 | Token-level selectivity → DTR ≈ 0.5 |
| 4 | `late_exit_boundary` | 3 | 12 | 20 | 64 | 128 | 0.01 | Exit layer in the deep portion for half the tokens |
| 5 | `minimal_edge_case` | 4 | 2 | 1 | 16 | 32 | 0.01 | Two-layer, single-token boundary |
| 6 | `high_threshold` | 5 | 8 | 15 | 32 | 64 | 100.0 | Very high threshold causes early exit → DTR = 0.0 |
| 7 | `realistic_partial` | 6 | 16 | 30 | 128 | 256 | 0.01 | Larger dims, partial deep-token set → DTR ≈ 0.333 |

### Common failure modes each case is designed to catch

- **Case 1** — final-layer target logic and late exit classification
- **Case 2** — early exit when all layers already match the final distribution
- **Case 3** — per-token selectivity; must not average across tokens before classifying
- **Case 4** — correct deep-region threshold with 12 layers
- **Case 5** — minimal layer count and 1-token sequence
- **Case 6** — high threshold must cause early exit rather than marking tokens deep
- **Case 7** — correct handling of larger hidden/vocab dims and partial deep-token sets

---

## Environment Spec

### Docker image

Base: `python:3.11-slim`
Packages: `torch==2.3.1` (CPU wheel), `pytest==8.4.1`
CPU dispatch is pinned to conservative single-threaded settings to avoid
architecture-specific illegal-instruction failures in optimized Torch kernels.
No GPU. Internet is enabled so installed CLI agents such as Codex can install
their runtime and reach model APIs; the task instructions still forbid loading
or downloading real model weights as part of the solution.
The image pre-creates `/solution` as writable so non-oracle agents can create
`/solution/dtr.py`.

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
| Internet | enabled |
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

Harbor runs Codex inside the task container with `CODEX_HOME=/logs/agent`, so
local `~/.codex/config.toml` is not automatically used. Pass OpenRouter through
Codex's OpenAI-compatible path:

```bash
export OPENAI_API_KEY="sk-or-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

---

### Running rollouts

With both CLIs configured, run Harbor against each agent. Rollout outputs are
written to `jobs/` automatically. The checked-in job configs implement the
required model/reasoning tiers from `Hillclimb_Take-Home.pdf`:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

Equivalent explicit commands:

```bash
# 10 rollouts — Claude Code, Opus 4.7, high reasoning
harbor run \
  -p tasks/deep-thinking-ratio \
  --agent claude-code \
  --model anthropic/claude-opus-4-7 \
  --n-attempts 10 \
  --n-concurrent 2 \
  --agent-kwarg reasoning_effort=high

# 10 rollouts — Codex, GPT-5.5, xhigh reasoning
harbor run \
  -p tasks/deep-thinking-ratio \
  --agent codex \
  --model gpt-5.5 \
  --n-attempts 10 \
  --n-concurrent 2 \
  --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" \
  --agent-env OPENAI_BASE_URL="$OPENAI_BASE_URL" \
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

**Tolerance of 0.01** — the verifier compares final DTR ratios, not raw JSD
values. A ratio tolerance of 0.01 is strict enough to catch wrong layer indexing,
wrong JSD direction, incorrect cumulative-min logic, and wrong deep-region
boundaries while still allowing minor floating-point differences.

**Self-contained verifier reference** — Harbor mounts `tests/` and `solution/` as
separate directories. Keeping the reference implementation self-contained in
`tests/dtr_reference.py` means the verifier never depends on what the agent wrote
in `/solution/`.
