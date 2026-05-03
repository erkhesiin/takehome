# Write Up

Kevin Tamir (amarin)

## Paper

This task is based on "Closed-form feedback-free learning with forward projection," a recent Forward Projection (FP) paper that proposes fitting neural layers without backpropagation feedback. I chose it because the task sits in a useful middle ground: the core equations are small enough for a verifier to judge deterministically, but recent and unusual enough that a strong model cannot simply lean on a familiar implementation template.

The task focuses on the paper's target-potential and closed-form fitting idea. For each layer, the agent must construct a target membrane potential from the previous activation and label projections:

```text
Z_tilde_l = g(A_prev @ Q_l) + g(Y @ U_l)
```

It then fits the layer with ridge regression:

```text
(A_prev.T @ A_prev + ridge * I) @ W_l = A_prev.T @ Z_tilde_l
```

After fitting a layer, the implementation computes `Z_l = A_prev @ W_l`, applies the requested activation, and feeds that activation into the next layer.

## Capability

The intended capability is paper-to-code translation for a recent ML algorithm. The task tests whether an agent can extract the algorithmic core from a paper, turn it into numerically stable PyTorch, and respect implementation constraints that are easy to miss under time pressure: no gradient descent, no autograd fitting path, dtype/device preservation, no input mutation, and consistent replay semantics for prediction.

This is not a benchmark of large-scale biomedical reproduction. It is a focused implementation task: can the agent correctly implement the FP equations and API contract in a CPU-only environment with deterministic tests?

## Agent Task

The agent must create exactly one importable module at:

```text
/solution/forward_projection.py
```

That module must define two top-level functions:

```python
def fit_forward_projection(X, Y, Qs, Us, ridge=1e-3, activation="tanh", target_nonlinearity="tanh"):
    ...


def forward_projection_predict(X, weights, activation="tanh"):
    ...
```

`fit_forward_projection` returns a dictionary with exactly `weights`, `membranes`, and `activations`. The `activations` list must start with the original input `X`. `forward_projection_predict` replays a fitted weight stack and returns `(membranes, activations)` with the same activation-list convention.

## Environment

The Docker image is based on `python:3.11-slim` and installs only:

```text
torch==2.3.1
pytest==8.4.1
```

The task is CPU-only with 2 CPUs, 2048 MB memory, no GPUs, and a 120 second verifier timeout. The image creates writable `/solution` and `/logs/verifier` directories and copies the source paper to `/paper/Forward_Projection_Paper.pdf`.

The Dockerfile also sets conservative CPU threading and dispatch environment variables. This matters because direct LAPACK-backed `torch.linalg.solve` can hit `Illegal instruction` in some CPU-only environments, so the instructions recommend a small PyTorch Gaussian-elimination style linear solver.

## Verifier And Reward

The verifier runs `tests/test.sh`, which calls `python /tests/run_verifier.py`. The verifier imports `/solution/forward_projection.py`, executes deterministic pytest cases, and writes the numeric reward to:

```text
/logs/verifier/reward.txt
```

Reward is partial credit:

```text
reward = passed_cases / 12
```

The 12 deterministic cases cover the required fitting and prediction behavior across multiple layer counts, nonlinearities, dtypes, rank-deficient inputs, high-dimensional systems, large-magnitude saturation cases, soft labels, dtype/device preservation, no input mutation, and the replay requirement that activations include the input as `activations[0]`.

The tests also statically check that the submitted module uses only `torch` plus the Python standard library and does not import or call optimizer/autograd fitting APIs. That keeps the verifier aligned with the instruction file's closed-form, no-gradient-descent constraint.

## Design Decisions And Tradeoffs

I originally explored easier or more culturally cached papers, including "Think Deep, Not Just Long" and Mamba-style implementation work. Those produced saturated or near-saturated rollout scores from frontier models, which made them poor fits for a task meant to expose capability gaps rather than reward boilerplate implementation.

Forward Projection was a better target because it is recent, algorithmically specific, and cuts against the backprop-first default that models often assume. The task focuses on the core FP equations rather than full dataset reproduction because full biomedical experiments would add noise from data loading, preprocessing, and training infrastructure. The chosen slice keeps the verifier crisp: either the target construction, ridge solve, sequential layer fitting, and replay semantics are correct, or they are not.

The main tradeoff is that the task is narrower than the full paper. That is intentional. A deterministic, numerically focused task gives more interpretable failures than a broad reproduction attempt where failures could come from unrelated environment or dataset issues.

## Pull Request Checklist

This mirrors the Terminal-Bench PR template checklist.

- [x] All behavior checked in `tests/` is described in `instruction.md`.
- [x] All behavior described in `instruction.md` is checked in `tests/`.
- [x] The `tests/` pytest functions have informative docstrings describing which behavior they check.
- [x] `instruction.md` was written by a human.
- [x] `solution/` was written by a human with minimal language-model assistance.
- [x] The task was run with strong models through Harbor rollout configs for Claude Opus 4.7 high and Codex GPT-5.5 xhigh.
- [x] The task is hard for the agent to cheat because the verifier imports only the submitted `/solution/forward_projection.py`, checks exact numerical behavior against an independent reference, uses multiple deterministic edge cases, and statically rejects external ML dependencies plus optimizer/autograd fitting APIs.
- [x] Failing runs are analyzed below, and the failures confirm the task is valid rather than underspecified.

The behavior coverage is intentionally two-way. `instruction.md` specifies the exact module path, top-level functions, FP target equation, closed-form ridge equation, sequential activation flow, return keys, activation-list convention, supported nonlinearities, dtype/device preservation, no mutation, CPU compatibility, and no optimizer/autograd fitting path. The tests exercise those requirements through reference comparisons, exact return-shape checks, source-contract checks, dtype/device assertions, mutation checks, and parameterized cases covering all four listed nonlinearities.

## Agent Run Analysis

I ran 10 rollouts each for Claude Code Opus 4.7 high and Codex GPT-5.5 xhigh on the Forward Projection task. The results show that the task is neither trivial nor broken: both strong agents can solve it, but they also produce structured failures that match real implementation pitfalls.

| Agent | Model / effort | Trials | Mean reward | Full-credit trials | Notable issue | Wall-clock |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Claude Code | `anthropic/claude-opus-4-7`, high | 10 | 0.800 | 4/10 | 6 trials missed prediction replay semantics | 11m 02s |
| Codex | `gpt-5.5`, xhigh | 10 | 0.933 | 9/10 | 1 trial failed most fitting cases; 1 full-credit artifact had an agent-timeout flag | 35m 58s |

Claude's failures were highly consistent. Six trials scored `0.6667`: they passed all eight fitting cases and failed all four prediction replay cases. The verifier failure was `predict activations has the wrong length`; those implementations returned only post-layer activations instead of `[A_0, A_1, ..., A_L]`. This reflects a real API-composition limitation, not a task flaw, because `instruction.md` explicitly says prediction activations must start with the input `X`, the checked-in oracle follows that convention, and four Claude rollouts implemented it correctly.

| Claude trial | Reward | Total duration | Main result |
| --- | ---: | ---: | --- |
| `5yGY6gL` | 1.000 | 1m 59s | full pass |
| `H3YGako` | 0.667 | 1m 45s | failed all prediction replay cases |
| `Pihgae9` | 0.667 | 2m 01s | failed all prediction replay cases |
| `Vt3AZ9E` | 1.000 | 1m 36s | full pass |
| `Xsv9WDb` | 0.667 | 1m 48s | failed all prediction replay cases |
| `aMKQsMU` | 0.667 | 2m 07s | failed all prediction replay cases |
| `gZY3fa3` | 0.667 | 1m 44s | failed all prediction replay cases |
| `k4sPMXA` | 1.000 | 3m 25s | full pass |
| `kEpprJC` | 1.000 | 2m 00s | full pass |
| `zmCkFsQ` | 0.667 | 2m 08s | failed all prediction replay cases |

Codex usually solved the full task but took longer per rollout. Nine of ten trials received reward `1.0`. The low-scoring trial, `Sj3QVfq`, scored `0.3333`: it passed the prediction tests but failed all eight fitting tests with large numerical mismatches in weights and membranes. That failure points to an incorrect closed-form fitting implementation or unstable ridge solve, which is exactly the numerical detail the task is meant to test. Another trial, `KqLdbDL`, produced a full-credit artifact but Harbor recorded an `AgentTimeoutError`; I kept it visible as an operational rollout issue rather than hiding it.

| Codex trial | Reward | Total duration | Main result |
| --- | ---: | ---: | --- |
| `6fH84MW` | 1.000 | 4m 40s | full pass |
| `HJ7sTej` | 1.000 | 5m 04s | full pass |
| `K6K2pnx` | 1.000 | 9m 09s | full pass |
| `KqLdbDL` | 1.000 | 10m 29s | full pass artifact; agent timeout flag |
| `Sj3QVfq` | 0.333 | 4m 24s | failed all fitting cases |
| `h6Amnmm` | 1.000 | 6m 04s | full pass |
| `hu4QvHn` | 1.000 | 4m 26s | full pass |
| `t6NKAu5` | 1.000 | 5m 18s | full pass |
| `uNnLjRQ` | 1.000 | 6m 51s | full pass |
| `zuYcubz` | 1.000 | 9m 32s | full pass |

The failures are useful because they are local and interpretable. Claude often understood the fitting equation but missed an output convention that matters for layer composition. Codex's single substantive failure preserved replay behavior but got the numerical fitting path wrong. Since the oracle passes, multiple independent model rollouts pass, and the failures cluster around exactly the specified requirements, the evidence supports the task's validity.

## Possible Improvements

A follow-up version could add streaming sufficient-statistics checks so agents must compute ridge systems from accumulated statistics rather than materializing every intermediate. Another extension could require local label decoding from the final activation, or a tiny end-to-end classification experiment using synthetic labels. Those would broaden the task while preserving deterministic verification, but I kept this version focused on the FP core so the reward signal stays clean.
