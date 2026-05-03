# Write Up

Kevin Tamir (amarin)

## Paper Chosen

I chose the Forward Projection paper, ["Closed-form feedback-free learning with forward projection."](https://arxiv.org/abs/2501.16476) The paper proposes a way to fit neural-network layers without backpropagating errors through the whole model. Instead, each layer receives a locally constructed target membrane potential and is fit with a closed-form ridge-regression solve.

I chose this paper because it is recent, specific, and less likely to be saturated in frontier-model training data than common implementation targets like Mamba. The core algorithm is also compact enough to judge deterministically. That made it a good fit for Hillclimb's goal: a task that can challenge strong agents without making the reward arbitrary or dependent on noisy external datasets.

The part of the paper I focused on is the layerwise Forward Projection update. For layer `l`, the implementation constructs a target membrane potential:

```text
Z_tilde_l = g(A_prev @ Q_l) + g(Y @ U_l)
```

It then fits the layer by closed-form ridge regression:

```text
(A_prev.T @ A_prev + ridge * I) @ W_l = A_prev.T @ Z_tilde_l
```

After fitting, it computes `Z_l = A_prev @ W_l`, applies the selected activation, and passes that activation into the next layer.

## Capability Tested

The environment tests paper-to-code translation for a recent ML algorithm. A successful agent has to read the FP description, identify the algorithmic core, and implement it as stable PyTorch rather than falling back to familiar training loops.

Concretely, the task tests whether the agent can:

- implement the target membrane equation correctly;
- fit each layer with closed-form ridge regression;
- process layers sequentially, feeding each activation into the next layer;
- support the required nonlinearities;
- preserve tensor dtype/device behavior;
- avoid mutating inputs;
- replay a fitted stack for prediction with the same activation convention;
- avoid optimizers, gradient descent, autograd-based fitting, NumPy, SciPy, and sklearn.

That combination is narrow enough to verify cleanly, but still exposes meaningful mistakes from frontier agents.

## Agent Task

The agent is asked to create exactly one file:

```text
/solution/forward_projection.py
```

That module must define two top-level functions:

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

`fit_forward_projection` returns a dictionary with exactly three keys: `weights`, `membranes`, and `activations`. The `activations` list must start with the original input `X`, so the expected convention is `[A_0, A_1, ..., A_L]`.

`forward_projection_predict` replays a fitted stack on a new `X` and returns `(membranes, activations)`, using the same activation-list convention.

## Environment

The task runs in a CPU-only Docker image based on `python:3.11-slim`. The image installs only:

```text
torch==2.3.1
pytest==8.4.1
```

The container gives the agent a writable `/solution` directory and gives the verifier a `/logs/verifier` directory for reward output. The paper PDF is copied into the environment as:

```text
/paper/Forward_Projection_Paper.pdf
```

Harbor runs the agent first. If the agent writes `/solution/forward_projection.py`, Harbor then runs the verifier. The task also collects the submitted solution into the trial artifacts, so failing and passing implementations can be inspected later under `jobs/<job-name>/<trial-id>/artifacts/solution/forward_projection.py`.

I kept the environment intentionally small. There are no real datasets, model downloads, GPUs, or training jobs. That keeps failures attributable to the FP implementation itself rather than to infrastructure or data-preparation issues.

The Dockerfile also sets conservative CPU threading and dispatch variables. This mattered because a direct LAPACK-backed `torch.linalg.solve` path can throw `Illegal instruction` on some CPU-only hosts. The instructions therefore recommend a small PyTorch Gaussian-elimination style solve or an equivalent closed-form linear-system solver.

## Verifier And Reward

The verifier entrypoint is:

```text
tasks/forward-projection/tests/test.sh
```

It calls `python /tests/run_verifier.py`, which imports `/solution/forward_projection.py`, runs pytest, counts passed test calls, and writes the reward to:

```text
/logs/verifier/reward.txt
```

The reward is partial credit:

```text
reward = passed_cases / 12
```

The 12 deterministic cases cover both fitting and prediction. They check multiple layer counts, all supported nonlinearities, float32 and float64 behavior, rank-deficient inputs, high-dimensional ridge systems, large-magnitude inputs, soft labels, no input mutation, and the requirement that prediction activations include the original input as `activations[0]`.

The tests also enforce the task contract around dependencies and approach: the submitted module should use PyTorch and the standard library, not external ML packages or optimizer/autograd fitting paths. This keeps the reward aligned with the paper capability rather than rewarding a generic training loop.

## Design Decisions And Tradeoffs

I originally explored easier or more culturally cached papers, including "Think Deep, Not Just Long" and Mamba-style implementation work. Those produced saturated or near-saturated scores from frontier models, which made them poor fits for a task meant to expose capability gaps.

Forward Projection was a better target because it is recent, algorithmically specific, and pushes against the default backprop-first pattern that models often reach for. The task uses the paper's core equations rather than a full paper reproduction because full reproduction would add noise from datasets, preprocessing, experiment configuration, and runtime limits.

The main tradeoff is scope. This task does not ask the agent to reproduce the paper's full empirical results. Instead, it isolates the part that matters most for a deterministic RL task: can the agent implement the new learning rule correctly? That narrower scope produces cleaner reward signal and more interpretable failures.

Another tradeoff is the reference-based verifier. It is strict, but the behavior is mathematical and deterministic, so reference comparison is appropriate here. The hidden challenge is not guessing magic constants; it is following the FP equations and API contract carefully.

## Pull Request Checklist

This mirrors the Terminal-Bench PR template checklist.

- [x] All behavior checked in `tests/` is described in `instruction.md`.
- [x] All behavior described in `instruction.md` is checked in `tests/`.
- [x] The task was run with strong models through Harbor rollout configs for Claude Opus 4.7 high and Codex GPT-5.5 xhigh.
- [x] The task is hard for the agent to cheat because the verifier imports only the submitted `/solution/forward_projection.py`, checks exact numerical behavior against an independent reference, uses multiple deterministic edge cases, and rejects external ML dependencies plus optimizer/autograd fitting APIs.
- [x] Failing runs were analyzed, and the failures point to real implementation mistakes rather than underspecification.

The coverage is intentionally two-way. `instruction.md` specifies the module path, top-level functions, FP target equation, ridge equation, sequential activation flow, return keys, activation-list convention, supported nonlinearities, dtype/device preservation, no mutation, CPU compatibility, and no optimizer/autograd fitting path. The tests exercise those requirements through reference comparisons, return-shape checks, source-contract checks, dtype/device assertions, mutation checks, and parameterized cases covering all listed nonlinearities.

## Sample Rollouts

I ran 10 rollouts each for Claude Code Opus 4.7 high and Codex GPT-5.5 xhigh on the Forward Projection task. The results show that the task is neither trivial nor broken: both strong agents can solve it, but they also produce structured failures that match real implementation pitfalls.

| Agent | Model / effort | Trials | Mean reward | Full-credit trials | Notable issue | Wall-clock |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Claude Code | `anthropic/claude-opus-4-7`, high | 10 | 0.800 | 4/10 | 6 trials missed prediction replay semantics | 11m 02s |
| Codex | `gpt-5.5`, xhigh | 10 | 0.933 | 9/10 | 1 trial failed most fitting cases; 1 full-credit artifact had an agent-timeout flag | 35m 58s |

Claude's failures were highly consistent. Six trials scored `0.6667`: they passed all eight fitting cases and failed all four prediction replay cases. The verifier failure was that `predict activations` had the wrong length. Those implementations returned only post-layer activations instead of `[A_0, A_1, ..., A_L]`. This reflects a real API-composition mistake because the instruction explicitly says prediction activations must start with the input `X`, the oracle follows that convention, and four Claude rollouts implemented it correctly.

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

## Future Improvements

With more time, I would extend the task in ways that preserve deterministic scoring while increasing coverage of the full paper:

- Add a streaming sufficient-statistics variant so agents must compute ridge systems from accumulated statistics rather than materializing every intermediate.
- Add a small synthetic classification check that verifies the fitted stack can be used for local label prediction.
- Add more shape and numerical-stability edge cases, especially around nearly singular systems and mixed activation choices.
- Add a second API surface for deeper FP-style layer construction so the task tests composition beyond the current two required functions.
- Run more models and reasoning tiers to better map the task's difficulty curve.

I kept those out of this version because the current task already gives useful signal while staying compact, deterministic, and easy to inspect.
