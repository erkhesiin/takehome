# Forward Projection Harbor Task Spec

## Overview

This task evaluates whether an agent can implement the core Forward Projection algorithm from "Closed-form feedback-free learning with forward projection." The agent receives the paper PDF in the task environment and must write a single importable Python module at `/solution/forward_projection.py`.

The verifier is deterministic, PyTorch-only, and rewards partial progress. The intended capability is paper-to-code translation for a recent ML method, including numerical care around closed-form linear solves.

## Deliverable

The submitted file must define these two top-level functions:

```python
import torch


def fit_forward_projection(
    X: torch.Tensor,
    Y: torch.Tensor,
    Qs: list[torch.Tensor],
    Us: list[torch.Tensor],
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict[str, list[torch.Tensor]]:
    ...


def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ...
```

Supported nonlinearities are exactly:

```text
identity, tanh, relu, sigmoid
```

## Inputs

`fit_forward_projection` receives:

| Name | Shape | Meaning |
| --- | --- | --- |
| `X` | `(num_samples, input_dim)` | Initial input activation matrix |
| `Y` | `(num_samples, label_dim)` | One-hot or soft label matrix |
| `Qs` | list | Fixed random input projections, one per layer |
| `Us` | list | Fixed random label projections, one per layer |
| `ridge` | scalar | Nonnegative ridge penalty |
| `activation` | string | Model activation used after fitted membranes |
| `target_nonlinearity` | string | Nonlinearity used for target membrane construction |

For layer `l`, `Q_l` has shape `(in_dim_l, out_dim_l)` and `U_l` has shape `(label_dim, out_dim_l)`.

## Algorithm

Let `A_prev` be the previous activation matrix, with `A_0 = X`. For each layer `l`, generate target membrane potentials:

```text
Z_tilde_l = g(A_prev @ Q_l) + g(Y @ U_l)
```

where `g` is selected by `target_nonlinearity`.

Fit a layer weight matrix by closed-form ridge regression:

```text
(A_prev.T @ A_prev + ridge * I) @ W_l = A_prev.T @ Z_tilde_l
```

Equivalently, `W_l` minimizes:

```text
||A_prev @ W_l - Z_tilde_l||_2^2 + ridge * ||W_l||_2^2
```

Then compute the fitted membrane potentials and activations:

```text
Z_l = A_prev @ W_l
A_l = f(Z_l)
```

where `f` is selected by `activation`. The next layer uses `A_l` as `A_prev`.

## Return Values

`fit_forward_projection` must return a dictionary with exactly these keys:

| Key | Value |
| --- | --- |
| `weights` | list of fitted `W_l` tensors, one per layer |
| `membranes` | list of training `Z_l` tensors, one per layer |
| `activations` | list `[A_0, A_1, ..., A_L]`, including the original `X` |

`forward_projection_predict` must replay a fitted stack on a new `X` and return:

```python
(membranes, activations)
```

The prediction `activations` list follows the same convention: `activations[0]` is the input `X`, followed by one activation per fitted layer.

## Constraints

- Use only `torch` and the Python standard library.
- Do not use NumPy, SciPy, scikit-learn, optimizers, gradient descent, epochs, or autograd-based fitting.
- Fit each layer with a closed-form ridge solve.
- Do not mutate `X`, `Y`, `Qs`, `Us`, or `weights` in place.
- Preserve input dtype and device for outputs and intermediates such as identity matrices.
- Keep the task CPU-compatible. CUDA is not available.
- A hand-written Gaussian-elimination or equivalent PyTorch closed-form linear-system solve is preferred. Direct `torch.linalg.solve` is not required because it can hit CPU dispatch issues in some Docker environments.

## Verifier

`tests/test.sh` runs `python /tests/run_verifier.py`, which executes pytest and writes the numeric reward to `/logs/verifier/reward.txt`.

Reward:

```text
reward = passed_cases / 12
```

The 12 deterministic cases cover fitting and prediction behavior across multiple layer counts, nonlinearities, dtypes, high-dimensional ridge systems, rank-deficient inputs, large-magnitude inputs, soft labels, no input mutation, and replay semantics.

## Environment

The Docker image is based on `python:3.11-slim` and installs:

```text
torch==2.3.1
pytest==8.4.1
```

The image pre-creates writable `/solution` and `/logs/verifier` directories, copies the paper PDF to `/paper/Forward_Projection_Paper.pdf`, and sets conservative CPU threading/dispatch environment variables.

Task resource limits:

| Resource | Limit |
| --- | --- |
| CPUs | 2 |
| Memory | 2048 MB |
| Storage | 10240 MB |
| GPUs | 0 |
| Internet | enabled |
| Agent timeout | 600s |
| Verifier timeout | 120s |

## Files

```text
tasks/forward-projection/
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

## Running

Oracle:

```bash
harbor run -p tasks/forward-projection --agent oracle --force-build
```

Local verifier against the checked-in solution:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --python 3.11 --with torch==2.3.1 --with pytest==8.4.1 python -m pytest -q -p no:cacheprovider tasks/forward-projection/tests/test_forward_projection.py
```

Rollout configs:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```
