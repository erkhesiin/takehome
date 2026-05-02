# Forward Projection

This Harbor task asks an agent to implement the core equations from `Forward_Projection_Paper.pdf` in `/solution/forward_projection.py`.

The implementation must fit a stack of layers without backpropagation: generate Forward Projection target membrane potentials from fixed random input and label projections, solve each layer with closed-form ridge regression, then replay the fitted stack for prediction.

## Required Functions

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

`fit_forward_projection` returns a dictionary with exactly:

```text
weights
membranes
activations
```

`activations` must include the original input as the first element: `[A_0, A_1, ..., A_L]`.

`forward_projection_predict` returns `(membranes, activations)` with the same activation convention.

## Verifier

The verifier imports `/solution/forward_projection.py`, runs deterministic pytest cases, and writes the reward to `/logs/verifier/reward.txt`.

```text
reward = passed_cases / 12
```

The cases check fitting accuracy, prediction replay, supported nonlinearities, dtype preservation, rank-deficient ridge systems, high-dimensional systems, large values, soft labels, and no input mutation.

## Run

Oracle:

```bash
harbor run -p tasks/forward-projection --agent oracle --force-build
```

Local pytest verifier against the checked-in solution:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --python 3.11 --with torch==2.3.1 --with pytest==8.4.1 python -m pytest -q -p no:cacheprovider tasks/forward-projection/tests/test_forward_projection.py
```

Rollouts:

```bash
harbor run -c configs/rollouts/claude-opus-4-7-high.yaml --yes
harbor run -c configs/rollouts/codex-gpt-5-5-xhigh-openrouter.yaml --yes
```

## Notes

The Docker environment is CPU-only. A direct `torch.linalg.solve` path can fail with `Illegal instruction` on some hosts, so the oracle uses a small PyTorch closed-form linear-system solver instead.
