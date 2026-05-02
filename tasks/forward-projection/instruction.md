# Forward Projection Implementation

Read `Forward_Projection_Paper.pdf` and implement the core Forward Projection fitting algorithm in one Python file at this exact absolute path:

```text
/solution/forward_projection.py
```

Do not put the implementation in the current directory, a notebook, or a differently named file. The file must be importable and must define both required functions at module top level.

## Required Signatures

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

## Algorithm

For fitting, process layers sequentially. Let `A_prev` start as `X`. For each layer `l`, use the corresponding `Q_l` and `U_l` to construct target membrane potentials:

```text
Z_tilde_l = g(A_prev @ Q_l) + g(Y @ U_l)
```

Here `g` is selected by `target_nonlinearity`.

Fit the layer weights with closed-form ridge regression:

```text
(A_prev.T @ A_prev + ridge * I) @ W_l = A_prev.T @ Z_tilde_l
```

Then compute the fitted membrane potentials and activations:

```text
Z_l = A_prev @ W_l
A_l = f(Z_l)
```

Here `f` is selected by `activation`. Use `A_l` as the input to the next layer.

## Expected Outputs

`fit_forward_projection` must return a dictionary with exactly these keys:

- `weights`: list of fitted weight tensors, one per layer.
- `membranes`: list of fitted training membrane potentials, one per layer.
- `activations`: list of activations starting with the original input `X`, followed by one activation per layer.

`forward_projection_predict` must replay the fitted weight stack on a new `X` and return:

```python
(membranes, activations)
```

The prediction `activations` list must also start with the input `X`.

## Supported Nonlinearities

Support exactly these names for both `activation` and `target_nonlinearity`:

```text
identity
tanh
relu
sigmoid
```

## Constraints

- Use only `torch` and the Python standard library.
- Use closed-form ridge regression. Do not use gradient descent, epochs, optimizers, or autograd-based fitting.
- Do not use NumPy, SciPy, scikit-learn, or external ML libraries.
- Do not mutate `X`, `Y`, `Qs`, `Us`, or `weights` in place.
- Preserve the dtype and device of the input tensors for outputs and intermediate tensors such as identity matrices.
- Keep the implementation CPU-compatible.
- Prefer a small Gaussian-elimination or equivalent PyTorch linear-system solver for the ridge system. A direct LAPACK-backed `torch.linalg.solve` can crash with `Illegal instruction` in this CPU-only Docker environment.
- Do not rely on printed output or an `if __name__ == "__main__"` block for correctness.

## Smoke Test

After writing the file, this should succeed inside the task container:

```bash
test -s /solution/forward_projection.py
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("forward_projection", "/solution/forward_projection.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert hasattr(module, "fit_forward_projection")
assert hasattr(module, "forward_projection_predict")
print("Smoke test passed")
PY
```
