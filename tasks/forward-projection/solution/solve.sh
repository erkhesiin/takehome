#!/bin/bash
set -euo pipefail

mkdir -p /solution

cat > /solution/forward_projection.py <<'PY'
import torch

def _get_activation(name: str):
    if name == "identity":
        return lambda x: x
    elif name == "tanh":
        return torch.tanh
    elif name == "relu":
        return torch.relu
    elif name == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError(f"Unsupported activation: {name}")



def _ridge_solve(A_T_A: torch.Tensor, rhs: torch.Tensor, ridge: float) -> torch.Tensor:
    n = A_T_A.shape[0]
    system = A_T_A + ridge * torch.eye(n, dtype=A_T_A.dtype, device=A_T_A.device)
    aug = torch.cat([system.clone(), rhs.clone()], dim=1)

    for col in range(n):
        pivot_offset = torch.argmax(torch.abs(aug[col:, col]))
        pivot_row = col + int(pivot_offset.item())
        if pivot_row != col:
            row = aug[col].clone()
            aug[col] = aug[pivot_row]
            aug[pivot_row] = row

        pivot = aug[col, col]
        if torch.abs(pivot).item() <= torch.finfo(aug.dtype).eps:
            raise ValueError("Ridge system is singular; increase ridge")

        aug[col] = aug[col] / pivot
        for row_idx in range(n):
            if row_idx != col:
                aug[row_idx] = aug[row_idx] - aug[row_idx, col] * aug[col]

    return aug[:, n:]

def fit_forward_projection(
    X: torch.Tensor,
    Y: torch.Tensor,
    Qs: list[torch.Tensor],
    Us: list[torch.Tensor],
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict[str, list[torch.Tensor]]:
    
    f = _get_activation(activation)
    g = _get_activation(target_nonlinearity)
    
    weights = []
    membranes = []
    activations = [X]
    
    A_prev = X
    
    for Q_l, U_l in zip(Qs, Us):
        # Generate the target membrane potential using the Forward Projection equation
        Z_tilde_l = g(A_prev @ Q_l) + g(Y @ U_l)
        
        # Calculate A_prev^T @ A_prev
        A_T_A = A_prev.T @ A_prev
        
        # Fit weights with closed-form ridge regression.
        W_l = _ridge_solve(A_T_A, A_prev.T @ Z_tilde_l, ridge)
        
        # Compute actual membrane potentials and activations
        Z_l = A_prev @ W_l
        A_l = f(Z_l)
        
        weights.append(W_l)
        membranes.append(Z_l)
        activations.append(A_l)
        
        # Update A_prev for the next layer
        A_prev = A_l
        
    return {
        "weights": weights,
        "membranes": membranes,
        "activations": activations,
    }

def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    
    f = _get_activation(activation)
    
    membranes = []
    activations = [X]
    
    A_prev = X
    
    for W_l in weights:
        Z_l = A_prev @ W_l
        A_l = f(Z_l)
        
        membranes.append(Z_l)
        activations.append(A_l)
        
        A_prev = A_l
        
    return membranes, activations
PY

python - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location("forward_projection", "/solution/forward_projection.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert hasattr(module, "fit_forward_projection")
assert hasattr(module, "forward_projection_predict")
PY
