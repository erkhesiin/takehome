import torch


def _apply_nonlinearity(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "identity":
        return x
    if kind == "tanh":
        return torch.tanh(x)
    if kind == "relu":
        return torch.relu(x)
    if kind == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"Unsupported nonlinearity: {kind}")


def _solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A X = B for X using Gaussian elimination with partial pivoting.

    A is (n, n); B is (n, k). Returns X of shape (n, k).
    Implemented purely with tensor operations to avoid LAPACK calls that
    can crash in certain CPU-only Docker environments.
    """
    n = A.shape[0]
    M = torch.cat([A, B], dim=1).clone()

    for i in range(n):
        # Partial pivoting: swap the row with largest |pivot|
        col = M[i:, i]
        pivot_rel = int(torch.argmax(torch.abs(col)).item())
        pivot_idx = i + pivot_rel
        if pivot_idx != i:
            tmp = M[i].clone()
            M[i] = M[pivot_idx]
            M[pivot_idx] = tmp

        pivot = M[i, i]
        # Normalize pivot row
        M[i] = M[i] / pivot

        # Eliminate entries below and above (Gauss-Jordan)
        if i > 0:
            factors_above = M[:i, i:i + 1].clone()
            M[:i] = M[:i] - factors_above * M[i:i + 1]
        if i < n - 1:
            factors_below = M[i + 1:, i:i + 1].clone()
            M[i + 1:] = M[i + 1:] - factors_below * M[i:i + 1]

    return M[:, n:]


def fit_forward_projection(
    X: torch.Tensor,
    Y: torch.Tensor,
    Qs: list[torch.Tensor],
    Us: list[torch.Tensor],
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict[str, list[torch.Tensor]]:
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must have the same length")

    L = len(Qs)
    dtype = X.dtype
    device = X.device

    A_prev = X
    weights: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    activations_list: list[torch.Tensor] = [X]

    for i in range(L):
        Q = Qs[i]
        U = Us[i]

        # Target membrane potential: Z_hat = g(A_{i-1} Q) + g(Y U)
        Z_hat = _apply_nonlinearity(A_prev @ Q, target_nonlinearity) + \
                _apply_nonlinearity(Y @ U, target_nonlinearity)

        # Ridge regression: W = (A^T A + lambda I)^{-1} (A^T Z_hat)
        AtA = A_prev.transpose(0, 1) @ A_prev
        m = AtA.shape[0]
        I = torch.eye(m, dtype=dtype, device=device)
        lhs = AtA + ridge * I
        rhs = A_prev.transpose(0, 1) @ Z_hat

        W = _solve_linear_system(lhs, rhs)

        # Forward through layer
        Z = A_prev @ W
        A_curr = _apply_nonlinearity(Z, activation)

        weights.append(W)
        membranes.append(Z)
        activations_list.append(A_curr)

        A_prev = A_curr

    return {
        "weights": weights,
        "membranes": membranes,
        "activations": activations_list,
    }


def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    membranes: list[torch.Tensor] = []
    activations_list: list[torch.Tensor] = []

    A_prev = X
    for W in weights:
        Z = A_prev @ W
        A_curr = _apply_nonlinearity(Z, activation)
        membranes.append(Z)
        activations_list.append(A_curr)
        A_prev = A_curr

    return membranes, activations_list
