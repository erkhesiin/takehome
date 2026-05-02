import torch


def _apply_nonlinearity(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "tanh":
        return torch.tanh(x)
    if name == "relu":
        return torch.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"Unsupported nonlinearity: {name}")


def _solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = B for X via Gaussian elimination with partial pivoting.

    Implemented with PyTorch tensor ops to avoid LAPACK illegal-instruction
    crashes observed with torch.linalg.solve in this environment.
    """
    n = A.shape[0]
    M = torch.cat([A, B], dim=1).clone()
    k_cols = B.shape[1]

    for i in range(n):
        pivot_idx = i + int(torch.argmax(torch.abs(M[i:, i])).item())
        if pivot_idx != i:
            tmp = M[i].clone()
            M[i] = M[pivot_idx]
            M[pivot_idx] = tmp

        pivot = M[i, i]
        if pivot == 0:
            continue

        M[i] = M[i] / pivot

        if i + 1 < n:
            factors = M[i + 1 :, i : i + 1].clone()
            M[i + 1 :] = M[i + 1 :] - factors * M[i : i + 1]

    X = torch.zeros((n, k_cols), dtype=M.dtype, device=M.device)
    for i in range(n - 1, -1, -1):
        X[i] = M[i, n : n + k_cols]
        if i + 1 < n:
            X[i] = X[i] - M[i, i + 1 : n] @ X[i + 1 :]

    return X


def fit_forward_projection(
    X: torch.Tensor,
    Y: torch.Tensor,
    Qs: list[torch.Tensor],
    Us: list[torch.Tensor],
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict[str, list[torch.Tensor]]:
    num_layers = len(Qs)
    if len(Us) != num_layers:
        raise ValueError("Qs and Us must have the same length")

    weights: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    A_prev = X
    dtype = X.dtype
    device = X.device

    for i in range(num_layers):
        Q_i = Qs[i]
        U_i = Us[i]

        Z_target = _apply_nonlinearity(A_prev @ Q_i, target_nonlinearity) + \
                   _apply_nonlinearity(Y @ U_i, target_nonlinearity)

        m_in = A_prev.shape[1]
        gram = A_prev.T @ A_prev
        reg = ridge * torch.eye(m_in, dtype=dtype, device=device)
        lhs = gram + reg
        rhs = A_prev.T @ Z_target

        W_i = _solve_linear_system(lhs, rhs)

        Z_i = A_prev @ W_i
        A_i = _apply_nonlinearity(Z_i, activation)

        weights.append(W_i)
        membranes.append(Z_i)
        activations.append(A_i)

        A_prev = A_i

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
    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    A_prev = X
    for W in weights:
        Z = A_prev @ W
        A = _apply_nonlinearity(Z, activation)
        membranes.append(Z)
        activations.append(A)
        A_prev = A

    return membranes, activations
