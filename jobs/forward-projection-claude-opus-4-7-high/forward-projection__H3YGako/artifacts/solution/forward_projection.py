import torch


_ACTIVATIONS = {"identity", "tanh", "relu", "sigmoid"}


def _apply_activation(Z: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return Z
    if name == "tanh":
        return torch.tanh(Z)
    if name == "relu":
        return torch.relu(Z)
    if name == "sigmoid":
        return torch.sigmoid(Z)
    raise ValueError(f"Unsupported activation: {name!r}")


def _solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A X = B for X using Gaussian elimination with partial pivoting.

    Implemented in pure torch tensor ops to avoid the LAPACK path that
    crashes in this CPU-only Docker environment.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device

    M = torch.cat([A.clone(), B.clone()], dim=1)
    k = B.shape[1]

    for i in range(n):
        pivot_row = i + int(torch.argmax(torch.abs(M[i:, i])).item())
        if pivot_row != i:
            tmp = M[i].clone()
            M[i] = M[pivot_row]
            M[pivot_row] = tmp

        pivot = M[i, i]
        if torch.abs(pivot) < torch.tensor(1e-30, dtype=dtype, device=device):
            pivot = pivot + torch.tensor(1e-30, dtype=dtype, device=device)
            M[i, i] = pivot

        M[i] = M[i] / pivot

        if i + 1 < n:
            factors = M[i + 1 :, i : i + 1]
            M[i + 1 :] = M[i + 1 :] - factors * M[i : i + 1]

    X = torch.zeros((n, k), dtype=dtype, device=device)
    for i in range(n - 1, -1, -1):
        if i + 1 < n:
            X[i] = M[i, n : n + k] - M[i, i + 1 : n] @ X[i + 1 :]
        else:
            X[i] = M[i, n : n + k]

    return X


def fit_forward_projection(
    X: torch.Tensor,
    Y: torch.Tensor,
    Qs: list,
    Us: list,
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict:
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation!r}")
    if target_nonlinearity not in _ACTIVATIONS:
        raise ValueError(f"Unsupported target_nonlinearity: {target_nonlinearity!r}")
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must have the same length")

    dtype = X.dtype
    device = X.device

    weights: list = []
    membranes: list = []
    activations_list: list = [X]

    A_prev = X
    Y_local = Y

    for layer_idx, (Q, U) in enumerate(zip(Qs, Us)):
        Q = Q.to(dtype=dtype, device=device)
        U = U.to(dtype=dtype, device=device)

        Z_hat = _apply_activation(A_prev @ Q, target_nonlinearity) + _apply_activation(
            Y_local @ U, target_nonlinearity
        )

        m_prev = A_prev.shape[1]
        AtA = A_prev.transpose(0, 1) @ A_prev
        AtZ = A_prev.transpose(0, 1) @ Z_hat

        I = torch.eye(m_prev, dtype=dtype, device=device)
        lhs = AtA + ridge * I

        W = _solve_linear_system(lhs, AtZ)

        Z = A_prev @ W
        A_next = _apply_activation(Z, activation)

        weights.append(W)
        membranes.append(Z)
        activations_list.append(A_next)

        A_prev = A_next

    return {
        "weights": weights,
        "membranes": membranes,
        "activations": activations_list,
    }


def forward_projection_predict(
    X: torch.Tensor,
    weights: list,
    activation: str = "tanh",
) -> tuple:
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation!r}")

    membranes: list = []
    activations_list: list = []

    A_prev = X
    for W in weights:
        W_cast = W.to(dtype=A_prev.dtype, device=A_prev.device)
        Z = A_prev @ W_cast
        A_next = _apply_activation(Z, activation)
        membranes.append(Z)
        activations_list.append(A_next)
        A_prev = A_next

    return membranes, activations_list
