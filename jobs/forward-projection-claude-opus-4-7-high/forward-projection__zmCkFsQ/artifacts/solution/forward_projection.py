import torch


def _apply_nonlinearity(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return tensor
    if name == "tanh":
        return torch.tanh(tensor)
    if name == "relu":
        return torch.relu(tensor)
    if name == "sigmoid":
        return torch.sigmoid(tensor)
    raise ValueError(f"Unsupported nonlinearity: {name}")


def _solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Solves A X = B for X using Gaussian elimination with partial pivoting.
    # A: (n, n), B: (n, k). Avoids torch.linalg.solve which can crash via LAPACK
    # on this CPU-only environment.
    n = A.shape[0]
    k = B.shape[1]
    M = torch.cat([A.clone(), B.clone()], dim=1)

    for i in range(n):
        pivot_row = i + int(torch.argmax(torch.abs(M[i:, i])).item())
        if pivot_row != i:
            tmp = M[i].clone()
            M[i] = M[pivot_row]
            M[pivot_row] = tmp

        pivot = M[i, i]
        if pivot == 0:
            continue

        M[i] = M[i] / pivot

        for j in range(n):
            if j == i:
                continue
            factor = M[j, i].clone()
            if factor != 0:
                M[j] = M[j] - factor * M[i]

    return M[:, n:n + k]


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
        raise ValueError("Qs and Us must have equal length")

    dtype = X.dtype
    device = X.device

    weights: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    A_prev = X
    Y_local = Y

    for Q, U in zip(Qs, Us):
        Q_cast = Q.to(dtype=dtype, device=device)
        U_cast = U.to(dtype=dtype, device=device)

        target_input_proj = _apply_nonlinearity(A_prev @ Q_cast, target_nonlinearity)
        target_label_proj = _apply_nonlinearity(Y_local @ U_cast, target_nonlinearity)
        Z_hat = target_input_proj + target_label_proj

        m_in = A_prev.shape[1]
        gram = A_prev.transpose(0, 1) @ A_prev
        identity = torch.eye(m_in, dtype=dtype, device=device)
        reg = gram + ridge * identity
        cross = A_prev.transpose(0, 1) @ Z_hat

        W = _solve_linear_system(reg, cross)

        Z = A_prev @ W
        A = _apply_nonlinearity(Z, activation)

        weights.append(W)
        membranes.append(Z)
        activations.append(A)

        A_prev = A

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
    dtype = X.dtype
    device = X.device

    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = []

    A_prev = X
    for W in weights:
        W_cast = W.to(dtype=dtype, device=device)
        Z = A_prev @ W_cast
        A = _apply_nonlinearity(Z, activation)
        membranes.append(Z)
        activations.append(A)
        A_prev = A

    return membranes, activations
