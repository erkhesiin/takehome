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
    """Solve A X = B for X using Gaussian elimination with partial pivoting.

    A is (n, n); B is (n, k). Returns X of shape (n, k). Does not mutate A or B.
    """
    n = A.shape[0]
    k = B.shape[1]
    M = torch.cat([A, B], dim=1).clone()

    for col in range(n):
        pivot_row = col + int(torch.argmax(torch.abs(M[col:, col])).item())
        if pivot_row != col:
            tmp = M[col].clone()
            M[col] = M[pivot_row]
            M[pivot_row] = tmp

        pivot_val = M[col, col]
        if pivot_val == 0:
            continue

        for row in range(n):
            if row == col:
                continue
            factor = M[row, col] / pivot_val
            if factor != 0:
                M[row] = M[row] - factor * M[col]

    X = torch.zeros(n, k, dtype=A.dtype, device=A.device)
    for row in range(n):
        diag = M[row, row]
        if diag == 0:
            X[row] = M[row, n:]
        else:
            X[row] = M[row, n:] / diag
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

    current = X
    for i in range(num_layers):
        Q = Qs[i]
        U = Us[i]

        target = _apply_nonlinearity(current @ Q, target_nonlinearity) + \
                 _apply_nonlinearity(Y @ U, target_nonlinearity)

        gram = current.transpose(0, 1) @ current
        in_dim = gram.shape[0]
        identity = torch.eye(in_dim, dtype=gram.dtype, device=gram.device)
        regularized = gram + ridge * identity
        cross = current.transpose(0, 1) @ target

        W = _solve_linear_system(regularized, cross)
        Z = current @ W
        A_next = _apply_nonlinearity(Z, activation)

        weights.append(W)
        membranes.append(Z)
        activations.append(A_next)
        current = A_next

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

    current = X
    for W in weights:
        Z = current @ W
        A_next = _apply_nonlinearity(Z, activation)
        membranes.append(Z)
        activations.append(A_next)
        current = A_next

    return membranes, activations
