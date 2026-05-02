import torch


_SUPPORTED_ACTIVATIONS = ("identity", "tanh", "relu", "sigmoid")


def _apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "tanh":
        return torch.tanh(x)
    if name == "relu":
        return torch.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(
        f"Unsupported activation {name!r}; expected one of {_SUPPORTED_ACTIVATIONS}."
    )


def _solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A X = B using Gauss-Jordan elimination with partial pivoting.

    Implemented with tensor ops rather than torch.linalg.solve to avoid a
    LAPACK crash observed in this CPU-only environment.
    """
    n = A.shape[0]
    M = torch.cat([A, B], dim=1).clone()

    for i in range(n):
        col_tail = M[i:, i].abs()
        pivot_offset = int(col_tail.argmax().item())
        pivot_row = i + pivot_offset
        if pivot_row != i:
            swap = M[i].clone()
            M[i] = M[pivot_row]
            M[pivot_row] = swap

        pivot = M[i, i].clone()
        M[i] = M[i] / pivot

        factors = M[:, i].clone()
        factors[i] = 0
        M = M - factors.unsqueeze(1) * M[i].unsqueeze(0)

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
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation {activation!r}; expected one of "
            f"{_SUPPORTED_ACTIVATIONS}."
        )
    if target_nonlinearity not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported target_nonlinearity {target_nonlinearity!r}; expected "
            f"one of {_SUPPORTED_ACTIVATIONS}."
        )
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must have the same length.")

    dtype = X.dtype
    device = X.device

    weights: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    A_prev = X
    for l in range(len(Qs)):
        Q_l = Qs[l].to(dtype=dtype, device=device)
        U_l = Us[l].to(dtype=dtype, device=device)
        Y_l = Y.to(dtype=dtype, device=device)

        target = _apply_activation(A_prev @ Q_l, target_nonlinearity) + \
            _apply_activation(Y_l @ U_l, target_nonlinearity)

        m_in = A_prev.shape[1]
        gram = A_prev.transpose(0, 1) @ A_prev
        identity = torch.eye(m_in, dtype=dtype, device=device)
        gram = gram + ridge * identity
        rhs = A_prev.transpose(0, 1) @ target

        W_l = _solve_linear_system(gram, rhs)

        Z_l = A_prev @ W_l
        A_l = _apply_activation(Z_l, activation)

        weights.append(W_l)
        membranes.append(Z_l)
        activations.append(A_l)

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
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation {activation!r}; expected one of "
            f"{_SUPPORTED_ACTIVATIONS}."
        )

    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    A_prev = X
    for W in weights:
        W_l = W.to(dtype=A_prev.dtype, device=A_prev.device)
        Z_l = A_prev @ W_l
        A_l = _apply_activation(Z_l, activation)
        membranes.append(Z_l)
        activations.append(A_l)
        A_prev = A_l

    return membranes, activations
