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
    raise ValueError(f"Unsupported non-linearity: {name!r}")


def _solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A @ X = B via Gauss-Jordan elimination with partial pivoting.

    A is (n, n); B is (n, m). Returns X of shape (n, m). Works purely with
    PyTorch tensor ops to avoid the LAPACK path used by ``torch.linalg.solve``.
    """
    n = A.shape[0]
    M = torch.cat([A, B], dim=1).clone()

    for i in range(n):
        # Partial pivoting: find row with largest absolute value in column i
        # among rows i..n-1.
        col = M[i:, i].abs()
        rel_pivot = int(torch.argmax(col).item())
        pivot = rel_pivot + i
        if pivot != i:
            tmp = M[i].clone()
            M[i] = M[pivot]
            M[pivot] = tmp

        piv_val = M[i, i].clone()
        M[i] = M[i] / piv_val

        # Eliminate column i from all other rows simultaneously.
        factors = M[:, i].clone().unsqueeze(1)
        factors[i, 0] = 0
        M = M - factors * M[i].unsqueeze(0)

    return M[:, n:]


def _target_potentials(
    A_prev: torch.Tensor,
    Y: torch.Tensor,
    Q: torch.Tensor,
    U: torch.Tensor,
    target_nonlinearity: str,
) -> torch.Tensor:
    g = lambda t: _apply_nonlinearity(t, target_nonlinearity)
    return g(A_prev @ Q) + g(Y @ U)


def fit_forward_projection(
    X: torch.Tensor,
    Y: torch.Tensor,
    Qs: list,
    Us: list,
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict:
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must have the same length.")

    L = len(Qs)
    dtype = X.dtype
    device = X.device

    weights: list = []
    membranes: list = []
    activations: list = [X]

    A_prev = X
    for l in range(L):
        Q_l = Qs[l].to(dtype=dtype, device=device) if (Qs[l].dtype != dtype or Qs[l].device != device) else Qs[l]
        U_l = Us[l].to(dtype=dtype, device=device) if (Us[l].dtype != dtype or Us[l].device != device) else Us[l]

        Z_tilde = _target_potentials(A_prev, Y, Q_l, U_l, target_nonlinearity)

        # Ridge regression: W_l = (A^T A + lambda I)^-1 (A^T Z_tilde)
        AtA = A_prev.transpose(0, 1) @ A_prev
        n_in = AtA.shape[0]
        eye = torch.eye(n_in, dtype=dtype, device=device)
        lhs = AtA + ridge * eye
        rhs = A_prev.transpose(0, 1) @ Z_tilde

        W_l = _solve_linear_system(lhs, rhs)

        Z_l = A_prev @ W_l
        A_l = _apply_nonlinearity(Z_l, activation)

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
    weights: list,
    activation: str = "tanh",
) -> tuple:
    membranes: list = []
    activations: list = [X]

    A_prev = X
    for W_l in weights:
        W = W_l.to(dtype=A_prev.dtype, device=A_prev.device) if (
            W_l.dtype != A_prev.dtype or W_l.device != A_prev.device
        ) else W_l
        Z_l = A_prev @ W
        A_l = _apply_nonlinearity(Z_l, activation)
        membranes.append(Z_l)
        activations.append(A_l)
        A_prev = A_l

    return membranes, activations
