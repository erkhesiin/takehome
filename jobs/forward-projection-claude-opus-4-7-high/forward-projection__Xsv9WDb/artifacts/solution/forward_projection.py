import torch


def _apply_nonlinearity(z: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return z
    if name == "tanh":
        return torch.tanh(z)
    if name == "relu":
        return torch.relu(z)
    if name == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(f"Unsupported nonlinearity: {name}")


def _solve_linear_system(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Solve A X = B using Gaussian elimination with partial pivoting.

    A: (n, n), B: (n, k). Returns X: (n, k).
    Avoids torch.linalg.solve which can crash with illegal instruction in
    CPU-only Docker environments lacking required LAPACK SIMD features.
    """
    n = A.shape[0]
    k = B.shape[1]
    M = torch.cat([A, B], dim=1).clone()

    for i in range(n):
        pivot_row = i + int(torch.argmax(torch.abs(M[i:, i])).item())
        if pivot_row != i:
            tmp = M[i].clone()
            M[i] = M[pivot_row]
            M[pivot_row] = tmp
        pivot = M[i, i]
        if torch.abs(pivot) < 1e-30:
            pivot = pivot + 1e-30
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
    Qs: list,
    Us: list,
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict:
    L = len(Qs)
    if len(Us) != L:
        raise ValueError("Qs and Us must have same length")

    dtype = X.dtype
    device = X.device

    weights = []
    membranes = []
    activations = [X]

    a_prev = X
    for i in range(L):
        Q_i = Qs[i]
        U_i = Us[i]

        Z_hat = _apply_nonlinearity(a_prev @ Q_i, target_nonlinearity) + \
                _apply_nonlinearity(Y @ U_i, target_nonlinearity)

        m_in = a_prev.shape[1]
        gram = a_prev.transpose(0, 1) @ a_prev
        I_mat = torch.eye(m_in, dtype=dtype, device=device)
        A_reg = gram + ridge * I_mat
        rhs = a_prev.transpose(0, 1) @ Z_hat

        W_i = _solve_linear_system(A_reg, rhs)

        z_i = a_prev @ W_i
        a_i = _apply_nonlinearity(z_i, activation)

        weights.append(W_i)
        membranes.append(z_i)
        activations.append(a_i)

        a_prev = a_i

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
    membranes = []
    activations = []

    a_prev = X
    for W_i in weights:
        z_i = a_prev @ W_i
        a_i = _apply_nonlinearity(z_i, activation)
        membranes.append(z_i)
        activations.append(a_i)
        a_prev = a_i

    return membranes, activations
