import torch


_ACTIVATIONS = {
    "identity": lambda x: x,
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
}


def _apply(name, x):
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported non-linearity: {name!r}. "
            "Expected one of: 'identity', 'tanh', 'relu', 'sigmoid'."
        )
    return _ACTIVATIONS[name](x)


def _gauss_solve(A, B):
    """Solve the linear system A X = B where A is (n, n) and B is (n, k).

    Uses Gaussian elimination with partial pivoting, implemented entirely in
    PyTorch tensor operations. A and B are not mutated; a fresh augmented
    matrix is constructed internally.
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    M = torch.cat([A, B], dim=1).clone()
    tiny = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-30

    for i in range(n):
        col = torch.abs(M[i:, i])
        pivot_offset = int(torch.argmax(col).item())
        pivot_row = i + pivot_offset
        if pivot_row != i:
            row_i = M[i].clone()
            M[i] = M[pivot_row]
            M[pivot_row] = row_i
        pivot = M[i, i]
        if float(pivot.abs()) <= tiny:
            continue
        if i + 1 < n:
            factors = M[i + 1:, i] / pivot
            M[i + 1:] = M[i + 1:] - factors.unsqueeze(1) * M[i].unsqueeze(0)

    X = torch.zeros((n, B.shape[1]), dtype=dtype, device=device)
    for i in range(n - 1, -1, -1):
        s = M[i, n:].clone()
        if i + 1 < n:
            s = s - M[i, i + 1:n] @ X[i + 1:]
        diag = M[i, i]
        if float(diag.abs()) <= tiny:
            X[i] = torch.zeros_like(s)
        else:
            X[i] = s / diag
    return X


def fit_forward_projection(
    X,
    Y,
    Qs,
    Us,
    ridge=1e-3,
    activation="tanh",
    target_nonlinearity="tanh",
):
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must have the same length.")

    dtype = X.dtype
    device = X.device

    weights = []
    membranes = []
    activations_list = [X]

    a_prev = X
    for layer_idx in range(len(Qs)):
        Q = Qs[layer_idx]
        U = Us[layer_idx]

        target = _apply(target_nonlinearity, a_prev @ Q) + _apply(
            target_nonlinearity, Y @ U
        )

        m_in = a_prev.shape[1]
        gram = a_prev.transpose(0, 1) @ a_prev
        eye = torch.eye(m_in, dtype=dtype, device=device)
        lhs = gram + float(ridge) * eye
        rhs = a_prev.transpose(0, 1) @ target

        W = _gauss_solve(lhs, rhs)

        z = a_prev @ W
        a = _apply(activation, z)

        weights.append(W)
        membranes.append(z)
        activations_list.append(a)

        a_prev = a

    return {
        "weights": weights,
        "membranes": membranes,
        "activations": activations_list,
    }


def forward_projection_predict(X, weights, activation="tanh"):
    membranes = []
    activations_list = [X]
    a_prev = X
    for W in weights:
        z = a_prev @ W
        a = _apply(activation, z)
        membranes.append(z)
        activations_list.append(a)
        a_prev = a
    return membranes, activations_list
