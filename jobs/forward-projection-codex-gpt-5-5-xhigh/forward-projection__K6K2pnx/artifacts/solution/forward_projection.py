import torch


def fit_forward_projection(
    X: torch.Tensor,
    Y: torch.Tensor,
    Qs: list[torch.Tensor],
    Us: list[torch.Tensor],
    ridge: float = 1e-3,
    activation: str = "tanh",
    target_nonlinearity: str = "tanh",
) -> dict[str, list[torch.Tensor]]:
    def apply_nonlinearity(values: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return values.clone()
        if name == "tanh":
            return torch.tanh(values)
        if name == "relu":
            return torch.relu(values)
        if name == "sigmoid":
            return torch.sigmoid(values)
        raise ValueError(
            "Unsupported nonlinearity. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    valid_nonlinearities = ("identity", "tanh", "relu", "sigmoid")
    if activation not in valid_nonlinearities:
        raise ValueError(
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if target_nonlinearity not in valid_nonlinearities:
        raise ValueError(
            "Unsupported target_nonlinearity. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    def solve_linear_system(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Ridge system matrix must be square.")
        if rhs.ndim != 2 or rhs.shape[0] != matrix.shape[0]:
            raise ValueError("Right-hand side has incompatible shape.")

        n = matrix.shape[0]
        a = matrix.clone()
        b = rhs.clone()
        tiny_value = float(torch.finfo(a.dtype).tiny)

        for col in range(n):
            pivot_offset = torch.argmax(torch.abs(a[col:, col])).item()
            pivot_row = col + pivot_offset
            if pivot_row != col:
                row_order = torch.arange(n, device=a.device)
                row_order[col] = pivot_row
                row_order[pivot_row] = col
                a = a.index_select(0, row_order)
                b = b.index_select(0, row_order)

            pivot = a[col, col]
            if torch.abs(pivot).item() <= tiny_value:
                replacement = tiny_value if pivot.item() >= 0 else -tiny_value
                pivot = torch.as_tensor(replacement, dtype=a.dtype, device=a.device)
                a[col, col] = pivot

            if col + 1 < n:
                factors = a[col + 1 :, col] / pivot
                a[col + 1 :] = a[col + 1 :] - factors.unsqueeze(1) * a[col : col + 1]
                b[col + 1 :] = b[col + 1 :] - factors.unsqueeze(1) * b[col : col + 1]

        solution = torch.empty_like(b)
        for row in range(n - 1, -1, -1):
            residual = b[row]
            if row + 1 < n:
                residual = residual - a[row, row + 1 :] @ solution[row + 1 :]
            pivot = a[row, row]
            if torch.abs(pivot).item() <= tiny_value:
                replacement = tiny_value if pivot.item() >= 0 else -tiny_value
                pivot = torch.as_tensor(replacement, dtype=a.dtype, device=a.device)
            solution[row] = residual / pivot
        return solution

    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional tensors.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    with torch.no_grad():
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X.clone()]

        for layer, (Q, U) in enumerate(zip(Qs, Us)):
            previous = activations[-1]
            if Q.ndim != 2 or U.ndim != 2:
                raise ValueError("Every Q and U tensor must be two-dimensional.")
            if previous.shape[1] != Q.shape[0]:
                raise ValueError(
                    f"Q at layer {layer} has incompatible input dimension."
                )
            if Y.shape[1] != U.shape[0]:
                raise ValueError(
                    f"U at layer {layer} has incompatible label dimension."
                )
            if Q.shape[1] != U.shape[1]:
                raise ValueError(
                    f"Q and U at layer {layer} must project to the same width."
                )
            if previous.device != Q.device or previous.device != U.device:
                raise ValueError("X, Qs, and Us must be on the same device per layer.")
            if Y.device != previous.device:
                raise ValueError("X and Y must be on the same device.")
            if previous.dtype != Q.dtype or previous.dtype != U.dtype:
                raise ValueError("X, Qs, and Us must share the same dtype per layer.")
            if Y.dtype != previous.dtype:
                raise ValueError("X and Y must share the same dtype.")
            if not previous.dtype.is_floating_point:
                raise TypeError("Forward Projection requires floating-point tensors.")

            target = apply_nonlinearity(previous @ Q, target_nonlinearity) + apply_nonlinearity(
                Y @ U, target_nonlinearity
            )
            gram = previous.transpose(0, 1) @ previous
            system = gram.clone()
            system.diagonal().add_(ridge)
            cross = previous.transpose(0, 1) @ target
            weight = solve_linear_system(system, cross)
            membrane = previous @ weight
            current_activation = apply_nonlinearity(membrane, activation)

            weights.append(weight)
            membranes.append(membrane)
            activations.append(current_activation)

    return {"weights": weights, "membranes": membranes, "activations": activations}


def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    def apply_nonlinearity(values: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return values.clone()
        if name == "tanh":
            return torch.tanh(values)
        if name == "relu":
            return torch.relu(values)
        if name == "sigmoid":
            return torch.sigmoid(values)
        raise ValueError(
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    if activation not in ("identity", "tanh", "relu", "sigmoid"):
        raise ValueError(
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional tensor.")

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X.clone()]

        for layer, weight in enumerate(weights):
            previous = activations[-1]
            if weight.ndim != 2:
                raise ValueError("Every weight tensor must be two-dimensional.")
            if previous.shape[1] != weight.shape[0]:
                raise ValueError(
                    f"Weight at layer {layer} has incompatible input dimension."
                )
            if previous.device != weight.device:
                raise ValueError("X and weights must be on the same device per layer.")
            if previous.dtype != weight.dtype:
                raise ValueError("X and weights must share the same dtype per layer.")
            if not previous.dtype.is_floating_point:
                raise TypeError("Forward Projection requires floating-point tensors.")

            membrane = previous @ weight
            current_activation = apply_nonlinearity(membrane, activation)
            membranes.append(membrane)
            activations.append(current_activation)

    return membranes, activations
