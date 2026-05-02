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
    """Fit Forward Projection layers with closed-form ridge regression."""

    def apply_nonlinearity(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return tensor
        if name == "tanh":
            return torch.tanh(tensor)
        if name == "relu":
            return torch.relu(tensor)
        if name == "sigmoid":
            return torch.sigmoid(tensor)
        raise ValueError(
            "Unsupported nonlinearity "
            f"{name!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    def solve_linear_system(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        size = matrix.shape[0]
        left = matrix.clone()
        right = rhs.clone()

        for column in range(size):
            pivot_offset = torch.argmax(left[column:, column].abs()).item()
            pivot_row = column + pivot_offset
            if pivot_row != column:
                order = torch.arange(size, device=left.device)
                order[column] = pivot_row
                order[pivot_row] = column
                left = left.index_select(0, order)
                right = right.index_select(0, order)

            pivot = left[column, column]
            if pivot.abs() == 0:
                raise RuntimeError(
                    "Ridge system is singular; use a positive ridge value."
                )

            if column + 1 < size:
                factors = left[column + 1 :, column] / pivot
                left[column + 1 :] = left[column + 1 :] - factors[:, None] * left[column]
                right[column + 1 :] = right[column + 1 :] - factors[:, None] * right[column]

        solution = torch.zeros_like(right)
        for row in range(size - 1, -1, -1):
            tail = left[row, row + 1 :] @ solution[row + 1 :]
            solution[row] = (right[row] - tail) / left[row, row]
        return solution

    def ridge_regression(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_2d = inputs.reshape(-1, inputs.shape[-1])
        target_2d = targets.reshape(-1, targets.shape[-1])
        gram = input_2d.transpose(0, 1) @ input_2d
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        system = gram + identity * ridge
        rhs = input_2d.transpose(0, 1) @ target_2d
        return solve_linear_system(system, rhs)

    if not torch.is_tensor(X) or not torch.is_tensor(Y):
        raise TypeError("X and Y must be torch.Tensor instances.")
    if not X.is_floating_point() or not Y.is_floating_point():
        raise TypeError("X and Y must use floating-point dtypes.")
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of tensors.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    supported_nonlinearities = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in supported_nonlinearities:
        raise ValueError(
            "Unsupported activation "
            f"{activation!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if target_nonlinearity not in supported_nonlinearities:
        raise ValueError(
            "Unsupported target_nonlinearity "
            f"{target_nonlinearity!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    weights: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    with torch.no_grad():
        current = X
        for layer_index, (Q, U) in enumerate(zip(Qs, Us)):
            if not torch.is_tensor(Q) or not torch.is_tensor(U):
                raise TypeError("Qs and Us must contain only torch.Tensor instances.")
            if Q.device != current.device or U.device != current.device or Y.device != current.device:
                raise ValueError("X, Y, Qs, and Us must be on the same device.")
            if Q.dtype != current.dtype or U.dtype != current.dtype or Y.dtype != current.dtype:
                raise ValueError("X, Y, Qs, and Us must use the same dtype.")
            if Q.ndim != 2 or U.ndim != 2:
                raise ValueError("Each Q and U tensor must be two-dimensional.")
            if current.shape[-1] != Q.shape[0]:
                raise ValueError(
                    f"Q at layer {layer_index} has incompatible input dimension."
                )
            if Y.shape[-1] != U.shape[0]:
                raise ValueError(
                    f"U at layer {layer_index} has incompatible label dimension."
                )
            if Q.shape[1] != U.shape[1]:
                raise ValueError(
                    f"Q and U at layer {layer_index} must have matching output dimensions."
                )

            data_projection = apply_nonlinearity(current @ Q, target_nonlinearity)
            label_projection = apply_nonlinearity(Y @ U, target_nonlinearity)
            target_membrane = data_projection + label_projection

            weight = ridge_regression(current, target_membrane)
            membrane = current @ weight
            current = apply_nonlinearity(membrane, activation)

            weights.append(weight)
            membranes.append(membrane)
            activations.append(current)

    return {"weights": weights, "membranes": membranes, "activations": activations}


def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Replay a fitted Forward Projection stack on input X."""

    def apply_nonlinearity(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return tensor
        if name == "tanh":
            return torch.tanh(tensor)
        if name == "relu":
            return torch.relu(tensor)
        if name == "sigmoid":
            return torch.sigmoid(tensor)
        raise ValueError(
            "Unsupported nonlinearity "
            f"{name!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    if not torch.is_tensor(X):
        raise TypeError("X must be a torch.Tensor instance.")
    if not X.is_floating_point():
        raise TypeError("X must use a floating-point dtype.")

    supported_nonlinearities = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in supported_nonlinearities:
        raise ValueError(
            "Unsupported activation "
            f"{activation!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    with torch.no_grad():
        current = X
        for layer_index, weight in enumerate(weights):
            if not torch.is_tensor(weight):
                raise TypeError("weights must contain only torch.Tensor instances.")
            if weight.device != current.device:
                raise ValueError("X and weights must be on the same device.")
            if weight.dtype != current.dtype:
                raise ValueError("X and weights must use the same dtype.")
            if weight.ndim != 2:
                raise ValueError("Each weight tensor must be two-dimensional.")
            if current.shape[-1] != weight.shape[0]:
                raise ValueError(
                    f"Weight at layer {layer_index} has incompatible input dimension."
                )

            membrane = current @ weight
            current = apply_nonlinearity(membrane, activation)
            membranes.append(membrane)
            activations.append(current)

    return membranes, activations
