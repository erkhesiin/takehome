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
    """Fit Forward Projection layer weights by closed-form ridge regression."""

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
            "Unsupported nonlinearity. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    def solve_linear_system(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        coefficient = matrix.clone()
        value = rhs.clone()
        size = coefficient.shape[0]
        tiny = torch.finfo(coefficient.dtype).eps

        for pivot_index in range(size):
            pivot_offset = torch.argmax(
                torch.abs(coefficient[pivot_index:, pivot_index])
            ).item()
            swap_index = pivot_index + int(pivot_offset)
            if swap_index != pivot_index:
                pivot_row = coefficient[pivot_index].clone()
                coefficient[pivot_index] = coefficient[swap_index]
                coefficient[swap_index] = pivot_row

                pivot_value = value[pivot_index].clone()
                value[pivot_index] = value[swap_index]
                value[swap_index] = pivot_value

            pivot = coefficient[pivot_index, pivot_index]
            if torch.abs(pivot).item() <= tiny:
                raise RuntimeError("Ridge system is singular or ill-conditioned.")

            if pivot_index + 1 < size:
                factors = coefficient[pivot_index + 1 :, pivot_index] / pivot
                coefficient[pivot_index + 1 :, pivot_index:] = (
                    coefficient[pivot_index + 1 :, pivot_index:]
                    - factors.unsqueeze(1) * coefficient[pivot_index, pivot_index:]
                )
                value[pivot_index + 1 :, :] = (
                    value[pivot_index + 1 :, :]
                    - factors.unsqueeze(1) * value[pivot_index : pivot_index + 1, :]
                )
                coefficient[pivot_index + 1 :, pivot_index] = 0

        solution = torch.empty_like(value)
        for row in range(size - 1, -1, -1):
            if row + 1 < size:
                known = coefficient[row, row + 1 :] @ solution[row + 1 :, :]
            else:
                known = torch.zeros(
                    value.shape[1], dtype=value.dtype, device=value.device
                )
            solution[row, :] = (value[row, :] - known) / coefficient[row, row]

        return solution

    supported = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in supported:
        raise ValueError(
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if target_nonlinearity not in supported:
        raise ValueError(
            "Unsupported target nonlinearity. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2-dimensional tensors.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")
    if not torch.is_floating_point(X) or not torch.is_floating_point(Y):
        raise TypeError("X and Y must be floating point tensors.")
    if X.dtype != Y.dtype or X.device != Y.device:
        raise ValueError("X and Y must share the same dtype and device.")

    with torch.no_grad():
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_index, (Q, U) in enumerate(zip(Qs, Us)):
            if Q.ndim != 2 or U.ndim != 2:
                raise ValueError("Each Q and U tensor must be 2-dimensional.")
            if not torch.is_floating_point(Q) or not torch.is_floating_point(U):
                raise TypeError("Each Q and U tensor must be floating point.")
            if Q.dtype != X.dtype or U.dtype != X.dtype:
                raise ValueError("X, Y, Qs, and Us must share the same dtype.")
            if Q.device != X.device or U.device != X.device:
                raise ValueError("X, Y, Qs, and Us must share the same device.")
            if Q.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Q at layer {layer_index} has incompatible input dimension."
                )
            if U.shape[0] != Y.shape[1]:
                raise ValueError(
                    f"U at layer {layer_index} has incompatible label dimension."
                )
            if Q.shape[1] != U.shape[1]:
                raise ValueError(
                    f"Q and U at layer {layer_index} must project to the same width."
                )

            target_membrane = apply_nonlinearity(
                current_activation @ Q, target_nonlinearity
            ) + apply_nonlinearity(Y @ U, target_nonlinearity)

            normal_matrix = current_activation.transpose(0, 1) @ current_activation
            identity = torch.eye(
                normal_matrix.shape[0],
                dtype=normal_matrix.dtype,
                device=normal_matrix.device,
            )
            normal_matrix = normal_matrix + identity * normal_matrix.new_tensor(ridge)
            cross_matrix = current_activation.transpose(0, 1) @ target_membrane
            layer_weight = solve_linear_system(normal_matrix, cross_matrix)
            layer_membrane = current_activation @ layer_weight
            current_activation = apply_nonlinearity(layer_membrane, activation)

            weights.append(layer_weight)
            membranes.append(layer_membrane)
            activations.append(current_activation)

    return {"weights": weights, "membranes": membranes, "activations": activations}


def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Replay a fitted Forward Projection stack on new inputs."""

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
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    if activation not in {"identity", "tanh", "relu", "sigmoid"}:
        raise ValueError(
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional tensor.")
    if not torch.is_floating_point(X):
        raise TypeError("X must be a floating point tensor.")

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_index, weight in enumerate(weights):
            if weight.ndim != 2:
                raise ValueError("Each weight tensor must be 2-dimensional.")
            if not torch.is_floating_point(weight):
                raise TypeError("Each weight tensor must be floating point.")
            if weight.dtype != X.dtype or weight.device != X.device:
                raise ValueError("X and weights must share the same dtype and device.")
            if weight.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Weight at layer {layer_index} has incompatible input dimension."
                )

            membrane = current_activation @ weight
            current_activation = apply_nonlinearity(membrane, activation)
            membranes.append(membrane)
            activations.append(current_activation)

    return membranes, activations
