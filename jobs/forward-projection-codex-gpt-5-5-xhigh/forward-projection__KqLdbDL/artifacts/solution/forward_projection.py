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
    def apply_nonlinearity(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return tensor.clone()
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

    def solve_linear_system(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        size = system.shape[0]
        if size == 0:
            return rhs.clone()

        left = system.clone()
        right = rhs.clone()
        scale = torch.max(torch.abs(left)).item()
        eps = torch.finfo(left.dtype).eps
        tolerance = eps * max(1, size) * max(1.0, scale)

        for column in range(size):
            pivot_offset = torch.argmax(torch.abs(left[column:, column])).item()
            pivot = column + int(pivot_offset)

            if pivot != column:
                left_row = left[column].clone()
                right_row = right[column].clone()
                left[column] = left[pivot]
                right[column] = right[pivot]
                left[pivot] = left_row
                right[pivot] = right_row

            pivot_value = left[column, column].clone()
            if torch.abs(pivot_value).item() <= tolerance:
                raise ValueError(
                    "Ridge system is singular or ill-conditioned; "
                    "increase ridge or check input rank."
                )

            left[column] = left[column] / pivot_value
            right[column] = right[column] / pivot_value

            for row in range(size):
                if row == column:
                    continue
                factor = left[row, column].clone()
                if torch.abs(factor).item() <= tolerance:
                    continue
                left[row] = left[row] - factor * left[column]
                right[row] = right[row] - factor * right[column]

        return right

    def ridge_regression(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feature_count = design.shape[1]
        identity = torch.eye(
            feature_count,
            dtype=design.dtype,
            device=design.device,
        )
        design_t = design.transpose(0, 1)
        system = design_t @ design + ridge * identity
        rhs = design_t @ target
        return solve_linear_system(system, rhs)

    if activation not in {"identity", "tanh", "relu", "sigmoid"}:
        raise ValueError(
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if target_nonlinearity not in {"identity", "tanh", "relu", "sigmoid"}:
        raise ValueError(
            "Unsupported target_nonlinearity. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if not torch.is_floating_point(X) or not torch.is_floating_point(Y):
        raise TypeError("X and Y must be floating-point tensors.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be 2-D tensors shaped as (samples, features).")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    with torch.no_grad():
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X.clone()]

        current_activation = activations[0]
        for layer_index, (Q, U) in enumerate(zip(Qs, Us)):
            if Q.ndim != 2 or U.ndim != 2:
                raise ValueError("Every Q and U tensor must be 2-D.")
            if not torch.is_floating_point(Q) or not torch.is_floating_point(U):
                raise TypeError("Every Q and U tensor must be floating-point.")
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
                    f"Q and U at layer {layer_index} must have the same output dimension."
                )

            projected_inputs = current_activation @ Q
            projected_labels = Y @ U
            target_membrane = apply_nonlinearity(
                projected_inputs,
                target_nonlinearity,
            ) + apply_nonlinearity(projected_labels, target_nonlinearity)

            weight = ridge_regression(current_activation, target_membrane)
            membrane = current_activation @ weight
            next_activation = apply_nonlinearity(membrane, activation)

            weights.append(weight)
            membranes.append(membrane)
            activations.append(next_activation)
            current_activation = next_activation

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
    def apply_activation(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return tensor.clone()
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
    if not torch.is_floating_point(X):
        raise TypeError("X must be a floating-point tensor.")
    if X.ndim != 2:
        raise ValueError("X must be a 2-D tensor shaped as (samples, features).")

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X.clone()]
        current_activation = activations[0]

        for layer_index, weight in enumerate(weights):
            if weight.ndim != 2:
                raise ValueError("Every weight tensor must be 2-D.")
            if not torch.is_floating_point(weight):
                raise TypeError("Every weight tensor must be floating-point.")
            if weight.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Weight at layer {layer_index} has incompatible input dimension."
                )

            membrane = current_activation @ weight
            current_activation = apply_activation(membrane, activation)
            membranes.append(membrane)
            activations.append(current_activation)

        return membranes, activations
