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
            return values
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

    def solve_by_gaussian_elimination(
        coefficients: torch.Tensor, right_hand_side: torch.Tensor
    ) -> torch.Tensor:
        size = coefficients.shape[0]
        if size == 0:
            return right_hand_side.clone()

        augmented = torch.cat(
            (coefficients.clone(), right_hand_side.clone()), dim=1
        )

        for pivot_index in range(size):
            pivot_offset = torch.argmax(
                torch.abs(augmented[pivot_index:, pivot_index])
            ).item()
            pivot_row = pivot_index + pivot_offset

            if pivot_row != pivot_index:
                pivot_copy = augmented[pivot_index].clone()
                augmented[pivot_index] = augmented[pivot_row]
                augmented[pivot_row] = pivot_copy

            pivot = augmented[pivot_index, pivot_index]
            if (not bool(torch.isfinite(pivot).item())) or pivot.item() == 0:
                raise ValueError(
                    "Ridge system is singular or non-finite; use a positive "
                    "ridge value or check the input tensors."
                )

            augmented[pivot_index] = augmented[pivot_index] / pivot

            if pivot_index + 1 < size:
                factors = augmented[pivot_index + 1 :, pivot_index].clone().unsqueeze(1)
                augmented[pivot_index + 1 :] = (
                    augmented[pivot_index + 1 :]
                    - factors * augmented[pivot_index].unsqueeze(0)
                )

        solution = augmented[:, size:].clone()

        for pivot_index in range(size - 1, -1, -1):
            if pivot_index > 0:
                factors = augmented[:pivot_index, pivot_index].clone().unsqueeze(1)
                solution[:pivot_index] = (
                    solution[:pivot_index]
                    - factors * solution[pivot_index].unsqueeze(0)
                )

        return solution

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
    if not isinstance(Qs, list) or not isinstance(Us, list):
        raise TypeError("Qs and Us must be lists of torch.Tensor objects.")
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must both be two-dimensional tensors.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not torch.is_floating_point(X) or not torch.is_floating_point(Y):
        raise TypeError("X and Y must be floating point tensors.")
    if X.dtype != Y.dtype or X.device != Y.device:
        raise ValueError("X and Y must have matching dtype and device.")

    with torch.no_grad():
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_index, (Q, U) in enumerate(zip(Qs, Us)):
            if not isinstance(Q, torch.Tensor) or not isinstance(U, torch.Tensor):
                raise TypeError("Each element of Qs and Us must be a torch.Tensor.")
            if Q.ndim != 2 or U.ndim != 2:
                raise ValueError("Each Q and U projection must be two-dimensional.")
            if Q.dtype != X.dtype or U.dtype != X.dtype:
                raise ValueError("All Q and U tensors must match X's dtype.")
            if Q.device != X.device or U.device != X.device:
                raise ValueError("All Q and U tensors must match X's device.")
            if current_activation.shape[1] != Q.shape[0]:
                raise ValueError(
                    f"Layer {layer_index}: Q has incompatible input dimension."
                )
            if Y.shape[1] != U.shape[0]:
                raise ValueError(
                    f"Layer {layer_index}: U has incompatible label dimension."
                )
            if Q.shape[1] != U.shape[1]:
                raise ValueError(
                    f"Layer {layer_index}: Q and U must project to the same width."
                )

            target_membrane = apply_nonlinearity(
                current_activation @ Q, target_nonlinearity
            ) + apply_nonlinearity(Y @ U, target_nonlinearity)

            gram = current_activation.transpose(0, 1) @ current_activation
            identity = torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            regularized_gram = gram + ridge * identity
            right_hand_side = current_activation.transpose(0, 1) @ target_membrane
            layer_weight = solve_by_gaussian_elimination(
                regularized_gram, right_hand_side
            )
            layer_membrane = current_activation @ layer_weight
            layer_activation = apply_nonlinearity(layer_membrane, activation)

            weights.append(layer_weight)
            membranes.append(layer_membrane)
            activations.append(layer_activation)
            current_activation = layer_activation

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
    def apply_nonlinearity(values: torch.Tensor, name: str) -> torch.Tensor:
        if name == "identity":
            return values
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

    if activation not in {"identity", "tanh", "relu", "sigmoid"}:
        raise ValueError(
            "Unsupported activation. Expected one of: "
            "'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if not isinstance(weights, list):
        raise TypeError("weights must be a list of torch.Tensor objects.")
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional tensor.")
    if not torch.is_floating_point(X):
        raise TypeError("X must be a floating point tensor.")

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_index, layer_weight in enumerate(weights):
            if not isinstance(layer_weight, torch.Tensor):
                raise TypeError("Each weight must be a torch.Tensor.")
            if layer_weight.ndim != 2:
                raise ValueError("Each weight tensor must be two-dimensional.")
            if layer_weight.dtype != X.dtype:
                raise ValueError("All weights must match X's dtype.")
            if layer_weight.device != X.device:
                raise ValueError("All weights must match X's device.")
            if current_activation.shape[1] != layer_weight.shape[0]:
                raise ValueError(
                    f"Layer {layer_index}: weight has incompatible input dimension."
                )

            layer_membrane = current_activation @ layer_weight
            layer_activation = apply_nonlinearity(layer_membrane, activation)

            membranes.append(layer_membrane)
            activations.append(layer_activation)
            current_activation = layer_activation

        return membranes, activations
