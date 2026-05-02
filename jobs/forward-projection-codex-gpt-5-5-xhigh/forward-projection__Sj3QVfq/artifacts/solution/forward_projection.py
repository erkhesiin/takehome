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
            "Unsupported nonlinearity "
            f"{name!r}; expected one of 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    def gaussian_solve(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if system.ndim != 2 or system.shape[0] != system.shape[1]:
            raise ValueError("The linear-system matrix must be square.")
        if rhs.ndim != 2 or rhs.shape[0] != system.shape[0]:
            raise ValueError("The right-hand side must have shape (n, k).")

        size = system.shape[0]
        if size == 0:
            return rhs.clone()

        matrix = system.clone()
        values = rhs.clone()
        tiny = torch.as_tensor(torch.finfo(matrix.dtype).eps, dtype=matrix.dtype, device=matrix.device)

        for pivot_idx in range(size):
            pivot_column = torch.abs(matrix[pivot_idx:, pivot_idx])
            pivot_row = pivot_idx + int(torch.argmax(pivot_column).item())

            if pivot_row != pivot_idx:
                row_order = torch.arange(size, device=matrix.device)
                row_order[pivot_idx] = pivot_row
                row_order[pivot_row] = pivot_idx
                matrix = matrix.index_select(0, row_order)
                values = values.index_select(0, row_order)

            pivot = matrix[pivot_idx, pivot_idx]
            if torch.abs(pivot) <= tiny:
                replacement = torch.where(pivot >= 0, tiny, -tiny)
                matrix[pivot_idx, pivot_idx] = replacement
                pivot = replacement

            matrix[pivot_idx] = matrix[pivot_idx] / pivot
            values[pivot_idx] = values[pivot_idx] / pivot

            if pivot_idx + 1 < size:
                factors = matrix[pivot_idx + 1 :, pivot_idx].clone()
                matrix[pivot_idx + 1 :] = (
                    matrix[pivot_idx + 1 :] - factors.unsqueeze(1) * matrix[pivot_idx].unsqueeze(0)
                )
                values[pivot_idx + 1 :] = (
                    values[pivot_idx + 1 :] - factors.unsqueeze(1) * values[pivot_idx].unsqueeze(0)
                )

        for pivot_idx in range(size - 1, -1, -1):
            if pivot_idx > 0:
                factors = matrix[:pivot_idx, pivot_idx].clone()
                matrix[:pivot_idx] = matrix[:pivot_idx] - factors.unsqueeze(1) * matrix[pivot_idx].unsqueeze(0)
                values[:pivot_idx] = values[:pivot_idx] - factors.unsqueeze(1) * values[pivot_idx].unsqueeze(0)

        return values

    def ridge_regression(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gram = inputs.transpose(0, 1).matmul(inputs)
        cross = inputs.transpose(0, 1).matmul(targets)
        identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        penalty = torch.as_tensor(ridge, dtype=gram.dtype, device=gram.device)
        return gaussian_solve(gram + identity * penalty, cross)

    allowed_nonlinearities = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in allowed_nonlinearities:
        raise ValueError(
            "Unsupported nonlinearity "
            f"{activation!r}; expected one of 'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if target_nonlinearity not in allowed_nonlinearities:
        raise ValueError(
            "Unsupported nonlinearity "
            f"{target_nonlinearity!r}; expected one of 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional tensors.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    if not X.dtype.is_floating_point or not Y.dtype.is_floating_point:
        raise TypeError("X and Y must be floating-point tensors.")

    with torch.no_grad():
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_idx, (Q, U) in enumerate(zip(Qs, Us)):
            if Q.ndim != 2 or U.ndim != 2:
                raise ValueError("Each projection matrix in Qs and Us must be two-dimensional.")
            if Q.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Qs[{layer_idx}] has incompatible input dimension: "
                    f"expected {current_activation.shape[1]}, got {Q.shape[0]}."
                )
            if U.shape[0] != Y.shape[1]:
                raise ValueError(
                    f"Us[{layer_idx}] has incompatible label dimension: "
                    f"expected {Y.shape[1]}, got {U.shape[0]}."
                )
            if Q.shape[1] != U.shape[1]:
                raise ValueError(f"Qs[{layer_idx}] and Us[{layer_idx}] must project to the same width.")
            if (
                Q.device != current_activation.device
                or U.device != current_activation.device
                or Y.device != current_activation.device
            ):
                raise ValueError("X, Y, Qs, and Us must be on the same device.")
            if (
                Q.dtype != current_activation.dtype
                or U.dtype != current_activation.dtype
                or Y.dtype != current_activation.dtype
            ):
                raise ValueError("X, Y, Qs, and Us must have the same dtype.")

            target_membrane = apply_nonlinearity(current_activation.matmul(Q), target_nonlinearity) + apply_nonlinearity(
                Y.matmul(U), target_nonlinearity
            )
            layer_weight = ridge_regression(current_activation, target_membrane)
            membrane = current_activation.matmul(layer_weight)
            current_activation = apply_nonlinearity(membrane, activation)

            weights.append(layer_weight)
            membranes.append(membrane)
            activations.append(current_activation)

    return {"weights": weights, "membranes": membranes, "activations": activations}


def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Replay a fitted Forward Projection stack on new inputs."""

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
            "Unsupported nonlinearity "
            f"{name!r}; expected one of 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    allowed_nonlinearities = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in allowed_nonlinearities:
        raise ValueError(
            "Unsupported nonlinearity "
            f"{activation!r}; expected one of 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional tensor.")
    if not X.dtype.is_floating_point:
        raise TypeError("X must be a floating-point tensor.")

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_idx, weight in enumerate(weights):
            if weight.ndim != 2:
                raise ValueError("Each weight tensor must be two-dimensional.")
            if weight.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"weights[{layer_idx}] has incompatible input dimension: "
                    f"expected {current_activation.shape[1]}, got {weight.shape[0]}."
                )
            if weight.device != current_activation.device:
                raise ValueError("X and weights must be on the same device.")
            if weight.dtype != current_activation.dtype:
                raise ValueError("X and weights must have the same dtype.")

            membrane = current_activation.matmul(weight)
            current_activation = apply_nonlinearity(membrane, activation)
            membranes.append(membrane)
            activations.append(current_activation)

    return membranes, activations
