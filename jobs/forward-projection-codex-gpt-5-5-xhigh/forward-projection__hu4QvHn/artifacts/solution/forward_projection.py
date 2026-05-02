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
            "Unsupported nonlinearity: "
            f"{name!r}. Expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    def solve_closed_form(system: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if system.ndim != 2 or system.shape[0] != system.shape[1]:
            raise ValueError("Ridge system matrix must be square.")
        if rhs.ndim != 2 or rhs.shape[0] != system.shape[0]:
            raise ValueError("Right-hand side has incompatible shape.")
        if not system.dtype.is_floating_point:
            raise TypeError("Forward projection requires floating-point tensors.")

        matrix = system.clone()
        solution_rhs = rhs.clone()
        rows = matrix.shape[0]

        if rows == 0:
            return solution_rhs

        finfo = torch.finfo(matrix.dtype)
        scale = matrix.abs().max().clamp_min(torch.tensor(1, dtype=matrix.dtype, device=matrix.device))
        tolerance = scale * finfo.eps * max(1, rows) * 16

        for pivot_index in range(rows):
            pivot_slice = matrix[pivot_index:, pivot_index].abs()
            pivot_row = pivot_index + int(torch.argmax(pivot_slice).item())

            if pivot_row != pivot_index:
                matrix_pivot = matrix[pivot_index].clone()
                rhs_pivot = solution_rhs[pivot_index].clone()
                matrix[pivot_index] = matrix[pivot_row]
                solution_rhs[pivot_index] = solution_rhs[pivot_row]
                matrix[pivot_row] = matrix_pivot
                solution_rhs[pivot_row] = rhs_pivot

            pivot = matrix[pivot_index, pivot_index]
            if bool((pivot.abs() <= tolerance).item()):
                raise RuntimeError(
                    "Ridge system is singular or ill-conditioned; increase ridge."
                )

            next_row = pivot_index + 1
            if next_row < rows:
                factors = matrix[next_row:, pivot_index] / pivot
                matrix[next_row:, pivot_index:] = (
                    matrix[next_row:, pivot_index:]
                    - factors.unsqueeze(1) * matrix[pivot_index, pivot_index:]
                )
                solution_rhs[next_row:] = (
                    solution_rhs[next_row:] - factors.unsqueeze(1) * solution_rhs[pivot_index]
                )

        result = torch.empty_like(solution_rhs)
        for row_index in range(rows - 1, -1, -1):
            row_rhs = solution_rhs[row_index]
            if row_index + 1 < rows:
                row_rhs = row_rhs - matrix[row_index, row_index + 1 :] @ result[row_index + 1 :]
            result[row_index] = row_rhs / matrix[row_index, row_index]

        return result

    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional tensors.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if not X.dtype.is_floating_point or not Y.dtype.is_floating_point:
        raise TypeError("Forward projection requires floating-point X and Y tensors.")

    weights: list[torch.Tensor] = []
    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    with torch.no_grad():
        current_activation = X
        for layer_index, (projection_q, projection_u) in enumerate(zip(Qs, Us)):
            if projection_q.ndim != 2 or projection_u.ndim != 2:
                raise ValueError("Each Q and U projection must be two-dimensional.")
            if projection_q.device != current_activation.device:
                raise ValueError("Q projection device must match its pre-synaptic activation device.")
            if projection_u.device != Y.device:
                raise ValueError("U projection device must match Y device.")
            if projection_q.dtype != current_activation.dtype:
                raise TypeError("Q projection dtype must match its pre-synaptic activation dtype.")
            if projection_u.dtype != Y.dtype:
                raise TypeError("U projection dtype must match Y dtype.")
            if current_activation.device != Y.device or current_activation.dtype != Y.dtype:
                raise TypeError("X-derived activations and Y must share dtype and device.")
            if projection_q.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Qs[{layer_index}] has incompatible input dimension."
                )
            if projection_u.shape[0] != Y.shape[1]:
                raise ValueError(
                    f"Us[{layer_index}] has incompatible label dimension."
                )
            if projection_q.shape[1] != projection_u.shape[1]:
                raise ValueError(
                    f"Qs[{layer_index}] and Us[{layer_index}] must project to the same layer width."
                )

            target_membrane = apply_nonlinearity(
                current_activation @ projection_q, target_nonlinearity
            ) + apply_nonlinearity(Y @ projection_u, target_nonlinearity)

            gram = current_activation.transpose(0, 1) @ current_activation
            identity = torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            system = gram + ridge * identity
            rhs = current_activation.transpose(0, 1) @ target_membrane
            layer_weight = solve_closed_form(system, rhs)
            layer_membrane = current_activation @ layer_weight
            layer_activation = apply_nonlinearity(layer_membrane, activation)

            weights.append(layer_weight)
            membranes.append(layer_membrane)
            activations.append(layer_activation)
            current_activation = layer_activation

    return {"weights": weights, "membranes": membranes, "activations": activations}


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
            "Unsupported activation: "
            f"{name!r}. Expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional tensor.")
    if not X.dtype.is_floating_point:
        raise TypeError("Forward projection requires floating-point X tensors.")

    membranes: list[torch.Tensor] = []
    activations: list[torch.Tensor] = [X]

    with torch.no_grad():
        current_activation = X
        for layer_index, layer_weight in enumerate(weights):
            if layer_weight.ndim != 2:
                raise ValueError("Each weight tensor must be two-dimensional.")
            if layer_weight.device != current_activation.device:
                raise ValueError("Weight device must match the current activation device.")
            if layer_weight.dtype != current_activation.dtype:
                raise TypeError("Weight dtype must match the current activation dtype.")
            if layer_weight.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"weights[{layer_index}] has incompatible input dimension."
                )

            layer_membrane = current_activation @ layer_weight
            layer_activation = apply_nonlinearity(layer_membrane, activation)
            membranes.append(layer_membrane)
            activations.append(layer_activation)
            current_activation = layer_activation

    return membranes, activations
