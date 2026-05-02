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
            "Unsupported non-linearity {!r}; expected 'identity', 'tanh', "
            "'relu', or 'sigmoid'.".format(name)
        )

    def gaussian_solve(system: torch.Tensor, right_hand_side: torch.Tensor) -> torch.Tensor:
        dimension = system.shape[0]
        upper = system.clone()
        transformed_rhs = right_hand_side.clone()
        finfo = torch.finfo(upper.dtype)
        tiny = torch.as_tensor(finfo.eps, dtype=upper.dtype, device=upper.device)

        for column in range(dimension):
            pivot_offset = torch.argmax(torch.abs(upper[column:, column]))
            pivot_row = int(pivot_offset.item()) + column
            if pivot_row != column:
                row_order = torch.arange(dimension, device=upper.device)
                row_order[column] = pivot_row
                row_order[pivot_row] = column
                upper = upper.index_select(0, row_order)
                transformed_rhs = transformed_rhs.index_select(0, row_order)

            pivot_value = upper[column, column]
            if torch.abs(pivot_value) <= tiny:
                pivot_sign = torch.where(
                    pivot_value < 0,
                    -torch.ones_like(pivot_value),
                    torch.ones_like(pivot_value),
                )
                upper[column, column] = pivot_value + pivot_sign * tiny
                pivot_value = upper[column, column]

            if column + 1 < dimension:
                elimination_factors = upper[column + 1 :, column] / pivot_value
                upper[column + 1 :, column:] = (
                    upper[column + 1 :, column:]
                    - elimination_factors.unsqueeze(1) * upper[column : column + 1, column:]
                )
                transformed_rhs[column + 1 :] = (
                    transformed_rhs[column + 1 :]
                    - elimination_factors.unsqueeze(1) * transformed_rhs[column : column + 1]
                )

        solution = torch.empty_like(transformed_rhs)
        for row in range(dimension - 1, -1, -1):
            residual = transformed_rhs[row]
            if row + 1 < dimension:
                residual = residual - upper[row, row + 1 :].matmul(solution[row + 1 :])
            diagonal = upper[row, row]
            if torch.abs(diagonal) <= tiny:
                diagonal_sign = torch.where(
                    diagonal < 0,
                    -torch.ones_like(diagonal),
                    torch.ones_like(diagonal),
                )
                diagonal = diagonal + diagonal_sign * tiny
            solution[row] = residual / diagonal

        return solution

    def ridge_regression(design: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feature_count = design.shape[1]
        gram = design.transpose(0, 1).matmul(design)
        projected_target = design.transpose(0, 1).matmul(target)
        identity = torch.eye(feature_count, dtype=design.dtype, device=design.device)
        ridge_value = torch.as_tensor(ridge, dtype=design.dtype, device=design.device)
        return gaussian_solve(gram + ridge_value * identity, projected_target)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D tensors.")
    if not X.dtype.is_floating_point or not Y.dtype.is_floating_point:
        raise TypeError("X and Y must use floating-point dtypes.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples.")
    if X.device != Y.device or X.dtype != Y.dtype:
        raise ValueError("X and Y must have the same dtype and device.")
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    apply_nonlinearity(torch.empty(0, dtype=X.dtype, device=X.device), activation)
    apply_nonlinearity(torch.empty(0, dtype=X.dtype, device=X.device), target_nonlinearity)

    with torch.no_grad():
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X.clone()]
        current_activation = activations[0]

        for layer_index, (projection, label_projection) in enumerate(zip(Qs, Us)):
            if projection.ndim != 2 or label_projection.ndim != 2:
                raise ValueError("Each Q and U tensor must be 2-D.")
            if projection.device != X.device or label_projection.device != X.device:
                raise ValueError("Qs and Us must be on the same device as X and Y.")
            if projection.dtype != X.dtype or label_projection.dtype != X.dtype:
                raise ValueError("Qs and Us must have the same dtype as X and Y.")
            if current_activation.shape[1] != projection.shape[0]:
                raise ValueError(
                    "Q at layer {} has an incompatible input dimension.".format(layer_index)
                )
            if Y.shape[1] != label_projection.shape[0]:
                raise ValueError(
                    "U at layer {} has an incompatible label dimension.".format(layer_index)
                )
            if projection.shape[1] != label_projection.shape[1]:
                raise ValueError(
                    "Q and U at layer {} must project to the same width.".format(layer_index)
                )

            input_component = apply_nonlinearity(
                current_activation.matmul(projection), target_nonlinearity
            )
            label_component = apply_nonlinearity(
                Y.matmul(label_projection), target_nonlinearity
            )
            target_membrane = input_component + label_component
            layer_weight = ridge_regression(current_activation, target_membrane)
            layer_membrane = current_activation.matmul(layer_weight)
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
            "Unsupported non-linearity {!r}; expected 'identity', 'tanh', "
            "'relu', or 'sigmoid'.".format(name)
        )

    if X.ndim != 2:
        raise ValueError("X must be a 2-D tensor.")
    if not X.dtype.is_floating_point:
        raise TypeError("X must use a floating-point dtype.")
    apply_nonlinearity(torch.empty(0, dtype=X.dtype, device=X.device), activation)

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X.clone()]
        current_activation = activations[0]

        for layer_index, layer_weight in enumerate(weights):
            if layer_weight.ndim != 2:
                raise ValueError("Each weight tensor must be 2-D.")
            if layer_weight.device != X.device:
                raise ValueError("All weights must be on the same device as X.")
            if layer_weight.dtype != X.dtype:
                raise ValueError("All weights must have the same dtype as X.")
            if current_activation.shape[1] != layer_weight.shape[0]:
                raise ValueError(
                    "Weight at layer {} has an incompatible input dimension.".format(
                        layer_index
                    )
                )

            layer_membrane = current_activation.matmul(layer_weight)
            current_activation = apply_nonlinearity(layer_membrane, activation)
            membranes.append(layer_membrane)
            activations.append(current_activation)

        return membranes, activations
