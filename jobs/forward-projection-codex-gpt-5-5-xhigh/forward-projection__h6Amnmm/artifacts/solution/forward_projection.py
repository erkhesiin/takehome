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
    """Fit Forward Projection layer weights with closed-form ridge regression."""

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

    def solve_linear_system(
        coefficient_matrix: torch.Tensor,
        right_hand_side: torch.Tensor,
    ) -> torch.Tensor:
        system_matrix = coefficient_matrix.clone()
        solution_rhs = right_hand_side.clone()
        dimension = system_matrix.shape[0]

        if dimension == 0:
            return solution_rhs

        for column_index in range(dimension):
            pivot_offset = torch.argmax(
                torch.abs(system_matrix[column_index:, column_index])
            ).item()
            pivot_index = column_index + int(pivot_offset)

            if pivot_index != column_index:
                matrix_row = system_matrix[column_index].clone()
                system_matrix[column_index] = system_matrix[pivot_index]
                system_matrix[pivot_index] = matrix_row

                rhs_row = solution_rhs[column_index].clone()
                solution_rhs[column_index] = solution_rhs[pivot_index]
                solution_rhs[pivot_index] = rhs_row

            pivot_value = system_matrix[column_index, column_index]
            if torch.abs(pivot_value).item() == 0.0:
                raise RuntimeError(
                    "Ridge regression system is singular; increase ridge."
                )

            if column_index + 1 < dimension:
                elimination_factors = (
                    system_matrix[column_index + 1 :, column_index] / pivot_value
                )
                system_matrix[column_index + 1 :, column_index:] = (
                    system_matrix[column_index + 1 :, column_index:]
                    - elimination_factors.unsqueeze(1)
                    * system_matrix[column_index : column_index + 1, column_index:]
                )
                solution_rhs[column_index + 1 :] = (
                    solution_rhs[column_index + 1 :]
                    - elimination_factors.unsqueeze(1)
                    * solution_rhs[column_index : column_index + 1]
                )

        solution = torch.empty_like(solution_rhs)
        for row_index in range(dimension - 1, -1, -1):
            if row_index + 1 < dimension:
                known_terms = (
                    system_matrix[row_index, row_index + 1 :]
                    @ solution[row_index + 1 :]
                )
            else:
                known_terms = torch.zeros(
                    solution_rhs.shape[1],
                    dtype=solution_rhs.dtype,
                    device=solution_rhs.device,
                )
            solution[row_index] = (
                solution_rhs[row_index] - known_terms
            ) / system_matrix[row_index, row_index]

        return solution

    valid_names = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in valid_names:
        raise ValueError(
            "Unsupported activation "
            f"{activation!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if target_nonlinearity not in valid_names:
        raise ValueError(
            "Unsupported target nonlinearity "
            f"{target_nonlinearity!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be rank-2 tensors.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers.")

    with torch.no_grad():
        labels = Y.to(dtype=X.dtype, device=X.device)
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_index, (input_projection, label_projection) in enumerate(zip(Qs, Us)):
            input_projection = input_projection.to(
                dtype=current_activation.dtype,
                device=current_activation.device,
            )
            label_projection = label_projection.to(
                dtype=current_activation.dtype,
                device=current_activation.device,
            )
            if input_projection.ndim != 2 or label_projection.ndim != 2:
                raise ValueError("Each Q and U tensor must be rank-2.")
            if input_projection.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Q at layer {layer_index} has incompatible input dimension."
                )
            if label_projection.shape[0] != labels.shape[1]:
                raise ValueError(
                    f"U at layer {layer_index} has incompatible label dimension."
                )
            if input_projection.shape[1] != label_projection.shape[1]:
                raise ValueError(
                    f"Q and U at layer {layer_index} must project to the same width."
                )

            projection_from_input = current_activation @ input_projection
            projection_from_label = labels @ label_projection
            target_membrane = apply_nonlinearity(
                projection_from_input, target_nonlinearity
            ) + apply_nonlinearity(projection_from_label, target_nonlinearity)

            gram_matrix = current_activation.transpose(0, 1) @ current_activation
            identity_matrix = torch.eye(
                gram_matrix.shape[0],
                dtype=gram_matrix.dtype,
                device=gram_matrix.device,
            )
            regularized_gram = gram_matrix + ridge * identity_matrix
            projected_targets = current_activation.transpose(0, 1) @ target_membrane
            layer_weight = solve_linear_system(regularized_gram, projected_targets)

            layer_membrane = current_activation @ layer_weight
            current_activation = apply_nonlinearity(layer_membrane, activation)

            weights.append(layer_weight)
            membranes.append(layer_membrane)
            activations.append(current_activation)

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
            "Unsupported activation "
            f"{name!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )

    valid_names = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in valid_names:
        raise ValueError(
            "Unsupported activation "
            f"{activation!r}; expected 'identity', 'tanh', 'relu', or 'sigmoid'."
        )
    if X.ndim != 2:
        raise ValueError("X must be a rank-2 tensor.")

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_index, layer_weight in enumerate(weights):
            layer_weight = layer_weight.to(
                dtype=current_activation.dtype,
                device=current_activation.device,
            )
            if layer_weight.ndim != 2:
                raise ValueError("Each weight tensor must be rank-2.")
            if layer_weight.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Weight at layer {layer_index} has incompatible input dimension."
                )

            layer_membrane = current_activation @ layer_weight
            current_activation = apply_nonlinearity(layer_membrane, activation)
            membranes.append(layer_membrane)
            activations.append(current_activation)

    return membranes, activations
