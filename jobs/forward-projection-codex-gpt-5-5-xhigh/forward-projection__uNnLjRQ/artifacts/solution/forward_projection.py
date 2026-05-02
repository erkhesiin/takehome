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
    supported_nonlinearities = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in supported_nonlinearities:
        raise ValueError(
            "activation must be one of: identity, tanh, relu, sigmoid"
        )
    if target_nonlinearity not in supported_nonlinearities:
        raise ValueError(
            "target_nonlinearity must be one of: identity, tanh, relu, sigmoid"
        )
    if len(Qs) != len(Us):
        raise ValueError("Qs and Us must contain the same number of layers")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional tensors")
    if not X.dtype.is_floating_point:
        raise TypeError("X must have a floating point dtype")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    with torch.no_grad():
        weights: list[torch.Tensor] = []
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X
        labels = Y.to(dtype=X.dtype, device=X.device)

        for layer_index, (Q_input, U_input) in enumerate(zip(Qs, Us)):
            if Q_input.ndim != 2 or U_input.ndim != 2:
                raise ValueError("all Qs and Us entries must be two-dimensional")

            Q = Q_input.to(dtype=current_activation.dtype, device=current_activation.device)
            U = U_input.to(dtype=current_activation.dtype, device=current_activation.device)

            if Q.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"Qs[{layer_index}] has input dimension {Q.shape[0]}, "
                    f"expected {current_activation.shape[1]}"
                )
            if U.shape[0] != labels.shape[1]:
                raise ValueError(
                    f"Us[{layer_index}] has label dimension {U.shape[0]}, "
                    f"expected {labels.shape[1]}"
                )
            if Q.shape[1] != U.shape[1]:
                raise ValueError(
                    f"Qs[{layer_index}] and Us[{layer_index}] must project "
                    "to the same layer width"
                )

            input_projection = current_activation @ Q
            label_projection = labels @ U

            if target_nonlinearity == "identity":
                input_target = input_projection
                label_target = label_projection
            elif target_nonlinearity == "tanh":
                input_target = torch.tanh(input_projection)
                label_target = torch.tanh(label_projection)
            elif target_nonlinearity == "relu":
                input_target = torch.relu(input_projection)
                label_target = torch.relu(label_projection)
            else:
                input_target = torch.sigmoid(input_projection)
                label_target = torch.sigmoid(label_projection)

            target_membrane = input_target + label_target
            transposed_activation = current_activation.transpose(0, 1)
            gram = transposed_activation @ current_activation
            cross = transposed_activation @ target_membrane
            feature_count = gram.shape[0]

            if feature_count == 0:
                layer_weight = cross.clone()
            else:
                identity = torch.eye(
                    feature_count,
                    dtype=current_activation.dtype,
                    device=current_activation.device,
                )
                system_matrix = gram + current_activation.new_tensor(ridge) * identity
                left = system_matrix.clone()
                right = cross.clone()

                for pivot_index in range(feature_count):
                    pivot_offset = torch.argmax(
                        torch.abs(left[pivot_index:, pivot_index])
                    ).item()
                    pivot_row = pivot_index + int(pivot_offset)

                    if pivot_row != pivot_index:
                        saved_left_row = left[pivot_index].clone()
                        saved_right_row = right[pivot_index].clone()
                        left[pivot_index] = left[pivot_row]
                        right[pivot_index] = right[pivot_row]
                        left[pivot_row] = saved_left_row
                        right[pivot_row] = saved_right_row

                    pivot = left[pivot_index, pivot_index]
                    if pivot.abs().item() == 0.0:
                        raise RuntimeError(
                            "ridge system is singular; use a positive ridge value"
                        )

                    if pivot_index + 1 < feature_count:
                        factors = left[pivot_index + 1 :, pivot_index] / pivot
                        left[pivot_index + 1 :, pivot_index:] = (
                            left[pivot_index + 1 :, pivot_index:]
                            - factors.unsqueeze(1)
                            * left[pivot_index : pivot_index + 1, pivot_index:]
                        )
                        right[pivot_index + 1 :, :] = (
                            right[pivot_index + 1 :, :]
                            - factors.unsqueeze(1)
                            * right[pivot_index : pivot_index + 1, :]
                        )

                layer_weight = torch.empty_like(right)
                for row_index in range(feature_count - 1, -1, -1):
                    row_solution = right[row_index].clone()
                    if row_index + 1 < feature_count:
                        row_solution = row_solution - (
                            left[row_index, row_index + 1 :] @ layer_weight[row_index + 1 :, :]
                        )
                    diagonal = left[row_index, row_index]
                    if diagonal.abs().item() == 0.0:
                        raise RuntimeError(
                            "ridge system is singular; use a positive ridge value"
                        )
                    layer_weight[row_index, :] = row_solution / diagonal

            membrane = current_activation @ layer_weight
            if activation == "identity":
                next_activation = membrane
            elif activation == "tanh":
                next_activation = torch.tanh(membrane)
            elif activation == "relu":
                next_activation = torch.relu(membrane)
            else:
                next_activation = torch.sigmoid(membrane)

            weights.append(layer_weight)
            membranes.append(membrane)
            activations.append(next_activation)
            current_activation = next_activation

        return {"weights": weights, "membranes": membranes, "activations": activations}


def forward_projection_predict(
    X: torch.Tensor,
    weights: list[torch.Tensor],
    activation: str = "tanh",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    supported_nonlinearities = {"identity", "tanh", "relu", "sigmoid"}
    if activation not in supported_nonlinearities:
        raise ValueError(
            "activation must be one of: identity, tanh, relu, sigmoid"
        )
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional tensor")
    if not X.dtype.is_floating_point:
        raise TypeError("X must have a floating point dtype")

    with torch.no_grad():
        membranes: list[torch.Tensor] = []
        activations: list[torch.Tensor] = [X]
        current_activation = X

        for layer_index, weight_input in enumerate(weights):
            if weight_input.ndim != 2:
                raise ValueError("all weights entries must be two-dimensional")

            weight = weight_input.to(
                dtype=current_activation.dtype,
                device=current_activation.device,
            )
            if weight.shape[0] != current_activation.shape[1]:
                raise ValueError(
                    f"weights[{layer_index}] has input dimension {weight.shape[0]}, "
                    f"expected {current_activation.shape[1]}"
                )

            membrane = current_activation @ weight
            if activation == "identity":
                next_activation = membrane
            elif activation == "tanh":
                next_activation = torch.tanh(membrane)
            elif activation == "relu":
                next_activation = torch.relu(membrane)
            else:
                next_activation = torch.sigmoid(membrane)

            membranes.append(membrane)
            activations.append(next_activation)
            current_activation = next_activation

        return membranes, activations
