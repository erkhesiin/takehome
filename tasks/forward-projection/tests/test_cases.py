from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FitCase:
    name: str
    seed: int
    samples: int
    input_dim: int
    label_dim: int
    layer_dims: tuple[int, ...]
    ridge: float
    activation: str
    target_nonlinearity: str
    dtype: torch.dtype = torch.float32
    x_scale: float = 1.0
    q_scale: float = 0.5
    u_scale: float = 0.5
    rank_deficient: bool = False
    soft_labels: bool = False


@dataclass(frozen=True)
class PredictCase:
    name: str
    seed: int
    samples: int
    input_dim: int
    layer_dims: tuple[int, ...]
    activation: str
    dtype: torch.dtype = torch.float32
    x_scale: float = 1.0
    weight_scale: float = 0.5


FIT_CASES = [
    FitCase("two_layer_tanh", 0, 12, 5, 3, (7, 4), 1e-2, "tanh", "tanh"),
    FitCase("identity_targets_relu_model", 1, 9, 4, 2, (6, 5, 3), 5e-2, "relu", "identity", soft_labels=True),
    FitCase("rank_deficient_sigmoid", 2, 8, 6, 3, (5, 5), 1e-1, "sigmoid", "tanh", rank_deficient=True),
    FitCase("high_dim_low_sample", 3, 5, 9, 4, (11, 6), 2e-1, "tanh", "relu", soft_labels=True),
    FitCase("large_magnitude_saturation", 4, 10, 4, 3, (8, 4), 1e-1, "tanh", "sigmoid", x_scale=25.0, q_scale=0.1, u_scale=2.0),
    FitCase("float64_precision", 5, 11, 5, 4, (6, 7, 3), 1e-3, "identity", "tanh", dtype=torch.float64, soft_labels=True),
    FitCase("single_layer_boundary", 6, 7, 3, 2, (2,), 1e-4, "identity", "identity"),
    FitCase("wide_rank_deficient_float64", 7, 6, 8, 3, (10, 4), 5e-2, "relu", "sigmoid", dtype=torch.float64, rank_deficient=True),
]

PREDICT_CASES = [
    PredictCase("predict_basic", 20, 6, 4, (5, 3), "tanh"),
    PredictCase("predict_relu_three_layer", 21, 5, 3, (7, 6, 2), "relu"),
    PredictCase("predict_identity_float64", 22, 4, 5, (5, 5), "identity", dtype=torch.float64),
    PredictCase("predict_sigmoid_large_inputs", 23, 7, 4, (4, 3), "sigmoid", x_scale=12.0, weight_scale=0.25),
]


def make_labels(generator: torch.Generator, samples: int, label_dim: int, dtype: torch.dtype, soft: bool) -> torch.Tensor:
    labels = torch.arange(samples) % label_dim
    y = F.one_hot(labels, num_classes=label_dim).to(dtype=dtype)
    if soft:
        noise = torch.rand(samples, label_dim, generator=generator, dtype=dtype)
        noise = noise / noise.sum(dim=1, keepdim=True)
        y = 0.75 * y + 0.25 * noise
    return y


def make_fit_inputs(case: FitCase) -> dict[str, object]:
    generator = torch.Generator().manual_seed(case.seed)
    X = case.x_scale * torch.randn(case.samples, case.input_dim, generator=generator, dtype=case.dtype)
    if case.rank_deficient and case.input_dim >= 3:
        X = X.clone()
        X[:, -1] = 0.5 * X[:, 0] - 0.25 * X[:, 1]

    Y = make_labels(generator, case.samples, case.label_dim, case.dtype, case.soft_labels)
    Qs = []
    Us = []
    in_dim = case.input_dim
    for out_dim in case.layer_dims:
        Qs.append(case.q_scale * torch.randn(in_dim, out_dim, generator=generator, dtype=case.dtype))
        Us.append(case.u_scale * torch.randn(case.label_dim, out_dim, generator=generator, dtype=case.dtype))
        in_dim = out_dim

    return {
        "X": X,
        "Y": Y,
        "Qs": Qs,
        "Us": Us,
        "ridge": case.ridge,
        "activation": case.activation,
        "target_nonlinearity": case.target_nonlinearity,
    }


def make_predict_inputs(case: PredictCase) -> dict[str, object]:
    generator = torch.Generator().manual_seed(case.seed)
    X = case.x_scale * torch.randn(case.samples, case.input_dim, generator=generator, dtype=case.dtype)
    weights = []
    in_dim = case.input_dim
    for out_dim in case.layer_dims:
        weights.append(case.weight_scale * torch.randn(in_dim, out_dim, generator=generator, dtype=case.dtype))
        in_dim = out_dim
    return {"X": X, "weights": weights, "activation": case.activation}
