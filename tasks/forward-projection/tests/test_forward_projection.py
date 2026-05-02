import importlib.util
from pathlib import Path

import pytest
import torch

from fp_reference import (
    fit_forward_projection as fit_reference,
    forward_projection_predict as predict_reference,
)
from test_cases import FIT_CASES, PREDICT_CASES, make_fit_inputs, make_predict_inputs


def load_agent_module():
    path = Path("/solution/forward_projection.py")
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "solution" / "forward_projection.py"
    assert path.exists(), "Expected implementation at /solution/forward_projection.py"

    spec = importlib.util.spec_from_file_location("agent_forward_projection", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "fit_forward_projection"), "Expected top-level fit_forward_projection function"
    assert hasattr(module, "forward_projection_predict"), "Expected top-level forward_projection_predict function"
    return module


def _clone_nested(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    return value


def _assert_nested_unchanged(current, original, name):
    if isinstance(original, torch.Tensor):
        torch.testing.assert_close(current, original, rtol=0, atol=0, msg=f"Input {name} was mutated")
    elif isinstance(original, list):
        assert len(current) == len(original)
        for idx, (cur_item, orig_item) in enumerate(zip(current, original)):
            _assert_nested_unchanged(cur_item, orig_item, f"{name}[{idx}]")


def _assert_tensor_lists_close(actual, expected, rtol, atol, label):
    assert isinstance(actual, list), f"{label} must be a list"
    assert len(actual) == len(expected), f"{label} has the wrong length"
    for idx, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        assert isinstance(actual_tensor, torch.Tensor), f"{label}[{idx}] must be a tensor"
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype
        assert torch.isfinite(actual_tensor).all()
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", FIT_CASES, ids=[case.name for case in FIT_CASES])
def test_fit_forward_projection_matches_reference(case):
    module = load_agent_module()
    kwargs = make_fit_inputs(case)
    originals = {key: _clone_nested(value) for key, value in kwargs.items() if key in {"X", "Y", "Qs", "Us"}}

    actual = module.fit_forward_projection(**kwargs)
    expected = fit_reference(**kwargs)

    assert isinstance(actual, dict), "fit_forward_projection must return a dict"
    assert set(actual.keys()) == {"weights", "membranes", "activations"}

    rtol, atol = (1e-8, 1e-10) if case.dtype is torch.float64 else (1e-4, 1e-5)
    _assert_tensor_lists_close(actual["weights"], expected["weights"], rtol, atol, "weights")
    _assert_tensor_lists_close(actual["membranes"], expected["membranes"], rtol, atol, "membranes")
    _assert_tensor_lists_close(actual["activations"], expected["activations"], rtol, atol, "activations")

    assert len(actual["weights"]) == len(case.layer_dims)
    assert len(actual["membranes"]) == len(case.layer_dims)
    assert len(actual["activations"]) == len(case.layer_dims) + 1

    for key, original in originals.items():
        _assert_nested_unchanged(kwargs[key], original, key)


@pytest.mark.parametrize("case", PREDICT_CASES, ids=[case.name for case in PREDICT_CASES])
def test_forward_projection_predict_matches_reference(case):
    module = load_agent_module()
    kwargs = make_predict_inputs(case)
    originals = {key: _clone_nested(value) for key, value in kwargs.items() if key in {"X", "weights"}}

    actual = module.forward_projection_predict(**kwargs)
    expected = predict_reference(**kwargs)

    assert isinstance(actual, tuple), "forward_projection_predict must return (membranes, activations)"
    assert len(actual) == 2
    actual_membranes, actual_activations = actual
    expected_membranes, expected_activations = expected

    rtol, atol = (1e-8, 1e-10) if case.dtype is torch.float64 else (1e-5, 1e-6)
    _assert_tensor_lists_close(actual_membranes, expected_membranes, rtol, atol, "predict membranes")
    _assert_tensor_lists_close(actual_activations, expected_activations, rtol, atol, "predict activations")

    assert len(actual_membranes) == len(case.layer_dims)
    assert len(actual_activations) == len(case.layer_dims) + 1

    for key, original in originals.items():
        _assert_nested_unchanged(kwargs[key], original, key)
