import importlib.util
import sys
from pathlib import Path

import pytest

from dtr_reference import compute_dtr as reference_compute_dtr
from test_cases import CASES, make_case

try:
    import torch

    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
except Exception:
    pass


def load_agent_compute_dtr():
    path = Path("/solution/dtr.py")
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / "solution" / "dtr.py"
    assert path.exists(), "Expected implementation at /solution/dtr.py"
    spec = importlib.util.spec_from_file_location("agent_dtr", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_dtr"] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "compute_dtr"), "dtr.py must define compute_dtr"
    return module.compute_dtr


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_case_matches_reference(case):
    compute_dtr = load_agent_compute_dtr()
    hidden_states, unembedding, threshold = make_case(case)

    expected = reference_compute_dtr(hidden_states, unembedding, threshold)
    actual = compute_dtr(hidden_states, unembedding, threshold)

    assert isinstance(actual, float), "compute_dtr must return a Python float"
    assert 0.0 <= actual <= 1.0
    assert abs(actual - expected) <= 0.01
