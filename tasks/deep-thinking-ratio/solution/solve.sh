#!/bin/bash
set -euo pipefail

mkdir -p /solution

cat > /solution/dtr.py <<'PY'
import math

import torch
import torch.nn.functional as F


def compute_dtr(
    hidden_states: list[torch.Tensor],
    unembedding_matrix: torch.Tensor,
    threshold: float = 0.01,
    depth_fraction: float = 0.25,
) -> float:
    """
    Computes the Deep-Thinking Ratio (DTR) for an entire sequence simultaneously
    using vectorized PyTorch operations.
    """
    L = len(hidden_states)
    if L == 0:
        return 0.0

    seq_len = hidden_states[0].shape[0]
    if seq_len == 0:
        return 0.0

    stacked_hiddens = torch.stack(hidden_states)
    all_logits = torch.matmul(stacked_hiddens, unembedding_matrix.T)
    all_probs = F.softmax(all_logits, dim=-1)

    p_L = all_probs[-1].unsqueeze(0)
    p_L_expanded = p_L.expand_as(all_probs)
    m = 0.5 * (p_L + all_probs)
    log_m = m.clamp(min=1e-10).log()

    kl_pL = F.kl_div(log_m, p_L_expanded, reduction="none").sum(dim=-1)
    kl_pl = F.kl_div(log_m, all_probs, reduction="none").sum(dim=-1)
    jsd = 0.5 * (kl_pL + kl_pl)

    cummin_jsd, _ = torch.cummin(jsd, dim=0)
    met_threshold = cummin_jsd <= threshold
    layer_indices = torch.arange(1, L + 1, device=jsd.device).unsqueeze(1)
    valid_indices = torch.where(
        met_threshold,
        layer_indices,
        torch.full_like(layer_indices, L),
    )
    c_t, _ = valid_indices.min(dim=0)

    depth_thresh = math.ceil((1.0 - depth_fraction) * L)
    deep_thinking_token_count = (c_t >= depth_thresh).sum().item()
    return deep_thinking_token_count / seq_len
PY

python - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location("dtr", "/solution/dtr.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert hasattr(module, "compute_dtr")
PY
