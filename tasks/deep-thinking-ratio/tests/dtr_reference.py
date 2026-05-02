import math

import torch


def compute_dtr(
    hidden_states: list[torch.Tensor],
    unembedding_matrix: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    if len(hidden_states) < 2:
        raise ValueError("DTR requires at least two layers")

    probs = [
        torch.softmax(hidden @ unembedding_matrix.T, dim=-1)
        for hidden in hidden_states
    ]

    jsd_values = []
    eps = 1e-10
    for p, q in zip(probs[:-1], probs[1:]):
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)), dim=-1)
        kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)), dim=-1)
        jsd_values.append(0.5 * kl_pm + 0.5 * kl_qm)

    jsd_matrix = torch.stack(jsd_values, dim=0)
    n_late = max(1, math.floor(jsd_matrix.shape[0] * 0.25))
    max_late_jsd = jsd_matrix[-n_late:].max(dim=0).values
    return float((max_late_jsd > threshold).float().mean().item())
