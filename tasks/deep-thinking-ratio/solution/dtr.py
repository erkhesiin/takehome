import math

import torch


def compute_dtr(
    hidden_states: list[torch.Tensor],
    unembedding_matrix: torch.Tensor,
    threshold: float = 0.01,
) -> float:
    if len(hidden_states) < 2:
        raise ValueError("DTR requires at least two layers")

    probabilities = []
    for hidden in hidden_states:
        logits = hidden @ unembedding_matrix.T
        probabilities.append(torch.softmax(logits, dim=-1))

    eps = 1e-10
    jsd_by_transition = []
    for p, q in zip(probabilities[:-1], probabilities[1:]):
        midpoint = 0.5 * (p + q)
        kl_p = torch.sum(p * (torch.log(p + eps) - torch.log(midpoint + eps)), dim=-1)
        kl_q = torch.sum(q * (torch.log(q + eps) - torch.log(midpoint + eps)), dim=-1)
        jsd_by_transition.append(0.5 * kl_p + 0.5 * kl_q)

    jsd_matrix = torch.stack(jsd_by_transition, dim=0)
    n_late = max(1, math.floor((len(hidden_states) - 1) * 0.25))
    max_late_jsd = jsd_matrix[-n_late:].max(dim=0).values
    return float((max_late_jsd > threshold).float().mean().item())
