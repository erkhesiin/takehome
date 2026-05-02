from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DTRCase:
    name: str
    seed: int
    layers: int
    seq_len: int
    hidden_dim: int
    vocab_size: int
    threshold: float
    deep_positions: tuple[int, ...]
    perturb_transition_offsets: tuple[int, ...] = (0,)


CASES = [
    DTRCase("all_deep_thinking", 0, 8, 10, 32, 64, 0.01, tuple(range(10))),
    DTRCase("no_deep_thinking", 1, 8, 10, 32, 64, 0.01, ()),
    DTRCase("half_deep_thinking", 2, 8, 10, 32, 64, 0.01, tuple(range(0, 10, 2))),
    DTRCase("multilayer_late_divergence", 3, 12, 20, 64, 128, 0.01, tuple(range(5, 15)), (0, 1)),
    DTRCase("minimal_edge_case", 4, 2, 1, 16, 32, 0.01, (0,)),
    DTRCase("high_threshold", 5, 8, 15, 32, 64, 100.0, tuple(range(15))),
    DTRCase("realistic_partial", 6, 16, 30, 128, 256, 0.01, tuple(range(10, 20)), (0, 2, 3)),
]


def make_case(case: DTRCase) -> tuple[list[torch.Tensor], torch.Tensor, float]:
    torch.manual_seed(case.seed)
    base = torch.randn(case.seq_len, case.hidden_dim) * 0.05
    hidden_states = [base.clone() for _ in range(case.layers)]
    unembedding = torch.randn(case.vocab_size, case.hidden_dim)

    if case.deep_positions:
        direction = torch.randn(case.hidden_dim)
        direction = direction / direction.norm()
        n_transitions = case.layers - 1
        n_late = max(1, int(n_transitions * 0.25))
        late_start_layer = case.layers - n_late

        for offset in case.perturb_transition_offsets:
            layer_idx = min(case.layers - 1, late_start_layer + offset)
            for token_idx in case.deep_positions:
                hidden_states[layer_idx][token_idx] += direction * 8.0

    return hidden_states, unembedding, case.threshold
