#!/bin/bash
set -euo pipefail

mkdir -p /solution

cat > /solution/dtr.py <<'PY'
import math


def compute_dtr(
    hidden_states: list,
    unembedding_matrix,
    threshold: float = 0.01,
) -> float:
    if len(hidden_states) < 2:
        raise ValueError("DTR requires at least two layers")

    unembedding = unembedding_matrix.detach().cpu().tolist()
    probabilities = []
    for hidden in hidden_states:
        token_probabilities = []
        for token in hidden.detach().cpu().tolist():
            logits = [
                sum(value * weight for value, weight in zip(token, vocab_row))
                for vocab_row in unembedding
            ]
            max_logit = max(logits)
            exp_logits = [math.exp(logit - max_logit) for logit in logits]
            total = sum(exp_logits)
            token_probabilities.append([value / total for value in exp_logits])
        probabilities.append(token_probabilities)

    eps = 1e-10
    jsd_by_transition = []
    for p, q in zip(probabilities[:-1], probabilities[1:]):
        token_jsds = []
        for p_token, q_token in zip(p, q):
            jsd = 0.0
            for p_value, q_value in zip(p_token, q_token):
                midpoint = 0.5 * (p_value + q_value)
                jsd += 0.5 * p_value * (
                    math.log(p_value + eps) - math.log(midpoint + eps)
                )
                jsd += 0.5 * q_value * (
                    math.log(q_value + eps) - math.log(midpoint + eps)
                )
            token_jsds.append(jsd)
        jsd_by_transition.append(token_jsds)

    n_late = max(1, math.floor((len(hidden_states) - 1) * 0.25))
    late_jsds = jsd_by_transition[-n_late:]
    seq_len = len(late_jsds[0])
    deep_count = 0
    for token_idx in range(seq_len):
        if max(transition[token_idx] for transition in late_jsds) > threshold:
            deep_count += 1
    return deep_count / seq_len
PY

python - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location("dtr", "/solution/dtr.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert hasattr(module, "compute_dtr")
PY
