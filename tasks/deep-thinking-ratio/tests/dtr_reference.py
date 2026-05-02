import math


def compute_dtr(
    hidden_states: list,
    unembedding_matrix,
    threshold: float = 0.01,
) -> float:
    if len(hidden_states) < 2:
        raise ValueError("DTR requires at least two layers")

    unembedding = unembedding_matrix.detach().cpu().tolist()
    layer_probs = []
    for hidden in hidden_states:
        hidden_rows = hidden.detach().cpu().tolist()
        token_probs = []
        for token in hidden_rows:
            logits = [
                sum(value * weight for value, weight in zip(token, vocab_row))
                for vocab_row in unembedding
            ]
            max_logit = max(logits)
            exp_logits = [math.exp(logit - max_logit) for logit in logits]
            total = sum(exp_logits)
            token_probs.append([value / total for value in exp_logits])
        layer_probs.append(token_probs)

    jsd_by_transition = []
    eps = 1e-10
    for prev_layer, next_layer in zip(layer_probs[:-1], layer_probs[1:]):
        token_jsds = []
        for p, q in zip(prev_layer, next_layer):
            jsd = 0.0
            for p_value, q_value in zip(p, q):
                midpoint = 0.5 * (p_value + q_value)
                jsd += 0.5 * p_value * (
                    math.log(p_value + eps) - math.log(midpoint + eps)
                )
                jsd += 0.5 * q_value * (
                    math.log(q_value + eps) - math.log(midpoint + eps)
                )
            token_jsds.append(jsd)
        jsd_by_transition.append(token_jsds)

    n_late = max(1, math.floor(len(jsd_by_transition) * 0.25))
    late_jsds = jsd_by_transition[-n_late:]
    seq_len = len(late_jsds[0])
    deep_count = 0
    for token_idx in range(seq_len):
        if max(transition[token_idx] for transition in late_jsds) > threshold:
            deep_count += 1
    return deep_count / seq_len
