# Deep-Thinking Ratio

Implement the Deep-Thinking Ratio (DTR) algorithm from the paper
`DTR_Paper.pdf` in a single Python file:

```text
/solution/dtr.py
```

Your file must define:

```python
def compute_dtr(
    hidden_states: list,
    unembedding_matrix,
    threshold: float = 0.01,
) -> float:
    ...
```

`hidden_states` is a list of PyTorch tensors, one per transformer layer. Each
tensor has shape `(seq_len, hidden_dim)`. `unembedding_matrix` is a PyTorch
tensor with shape `(vocab_size, hidden_dim)`.

Return a Python float in `[0.0, 1.0]` representing the fraction of token
positions classified as deep-thinking tokens.

The algorithm is:

1. For each layer, apply the logit lens:
   `logits = hidden_state @ unembedding_matrix.T`, then `softmax` over the
   vocabulary dimension.
2. Compute Jensen-Shannon divergence between the probability distributions for
   every pair of adjacent layers, independently for every token position. Use
   natural logarithms.
3. Treat the late regime as the last 25% of layer transitions:
   `max(1, floor((num_layers - 1) * 0.25))`.
4. A token is deep-thinking if its maximum JSD over the late regime is strictly
   greater than `threshold`.
5. Return `deep_thinking_token_count / seq_len`.

Constraints:

- Use only PyTorch and the Python standard library for tensor/math operations.
- Inputs are small; prioritize a clear CPU implementation over vectorized
  performance.
- Do not use scipy, sklearn, transformer libraries, downloaded models, or real
  model weights.
- Do not require CUDA.
- Keep the implementation importable with `from dtr import compute_dtr`.
