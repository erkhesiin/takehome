# Deep-Thinking Ratio (DTR)

Implement the Deep-Thinking Ratio (DTR) algorithm from the paper `DTR_Paper.pdf` in a single Python file:

```text
/solution/dtr.py
```

Your file must define the following function:

```python
import torch

def compute_dtr(
    hidden_states: list[torch.Tensor],
    unembedding_matrix: torch.Tensor,
    threshold: float = 0.01,
    depth_fraction: float = 0.25
) -> float:
    ...
```

### Inputs
* `hidden_states`: A list of PyTorch tensors, one for each transformer layer ($L$ total layers). Each tensor has shape `(seq_len, hidden_dim)`. The last tensor in the list represents the final layer.
* `unembedding_matrix`: A PyTorch tensor with shape `(vocab_size, hidden_dim)`.
* `threshold`: The Jensen-Shannon Divergence (JSD) threshold ($g$).
* `depth_fraction`: The fraction of layers considered "deep" ($\rho$).

### The Algorithm
You are computing the ratio of generated tokens that require "deep thinking." For every token position $t$ in the sequence (`seq_len`), perform the following steps independently:

1. **Logit Lens & Final Distribution:** Apply the unembedding matrix to the hidden states of the **final layer** $L$ to get the logits: `logits = hidden_state @ unembedding_matrix.T`. Apply a `softmax` over the vocabulary dimension to get the target probability distribution $p_{t,L}$.
2. **Intermediate Distributions & JSD:**
   For every layer $l$ from $1$ to $L$ (including the final layer):
   * Compute the intermediate probability distribution $p_{t,l}$ using the same logit lens and softmax method.
   * Compute the Jensen-Shannon Divergence ($D_{t,l}$) between the final layer's distribution ($p_{t,L}$) and the current layer's distribution ($p_{t,l}$). Use natural logarithms.
3. **Determine the Exit Layer ($c_t$):**
   Find the **earliest** layer $l$ (using 1-based indexing, where the first layer is 1 and the final layer is $L$) where the *cumulative minimum* JSD up to that layer is less than or equal to `threshold`. 
   * Mathematically: $c_t = \min \{ l : \min_{j \le l} D_{t,j} \le \text{threshold} \}$.
   * *Note: If the threshold is never met, default the exit layer to $L$.*
4. **Deep-Thinking Classification:**
   A token is classified as a deep-thinking token if its exit layer $c_t$ happens late in the network, specifically if: 
   $c_t \ge \lceil (1 - \text{depth\_fraction}) \times L \rceil$
5. **Return:**
   Return a Python `float` in `[0.0, 1.0]` representing `deep_thinking_token_count / seq_len`.

### Constraints
- Use only PyTorch and the Python standard library.
- You may use `torch.nn.functional` for JSD math or implement it manually using PyTorch tensor operations.
- Avoid slow, nested Python `for` loops where PyTorch vectorized operations can be used (e.g., computing logits and JSD for the whole sequence at once).
- Do not use scipy, sklearn, transformer libraries, downloaded models, or real model weights.
- Do not require CUDA (ensure it runs on CPU).
- Keep the implementation importable with `from dtr import compute_dtr`.