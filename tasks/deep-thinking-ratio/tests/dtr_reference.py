import math
import torch
import torch.nn.functional as F

def compute_dtr(
    hidden_states: list[torch.Tensor],
    unembedding_matrix: torch.Tensor,
    threshold: float = 0.01,
    depth_fraction: float = 0.25
) -> float:
    """
    Computes the Deep-Thinking Ratio (DTR) for an entire sequence simultaneously 
    using vectorized PyTorch operations.
    """
    L_hs = len(hidden_states)
    if L_hs == 0:
        return 0.0
        
    seq_len = hidden_states[0].shape[0]
    if seq_len == 0:
        return 0.0

    # Stack hidden states: Shape -> (L_hs, seq_len, hidden_dim)
    stacked_hiddens = torch.stack(hidden_states)
    
    # 1. Logit Lens & Final Distribution
    # Compute all logits at once: Shape -> (L_hs, seq_len, vocab_size)
    all_logits = torch.matmul(stacked_hiddens, unembedding_matrix.T)
    
    # Compute probabilities for all layers: Shape -> (L_hs, seq_len, vocab_size)
    all_probs = F.softmax(all_logits, dim=-1)
    
    # Extract the final layer's probabilities and expand to match shape
    # Shape -> (1, seq_len, vocab_size)
    p_L = all_probs[-1].unsqueeze(0) 

    # 2. Intermediate Distributions & JSD
    # m = 0.5 * (p_L + p_l)
    m = 0.5 * (p_L + all_probs)
    
    # Clamp to avoid log(0)
    log_m = m.clamp(min=1e-10).log()
    
    # PyTorch F.kl_div(input, target) computes: target * (log(target) - input)
    # So by passing input=log_m and target=p, it natively calculates p * log(p/m)
    kl_pL = F.kl_div(log_m, p_L, reduction='none').sum(dim=-1)       # D_KL(p_L || m)
    kl_pl = F.kl_div(log_m, all_probs, reduction='none').sum(dim=-1) # D_KL(p_l || m)
    
    # JSD Shape -> (L, seq_len)
    jsd = 0.5 * (kl_pL + kl_pl)
    
    # 3. Determine the Exit Layer (c_t)
    # Compute the cumulative minimum of JSD across the layer dimension (dim=0)
    cummin_jsd, _ = torch.cummin(jsd, dim=0)
    
    # Create a boolean mask of where the threshold condition is met
    # Shape -> (L_hs, seq_len)
    met_threshold = cummin_jsd <= threshold
    
    # Create layer indices (1 to L_hs) and reshape to broadcast: Shape -> (L_hs, 1)
    layer_indices = torch.arange(1, L_hs + 1, device=jsd.device).unsqueeze(1)
    
    # If a layer meets the threshold, use its index. Otherwise, default to L.
    valid_indices = torch.where(met_threshold, layer_indices, torch.tensor(L_hs, device=jsd.device))
    
    # Get the minimum valid index across the layer dimension: Shape -> (seq_len,)
    c_t, _ = valid_indices.min(dim=0)
    
    # 4. Deep-Thinking Classification
    # Calculate the layer integer threshold
    depth_thresh = math.ceil((1.0 - depth_fraction) * L_hs)
    
    # Count how many tokens exited late in the network
    deep_thinking_token_count = (c_t >= depth_thresh).sum().item()
    
    # 5. Return the ratio
    return deep_thinking_token_count / seq_len