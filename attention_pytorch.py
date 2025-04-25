import torch

def attention_pytorch(q, k, v):
    """
    Compute the attention mechanism given input and weights.
    
    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, seq_len, input_dim).
        k (torch.Tensor): Key tensor of shape (batch_size, seq_len, input_dim).
        v (torch.Tensor): Value tensor of shape (batch_size, seq_len, input_dim).
    
    Returns:
        torch.Tensor: Output tensor after applying attention.
    """
    # Compute queries, keys, and values
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output