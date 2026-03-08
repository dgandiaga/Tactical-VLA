import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 16 
block_size = 32 # Context window (flight history)
n_embd = 64     # Embedding dimension
head_size = 16

class TacticalAttention(nn.Module):
    """ Single head attention implementation """
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # The Mask: A physical constraint ensuring causality
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # Calculate Affinity Scores
        # This is the 'coupling' between time-steps
        wei = q @ k.transpose(-2, -1) * head_size**-0.5  # Scaled Dot-Product to avoid softmax saturation
        
        # Apply the autoregressive mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # Probabilistic distribution
        
        # Weighted Aggregation
        v = self.value(x)
        out = wei @ v
        return out

# --- Shape Audit ---
x = torch.randn(batch_size, block_size, n_embd)
head = TacticalAttention()
out = head(x)
print(f"Input shape: {x.shape}")
print(f"Output shape (Contextualized): {out.shape}")