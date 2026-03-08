import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, max_seq_len=2048):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Single projection for efficiency: maps input to Q, K, and V
        self.qkv_projection = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.output_projection = nn.Linear(n_embd, n_embd)
        
        # Autoregressive mask
        # Pre-compute the mask for the MAXIMUM expected mission length, slice it later
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))
        
    def forward(self, x):
        B, T, C = x.shape # Batch, Sequence Length, Embedding Dim
        
        # Linear Projection to Q, K, V
        qkv = self.qkv_projection(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)  # (B, T, C) each
        
        # Head Split Reshape
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # Interaction: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        
        # Apply Causality Constrain
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # Compute attention weights as a probability distribution
        att = F.softmax(att, dim=-1)
        
        # Weighted sum over the values
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ v
        
        # Concatenate & Final Projection
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.output_projection(out)

# --- Shape Audit ---
model = MultiHeadAttention(n_embd=128, n_head=8)
x = torch.randn(1, 16, 128) # 1 drone mission, 16 state steps, 128-dim state
y = model(x)
print(f"Input Shape:  {x.shape}")
print(f"Output Shape: {y.shape}")