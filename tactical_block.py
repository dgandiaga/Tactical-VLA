import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class FeedForward(nn.Module):
    """ The 'Thinking' part of the transformer """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # Expansion
            nn.GELU(),                     # Modern ReLU (smoother gradients)
            nn.Linear(4 * n_embd, n_embd), # Projection back
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ A full Transformer Layer: Communication + Computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd) # LayerNorm instead of BatchNorm
        self.attn = MultiHeadAttention(n_embd, n_head) # Custom MHA from multi_head_attention.py
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # Apply residual connections
        # Unlike the original 2017 attention paper (which did Norm after the addition)
        # modern SOTA like Qwen and Llama put the Norm before the MHA and MLP
        x = x + self.attn(self.ln1(x))
        
        # Computation (MLP) + Residual
        x = x + self.ffwd(self.ln2(x))
        return x

# --- Shape Audit ---
n_embd, n_head = 128, 8
block = Block(n_embd, n_head)
x = torch.randn(1, 32, 128)
y = block(x)
print(f"Output shape is preserved: {y.shape}") # (1, 32, 128)