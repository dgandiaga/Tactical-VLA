import torch
import torch.nn as nn
from tactical_block import Block

class TacticalVLA(nn.Module):
    def __init__(self, n_embd, n_head, n_layers, vision_dim):
        super().__init__()
        # Vision Bridge: Map raw sensor features to model dimension
        self.vision_proj = nn.Linear(vision_dim, n_embd)
        
        # Transformer Backbone: Stacking the Blocks from tactical_block.py
        # Each block has: MHA + Residual + FFN + Residual
        self.backbone = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layers)]
        )
        
        # 3. Final Layer Norm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # 4. Action Head: Tanh-constrained 6-DOF control
        self.action_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.GELU(),
            nn.Linear(n_embd // 2, 6),
            nn.Tanh() 
        )

    def forward(self, vision_features):
        # vision_features: (B, T, vision_dim)
        
        # Project to latent space
        x = self.vision_proj(vision_features)
        
        # Pass through the stack of Transformer Blocks
        x = self.backbone(x)
        
        # Use only the last token for the current action prediction
        x = self.ln_f(x[:, -1, :])
        
        # Output flight deltas
        return self.action_head(x)
    
# --- Shape Audit ---
vla = TacticalVLA(n_embd=128, n_head=8, n_layers=4, vision_dim=1152)
dummy_sensor = torch.randn(1, 32, 1152) # 32 timesteps of sensor patches
actions = vla(dummy_sensor)

print(f"Backbone layers: {len(vla.backbone)}")
print(f"Final Action Shape: {actions.shape}") # (1, 6)