import torch
import torch.optim as optim
import torch.nn as nn
from tactical_VLA import TacticalVLA

# 1. Setup the Model 
n_embd, n_head, n_layers, vision_dim = 128, 8, 4, 1152
model = TacticalVLA(n_embd, n_head, n_layers, vision_dim)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss() # Standard for continuous regression

# 2. Simulate a "Mission" (100 steps of flight)
# Input: (Batch, Time, Vision_Dim)
# Target: (Batch, 6-DOF)
fake_vision_stream = torch.randn(100, 32, 1152) 
fake_expert_actions = torch.tanh(torch.randn(100, 6)) # Expert's "flight commands"

print("--- Starting Architecture Smoke Test ---")

for epoch in range(20):
    total_loss = 0
    for i in range(len(fake_vision_stream)):
        # Get current window of 32 frames
        x = fake_vision_stream[i].unsqueeze(0) 
        target = fake_expert_actions[i].unsqueeze(0)
        
        # Forward Pass
        prediction = model(x)
        loss = criterion(prediction, target)
        
        # Backward Pass (The Physics of Gradients)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch == 0:
        initial_loss = total_loss
    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/100:.6f}")

print(f"Loss: {initial_loss/100:.6f} -> {total_loss/100:.6f}")
if initial_loss/2 > total_loss:
    print("--- Smoke Test Complete: Architecture is Mathematically Sound ---")