# Re-run after environment reset

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Mock loading since file path is inaccessible here
# Replace this with your actual file load:
# lfp_data = np.load('/home/maria/Flower/data/lfp_signal.npy')
# For demo, simulate it:
lfp_data = np.random.randn(31250, 95)[:5000]  # shape (time, channels)
lfp_data=np.load('/home/maria/Flower/data/lfp_signal.npy')[:5000][:,:10] 

# Normalize LFP data
scaler = MinMaxScaler()
lfp_data_scaled = scaler.fit_transform(lfp_data)

# Convert to torch tensor
lfp_tensor = torch.tensor(lfp_data_scaled, dtype=torch.float32)

# Create sliding windows
window_size = 10
lfp_windows = torch.stack([
    lfp_tensor[i:i + window_size].reshape(-1) for i in range(len(lfp_tensor) - window_size)
])  # shape: (31240, 950)

# Show shape
lfp_windows.shape

# Define a smaller ChaoticFlower network for large LFP input
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LFPChaoticFlowerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, iter_k=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.r_unbounded = nn.Parameter(torch.tensor(0.0))  # learnable r
        self.k = iter_k
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def r(self):
        # Bounds r between 3.5 and 4.0
        return 3.5 + 0.5 * torch.sigmoid(self.r_unbounded)

    def logistic_map(self, x):
        for _ in range(self.k):
            x = self.r() * x * (1 - x)
        return x

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))  # squash to [0, 1]
        x = self.logistic_map(x)           # apply iterated map
        return self.output_layer(x)

# Time-contrastive loss function
def time_contrastive_loss(z, tau_pos=1, tau_neg=100):
    pos_loss = ((z[:-tau_pos] - z[tau_pos:]) ** 2).mean()
    neg_loss = -(z[:-tau_neg] * z[tau_neg:]).mean()
    return pos_loss + neg_loss

# Initialize model and optimizer
input_dim = lfp_windows.shape[1]
model = LFPChaoticFlowerNet(input_dim=input_dim, hidden_dim=64, output_dim=3, iter_k=8)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train briefly
for epoch in range(100):
    model.train()
    embeddings = model(lfp_windows)
    loss = time_contrastive_loss(embeddings)
    optimizer.zero_grad()
    loss.backward()
    #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent explosion
    optimizer.step()
    print(f"Epoch {epoch+1}: loss = {loss.item():.5f}, r = {model.r().item():.4f}")

# Final embeddings
model.eval()
with torch.no_grad():
    lfp_embeddings = model(lfp_windows).numpy()

# Plot the 3D embedding
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(lfp_embeddings[:, 0], lfp_embeddings[:, 1], lfp_embeddings[:, 2], lw=0.5)
ax.set_title("ChaoticFlower Embedding of LFP Signal")
ax.set_xlabel("Z₁")
ax.set_ylabel("Z₂")
ax.set_zlabel("Z₃")
plt.tight_layout()
plt.show()


