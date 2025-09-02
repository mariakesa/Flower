# # Re-import everything after state reset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Time-Contrastive Sampler and Loss
class TimeContrastiveSampler:
    def __init__(self, data, window_size=5, batch_size=128):
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size
        self.T = data.shape[0]

    def sample(self):
        indices = np.random.randint(self.window_size, self.T - self.window_size, size=self.batch_size)
        ref = self.data[indices]
        pos = self.data[indices + np.random.randint(1, self.window_size + 1)]
        neg_indices = np.random.randint(0, self.T, size=self.batch_size)
        neg = self.data[neg_indices]
        return torch.tensor(ref, dtype=torch.float32), torch.tensor(pos, dtype=torch.float32), torch.tensor(neg, dtype=torch.float32)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_ref, z_pos, z_neg):
        z_ref = F.normalize(z_ref, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        z_neg = F.normalize(z_neg, dim=1)

        pos_sim = torch.sum(z_ref * z_pos, dim=1) / self.temperature
        neg_sim = torch.sum(z_ref * z_neg, dim=1) / self.temperature

        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(z_ref.size(0), dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# FlowerNet Definition
class LFPChaoticFlowerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, iter_k=5):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.r_unbounded = nn.Parameter(torch.tensor(4.0).logit())  # bounded r
        self.k = iter_k
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def r(self):
        return 3.5 + 0.5 * torch.sigmoid(self.r_unbounded)

    def logistic_map(self, x):
        for _ in range(self.k):
            x = torch.clamp(x, 0.0001, 0.9999)
            x = self.r() * x * (1 - x)
        return x

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = self.norm(x)
        x = self.logistic_map(x)
        return self.output_layer(x)

# ─────────────────────────────────────────────────────────────────────────────
# Time-Contrastive Sampler and Loss
class TimeContrastiveSampler:
    def __init__(self, data, window_size=5, batch_size=128):
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size
        self.T = data.shape[0]

    def sample(self):
        indices = np.random.randint(self.window_size, self.T - self.window_size, size=self.batch_size)
        ref = self.data[indices]
        pos = self.data[indices + np.random.randint(1, self.window_size + 1)]
        neg_indices = np.random.randint(0, self.T, size=self.batch_size)
        neg = self.data[neg_indices]
        return torch.tensor(ref, dtype=torch.float32), torch.tensor(pos, dtype=torch.float32), torch.tensor(neg, dtype=torch.float32)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_ref, z_pos, z_neg):
        z_ref = F.normalize(z_ref, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        z_neg = F.normalize(z_neg, dim=1)

        pos_sim = torch.sum(z_ref * z_pos, dim=1) / self.temperature
        neg_sim = torch.sum(z_ref * z_neg, dim=1) / self.temperature

        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(z_ref.size(0), dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        return loss

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# Mock loading since file path is inaccessible here
# Replace this with your actual file load:
# lfp_data = np.load('/home/maria/Flower/data/lfp_signal.npy')
# For demo, simulate it:
lfp_data = np.random.randn(31250, 95)[:5000]  # shape (time, channels)
lfp_data=np.load('/home/maria/Flower/data/lfp_signal.npy')[:1000]  

# Normalize LFP data
scaler = MinMaxScaler()
lfp_data_scaled = scaler.fit_transform(lfp_data)

# Convert to torch tensor
lfp_tensor = torch.tensor(lfp_data_scaled, dtype=torch.float32)
# ─────────────────────────────────────────────────────────────────────────────
# Create LFP windows
window_len = 10
lfp_tensor = torch.tensor(lfp_scaled, dtype=torch.float32)
lfp_windows = torch.stack([
    lfp_tensor[i:i+window_len].reshape(-1) for i in range(len(lfp_tensor) - window_len)
])  # shape: (31240, 950)

# ─────────────────────────────────────────────────────────────────────────────
# Training
sampler = TimeContrastiveSampler(data=lfp_windows.numpy(), window_size=10, batch_size=256)
model = LFPChaoticFlowerNet(input_dim=950, hidden_dim=64, output_dim=3, iter_k=5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
contrastive_loss = ContrastiveLoss(temperature=0.1)

logs = []
for epoch in range(100):
    model.train()
    ref, pos, neg = sampler.sample()
    z_ref = model(ref)
    z_pos = model(pos)
    z_neg = model(neg)

    loss = contrastive_loss(z_ref, z_pos, z_neg)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(loss)
    logs.append((epoch+1, loss.item(), model.r().item()))

import matplotlib.pyplot as plt
# 5. Visualize embedding
model.eval()
with torch.no_grad():
    final_embeddings = model(lfp_windows).numpy()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(final_embeddings[:, 0], final_embeddings[:, 1], final_embeddings[:, 2], lw=0.7)
ax.set_title("ChaoticFlower Embedding of Lorenz Attractor (Tiny Prototype)")
ax.set_xlabel("Z₁")
ax.set_ylabel("Z₂")
ax.set_zlabel("Z₃")
plt.tight_layout()
plt.show()