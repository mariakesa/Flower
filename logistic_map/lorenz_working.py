# Simplify to fit execution time constraints
# Retry with fewer training steps to reduce execution time
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def simulate_lorenz(T=40, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
    t_span = (0, T)
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    return sol.t, sol.y.T  # returns time and trajectory [T, 3]

# 1. Generate shorter Lorenz trajectory
t, data = simulate_lorenz(T=50)  # shorter time
print(data.shape)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
X = torch.tensor(data_scaled, dtype=torch.float32)

# 2. Windowed input
window_size = 10
X_windows = torch.stack([X[i:i+window_size].reshape(-1) for i in range(len(X)-window_size)])

# 3. Smaller model for speed
class SmallChaoticFlower(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, iter_k=7):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.r = nn.Parameter(torch.tensor(3.9))
        self.k = iter_k
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def logistic_map(self, x):
        for _ in range(self.k):
            x = self.r * x * (1 - x)
        return x

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = self.logistic_map(x)
        return self.output_layer(x)

# Reinit model
model = SmallChaoticFlower(input_dim=30, hidden_dim=32, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def time_contrastive_loss(z, tau_pos=1, tau_neg=10):
    """
    z: Tensor of shape [T, D], where T is time, D is embedding dim.
    tau_pos: temporal shift for positive pairs (usually small, e.g. 1)
    tau_neg: temporal shift for negative pairs (usually large, e.g. 10)

    Returns: scalar loss combining positive closeness + negative repulsion
    """
    # Positive pair loss: embeddings close in time should be close in space
    pos_loss = ((z[:-tau_pos] - z[tau_pos:])**2).mean()

    # Negative pair loss: embeddings far in time should be pushed apart
    neg_loss = -(z[:-tau_neg] * z[tau_neg:]).mean()

    return pos_loss + neg_loss

# 4. Short training
for epoch in range(2):
    model.train()
    embeddings = model(X_windows)
    loss = time_contrastive_loss(embeddings)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. Visualize embedding
model.eval()
with torch.no_grad():
    final_embeddings = model(X_windows).numpy()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(final_embeddings[:, 0], final_embeddings[:, 1], final_embeddings[:, 2], lw=0.7)
ax.set_title("ChaoticFlower Embedding of Lorenz Attractor (Tiny Prototype)")
ax.set_xlabel("Z₁")
ax.set_ylabel("Z₂")
ax.set_zlabel("Z₃")
plt.tight_layout()
plt.show()
