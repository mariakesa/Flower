# Retry with fewer training steps to reduce execution time
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
import torch

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
# Re-generate Lorenz data and windowed input
data = simulate_lorenz(n_steps=3000)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
X = torch.tensor(data_scaled, dtype=torch.float32)
window_size = 10
X_windows = torch.stack([X[i:i+window_size].reshape(-1) for i in range(len(X)-window_size)])

# Reinitialize model
model = ChaoticFlowerEmbedding(input_dim=30, hidden_dim=64, output_dim=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Shorter training loop
for epoch in range(10):
    model.train()
    embeddings = model(X_windows)
    loss = dummy_loss(embeddings)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get embeddings and visualize
model.eval()
with torch.no_grad():
    final_embeddings = model(X_windows).numpy()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(final_embeddings[:, 0], final_embeddings[:, 1], final_embeddings[:, 2], lw=0.7)
ax.set_title("ChaoticFlower Embedding of Lorenz Attractor")
ax.set_xlabel("Z₁")
ax.set_ylabel("Z₂")
ax.set_zlabel("Z₃")
plt.tight_layout()
plt.show()
