import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Lorenz System Dynamics
def lorenz(t, state, sigma=10., rho=28., beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# 2. Embed input by running Lorenz dynamics
def chaotic_embed(x0_batch, T=2.0, steps=200):
    final_states = []
    for x0 in x0_batch:
        sol = solve_ivp(lorenz, [0, T], x0.numpy(), t_eval=[T])
        final_states.append(torch.tensor(sol.y[:, -1], dtype=torch.float32))
    return torch.stack(final_states)

# 3. Simple Decoder (Inverse Model)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.net(x)

# 4. Train the inverse
def train_inverse_model(n_epochs=1000, batch_size=128, device='cpu'):
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        # Sample random initial points
        x0_batch = torch.rand(batch_size, 3) * 20 - 10  # Uniform in [-10, 10]^3
        x0_batch = x0_batch.to(device)

        with torch.no_grad():
            z_batch = chaotic_embed(x0_batch)

        x_recon = decoder(z_batch.to(device))
        loss = loss_fn(x_recon, x0_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    return decoder

# 5. Visualize
def visualize(decoder, device='cpu'):
    x0_batch = torch.rand(1000, 3) * 20 - 10
    z_batch = chaotic_embed(x0_batch)
    x_recon = decoder(z_batch.to(device)).cpu().detach()

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*x0_batch.T, alpha=0.4, label="Original")
    ax.set_title("Original x0")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(*x_recon.T, alpha=0.4, label="Reconstructed", color='r')
    ax2.set_title("Reconstructed x0")

    plt.show()

# Run everything
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = train_inverse_model(device=device)
visualize(decoder, device=device)
