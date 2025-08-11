# Re-run the code after state reset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
# Reversible Activation Layer with tanh normalization
class ReversibleFlowerLayer(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=4, is_first=False, omega_init_range=(0.5, 10.0)):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first

        low, high = omega_init_range
        self.omega = nn.Parameter(
            torch.linspace(low, high, steps=num_frequencies).view(1, 1, -1)
        )

        self._init_weights()
        self.out_features = out_features
        self.num_frequencies = num_frequencies

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = 6 / self.linear.in_features
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        z = self.linear(x).unsqueeze(-1)  # [B, out_features, 1]

        # Normalize z across batch
        z_mean = z.mean(dim=0, keepdim=True)
        z_std = z.std(dim=0, keepdim=True) + 1e-6
        z = (z - z_mean) / z_std

        tanh_scaled = torch.tanh(self.omega * z)
        safe_input = torch.clamp(tanh_scaled, min=-0.999999, max=0.999999)
        out = torch.arcsin(safe_input)
        return out.view(x.shape[0], -1)


# ─────────────────────────────────────────────────────────────────────────────
# Reversible Flower Encoder
class ReversibleFlowerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=3, num_frequencies=4, omega_init_range=(0.5, 10.0)):
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            is_first = i == 0
            layer = ReversibleFlowerLayer(
                in_features=in_dim,
                out_features=hidden_dim,
                num_frequencies=num_frequencies,
                is_first=is_first,
                omega_init_range=omega_init_range
            )
            layers.append(layer)
            in_dim = hidden_dim * num_frequencies  # update for next layer

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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



# ─────────────────────────────────────────────────────────────────────────────
# Training Function
def train_reversible_flower(
    data,
    input_dim,
    hidden_dim,
    output_dim,
    num_layers=3,
    num_frequencies=4,
    omega_init_range=(1.0, 60.0),
    epochs=1000,
    batch_size=128,
    learning_rate=1e-3,
    model_path="reversible_flower.pt"
):
    encoder = ReversibleFlowerEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_frequencies=num_frequencies,
        omega_init_range=omega_init_range
    )

    sampler = TimeContrastiveSampler(data, window_size=5, batch_size=batch_size)
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        ref, pos, neg = sampler.sample()
        z_ref = encoder(ref)
        z_pos = encoder(pos)
        z_neg = encoder(neg)

        loss = criterion(z_ref, z_pos, z_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[ReversibleFlower] Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(encoder.state_dict(), model_path)
    print(f"Reversible Flower model saved to {model_path}")
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# 3D Embedding Plot with Rotation
def animate_embedding_3d_reversible_flower(data, encoder_path, input_dim, hidden_dim, output_dim,
                                           num_layers=3, num_frequencies=6,
                                           omega_init_range=(1.0, 60.0),
                                           title="Reversible Flower embedding",
                                           fps=20, save_path='reversible_flower_embedding.mp4'):
    encoder = ReversibleFlowerEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_frequencies=num_frequencies,
        omega_init_range=omega_init_range
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(torch.tensor(data, dtype=torch.float32))

    if output_dim > 3:
        embeddings = PCA(n_components=3).fit_transform(embeddings.numpy())
    else:
        embeddings = embeddings.numpy()

    points = embeddings.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(0, len(segments))
    colors = cm.rainbow(norm(np.arange(len(segments))))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Latent 1", fontsize=12)
    ax.set_ylabel("Latent 2", fontsize=12)
    ax.set_zlabel("Latent 3", fontsize=12)

    lc = Line3DCollection(segments, colors=colors, linewidth=2)
    ax.add_collection3d(lc)
    ax.set_xlim(embeddings[:, 0].min(), embeddings[:, 0].max())
    ax.set_ylim(embeddings[:, 1].min(), embeddings[:, 1].max())
    ax.set_zlim(embeddings[:, 2].min(), embeddings[:, 2].max())

    def update(angle):
        ax.view_init(elev=30, azim=angle)
        return fig,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False, interval=1000/fps)

    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"Saved rotating embedding animation to {save_path}")
    else:
        plt.close(fig)
        return ani