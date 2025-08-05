import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from IPython.display import HTML
from sklearn.decomposition import PCA

class TimeContrastiveSampler:
    def __init__(self, data, window_size=5, batch_size=128):
        self.data = data  # Shape: [T, D]
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
    
class LearnableFreqSIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=4, is_first=False, omega_init_range=(1.0, 60.0)):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_first = is_first

        # Learnable frequencies: shape [num_frequencies]
        low, high = omega_init_range
        self.omega = nn.Parameter(
            torch.linspace(low, high, steps=num_frequencies).view(1, 1, -1)
        )  # shape [1, 1, K]

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
        """
        x: [B, in_features]
        output: [B, out_features * num_frequencies]
        """
        z = self.linear(x)  # shape: [B, out_features]
        z = z.unsqueeze(-1)  # [B, out_features, 1]

        # Apply learned frequencies
        out = torch.sin(self.omega * z)  # [B, out_features, K]
        return out.view(x.shape[0], -1)  # flatten to [B, out_features * K]

class LearnableFreqSIRENEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=3, num_frequencies=4, omega_init_range=(1.0, 60.0)):
        super().__init__()

        layers = [
            LearnableFreqSIRENLayer(input_dim, hidden_dim, num_frequencies, is_first=True, omega_init_range=omega_init_range)
        ]

        for _ in range(num_layers - 2):
            layers.append(
                LearnableFreqSIRENLayer(hidden_dim * num_frequencies, hidden_dim,
                                        num_frequencies, is_first=False, omega_init_range=omega_init_range)
            )

        layers.append(nn.Linear(hidden_dim * num_frequencies, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_cebra_time_learnable_siren(
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
    model_path="cebra_learnable_siren.pt"
):

    encoder = LearnableFreqSIRENEncoder(
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
            print(f"[LearnableFreqSIREN] Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(encoder.state_dict(), model_path)
    print(f"Learnable frequency SIREN model saved to {model_path}")

    return encoder



def animate_embedding_3d_learnable_siren(data, encoder_path, input_dim, hidden_dim, output_dim,
                                         num_layers=3, num_frequencies=6,
                                         omega_init_range=(1.0, 60.0),
                                         title="Flower embedding",
                                         fps=20, save_path='flower_embedding_line_more_points.mp4'):
    # Load trained model
    encoder = LearnableFreqSIRENEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_frequencies=num_frequencies,
        omega_init_range=omega_init_range
    )
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    encoder.eval()

    # Get embeddings
    with torch.no_grad():
        embeddings = encoder(torch.tensor(data, dtype=torch.float32))

    if output_dim > 3:
        embeddings = PCA(n_components=3).fit_transform(embeddings.numpy())
    else:
        embeddings = embeddings.numpy()

    # Prepare line segments for time-colored line
    points = embeddings.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(0, len(segments))
    colors = cm.rainbow(norm(np.arange(len(segments))))

    # Setup figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Latent 1", fontsize=12)
    ax.set_ylabel("Latent 2", fontsize=12)
    ax.set_zlabel("Latent 3", fontsize=12)

    # Create line collection
    lc = Line3DCollection(segments, colors=colors, linewidth=2)
    ax.add_collection3d(lc)
    ax.set_xlim(embeddings[:, 0].min(), embeddings[:, 0].max())
    ax.set_ylim(embeddings[:, 1].min(), embeddings[:, 1].max())
    ax.set_zlim(embeddings[:, 2].min(), embeddings[:, 2].max())

    # Rotation update function
    def update(angle):
        ax.view_init(elev=30, azim=angle)
        return fig,

    # Animate
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False, interval=1000/fps)

    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"Saved rotating embedding animation to {save_path}")
    else:
        plt.close(fig)
        return HTML(ani.to_jshtml())
