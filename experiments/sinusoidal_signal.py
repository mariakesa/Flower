import numpy as np
from scipy.integrate import solve_ivp
from flower_cone import train_reversible_flower, animate_embedding_3d_reversible_flower
from siren import train_cebra_time_learnable_siren, animate_embedding_3d_learnable_siren

def generate_sinusoidal_signal(T=1000, dt=0.01):
    t = np.linspace(0, T * dt, T)
    x = np.stack([
        np.sin(2 * np.pi * 1 * t),  # 1 Hz
        np.sin(2 * np.pi * 3 * t + 0.5),  # 3 Hz with phase shift
        np.sin(2 * np.pi * 5 * t + 1.0)   # 5 Hz with phase shift
    ], axis=1)  # Shape [T, 3]
    return t, x

t_vals, signal_data = generate_sinusoidal_signal(T=3000)

# Normalize
signal_data -= signal_data.mean(axis=0)
signal_data /= signal_data.std(axis=0)

flower_model = train_reversible_flower(
    data=signal_data,
    input_dim=3,
    hidden_dim=32,
    output_dim=8,
    num_layers=2,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    epochs=2000,
    learning_rate=1e-3,
    model_path="flower_sinusoid.pt"
)

animate_embedding_3d_reversible_flower(
    data=signal_data,
    encoder_path="flower_sinusoid.pt",
    input_dim=3,
    hidden_dim=32,
    output_dim=8,
    num_layers=2,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    title="Summed Sinusoids Flow in Flower Latent Space",
    fps=20,
    save_path="flower_sinusoids_embedding.mp4"
)
'''

flower_model = train_cebra_time_learnable_siren(
    data=signal_data,
    input_dim=3,
    hidden_dim=32,
    output_dim=8,
    num_layers=2,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    epochs=2000,
    learning_rate=1e-3,
    model_path="siren_sinusoid.pt"
)
animate_embedding_3d_learnable_siren(
    data=signal_data,
    encoder_path="siren_sinusoid.pt",
    input_dim=3,
    hidden_dim=32,
    output_dim=8,
    num_layers=2,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    title="Summed Sinusoids Flow in Flower Latent Space",
    fps=20,
    save_path="siren_sinusoids_embedding.mp4"
)
'''