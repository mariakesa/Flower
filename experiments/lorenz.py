import numpy as np
from scipy.integrate import solve_ivp
from flower_cone import train_reversible_flower, animate_embedding_3d_reversible_flower
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

# Example usage:
t_vals, lorenz_data = simulate_lorenz()
print("Lorenz data shape:", lorenz_data.shape)

lorenz_data -= lorenz_data.mean(axis=0)
lorenz_data /= lorenz_data.std(axis=0)


flower_model = train_reversible_flower(
    data=lorenz_data,
    input_dim=3,           # x, y, z
    hidden_dim=32,
    output_dim=8,          # or 3 if you want to directly visualize
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    epochs=2000,
    learning_rate=1e-3,
    model_path="flower_lorenz.pt"
)

animate_embedding_3d_reversible_flower(
    data=lorenz_data,
    encoder_path="flower_lorenz.pt",
    input_dim=3,
    hidden_dim=32,
    output_dim=8,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    title="Lorenz Flow in Flower Latent Space",
    fps=20,
    save_path="flower_lorenz_embedding.mp4"
)

