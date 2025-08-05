from flower import train_reversible_flower, animate_embedding_3d_reversible_flower
import numpy as np

lfp_data=np.load('/home/maria/Flower/data/lfp_signal.npy')

data=lfp_data
input_dim=95
hidden_dim=100
output_dim=3
train_reversible_flower(
    data,
    input_dim,
    hidden_dim,
    output_dim,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(0.5, 10.0),
    epochs=1000,
    batch_size=128,
    learning_rate=1e-3
)



animate_embedding_3d_reversible_flower(
    data=lfp_data[10000:30000],
    encoder_path="reversible_flower.pt",
    input_dim=95,
    hidden_dim=100,
    output_dim=3,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    title="Flower embedding")