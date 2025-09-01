from siren import train_cebra_time_learnable_siren,animate_embedding_3d_learnable_siren
import numpy as np
lfp_data=np.load('/home/maria/Flower/data/lfp_signal_2probes.npy')

data=lfp_data
input_dim=95
input_dim=190
hidden_dim=100
output_dim=3
train_cebra_time_learnable_siren(
    data,
    input_dim,
    hidden_dim,
    output_dim,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    epochs=5000,
    batch_size=128,
    learning_rate=1e-3,
    model_path="cebra_learnable_siren.pt"
)



animate_embedding_3d_learnable_siren(
    data=lfp_data[10000:30000],
    encoder_path="cebra_learnable_siren.pt",
    input_dim=190,
    hidden_dim=100,
    output_dim=3,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    title="Flower embedding"
)


