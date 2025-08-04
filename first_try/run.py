from siren import animate_embedding_3d_learnable_siren
import numpy as np
lfp_data=np.load('/home/maria/Flower/data/lfp_signal.npy')
animate_embedding_3d_learnable_siren(
    data=lfp_data[10000:30000],
    encoder_path="cebra_learnable_siren.pt",
    input_dim=95,
    hidden_dim=100,
    output_dim=3,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    title="Flower embedding"
)
