def animate_embedding_3d_learnable_siren(data, encoder_path, input_dim, hidden_dim, output_dim,
                                         num_layers=3, num_frequencies=6,
                                         omega_init_range=(1.0, 60.0),
                                         title="Flower embedding",
                                         fps=20, save_path='flower_embedding_line_scrambled_small.mp4',
                                         scramble=True, seed=42):
    # Optional: scramble data along time axis
    if scramble:
        np.random.seed(seed)
        data = data.copy()
        np.random.shuffle(data)  # Scramble time order

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
    
animate_embedding_3d_learnable_siren(
    data=lfp_data[10000:13000],
    encoder_path="cebra_learnable_siren.pt",
    input_dim=95,
    hidden_dim=100,
    output_dim=3,
    num_layers=3,
    num_frequencies=6,
    omega_init_range=(1.0, 60.0),
    title="Flower embedding"
)

