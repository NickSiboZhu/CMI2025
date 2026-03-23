# Configuration v2: IMU-only model with demographics and spectrogram features.

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='imu',  # Must stay aligned with the preprocessing and model-selection logic.
    max_length=100,
    batch_size=64,
)

# -------------------------- Model Architecture -----------------------
model = dict(
    type='MultimodalityModel',
    num_classes=18,
    sequence_length=data['max_length'],

    imu_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # Filled from the prepared dataset.
        sequence_length=data['max_length'],
        filters=[64, 128, 256],
        kernel_sizes=[5, 5, 3],
        # Global pooling keeps the IMU-only baseline cheaper and easier to tune.
        temporal_aggregation='global_pool',
        use_residual=True,
        use_se=True,
        se_reduction=16
    ),

    # Keep a lightweight side branch for demographics and missingness flags.
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # Filled from the prepared dataset.
        hidden_dims=[64],
        output_dim=32,
        dropout_rate=0.5
    ),

    spec_branch_cfg=dict(
        type='SpectrogramCNN',
        in_channels=6,
        filters=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        use_residual=True,
    ),
    # Spectrograms add a complementary view when THM and ToF are disabled.
    use_spec=True,
    use_tof=False,
    use_thm=False,

    # A smaller fusion head is enough once only IMU, spectrogram, and MLP branches remain.
    fusion_head_cfg=dict(
        type='FusionHead',
        hidden_dims=[256, 128],
        dropout_rates=[0.4, 0.3]
    )
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=100,
    patience=15,
    weight_decay=1e-2,
    use_amp=True, 
    mixup_enabled=False,
    mixup_alpha=0.2,
    loss=dict(type='CrossEntropyLoss'),

    # Keep scheduler settings explicit so standalone runs and Optuna share one interface.
    scheduler_cfg=dict(
        type='cosine',
        warmup_ratio=0.1,
        layer_lrs=dict(
            imu=5e-4,
            mlp=1e-3,
            fusion=1e-3,
            spec=5e-4,
        )
    ),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42, num_workers=4) 

# -------------------------- Spectrogram Params ------------------------
spec_params = dict(
    fs=10.0,
    nperseg=20,
    noverlap_ratio=0.75,
    max_length=data['max_length'],
)
