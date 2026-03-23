# Configuration v3: full multimodal model with IMU, THM, ToF, and demographics.

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='full',
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
        # The full variant can afford a richer temporal encoder in the IMU branch.
        temporal_aggregation='temporal_encoder',
        temporal_mode='lstm',
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=False,
        use_residual=True,
        use_se=True,
        se_reduction=16
    ),

    # Keep the THM branch narrower because the thermopile stream is lower dimensional.
    thm_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # Filled from the prepared dataset.
        sequence_length=data['max_length'],
        filters=[32, 64, 128],
        kernel_sizes=[5, 5, 3],
        temporal_aggregation='temporal_encoder',
        temporal_mode='lstm',
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=False,
        use_residual=True,
        use_se=True,
        se_reduction=16
    ),

    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,
        seq_len=100,
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 2],
        temporal_mode='lstm',
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=False,
        use_residual=True,
        # Keep sensor gating off by default to avoid over-suppressing sparse sensors.
        use_se=True,
        se_reduction=16,
        use_sensor_gate=False,
        sensor_gate_adaptive=False
    ),

    # Static demographics and missingness indicators flow through a lightweight MLP.
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

    use_thm=True,
    use_tof=True,
    use_spec=True,

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
    use_amp=False, 
    mixup_enabled=True,
    mixup_alpha=0.2,
    loss=dict(type='CrossEntropyLoss'),

    # Keep scheduler settings explicit so standalone runs and Optuna share one interface.
    scheduler_cfg=dict(
        type='cosine',
        warmup_ratio=0.1,
        layer_lrs=dict(
            imu=1e-3,
            thm=1e-3,
            tof=5e-4,
            mlp=2e-3,
            spec=2e-3,
            fusion=2e-3,
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
