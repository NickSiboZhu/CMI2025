# ===================================================================
#   Configuration v2 – IMU-Only (no TOF, with DEMO)
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='imu',  # Fixed: must match data preprocessing logic
    max_length=100,
    batch_size=64,
)

# -------------------------- Model Architecture -----------------------
model = dict(
    type='MultimodalityModel',
    num_classes=18,
    sequence_length=data['max_length'],

    # IMU branch - smaller since IMU-only has less data
    imu_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[64, 128, 256],
        kernel_sizes=[5, 5, 3],
        # NEW: Temporal aggregation options
        temporal_aggregation='global_pool',  # 'global_pool' or 'temporal_encoder'
        # temporal_mode='lstm',  # 'lstm' or 'transformer' (when using temporal_encoder)
        # lstm_hidden=128,
        # lstm_layers=1,
        # bidirectional=False,
        # # NEW: ResNet-style residual connections
        use_residual=True,
        # NEW: Channel attention (SE)
        use_se=True,
        se_reduction=16
    ),

    # MLP branch blueprint (same as full)
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # will be filled dynamically from data
        hidden_dims=[64],           # simpler since less total data
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
    # Explicitly enable Spectrogram branch
    use_spec=True,
    # Disable TOF and THM branches entirely for IMU variant
    use_tof=False,
    use_thm=False,

    # Fusion head blueprint - smaller combined feature size
    fusion_head_cfg=dict(
        type='FusionHead',  # Use default LinearFusionHead
        hidden_dims=[256, 128],
        dropout_rates=[0.4, 0.3]
    )
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=100,
    patience=15,
    # start_lr is no longer used; learning rates are defined per-layer below
    weight_decay=1e-2,
    use_amp=False, 
    mixup_enabled=False,
    mixup_alpha=0.2,
    loss=dict(type='CrossEntropyLoss'),

    # --- NEW: Learning Rate Scheduler Configuration ---
    # Choose 'cosine' or 'reduce_on_plateau'
    scheduler_cfg=dict(
        type='cosine',  # Default is cosine annealing
        warmup_ratio=0.1, # Optional warmup for ReduceLROnPlateau
        # --- NEW: Specific Learning Rates per Branch ---
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
