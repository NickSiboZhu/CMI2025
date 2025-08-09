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

    # IMU branch - smaller since IMU-only has less data
    imu_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[64, 128, 256],
        kernel_sizes=[5, 5, 3],
        # NEW: Temporal aggregation options
        temporal_aggregation='global_pool',  # 'global_pool' or 'temporal_encoder'
        use_residual=True
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
    # use_spec=False,
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
    # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),

    # --- NEW: Learning Rate Scheduler Configuration ---
    # Choose 'cosine' or 'reduce_on_plateau'
    scheduler_cfg=dict(
        # type='cosine',  # Default is cosine annealing
        type='cosine',
        warmup_ratio=0.1, # Optional warmup for ReduceLROnPlateau
        # --- NEW: Specific Learning Rates per Branch ---
        layer_lrs=dict(
            imu=1e-3,
            mlp=1e-3,
            fusion=1e-3,
            spec=1e-3,
        )
    ),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42) 
