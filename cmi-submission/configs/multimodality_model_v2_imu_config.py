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
        kernel_sizes=[7, 5, 3],
        # NEW: Temporal aggregation options
        temporal_aggregation='temporal_encoder',  # 'global_pool' or 'temporal_encoder'
        temporal_mode='lstm',  # 'lstm' or 'transformer' (when using temporal_encoder)
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=False,
        # NEW: ResNet-style residual connections
        use_residual=True,
        # NEW: Channel attention
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
    mixup_enabled=True,
    mixup_alpha=0.2,
    loss=dict(type='CrossEntropyLoss'),

    # --- NEW: Learning Rate Scheduler Configuration ---
    # Choose 'cosine' or 'reduce_on_plateau'
    scheduler_cfg=dict(
        type='cosine',  # Default is cosine annealing
        # type='reduce_on_plateau',
        # --- Settings for 'reduce_on_plateau' ---
        # factor=0.2,   # Factor to reduce LR by (e.g., new_lr = lr * factor)
        # patience=5,   # Epochs to wait for improvement before reducing LR
        # min_lr=1e-6,  # Minimum learning rate
        warmup_ratio=0.1, # Optional warmup for ReduceLROnPlateau
        
        # --- NEW: Specific Learning Rates per Branch ---
        layer_lrs=dict(
            imu=1e-3,
            mlp=2e-3,
            fusion=2e-3,
        )
    ),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42, num_workers=4) 
