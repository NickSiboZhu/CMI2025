# ===================================================================
#   Configuration v3 – FULL Multimodal (IMU + THM + TOF + DEMO)
# ===================================================================

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

    # IMU branch (inertial measurement unit)
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

    # THM branch (thermopile sensors)
    thm_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[32, 64, 128],           # different architecture for THM
        kernel_sizes=[5, 5, 3],
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

    # TOF 2D CNN branch config
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,
        seq_len=100,
        # out_features is now determined by conv_channels[-1]
        conv_channels=[32, 64, 128],  # Stronger spatial feature extraction
        kernel_sizes=[3, 3, 2],       # Matching kernel sizes
        # Temporal encoder configuration
        temporal_mode='lstm',  
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=False,
        # NEW: ResNet-style residual connections for spatial CNN
        use_residual=True,
        # NEW: Channel attention and optional sensor gate
        use_se=True,
        se_reduction=16,
        use_sensor_gate=False,
        sensor_gate_adaptive=False
    ),

    # MLP branch blueprint
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # will be filled dynamically from data
        hidden_dims=[64],
        output_dim=32,
        dropout_rate=0.5
    ),

    # Enable THM and TOF branch processing
    use_thm=True,
    use_tof=True,

    # Fusion head blueprint
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
        warmup_ratio=0.1, # 10% of total epochs for warmup before plateau scheduler takes over

        # --- NEW: Specific Learning Rates per Branch ---
        layer_lrs=dict(
            imu=1e-3,
            thm=1e-3,
            tof=5e-4,
            mlp=2e-3,
            fusion=2e-3,
        )
    ),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42, num_workers=4) 
