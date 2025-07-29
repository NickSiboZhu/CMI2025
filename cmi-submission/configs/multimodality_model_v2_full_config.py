# ===================================================================
#   Configuration v2 – FULL Multimodal (user-requested sizes)
#   • MLP hidden dims:           [32]
#   • 1-D CNN filters:           [32, 64]
#   • TOF 2-D CNN channels:      [32, 64, 128]
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

    # IMU branch (inertial measurement unit)
    imu_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # filled dynamically at runtime
        sequence_length=data['max_length'],
        filters=[32, 64],               # <-- user setting
        kernel_sizes=[5, 3]
    ),

    # THM branch (thermopile sensors)
    thm_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # filled dynamically at runtime
        sequence_length=data['max_length'],
        filters=[16, 32, 64],           # separate architecture for THM
        kernel_sizes=[7, 5, 3]
    ),

    # TOF 2-D CNN branch (depth grids)
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,  # Number of TOF sensors
        seq_len=data['max_length'],
        out_features=192,            # Note: ignored, actual output = conv_channels[-1] = 128 
        conv_channels=[32, 64, 128],    # <-- user setting
        kernel_sizes=[3, 3, 2],
        # Temporal encoder configuration
        temporal_mode='transformer',  # Using transformer for temporal modeling
        # Transformer parameters for temporal modeling
        num_heads=8,
        num_layers=2,
        ff_dim=768,  # Larger ff_dim for richer representations
        dropout=0.1
    ),

    # MLP branch (demographics)
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # filled dynamically at runtime
        hidden_dims=[32],             # <-- user setting
        output_dim=32,
        dropout_rate=0.5
    ),

    # Enable THM branch processing
    use_thm=True,
    use_tof=True,

    # Fusion head - adjusted for 4 branches: IMU + THM + TOF + Static
    fusion_head_cfg=dict(
        type='FusionHead',
        hidden_dims=[384, 192, 96],
        dropout_rates=[0.5, 0.4, 0.3]
    )
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=100,
    patience=15,
    start_lr=1e-3,
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    # Default to Cross-Entropy; add loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25) to switch.
    scheduler=dict(type='CosineAnnealingWarmRestarts', warmup_ratio=0.1)
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42) 