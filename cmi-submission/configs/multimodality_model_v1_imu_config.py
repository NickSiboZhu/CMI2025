# ===================================================================
#   Configuration v1 – IMU-Only (no TOF, with DEMO)
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='imu',  # Fixed: must match data preprocessing logic
    max_length=100,
    batch_size=32,
    seq_input_channels=7,   # Actual IMU features: acc_x,y,z + rot_w,x,y,z
    static_input_features=7
)

# -------------------------- Model Architecture -----------------------
model = dict(
    type='MultimodalityModel',
    num_classes=18,

    # CNN branch blueprint - smaller since IMU-only has less data
    cnn_branch_cfg=dict(
        type='CNN1D',
        input_channels=data['seq_input_channels'],
        sequence_length=data['max_length'],
        filters=[32, 64, 128],      # smaller than full config
        kernel_sizes=[5, 5, 3]
    ),

    # MLP branch blueprint (same as full)
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=data['static_input_features'],
        hidden_dims=[64],           # simpler since less total data
        output_dim=32,
        dropout_rate=0.5
    ),

    # No tof_branch_cfg - will use dummy/zero features

    # Fusion head blueprint - smaller combined feature size
    fusion_head_cfg=dict(
        type='FusionHead',  # Use default LinearFusionHead
        hidden_dims=[256, 128],
        dropout_rates=[0.5, 0.3]
    )
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=100,
    patience=15,
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
    scheduler=dict(type='CosineAnnealingWarmRestarts', warmup_ratio=0.1)
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42) 