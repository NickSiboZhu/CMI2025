# ===================================================================
#   Configuration v2 – IMU-Only (no TOF, with DEMO)
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='imu',  # Fixed: must match data preprocessing logic
    max_length=100,
    batch_size=32,
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
        kernel_sizes=[7, 5, 3]
    ),

    # MLP branch blueprint (same as full)
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # will be filled dynamically from data
        hidden_dims=[128, 64], 
        output_dim=64,
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
    patience=20,
    start_lr=1e-3,
    weight_decay=1e-2,
    # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42) 