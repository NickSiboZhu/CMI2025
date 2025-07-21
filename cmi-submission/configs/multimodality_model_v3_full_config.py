# ===================================================================
#   Configuration v1 – FULL Multimodal (IMU + THM + TOF + DEMO)
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='full',
    max_length=100,
    batch_size=128,
)

# -------------------------- Model Architecture -----------------------
model = dict(
    type='MultimodalityModel',
    num_classes=18,

    # CNN branch blueprint (can switch to LSTM etc. just by changing type)
    cnn_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[64, 128, 256],
        kernel_sizes=[5, 5, 3]
    ),

    # MLP branch blueprint
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # will be filled dynamically from data
        hidden_dims=[64],
        output_dim=32,
        dropout_rate=0.5
    ),

    # -------- 2-D CNN (TOF grids) ------
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        num_tof_sensors=5,
        seq_len=data['max_length'],
        out_features=128,
        conv_channels=[32, 64, 128],
        lstm_layers=1,
        lstm_hidden=128,
        kernel_sizes=[3, 3, 2]
    ),

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
    start_lr=1e-3,
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=1e-4),
    # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
    scheduler=dict(type='CosineAnnealingWarmRestarts', warmup_ratio=0.1)
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42) 