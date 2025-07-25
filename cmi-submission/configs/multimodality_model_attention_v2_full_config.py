# ===================================================================
#   Configuration – FULL Multimodal with TransformerFusionHead (v2)
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='full',
    max_length=100,
    batch_size=128,
)

# -------------------------- Model Architecture -----------------------
# Branch output sizes must be reflected in `branch_dims`.
#  * CNN1D last filter            → 64
#  * TemporalTOF2DCNN out_features→ 192
#  * MLP output_dim               → 32
model = dict(
    type='MultimodalityModel',
    num_classes=18,

    # --- IMU branch (inertial measurement unit) ---
    imu_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # filled dynamically at runtime
        sequence_length=data['max_length'],
        filters=[32, 64],
        kernel_sizes=[5, 3],
    ),

    # --- MLP branch (static demographics) ---
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # filled dynamically at runtime
        hidden_dims=[32],
        output_dim=32,
        dropout_rate=0.5,
    ),

    # --- TOF 2-D CNN + LSTM branch ---
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,  # Number of TOF sensors
        seq_len=data['max_length'],
        out_features=192,
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 2],
        lstm_hidden=192,
        lstm_layers=1,
        bidirectional=False,
    ),

    # --- Transformer-based fusion head with CLS token ---
    fusion_head_cfg=dict(
        type='TransformerFusionHead',
        branch_dims=[64, 192, 32],   # cnn, tof, mlp
        embed_dim=256,
        num_heads=4,
        depth=2,
        dropout=0.1,
    ),
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=50,
    patience=15,
    start_lr=1e-3,
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    # Uncomment to switch to Focal Loss
    # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
    scheduler=dict(type='CosineAnnealingWarmRestarts', warmup_ratio=0.1),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=1, seed=42) 