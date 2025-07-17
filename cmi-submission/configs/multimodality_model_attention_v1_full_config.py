# ===================================================================
#   Configuration – FULL Multimodal with AttentionFusionHead
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='full',
    max_length=100,
    batch_size=32,
)

# -------------------------- Model Architecture -----------------------
# NOTE: branch_dims must match the output sizes of each branch.
#       cnn_output_size    = filters[-1] (here 64)
#       tof_out_features   = 192
#       mlp_output_dim     = 32
model = dict(
    type='MultimodalityModel',
    num_classes=18,

    # 1-D CNN branch (IMU / THM)
    cnn_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # filled dynamically at runtime
        sequence_length=data['max_length'],
        filters=[32, 64],       # cnn_output_size = 64
        kernel_sizes=[5, 3]
    ),

    # MLP branch (static demographics)
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,   # filled dynamically
        hidden_dims=[32],
        output_dim=32,
        dropout_rate=0.5,
    ),

    # TOF 2-D CNN branch
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        num_tof_sensors=5,
        seq_len=data['max_length'],
        out_features=192,         # bigger than cnn_output_size
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 2],
        lstm_hidden=192,
        lstm_layers=1,
        bidirectional=False,
    ),

    # Fusion head – attention based
    fusion_head_cfg=dict(
        type='AttentionFusionHead',
        branch_dims=[64, 192, 32],   # [cnn_dim, tof_dim, mlp_dim]
        hidden_dims=[256, 128],
        dropout_rates=[0.5, 0.3]
    ),
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=100,
    patience=15,
    start_lr=1e-3,
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    # Uncomment to use Focal Loss instead of CE
    # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
    scheduler=dict(type='CosineAnnealingWarmRestarts', warmup_ratio=0.1),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=1, seed=42) 