# ===================================================================
#   Configuration – FULL Multimodal with TransformerFusionHead (v2)
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='full',
    max_length=100,
    batch_size=64,
)

# -------------------------- Model Architecture -----------------------
# Branch output sizes must be reflected in `branch_dims`.
#  * CNN1D last filter            → 64
#  * TemporalTOF2DCNN out_features→ 192
#  * MLP output_dim               → 32
model = dict(
    type='MultimodalityModel',
    num_classes=18,


    # IMU branch (inertial measurement unit)
    imu_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[64, 128, 256],
        kernel_sizes=[7, 5, 3]
    ),

    # THM branch (thermopile sensors)
    thm_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[32, 64, 128],           # different architecture for THM
        kernel_sizes=[5, 5, 3]
    ),

    # TOF 2D CNN branch config
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,
        seq_len=100,
        out_features=128,  # Note: actual output = conv_channels[-1] = 128
        conv_channels=[32, 64, 128],  # Stronger spatial feature extraction
        kernel_sizes=[3, 3, 2],       # Matching kernel sizes
        # Temporal encoder configuration
        temporal_mode='lstm',  
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=False,
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

    # Fusion head – attention based
    fusion_head_cfg = dict(
        type='AttentionFusionHead',
        hidden_dims=[256, 128],
        dropout_rates=[0.5, 0.3]
    )
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=100,
    patience=15,
    start_lr=1e-3,
    weight_decay=1e-2,
    use_amp=False, 
    mixup_enabled=True,
    mixup_alpha=0.2,
    # loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42) 