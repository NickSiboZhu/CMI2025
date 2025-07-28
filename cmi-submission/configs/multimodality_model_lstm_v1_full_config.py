# Configuration for Multimodality Model with LSTM-based TOF processing
# This variant uses LSTM for temporal modeling instead of Transformer

# Data configuration
data = dict(
    num_classes=18,
    max_length=100,
)

# Model configuration
model = dict(
    type='MultimodalityModel',
    num_classes=data['num_classes'],
    sequence_length=data['max_length'],
    
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
        kernel_sizes=[7, 5, 3]
    ),
    
    # TOF 2D CNN + LSTM branch config
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,  # 5 TOF sensors
        seq_len=data['max_length'],
        out_features=128,  # Note: ignored, actual output = lstm_hidden * 2 (if bidirectional) = 512
        # Spatial CNN parameters
        conv_channels=[32, 64, 128],
        kernel_sizes=[3, 3, 2],
        # Temporal encoder mode
        temporal_mode='lstm',  # Using LSTM instead of transformer
        # LSTM parameters
        lstm_hidden=256,       # Hidden size for LSTM
        lstm_layers=2,         # 2-layer LSTM
        bidirectional=False,    # Bidirectional: output = 256*2 = 512
    ),
    
    # MLP branch blueprint
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # will be filled dynamically from data
        hidden_dims=[64],
        output_dim=32,
        dropout_rate=0.5
    ),
    
    # Fusion head blueprint
    fusion_head_cfg=dict(
        type='FusionHead',  # Use default LinearFusionHead
        hidden_dims=[256, 128],
        dropout_rates=[0.4, 0.3]
    ),
    
    # Use all modalities
    use_tof=True,
    use_thm=True
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