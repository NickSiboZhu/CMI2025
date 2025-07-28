# Configuration for Multimodality Model with Transformer-based TOF processing
# This variant uses a Transformer encoder with [CLS] token for temporal modeling
# instead of LSTM for the TOF branch

# Data configuration
data = dict(
    num_classes=18,
    max_length=100,
    imu_features=8,   # 3 acc + 3 angular_vel + 2 engineered
    thm_features=32,  # 32 THM sensors
    tof_features=320, # 5 sensors x 64 pixels each
    static_features=7 # demographic features
)

# Model configuration
model = dict(
    type='MultimodalityModel',
    num_classes=data['num_classes'],
    sequence_length=data['max_length'],
    
    # IMU 1D CNN branch config
    imu_branch_cfg=dict(
        type='CNN1D',
        in_channels=data['imu_features'],
        cnn_channels=[64, 128, 256, 256],
        kernel_sizes=[7, 5, 3, 3],
        cnn_output_size=256,
        sequence_length=data['max_length']
    ),
    
    # THM 1D CNN branch config
    thm_branch_cfg=dict(
        type='CNN1D',
        in_channels=data['thm_features'],
        cnn_channels=[64, 128, 128],
        kernel_sizes=[5, 3, 3],
        cnn_output_size=128,
        sequence_length=data['max_length']
    ),
    
    # TOF 2D CNN + Transformer branch config
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,  # 5 TOF sensors
        seq_len=data['max_length'],
        out_features=128,  # Note: actual output = conv_channels[-1] = 128
        # Spatial CNN parameters
        conv_channels=[32, 64, 128],  # Stronger spatial feature extraction
        kernel_sizes=[3, 3, 2],       # Appropriate kernel sizes for 8x8 input
        # Temporal encoder configuration
        temporal_mode='transformer',  # Using transformer for temporal modeling
        # Transformer parameters for temporal modeling
        num_heads=8,        # 8 attention heads
        num_layers=3,       # 3 transformer layers for better temporal modeling
        ff_dim=512,         # Feedforward dimension
        dropout=0.1         # Dropout for regularization
    ),
    
    # Static features MLP branch config
    mlp_branch_cfg=dict(
        type='MLP',
        input_dim=data['static_features'],
        hidden_dims=[32, 32],
        output_dim=32
    ),
    
    # Fusion head configuration
    fusion_head_cfg=dict(
        type='FusionHead',  # Use standard FusionHead
        # Fusion parameters - dimensions will be auto-detected
        hidden_dims=[256, 128],
        dropout_rates=[0.4, 0.3]
    ),
    
    # Use all modalities
    use_tof=True,
    use_thm=True
) 