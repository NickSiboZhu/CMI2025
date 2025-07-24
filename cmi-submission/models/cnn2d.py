import torch
import torch.nn as nn
import numpy as np
from . import MODELS
from torch.nn.utils.rnn import pack_padded_sequence

class TOF2DCNN(nn.Module):
    """
    2D CNN for processing Time-of-Flight (TOF) sensor grids.
    
    Each TOF sensor provides an 8x8 grid of depth values.
    This module processes multiple TOF sensors and extracts spatial features.
    """
    
    def __init__(self, num_tof_sensors=5, out_features=128, 
                 conv_channels=None, kernel_sizes=None):
        super(TOF2DCNN, self).__init__()
        
        self.num_tof_sensors = num_tof_sensors
        self.out_features = out_features
        
        # Dynamic 2D conv stack
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 2]
        assert len(conv_channels) == len(kernel_sizes), "conv_channels and kernel_sizes length mismatch"

        layers = []
        in_channels = num_tof_sensors
        for i, (out_c, k) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.append(nn.Conv2d(in_channels, out_c, kernel_size=k, padding=k//2 if k > 2 else 0))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            # Only add MaxPool2d for first layers, avoid over-pooling small spatial dimensions
            if i < len(conv_channels) - 1:  # Don't pool on the last layer
                layers.append(nn.MaxPool2d(2)) # 2x2 max pool
            in_channels = out_c
        
        self.conv_layers = nn.Sequential(*layers)
        self.last_conv_channels = conv_channels[-1]
        
        # Global average pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final projection layer
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.last_conv_channels, out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, tof_grids):
        """
        Forward pass for TOF 2D CNN.
        
        Args:
            tof_grids: Tensor of shape (batch_size, num_sensors, 8, 8)
                      Each 8x8 grid represents one TOF sensor's depth map
        
        Returns:
            Tensor of shape (batch_size, out_features)
        """
        # Process through 2D convolutions
        x = self.conv_layers(tof_grids)  # (batch, 128, 1, 1) after conv layers
        
        # Global pooling (redundant here since we already have 1x1, but good practice)
        x = self.global_pool(x)  # (batch, 128, 1, 1)
        
        # Project to final feature dimension
        x = self.projection(x)  # (batch, out_features)
        
        return x
    
    def get_model_info(self):
        """Get model parameter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
        }


def reshape_tof_features(tof_features, num_sensors=5):
    """
    Reshape flattened TOF features back to 8x8 grids.
    
    Args:
        tof_features: Tensor of shape (batch_size, seq_len, num_sensors * 64)
                     or (batch_size, num_sensors * 64)
        num_sensors: Number of TOF sensors (default 5)
    
    Returns:
        Tensor of shape (batch_size, seq_len, num_sensors, 8, 8)
        or (batch_size, num_sensors, 8, 8) for single timestep
    """
    if tof_features.dim() == 3:
        # Sequential data: (batch_size, seq_len, num_sensors * 64)
        batch_size, seq_len, _ = tof_features.shape
        # Reshape to (batch_size, seq_len, num_sensors, 8, 8)
        tof_grids = tof_features.view(batch_size, seq_len, num_sensors, 8, 8)
    elif tof_features.dim() == 2:
        # Single timestep: (batch_size, num_sensors * 64)
        batch_size, _ = tof_features.shape
        # Reshape to (batch_size, num_sensors, 8, 8)
        tof_grids = tof_features.view(batch_size, num_sensors, 8, 8)
    else:
        raise ValueError(f"Unexpected input shape: {tof_features.shape}")
    
    return tof_grids


@MODELS.register_module()
class TemporalTOF2DCNN(nn.Module):
    """
    2-stage module for sequential TOF data.

    Stage 1: `TOF2DCNN` extracts spatial features for every timestep.
    Stage 2: an optional **learnable** temporal model (default LSTM) replaces
    the previous mean-pooling.  The LSTM's last hidden state is projected back
    to `out_features`, so the branch interface towards the fusion head stays
    unchanged.
    
    Args (new):
        lstm_hidden (int|None): Hidden size of LSTM.  If None ⇒ `out_features`.
        lstm_layers (int): Number of stacked LSTM layers.
        bidirectional (bool): Use bidirectional LSTM if True.
    """

    def __init__(self, num_tof_sensors=5, seq_len=100, out_features=128,
                 conv_channels=None, kernel_sizes=None,
                 lstm_hidden=None, lstm_layers=1, bidirectional=False):
        super(TemporalTOF2DCNN, self).__init__()

        self.num_tof_sensors = num_tof_sensors
        self.seq_len = seq_len
        self.out_features = out_features

        # ------------- Spatial CNN applied per-frame -------------
        self.spatial_cnn = TOF2DCNN(num_tof_sensors, out_features,
                                    conv_channels, kernel_sizes)

        # ------------- Temporal model (LSTM) ---------------------
        lstm_hidden = lstm_hidden or out_features
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=out_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        proj_in_dim = lstm_hidden * (2 if bidirectional else 1)
        self.projection = nn.Sequential(
            nn.Linear(proj_in_dim, out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, tof_sequence, mask=None):
        """
        Forward pass with BATCH-WISE alignment for sequences padded at the BEGINNING.
        """
        batch_size, seq_len, _ = tof_sequence.shape

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=tof_sequence.device)

        # --- 空间特征提取部分 (不变) ---
        # ... (这部分代码完全不变) ...
        tof_grids = reshape_tof_features(tof_sequence, self.num_tof_sensors)
        tof_grids_flat = tof_grids.view(batch_size * seq_len, self.num_tof_sensors, 8, 8)
        spatial_features = self.spatial_cnn(tof_grids_flat)
        spatial_features = spatial_features.view(batch_size, seq_len, self.out_features)

        # --- ✨ 数据对齐与时序处理 (核心修改) ---

        # 1. 计算真实长度
        lengths = mask.sum(dim=1).to(torch.int64)

        # 2. 创建一个新的空张量用于存放对齐后的数据
        aligned_features = torch.zeros_like(spatial_features)

        # 3. 对批次中的每个序列进行数据对齐（搬运）
        # 这个循环在GPU上操作，比在CPU上逐个处理高效得多
        for i in range(batch_size):
            L = lengths[i]
            if L > 0:
                # 将原序列的尾部真实数据，复制到新序列的头部
                aligned_features[i, :L] = spatial_features[i, -L:]

        # 4. 使用对齐后的数据进行打包
        packed_input = pack_padded_sequence(
            aligned_features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # 5. 将打包数据送入LSTM
        _, (h_n, _) = self.lstm(packed_input)

        # --- 特征提取 (不变) ---
        if self.bidirectional:
            temporal_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            temporal_features = h_n[-1,:,:]

        output = self.projection(temporal_features)

        return output
    
    def get_model_info(self):
        """Get model parameter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
        }