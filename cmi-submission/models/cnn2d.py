import torch
import torch.nn as nn
import numpy as np
from . import MODELS

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock2D, self).__init__()
        
        # 定义层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 定义捷径
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 使用非原地加法，以防万一
        out = out + identity
        out = self.relu(out)
        
        return out

# 2. 基于残差块的2D CNN (The spatial feature extractor)
# ==============================================================================
@MODELS.register_module()
class TOF2DCNN(nn.Module):
    def __init__(self, num_tof_sensors=5, out_features=128, 
                 conv_channels=None,
                 # ✨ 变化点 1: 恢复kernel_sizes参数
                 kernel_sizes=None):
        super(TOF2DCNN, self).__init__()
        
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        # ✨ 为kernel_sizes提供一个默认值，使其与conv_channels对齐
        if kernel_sizes is None:
            # 默认所有残差块的kernel_size都为3
            kernel_sizes = [3] * (len(conv_channels) -1)

        self.stem = nn.Sequential(
            nn.Conv2d(num_tof_sensors, conv_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        self.residual_layers = nn.ModuleList()
        in_c = conv_channels[0]
        # ✨ 变化点 2: 迭代时同时使用conv_channels和kernel_sizes
        # 我们假设kernel_sizes的长度对应于残差块的数量
        num_blocks = len(conv_channels) - 1
        assert len(kernel_sizes) == num_blocks, \
            f"Length of kernel_sizes ({len(kernel_sizes)}) must match the number of residual blocks ({num_blocks})."

        for i in range(num_blocks):
            out_c = conv_channels[i+1]
            k = kernel_sizes[i]
            stride = 2 if i < 2 else 1
            # ✨ 将动态的核大小k传递给残差块
            self.residual_layers.append(ResidualBlock2D(in_c, out_c, kernel_size=k, stride=stride))
            in_c = out_c
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[-1], out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, tof_grids):
        x = self.stem(tof_grids)
        for layer in self.residual_layers:
            x = layer(x)
        x = self.global_pool(x)
        x = self.projection(x)
        return x


@MODELS.register_module()
class TemporalTOF2DCNN(nn.Module):
    def __init__(self, num_tof_sensors=5, seq_len=100, out_features=128,
                 conv_channels=None,
                 # ✨ 变化点 3: 恢复kernel_sizes参数
                 kernel_sizes=None,
                 lstm_hidden=None, lstm_layers=1, lstm_bidirectional=False):
        super(TemporalTOF2DCNN, self).__init__()

        self.num_tof_sensors = num_tof_sensors
        self.seq_len = seq_len
        self.out_features = out_features

        # ✨ 变化点 4: 将kernel_sizes传递给底层的TOF2DCNN
        self.spatial_cnn = TOF2DCNN(
            num_tof_sensors=num_tof_sensors, 
            out_features=out_features, 
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes
        )

        # LSTM部分保持不变
        lstm_hidden = lstm_hidden or out_features
        self.bidirectional = lstm_bidirectional
        self.lstm = nn.LSTM(
            input_size=out_features, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=lstm_bidirectional
        )
        proj_in_dim = lstm_hidden * (2 if lstm_bidirectional else 1)
        self.projection = nn.Sequential(
            nn.Linear(proj_in_dim, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, tof_sequence):
        """
        前向传播
        Args:
            tof_sequence: Tensor of shape (batch_size, seq_len, num_sensors * 64)
        """
        batch_size, seq_len, _ = tof_sequence.shape
        
        # Reshape为 (batch_size, seq_len, num_sensors, 8, 8)
        tof_grids = reshape_tof_features(tof_sequence, self.num_tof_sensors)
        
        # 将时间和批次维度合并，以便一次性通过2D CNN
        # (batch_size, seq_len, C, H, W) -> (batch_size * seq_len, C, H, W)
        tof_grids_flat = tof_grids.view(batch_size * seq_len, self.num_tof_sensors, 8, 8)
        
        # 对所有时间步应用2D CNN提取空间特征
        spatial_features = self.spatial_cnn(tof_grids_flat)  # (batch_size * seq_len, out_features)
        
        # Reshape回时序格式 (batch_size, seq_len, out_features)
        spatial_features_seq = spatial_features.view(batch_size, seq_len, self.out_features)

        # 通过LSTM处理时间依赖性
        lstm_out, (h_n, _) = self.lstm(spatial_features_seq)

        # 提取LSTM的最终输出作为时间序列的表示
        if self.bidirectional:
            # 拼接双向LSTM的最后一个时间步的前向和后向隐藏状态
            temporal_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            temporal_features = h_n[-1,:,:]
            
        # 最终投影，得到该分支的输出
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
    
# 4. 辅助函数 (Utility function)
# ==============================================================================
def reshape_tof_features(tof_features, num_sensors=5):
    """
    将扁平化的TOF特征重塑为8x8的网格。
    """
    if tof_features.dim() == 3:
        batch_size, seq_len, _ = tof_features.shape
        tof_grids = tof_features.view(batch_size, seq_len, num_sensors, 8, 8)
    elif tof_features.dim() == 2:
        batch_size, _ = tof_features.shape
        tof_grids = tof_features.view(batch_size, num_sensors, 8, 8)
    else:
        raise ValueError(f"Unexpected input shape: {tof_features.shape}")
    
    return tof_grids