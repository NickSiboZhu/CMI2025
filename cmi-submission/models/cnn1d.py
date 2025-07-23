import torch
import torch.nn as nn
from . import MODELS  # Import registry

class ResidualBlock1D(nn.Module):
    """
    支持带步长下采样的1D残差块。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        # 主路径
        self.main_path = nn.Sequential(
            # 第一个卷积层使用传入的stride参数，实现下采样
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # 第二个卷积层通常保持维度不变 (stride=1)
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        # 捷径（Shortcut）
        self.shortcut = nn.Sequential()
        # 如果需要下采样或通道数变化，捷径也需要同步处理
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path(x)
        shortcut = self.shortcut(x)
        return self.final_relu(out + shortcut)


@MODELS.register_module()
class CNN1D(nn.Module):
    """
    一个灵活的1D CNN模型，可以选择使用全局池化或LSTM来处理时间序列。
    """
    def __init__(self,
                 input_channels: int,
                 filters: list = None,
                 kernel_sizes: list = None,
                 strides: list = None,
                 # ✨ 变化点 1: 新增核心参数，用于选择时间处理模块
                 temporal_module: str = 'global_avg',
                 # ✨ 变化点 2: 为LSTM模块新增配置参数
                 lstm_hidden: int = 128,
                 lstm_layers: int = 1,
                 lstm_bidirectional: bool = True,
                 # 其他旧参数...
                 sequence_length: int = 100,
                 num_classes: int = 18):
        super(CNN1D, self).__init__()
        
        # --- 1. CNN主干部分 (与之前相同) ---
        if filters is None: filters = [64, 128, 256]
        if kernel_sizes is None: kernel_sizes = [3, 3, 3]
        if strides is None: strides = [2, 2, 1]
        assert len(filters) == len(kernel_sizes) == len(strides)

        self.blocks = nn.ModuleList()
        in_c = input_channels
        for i in range(len(filters)):
            self.blocks.append(ResidualBlock1D(in_c, filters[i], kernel_sizes[i], strides[i]))
            in_c = filters[i]
        
        # --- 2. 可选的时间处理模块 ---
        self.temporal_module = temporal_module
        if self.temporal_module == 'global_avg':
            self.final_module = nn.AdaptiveAvgPool1d(1)
            # ✨ 变化点 3: 动态设置模型的最终输出维度
            self.cnn_output_size = filters[-1]
        elif self.temporal_module == 'global_max':
            self.final_module = nn.AdaptiveMaxPool1d(1)
            self.cnn_output_size = filters[-1]
        elif self.temporal_module == 'lstm':
            self.final_module = nn.LSTM(
                input_size=filters[-1], # LSTM的输入维度是CNN主干的输出通道数
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True, # 必须为True，方便处理
                bidirectional=lstm_bidirectional
            )
            # 动态设置模型的最终输出维度
            self.lstm_bidirectional = lstm_bidirectional
            self.cnn_output_size = lstm_hidden * 2 if lstm_bidirectional else lstm_hidden
        else:
            raise ValueError(f"Unknown temporal_module: {temporal_module}. Choose 'global_pool' or 'lstm'.")

    def forward(self, x):
        # x 初始形状: (batch, seq_len, channels)
        x = x.transpose(1, 2) # -> (batch, channels, seq_len)

        # 1. 通过CNN主干提取特征序列
        for block in self.blocks:
            x = block(x)
        # 此时 x 的形状为 (batch, cnn_out_channels, final_seq_len)

        # 根据选择的模块进行后续处理
        if self.temporal_module == 'global_avg':
            # 使用全局池化
            x = self.final_module(x) # -> (batch, cnn_out_channels, 1)
            x = x.squeeze(-1) # -> (batch, cnn_out_channels)
        elif self.temporal_module == 'global_max':
            # 使用全局最大池化
            x = self.final_module(x)
            x = x.squeeze(-1) # -> (batch, cnn_out_channels)
        elif self.temporal_module == 'lstm':
            # 使用LSTM
            # LSTM需要输入形状为 (batch, seq_len, features)
            x = x.transpose(1, 2) # -> (batch, final_seq_len, cnn_out_channels)
            
            # lstm_out 形状: (batch, seq_len, num_directions * hidden_size)
            # h_n 形状: (num_layers * num_directions, batch, hidden_size)
            lstm_out, (h_n, c_n) = self.final_module(x)
            
            # 我们通常取最后一个时间步的隐藏状态作为整个序列的表示
            if self.lstm_bidirectional:
                # 拼接双向LSTM的最后一个前向和后向隐藏状态
                # h_n[-2] 是最后一个前向, h_n[-1] 是最后一个后向
                x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            else:
                # 只取单向LSTM的最后一个隐藏状态
                x = h_n[-1,:,:]
        
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
