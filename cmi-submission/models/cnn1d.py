import torch
import torch.nn as nn
from . import MODELS  # Import registry

@MODELS.register_module()
class CNN1D(nn.Module):
    """
    1D CNN for gesture classification, specifically optimized for padded variable-length sequences.
    
    This version:
    - Uses masking at every convolutional step to prevent edge effects from padding.
    - Exclusively uses a correctly masked Global Average Pooling.
    """
    def __init__(self,
                 input_channels: int,
                 sequence_length: int = 100,
                 num_classes: int = 18,
                 filters: list = None,
                 kernel_sizes: list = None):
        super(CNN1D, self).__init__()
        
        if filters is None:
            filters = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]
        assert len(filters) == len(kernel_sizes), "filters and kernel_sizes length mismatch"

        layers = []
        self.bn_layers = nn.ModuleList()
        in_channels = input_channels
        for out_c, k in zip(filters, kernel_sizes):
            # 使用 'same' padding 可以在不缩减序列长度的情况下更容易处理mask
            layers.append(nn.Conv1d(in_channels, out_c, kernel_size=k, padding='same'))
            self.bn_layers.append(nn.BatchNorm1d(out_c))
            in_channels = out_c
        self.conv_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # ✨ 输出维度固定为最后一层卷积的滤波器数量
        self.cnn_output_size = filters[-1]
    
    def forward(self, x, mask=None):
        """
        Defines the forward pass with robust masking at each step.
        
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, channels)
            mask (Tensor, optional): Mask tensor of shape (batch, seq_len) where 1s are real data.
        """
        # (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # ✨ 如果没有提供mask，创建一个全为1的默认mask
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[2], device=x.device)

        for i, conv in enumerate(self.conv_layers):
            # ✨ 核心步骤 1: 在卷积前，确保被mask的输入为0
            # unsqueeze(1) 将 mask 从 (B, S) 变为 (B, 1, S) 以便广播
            x = x * mask.unsqueeze(1)
            
            x = conv(x)
            x = self.bn_layers[i](x)
            x = self.activation(x)
            
            # ✨ 核心步骤 2: 在激活后，再次应用mask，消除卷积在padding区域产生的任何边缘效应
            x = x * mask.unsqueeze(1)

            # 在最后一层卷积后不进行池化
            if i < len(self.conv_layers) - 1:
                x = self.pool(x)
                # ✨ 核心步骤 3: MaxPool后，mask的长度也必须相应缩减
                mask_for_pooling = mask.unsqueeze(1).float() 
                # 2. 应用池化操作
                pooled_mask = self.pool(mask_for_pooling)
                # 3. 移除临时的通道维度 (B, 1, L_new) -> (B, L_new)
                mask = pooled_mask.squeeze(1)

        # --- 正确实现的、带Mask的全局平均池化 ---
        # x shape: (batch, channels, reduced_seq_len)
        # mask shape: (batch, reduced_seq_len)
        
        # 扩展mask维度以便广播: (B, reduced_seq_len) -> (B, 1, reduced_seq_len)
        mask_expanded = mask.unsqueeze(1)
        
        # 分子: 对所有未被mask的元素求和
        # (x * mask_expanded) 确保我们只对真实数据求和
        numerator = torch.sum(x * mask_expanded, dim=2)
        
        # 分母: 计算每个序列的真实长度（即mask中1的个数）
        denominator = torch.sum(mask, dim=1).unsqueeze(1).clamp(min=1e-9)
        
        # 计算带掩码的平均值
        pooled = numerator / denominator

        # pooled 的形状已经是 (batch, channels), 无需 view 或 flatten
        return pooled

    def get_model_info(self):
        """Get model parameter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
        }
