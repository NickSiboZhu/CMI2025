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
                 kernel_sizes: list = None,
                 # NEW: Temporal aggregation options
                 temporal_aggregation: str = 'global_pool',  # 'global_pool' or 'temporal_encoder'
                 temporal_mode: str = 'lstm',  # 'lstm' or 'transformer' (when using temporal_encoder)
                 lstm_hidden: int = 256,
                 lstm_layers: int = 1,
                 bidirectional: bool = True,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 ff_dim: int = 512,
                 dropout: float = 0.1,
                 # NEW: ResNet-style residual connections
                 use_residual: bool = False):
        super(CNN1D, self).__init__()
        
        if filters is None:
            filters = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]
        assert len(filters) == len(kernel_sizes), "filters and kernel_sizes length mismatch"

        layers = []
        self.bn_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()  # For dimension matching in residual connections
        
        self.use_residual = use_residual
        
        in_channels = input_channels
        for i, (out_c, k) in enumerate(zip(filters, kernel_sizes)):
            # 使用 'same' padding 可以在不缩减序列长度的情况下更容易处理mask
            layers.append(nn.Conv1d(in_channels, out_c, kernel_size=k, padding='same'))
            self.bn_layers.append(nn.BatchNorm1d(out_c))
            
            # Add 1x1 projection for residual connections when dimensions don't match
            if use_residual and in_channels != out_c:
                self.residual_projections.append(nn.Conv1d(in_channels, out_c, kernel_size=1))
            else:
                self.residual_projections.append(None)
            
            in_channels = out_c
            
        self.conv_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Store temporal aggregation choice
        self.temporal_aggregation = temporal_aggregation
        
        if temporal_aggregation == 'temporal_encoder':
            # Import and setup temporal encoder (reuse from cnn2d.py)
            from .cnn2d import TemporalEncoder
            
            # Calculate sequence length after conv layers (with pooling)
            reduced_seq_len = sequence_length
            for i in range(len(filters) - 1):  # Pool after all but last layer
                reduced_seq_len = reduced_seq_len // 2
            
            self.temporal_encoder = TemporalEncoder(
                mode=temporal_mode,
                input_dim=filters[-1],  # Last conv filter count
                seq_len=reduced_seq_len,
                # LSTM params
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional,
                # Transformer params
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                dropout=dropout
            )
            self.cnn_output_size = self.temporal_encoder.output_dim
        else:
            # Use traditional global pooling
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
            
            # Store input for potential residual connection
            if self.use_residual:
                residual = x
            
            x = conv(x)
            x = self.bn_layers[i](x)
            
            # Apply residual connection if enabled
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    # Project residual to match output dimensions
                    residual = self.residual_projections[i](residual)
                    # Apply mask to projected residual
                    residual = residual * mask.unsqueeze(1)
                
                # Add residual connection before activation
                x = x + residual
            
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

        # --- Choose temporal aggregation method ---
        if self.temporal_aggregation == 'temporal_encoder':
            # Use LSTM/Transformer temporal encoder
            # x shape: (batch, channels, reduced_seq_len)
            # Need to transpose for temporal encoder: (batch, reduced_seq_len, channels)
            x = x.transpose(1, 2)
            
            # Apply temporal encoder with mask
            output = self.temporal_encoder(x, mask)
            
        else:
            # --- Original: 正确实现的、带Mask的全局平均池化 ---
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
            output = numerator / denominator

        # output shape is (batch, output_dim) for both methods
        return output
