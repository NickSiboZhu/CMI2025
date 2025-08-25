import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import MODELS
from torch.nn.utils.rnn import pack_padded_sequence

from timm.layers import SqueezeExcite as TimmSE


class MaskedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            # 运行统计固定 float32，更抗抖动
            self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float32))
            self.register_buffer("running_var",  torch.ones(num_features,  dtype=torch.float32))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var",  None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor = None, channel_mask: torch.Tensor = None):
        # x: (N, C, H, W)
        # time_mask:    (N, W)  取值{0,1} —— 用于掩蔽 padding 的时间步
        # channel_mask: (N, C)  取值{0,1} —— 用于掩蔽整通道（例如缺失的 TOF 传感器）
        N, C, H, W = x.shape
        if time_mask is None:
            m_t = torch.ones(N, 1, 1, W, device=x.device, dtype=torch.float32)
        else:
            m_t = time_mask.view(N, 1, 1, W).to(torch.float32)

        if channel_mask is None:
            m_c = torch.ones(N, C, 1, 1, device=x.device, dtype=torch.float32)
        else:
            m_c = channel_mask.view(N, C, 1, 1).to(torch.float32)

        # 组合掩码：按样本-通道-时间的有效位置统计，空间高度 H 无掩码
        M = m_t * m_c  # (N, C, 1, W)

        # 每个通道的有效元素数（标量/通道）：sum_{N,W}(M) × H
        denom = (M.sum(dim=(0, 2, 3)) * float(H)).clamp_(min=1.0)  # (C,)

        x_f32 = x.to(torch.float32)
        sum_  = (x_f32 * M).sum(dim=(0, 2, 3))                     # (C,)
        mean  = sum_ / denom                                       # (C,)
        var   = ((x_f32 - mean.view(1, C, 1, 1))**2 * M).sum(dim=(0, 2, 3)) / denom

        if self.training and self.track_running_stats:
            # 纯张量条件：有效比例 = denom / (N*H*W)
            total_pos   = torch.tensor(N * H * W, dtype=torch.float32, device=x.device)
            # denom 是 (C,), 计算每通道的有效比例
            valid_ratio = (denom / total_pos)                        # (C,)
            cond        = (valid_ratio > 1e-3).to(torch.float32)     # (C,)

            # 纯张量更新，避免 Python 调用导致的 CALL 字节码
            with torch.no_grad():
                mom = torch.tensor(self.momentum, dtype=torch.float32, device=x.device)
                new_rm = self.running_mean * (1 - mom) + mean * mom
                new_rv = self.running_var  * (1 - mom) + var  * mom
                # cond==0 时保持原值；cond==1 时更新（逐通道）
                self.running_mean.copy_(torch.lerp(self.running_mean, new_rm, cond))
                self.running_var.copy_( torch.lerp(self.running_var,  new_rv, cond))
                # 若至少一个通道有效，则计数 +1
                self.num_batches_tracked.add_( (cond.max() > 0).to(torch.long) )

        ref_mean = mean if (self.training or not self.track_running_stats) else self.running_mean
        ref_var  = var  if (self.training or not self.track_running_stats) else self.running_var

        x_hat = (x_f32 - ref_mean.view(1, C, 1, 1)) / torch.sqrt(ref_var.view(1, C, 1, 1) + self.eps)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, C, 1, 1).to(x_hat.dtype) + self.bias.view(1, C, 1, 1).to(x_hat.dtype)

        # cast 回输入 dtype，并零掉无效时间位
        return x_hat.to(x.dtype) * M.to(x.dtype)

@MODELS.register_module()
class SpectrogramCNN(nn.Module):
    """
    A highly configurable 2D CNN for processing spectrograms.
    MODIFIED: Fixed a bug by using MaxPool1d for mask downsampling and
              implementing a mask-aware final global average pooling.
    """
    def __init__(self,
                 in_channels: int,
                 filters: list,
                 kernel_sizes: list,
                 use_residual: bool):
        super().__init__()

        assert len(filters) == len(kernel_sizes), "filters and kernel_sizes length mismatch"

        self.use_residual = use_residual
        self.out_features = filters[-1]

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        current_channels = in_channels
        for i, (out_c, k) in enumerate(zip(filters, kernel_sizes)):
            self.conv_layers.append(nn.Conv2d(current_channels, out_c, kernel_size=k, padding='same', bias=True))
            self.bn_layers.append(MaskedBatchNorm2d(out_c))
            if use_residual and current_channels != out_c:
                self.residual_projections.append(nn.Conv2d(current_channels, out_c, kernel_size=1, stride=1, padding=0, bias=True))
            else:
                self.residual_projections.append(None)
            current_channels = out_c
        
        self.activation = nn.ReLU()
        # 2D池化层，用于处理特征图
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # --- 为1D掩码专门定义一个1D池化层 ---
        self.mask_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # <<< 修改: 移除了不支持mask的AdaptiveAvgPool2d >>>
        # self.final_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor = None) -> torch.Tensor:
        if time_mask is None:
            time_mask = torch.ones(x.shape[0], x.shape[3], device=x.device, dtype=torch.float)

        # 确保 time_mask 是浮点型以便乘法和池化
        time_mask = time_mask.float()

        for i, conv in enumerate(self.conv_layers):
            # 将1D时间掩码扩展为4D以便广播 (B, 1, 1, T)
            mask_4d = time_mask.view(x.shape[0], 1, 1, time_mask.shape[1])

            x = x * mask_4d
            
            residual = x if self.use_residual else None
            
            x = conv(x)
            # Spectrograms没有按通道的缺失掩码，这里传 None
            x = self.bn_layers[i](x, time_mask, None)
            
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    residual = self.residual_projections[i](residual)
                    residual = residual * mask_4d
                x = x + residual
            
            x = self.activation(x)
            x = x * mask_4d
            
            # 特征图使用2D池化（动态保护：在时间或频率维度过短时不再降采样）
            h, w = x.shape[2], x.shape[3]
            kh = 2 if h >= 2 else 1
            kw = 2 if w >= 2 else 1
            if kh > 1 or kw > 1:
                x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
            
            # 使用正确的1D池化来缩减掩码（仅当时间维长度>=2时）
            if kw > 1:
                mask_for_pooling = time_mask.unsqueeze(1) # (B, 1, T)
                pooled_mask = self.mask_pool(mask_for_pooling)
                time_mask = pooled_mask.squeeze(1) # (B, T_new)
        
        # <<< 修改: 使用手动实现的、支持mask的全局平均池化替换原有的 final_pool >>>
        # 这是为了修复原代码中 `AdaptiveAvgPool2d` 会错误地将填充区域计入平均值的问题。
        
        # 1. 再次将最终的掩码扩展为4D，以确保广播正确
        final_mask_4d = time_mask.view(x.shape[0], 1, 1, time_mask.shape[1])

        # 2. 计算分子：对特征图的有效区域求和 (在H和W维度上)
        #    乘以 final_mask_4d 是一个安全的冗余操作，确保填充区为零
        numerator = torch.sum(x * final_mask_4d, dim=(2, 3)) # 结果形状: (B, C)

        # 3. 计算分母：计算每个样本的有效面积
        #    有效面积 = 特征图高度 * 每个样本的有效时间步数
        feature_height = x.shape[2]
        valid_time_steps = torch.sum(time_mask, dim=1) # 结果形状: (B)
        
        # 分母形状: (B, 1)，以便和 (B, C) 的分子进行广播
        denominator = (feature_height * valid_time_steps).unsqueeze(1).clamp(min=1e-9)

        # 4. 计算正确的、带掩码的平均值
        output = numerator / denominator
        # <<< 修改结束 >>>
        
        return output
    

class TemporalEncoder(nn.Module):
    """
    时序编码器：支持 'lstm' 与 'transformer'
    - LSTM 分支：用 @dynamo.disable 将 pack(lengths.cpu()) 放到 eager，避免 device_put
    - Transformer 分支：保持你原来的实现不变
    """

    def __init__(self,
                 mode: str,
                 input_dim: int,
                 seq_len: int,
                 # LSTM
                 lstm_hidden: int = None,
                 lstm_layers: int = None,
                 bidirectional: bool = None,
                 # Transformer
                 num_heads: int = None,
                 num_layers: int = None,
                 ff_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()

        self.mode = mode
        self.input_dim = input_dim
        self.seq_len = seq_len

        if mode == 'lstm':
            if not all([lstm_hidden, lstm_layers is not None, bidirectional is not None]):
                raise ValueError("`lstm_hidden`, `lstm_layers`, `bidirectional` 必须在 LSTM 模式下提供。")
            self.bidirectional = bool(bidirectional)
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
            self.output_dim = lstm_hidden * (2 if self.bidirectional else 1)

        elif mode == 'transformer':
            if not all([num_heads, num_layers, ff_dim]):
                raise ValueError("`num_heads`, `num_layers`, `ff_dim` 必须在 Transformer 模式下提供。")

            # 保持原样：可学习 CLS 与位置编码（最大长度 seq_len+1）
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
            self.pos_encoding = nn.Parameter(torch.randn(1, seq_len + 1, input_dim))
            self.dropout = nn.Dropout(dropout)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
            self.output_dim = input_dim

            # 保持你原先的分类头（如果你原来就是直接输出，也可删除这个）
            self.classifier = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.output_dim)
            )
        else:
            raise ValueError(f"Unknown temporal encoder mode: {mode}")

    # --- 仅把 pack(lengths.cpu()) 放到 eager，避免编译图里出现 CPU 迁移 ---
    @staticmethod
    @torch._dynamo.disable
    def _pack_padded_sequence_eager(x_padded: torch.Tensor,
                                    lengths_gpu_int64: torch.Tensor):
        lengths_cpu = lengths_gpu_int64.to('cpu', dtype=torch.int64)
        return pack_padded_sequence(x_padded, lengths_cpu, batch_first=True, enforce_sorted=False)

    def forward(self, features, mask=None):
        """
        features: (B, S, C)
        mask:     (B, S)  1=有效, 0=padding
        return:   (B, D)  D=self.output_dim
        """
        B, S, C = features.shape

        if mask is None:
            mask = torch.ones(B, S, device=features.device)

        if self.mode == 'lstm':
            # --- 计算有效长度（保持在 GPU）---
            lengths = mask.sum(dim=1).to(torch.int64).clamp_(min=0, max=S)  # (B,)

            # --- 将“左填充”对齐为“右填充”的有效片段（纯 GPU）---
            t = torch.arange(S, device=features.device).unsqueeze(0).expand(B, S)  # (B,S)
            start = (S - lengths).clamp(min=0)                                     # (B,)
            src_idx = (start.unsqueeze(1) + t).clamp(max=S - 1)                    # (B,S)
            aligned = torch.gather(features, 1, src_idx.unsqueeze(-1).expand(-1, -1, C))
            keep = (t < lengths.unsqueeze(1)).unsqueeze(-1)                        # (B,S,1)
            aligned = aligned * keep

            # --- 在 eager 中执行 pack（内部会把 lengths 移到 CPU）---
            packed = self._pack_padded_sequence_eager(aligned, lengths)

            # --- LSTM 前向（可被编译/图捕获）---
            _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers*dirs, B, H)

            if self.bidirectional:
                out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2H)
            else:
                out = h_n[-1]                               # (B, H)
            return out

        else:
            # --- Transformer 分支保持不变 ---
            cls = self.cls_token.expand(B, 1, -1).to(features.device)  # (B,1,C)
            x = torch.cat([cls, features], dim=1)                       # (B,S+1,C)

            # 动态适配位置编码（与原实现一致）
            cur_len = x.size(1)
            if self.pos_encoding.size(1) != cur_len:
                if self.pos_encoding.size(1) > cur_len:
                    pos_enc = self.pos_encoding[:, :cur_len]
                else:
                    pad_len = cur_len - self.pos_encoding.size(1)
                    pad = self.pos_encoding[:, -1:].repeat(1, pad_len, 1)
                    pos_enc = torch.cat([self.pos_encoding, pad], dim=1)
            else:
                pos_enc = self.pos_encoding

            x = x + pos_enc.to(x.device)
            x = self.dropout(x)

            cls_mask = torch.ones(B, 1, device=mask.device)
            transformer_mask = torch.cat([cls_mask, mask], dim=1)
            attention_mask = (transformer_mask == 0)

            x = self.transformer(x, src_key_padding_mask=attention_mask)  # (B,S+1,C)
            cls_out = x[:, 0]
            return self.classifier(cls_out)


class TOF2DCNN(nn.Module):
    """
    2D CNN for processing Time-of-Flight (TOF) sensor grids.
    
    Each TOF sensor provides an 8x8 grid of depth values.
    This module processes multiple TOF sensors and extracts spatial features.
    """
    
    def __init__(self, input_channels: int,
                 conv_channels: list, kernel_sizes: list,
                 use_residual: bool,
                 # NEW: Channel attention and pre-conv sensor gate
                 use_se: bool,
                 se_reduction: int = None,
                 use_sensor_gate: bool = False,
                 sensor_gate_adaptive: bool = None,
                 sensor_gate_init: float = None):
        super(TOF2DCNN, self).__init__()
        
        self.input_channels = input_channels
        self.use_residual = use_residual
        self.use_se = use_se
        self.use_sensor_gate = use_sensor_gate
        self.sensor_gate_adaptive = sensor_gate_adaptive
        # Note: out_features parameter kept for compatibility, but actual output
        # will be determined by the last conv channel
        
        # Strict config: require conv_channels and kernel_sizes
        assert len(conv_channels) == len(kernel_sizes), "conv_channels and kernel_sizes length mismatch"

        # Strict config: require related params based on toggles
        if self.use_se and se_reduction is None:
            raise ValueError("'se_reduction' must be provided when 'use_se' is True.")
        if self.use_sensor_gate and sensor_gate_adaptive is None:
            raise ValueError("'sensor_gate_adaptive' must be provided when 'use_sensor_gate' is True.")
        if self.use_sensor_gate and (not sensor_gate_adaptive) and sensor_gate_init is None:
            raise ValueError("'sensor_gate_init' must be provided when 'use_sensor_gate' is True and not adaptive.")

        # --- 修改：将 Conv+BN 拆开，BN 改为 MaskedBatchNorm2d ---
        self.conv_layers = nn.ModuleList()
        self.bn_layers   = nn.ModuleList()
        self.se_layers   = nn.ModuleList() if use_se else None
        self.residual_projections = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_c, k) in enumerate(zip(conv_channels, kernel_sizes)):
            self.conv_layers.append(nn.Conv2d(in_channels, out_c, kernel_size=k, padding='same', bias=True))  # bias=True
            self.bn_layers.append(MaskedBatchNorm2d(out_c))  # ← 关键替换

            if self.se_layers is not None:
                # timm 1.0.15 signature: SqueezeExcite(channels, rd_ratio=1/16, rd_channels=None, ...)
                rd_ratio = 1.0 / max(1, se_reduction)
                self.se_layers.append(TimmSE(channels=out_c, rd_ratio=rd_ratio))
            
            # Add residual projection if needed
            if use_residual and in_channels != out_c:
                self.residual_projections.append(nn.Conv2d(in_channels, out_c, kernel_size=1, bias=True))  # bias=True
            else:
                self.residual_projections.append(None)
            
            in_channels = out_c
        
        self.activation = nn.ReLU()
        self.conv_channels = conv_channels
        self.pool = nn.MaxPool2d(2)
        self.last_conv_channels = conv_channels[-1]
        
        # Global average pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Optional static or adaptive pre-conv sensor gate (on raw 5 channels)
        if use_sensor_gate:
            if sensor_gate_adaptive:
                # Adaptive gate via SE on raw sensor channels (5)
                self.sensor_gate = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.input_channels, max(1, self.input_channels // 2), kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(1, self.input_channels // 2), self.input_channels, kernel_size=1),
                    nn.Sigmoid()
                )
            else:
                # Static learnable per-sensor scale
                self.sensor_gate = nn.Parameter(torch.full((1, self.input_channels, 1, 1), float(sensor_gate_init)))
        
        # Final projection layer
        # COMMENTED OUT: Let the CNN features speak for themselves
        # Linear projection can harm the spatial information extracted by convolutions
        # self.projection = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(self.last_conv_channels, out_features),
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )
        
        # Store the actual output dimension
        self.out_features = self.last_conv_channels  # Use last conv channel count as output
    
    def forward(self, tof_grids, frame_mask: torch.Tensor = None, channel_mask: torch.Tensor = None):
        """
        Forward pass for TOF 2D CNN with optional residual connections.
        
        Args:
            tof_grids: Tensor of shape (batch_size, num_sensors, 8, 8)
                      Each 8x8 grid represents one TOF sensor's depth map
            frame_mask: (batch_size,) or (batch_size, 1) —— 1 表示该帧有效，0 表示该帧是 padding
        Returns:
            Tensor of shape (batch_size, last_conv_channels)
        """
        x = tof_grids  # (BL, C, 8, 8)

        # Apply optional pre-conv sensor gate
        if self.use_sensor_gate:
            if self.sensor_gate_adaptive:
                gate = self.sensor_gate(x)
                x = x * gate
            else:
                x = x * self.sensor_gate
        
        # Zero-out missing sensor channels at input according to channel_mask (shape: (BL, C_in))
        if channel_mask is not None:
            x = x * channel_mask.view(x.size(0), self.input_channels, 1, 1)
        
        # Process through conv blocks with optional residual connections
        for i in range(len(self.conv_layers)):
            if self.use_residual:
                residual = x
            
            # Apply conv
            x = self.conv_layers[i](x)

            # <<< 关键：将帧级 mask 扩展到当前 W 维，作为时间掩码传入 BN >>>
            if frame_mask is None:
                tm = None
            else:
                # frame_mask: (N,) or (N,1) -> (N, W)
                W = x.size(3)
                tm = frame_mask.view(-1, 1).float().expand(-1, W)

            # Apply masked BN
            # BN 仅按时间掩码统计；传感器缺失已在输入处置零
            x = self.bn_layers[i](x, tm, None)

            # 构造帧级 4D 掩码，用于在残差与激活后强制零化整帧无效样本
            if frame_mask is None:
                fm4d = None
            else:
                fm4d = frame_mask.view(-1, 1, 1, 1).float()

            # Apply SE before residual addition (SENet-style)
            if self.se_layers is not None:
                x = self.se_layers[i](x)
            
            # Apply residual connection if enabled
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    # Project residual to match output dimensions
                    residual = self.residual_projections[i](residual)
                # 仅在有效帧上引入残差，避免无效帧泄漏
                if fm4d is not None:
                    residual = residual * fm4d
                x = x + residual
            
            x = self.activation(x)
            # 激活后再次施加帧级掩码，确保无效帧保持为零
            if fm4d is not None:
                x = x * fm4d
            
            # Apply pooling (except for last layer)
            if i < len(self.conv_layers) - 1:
                x = self.pool(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, last_conv_channels, 1, 1)
        
        # Flatten to get feature vector
        x = x.view(x.size(0), -1)  # (batch, last_conv_channels)
        
        return x

@MODELS.register_module()
class TemporalTOF2DCNN(nn.Module):
    """
    2-stage module for sequential TOF data.

    Stage 1: `TOF2DCNN` extracts spatial features for every timestep.
    Stage 2: Temporal encoder (LSTM or Transformer) for temporal modeling.
    
    Args:
        temporal_mode (str): 'lstm' or 'transformer' - temporal modeling approach
        
        # Common temporal args:
        input_channels (int): Number of TOF sensors
        seq_len (int): Maximum sequence length
        out_features (int): Desired output dimension (ignored, uses conv_channels[-1])
        conv_channels (list): Channel sizes for spatial CNN
        kernel_sizes (list): Kernel sizes for spatial CNN
        
        # LSTM specific (used when temporal_mode='lstm'):
        lstm_hidden (int): Hidden size for LSTM
        lstm_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        
        # Transformer specific (used when temporal_mode='transformer'):
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        ff_dim (int): Dimension of feedforward network
        dropout (float): Dropout rate
    """

    def __init__(self,
                 input_channels: int,
                 seq_len: int,
                 conv_channels: list,
                 kernel_sizes: list,
                 temporal_mode: str,
                 # --- Optional Features for Spatial CNN (required flags, strict) ---
                 use_residual: bool,
                 use_se: bool,
                 use_sensor_gate: bool,
                 se_reduction: int = None,
                 sensor_gate_adaptive: bool = None,
                 sensor_gate_init: float = None,
                 # --- Temporal Encoder Params (required based on mode) ---
                 lstm_hidden: int = None,
                 lstm_layers: int = None,
                 bidirectional: bool = None,
                 num_heads: int = None,
                 num_layers: int = None,
                 ff_dim: int = None,
                 dropout: float = None):
        super(TemporalTOF2DCNN, self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len
        self.temporal_mode = temporal_mode

        # ------------- Spatial CNN applied per-frame -------------
        self.spatial_cnn = TOF2DCNN(
            input_channels=input_channels,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            use_residual=use_residual,
            use_se=use_se,
            se_reduction=se_reduction,
            use_sensor_gate=use_sensor_gate,
            sensor_gate_adaptive=sensor_gate_adaptive,
            sensor_gate_init=sensor_gate_init,
        )
        
        # Get the actual output dimension from spatial CNN
        self.spatial_out_dim = self.spatial_cnn.out_features
        
        # ------------- Temporal Encoder ---------------------
        # 严格校验保持不变（此处略，复用你已有的 TemporalEncoder 实现）
        from .cnn2d import TemporalEncoder  # 复用你现有的实现

        if temporal_mode == 'lstm' and (lstm_hidden is None or lstm_layers is None or bidirectional is None):
            raise ValueError("For LSTM temporal_mode, provide lstm_hidden, lstm_layers, bidirectional.")
        if temporal_mode == 'transformer' and (num_heads is None or num_layers is None or ff_dim is None or dropout is None):
            raise ValueError("For transformer temporal_mode, provide num_heads, num_layers, ff_dim, dropout.")

        self.temporal_encoder = TemporalEncoder(
            mode=temporal_mode,
            input_dim=self.spatial_out_dim,
            seq_len=seq_len,
            # LSTM params
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            # Transformer params
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
        # Update output features to match temporal encoder output
        self.out_features = self.temporal_encoder.output_dim
        
        # Note: No projection layer - direct temporal encoder output is used
    
    def forward(self, tof_sequence, mask=None, channel_mask: torch.Tensor = None):
        """
        Forward pass with unified temporal encoder.
        
        Args:
            tof_sequence: (batch_size, seq_len, features) - flattened TOF data
            mask: (batch_size, seq_len) - 1 for valid positions, 0 for padding
            
        Returns:
            (batch_size, output_dim) - temporal aggregation of spatial features
        """
        batch_size, seq_len, _ = tof_sequence.shape

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=tof_sequence.device)

        # --- Spatial feature extraction ---
        # Reshape to (batch_size * seq_len, num_sensors, 8, 8) for 2D CNN processing
        tof_grids_flat = tof_sequence.view(batch_size * seq_len, self.input_channels, 8, 8)

        # <<< 将时序 mask 展平为帧级 mask，传入 spatial_cnn 的 BN >>>
        frame_mask = mask.reshape(batch_size * seq_len)  # (BL,)

        # Apply 2D CNN to all timesteps (masked BN inside)
        # 将 per-sample 的通道掩码复制到每个时间步： (B, C) -> (BL, C)
        if channel_mask is not None:
            channel_mask_flat = channel_mask.repeat_interleave(seq_len, dim=0)  # (BL, C)
        else:
            channel_mask_flat = None

        spatial_features = self.spatial_cnn(
            tof_grids_flat,
            frame_mask=frame_mask,
            channel_mask=channel_mask_flat,
        )  # (BL, out_features)
        
        # Reshape back to sequential format
        spatial_features = spatial_features.view(batch_size, seq_len, self.spatial_out_dim)  # (B, L, C)

        # --- Temporal feature aggregation ---
        temporal_features = self.temporal_encoder(spatial_features, mask)  # (B, output_dim)
        
        return temporal_features