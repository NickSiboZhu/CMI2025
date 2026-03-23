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
            # Keep running stats in float32 for stability under mixed precision.
            self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float32))
            self.register_buffer("running_var",  torch.ones(num_features,  dtype=torch.float32))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var",  None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor = None, channel_mask: torch.Tensor = None):
        N, C, H, W = x.shape
        if time_mask is None:
            m_t = torch.ones(N, 1, 1, W, device=x.device, dtype=torch.float32)
        else:
            m_t = time_mask.view(N, 1, 1, W).to(torch.float32)

        if channel_mask is None:
            m_c = torch.ones(N, C, 1, 1, device=x.device, dtype=torch.float32)
        else:
            m_c = channel_mask.view(N, C, 1, 1).to(torch.float32)

        M = m_t * m_c

        denom = (M.sum(dim=(0, 2, 3)) * float(H)).clamp_(min=1.0)

        x_f32 = x.to(torch.float32)
        sum_  = (x_f32 * M).sum(dim=(0, 2, 3))
        mean  = sum_ / denom
        var   = ((x_f32 - mean.view(1, C, 1, 1))**2 * M).sum(dim=(0, 2, 3)) / denom

        if self.training and self.track_running_stats:
            total_pos   = torch.tensor(N * H * W, dtype=torch.float32, device=x.device)
            valid_ratio = (denom / total_pos)
            cond        = (valid_ratio > 1e-3).to(torch.float32)

            with torch.no_grad():
                mom = torch.tensor(self.momentum, dtype=torch.float32, device=x.device)
                new_rm = self.running_mean * (1 - mom) + mean * mom
                new_rv = self.running_var  * (1 - mom) + var  * mom
                self.running_mean.copy_(torch.lerp(self.running_mean, new_rm, cond))
                self.running_var.copy_( torch.lerp(self.running_var,  new_rv, cond))
                self.num_batches_tracked.add_( (cond.max() > 0).to(torch.long) )

        ref_mean = mean if (self.training or not self.track_running_stats) else self.running_mean
        ref_var  = var  if (self.training or not self.track_running_stats) else self.running_var

        x_hat = (x_f32 - ref_mean.view(1, C, 1, 1)) / torch.sqrt(ref_var.view(1, C, 1, 1) + self.eps)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, C, 1, 1).to(x_hat.dtype) + self.bias.view(1, C, 1, 1).to(x_hat.dtype)

        # Return zeros for invalid positions so later residual paths stay clean.
        return x_hat.to(x.dtype) * M.to(x.dtype)

@MODELS.register_module()
class SpectrogramCNN(nn.Module):
    """
    A highly configurable 2D CNN for processing spectrograms.

    The branch keeps its temporal mask synchronized with pooling so the final
    global average never mixes in padded time bins.
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.mask_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor = None) -> torch.Tensor:
        if time_mask is None:
            time_mask = torch.ones(x.shape[0], x.shape[3], device=x.device, dtype=torch.float)

        time_mask = time_mask.float()

        for i, conv in enumerate(self.conv_layers):
            mask_4d = time_mask.view(x.shape[0], 1, 1, time_mask.shape[1])

            x = x * mask_4d
            
            residual = x if self.use_residual else None
            
            x = conv(x)
            x = self.bn_layers[i](x, time_mask, None)
            
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    residual = self.residual_projections[i](residual)
                    residual = residual * mask_4d
                x = x + residual
            
            x = self.activation(x)
            x = x * mask_4d
            
            # Stop downsampling once either spatial axis is already length 1.
            h, w = x.shape[2], x.shape[3]
            kh = 2 if h >= 2 else 1
            kw = 2 if w >= 2 else 1
            if kh > 1 or kw > 1:
                x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
            
            if kw > 1:
                mask_for_pooling = time_mask.unsqueeze(1)
                pooled_mask = self.mask_pool(mask_for_pooling)
                time_mask = pooled_mask.squeeze(1)
        
        final_mask_4d = time_mask.view(x.shape[0], 1, 1, time_mask.shape[1])

        numerator = torch.sum(x * final_mask_4d, dim=(2, 3))

        feature_height = x.shape[2]
        valid_time_steps = torch.sum(time_mask, dim=1)
        denominator = (feature_height * valid_time_steps).unsqueeze(1).clamp(min=1e-9)

        output = numerator / denominator
        
        return output
    

class TemporalEncoder(nn.Module):
    """
    Temporal encoder shared by the 1D and 2D branches.

    The LSTM path pushes ``pack_padded_sequence`` into eager mode so compiled
    graphs do not have to materialize CPU length tensors.
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
                raise ValueError("`lstm_hidden`, `lstm_layers`, and `bidirectional` must be provided in LSTM mode.")
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
                raise ValueError("`num_heads`, `num_layers`, and `ff_dim` must be provided in transformer mode.")

            # A learnable CLS token keeps the transformer path aligned with the
            # classifier interface used by the LSTM path.
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

            self.classifier = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.output_dim)
            )
        else:
            raise ValueError(f"Unknown temporal encoder mode: {mode}")

    @staticmethod
    @torch._dynamo.disable
    def _pack_padded_sequence_eager(x_padded: torch.Tensor,
                                    lengths_gpu_int64: torch.Tensor):
        lengths_cpu = lengths_gpu_int64.to('cpu', dtype=torch.int64)
        return pack_padded_sequence(x_padded, lengths_cpu, batch_first=True, enforce_sorted=False)

    def forward(self, features, mask=None):
        """
        Encode ``features`` with shape ``(B, S, C)`` and a ``(B, S)`` mask.
        """
        B, S, C = features.shape

        if mask is None:
            mask = torch.ones(B, S, device=features.device)

        if self.mode == 'lstm':
            lengths = mask.sum(dim=1).to(torch.int64).clamp_(min=0, max=S)

            # Training uses left padding, while ``pack_padded_sequence`` expects
            # valid tokens at the front of each sequence.
            t = torch.arange(S, device=features.device).unsqueeze(0).expand(B, S)
            start = (S - lengths).clamp(min=0)
            src_idx = (start.unsqueeze(1) + t).clamp(max=S - 1)
            aligned = torch.gather(features, 1, src_idx.unsqueeze(-1).expand(-1, -1, C))
            keep = (t < lengths.unsqueeze(1)).unsqueeze(-1)
            aligned = aligned * keep

            packed = self._pack_padded_sequence_eager(aligned, lengths)

            _, (h_n, _) = self.lstm(packed)

            if self.bidirectional:
                out = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                out = h_n[-1]
            return out

        else:
            cls = self.cls_token.to(dtype=features.dtype, device=features.device).expand(B, 1, -1)
            x = torch.cat([cls, features], dim=1)

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
            x = x + pos_enc.to(dtype=x.dtype, device=x.device)
            x = self.dropout(x)

            cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            transformer_mask = torch.cat([cls_mask, mask], dim=1)
            attention_mask = (transformer_mask == 0).to(torch.bool).contiguous()

            # Force the math kernel here because some fused attention kernels are
            # stricter about tensor layout than this branch guarantees.
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                x = self.transformer(x, src_key_padding_mask=attention_mask)

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
        # ``out_features`` is implicit: the last conv width defines the branch output.
        assert len(conv_channels) == len(kernel_sizes), "conv_channels and kernel_sizes length mismatch"

        if self.use_se and se_reduction is None:
            raise ValueError("'se_reduction' must be provided when 'use_se' is True.")
        if self.use_sensor_gate and sensor_gate_adaptive is None:
            raise ValueError("'sensor_gate_adaptive' must be provided when 'use_sensor_gate' is True.")
        if self.use_sensor_gate and (not sensor_gate_adaptive) and sensor_gate_init is None:
            raise ValueError("'sensor_gate_init' must be provided when 'use_sensor_gate' is True and not adaptive.")

        self.conv_layers = nn.ModuleList()
        self.bn_layers   = nn.ModuleList()
        self.se_layers   = nn.ModuleList() if use_se else None
        self.residual_projections = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_c, k) in enumerate(zip(conv_channels, kernel_sizes)):
            self.conv_layers.append(nn.Conv2d(in_channels, out_c, kernel_size=k, padding='same', bias=True))
            self.bn_layers.append(MaskedBatchNorm2d(out_c))

            if self.se_layers is not None:
                rd_ratio = 1.0 / max(1, se_reduction)
                self.se_layers.append(TimmSE(channels=out_c, rd_ratio=rd_ratio))
            
            if use_residual and in_channels != out_c:
                self.residual_projections.append(nn.Conv2d(in_channels, out_c, kernel_size=1, bias=True))
            else:
                self.residual_projections.append(None)
            
            in_channels = out_c
        
        self.activation = nn.ReLU()
        self.conv_channels = conv_channels
        self.pool = nn.MaxPool2d(2)
        self.last_conv_channels = conv_channels[-1]
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if use_sensor_gate:
            if sensor_gate_adaptive:
                self.sensor_gate = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.input_channels, max(1, self.input_channels // 2), kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(1, self.input_channels // 2), self.input_channels, kernel_size=1),
                    nn.Sigmoid()
                )
            else:
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
        
        self.out_features = self.last_conv_channels
    
    def forward(self, tof_grids, frame_mask: torch.Tensor = None, channel_mask: torch.Tensor = None):
        """
        Forward pass for TOF 2D CNN with optional residual connections.
        
        Args:
            tof_grids: Tensor of shape (batch_size, num_sensors, 8, 8)
                      Each 8x8 grid represents one TOF sensor's depth map
            frame_mask: ``(batch_size,)`` or ``(batch_size, 1)`` with 1 for valid frames.
        Returns:
            Tensor of shape (batch_size, last_conv_channels)
        """
        x = tof_grids

        if self.use_sensor_gate:
            if self.sensor_gate_adaptive:
                gate = self.sensor_gate(x)
                x = x * gate
            else:
                x = x * self.sensor_gate
        
        # Zero missing sensors before the first convolution so they never affect BN.
        if channel_mask is not None:
            x = x * channel_mask.view(x.size(0), self.input_channels, 1, 1)
        
        for i in range(len(self.conv_layers)):
            if self.use_residual:
                residual = x
            
            x = self.conv_layers[i](x)

            if frame_mask is None:
                tm = None
            else:
                W = x.size(3)
                tm = frame_mask.view(-1, 1).float().expand(-1, W)

            x = self.bn_layers[i](x, tm, None)

            if frame_mask is None:
                fm4d = None
            else:
                fm4d = frame_mask.view(-1, 1, 1, 1).float()

            if self.se_layers is not None:
                x = self.se_layers[i](x)
            
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    residual = self.residual_projections[i](residual)
                if fm4d is not None:
                    residual = residual * fm4d
                x = x + residual
            
            x = self.activation(x)
            if fm4d is not None:
                x = x * fm4d
            
            if i < len(self.conv_layers) - 1:
                x = self.pool(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
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
                 use_residual: bool,
                 use_se: bool,
                 use_sensor_gate: bool,
                 se_reduction: int = None,
                 sensor_gate_adaptive: bool = None,
                 sensor_gate_init: float = None,
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
        
        self.spatial_out_dim = self.spatial_cnn.out_features
        
        from .cnn2d import TemporalEncoder

        if temporal_mode == 'lstm' and (lstm_hidden is None or lstm_layers is None or bidirectional is None):
            raise ValueError("For LSTM temporal_mode, provide lstm_hidden, lstm_layers, bidirectional.")
        if temporal_mode == 'transformer' and (num_heads is None or num_layers is None or ff_dim is None or dropout is None):
            raise ValueError("For transformer temporal_mode, provide num_heads, num_layers, ff_dim, dropout.")

        self.temporal_encoder = TemporalEncoder(
            mode=temporal_mode,
            input_dim=self.spatial_out_dim,
            seq_len=seq_len,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        
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

        tof_grids_flat = tof_sequence.view(batch_size * seq_len, self.input_channels, 8, 8)

        frame_mask = mask.reshape(batch_size * seq_len)

        # Expand per-sequence sensor masks to the frame axis before the spatial CNN.
        if channel_mask is not None:
            channel_mask_flat = channel_mask.repeat_interleave(seq_len, dim=0)
        else:
            channel_mask_flat = None

        spatial_features = self.spatial_cnn(
            tof_grids_flat,
            frame_mask=frame_mask,
            channel_mask=channel_mask_flat,
        )
        
        spatial_features = spatial_features.view(batch_size, seq_len, self.spatial_out_dim)

        temporal_features = self.temporal_encoder(spatial_features, mask)
        
        return temporal_features
