import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import MODELS
from torch.nn.utils.rnn import pack_padded_sequence

from timm.layers import SqueezeExcite as TimmSE

    

@MODELS.register_module()
class SpectrogramCNN(nn.Module):
    """
    A highly configurable 2D CNN for processing spectrograms.
    MODIFIED: Fixed a bug by using MaxPool1d for mask downsampling.
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
            self.conv_layers.append(nn.Conv2d(current_channels, out_c, kernel_size=k, padding='same'))
            self.bn_layers.append(nn.BatchNorm2d(out_c))
            if use_residual and current_channels != out_c:
                self.residual_projections.append(nn.Conv2d(current_channels, out_c, kernel_size=1))
            else:
                self.residual_projections.append(None)
            current_channels = out_c
        
        self.activation = nn.ReLU()
        # 2D池化层，用于处理特征图
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # --- NEW: 为1D掩码专门定义一个1D池化层 ---
        self.mask_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor = None) -> torch.Tensor:
        if time_mask is None:
            time_mask = torch.ones(x.shape[0], x.shape[3], device=x.device)

        for i, conv in enumerate(self.conv_layers):
            # 将1D时间掩码扩展为4D以便广播
            mask_4d = time_mask.view(x.shape[0], 1, 1, x.shape[3])

            x = x * mask_4d
            
            residual = x if self.use_residual else None
            
            x = conv(x)
            x = self.bn_layers[i](x)
            
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    residual = self.residual_projections[i](residual)
                    residual = residual * mask_4d
                x = x + residual
            
            x = self.activation(x)
            x = x * mask_4d
            
            # 特征图使用2D池化（动态保护：在时间维度过短时不再沿时间降采样）
            h, w = x.shape[2], x.shape[3]
            kh = 2 if h >= 2 else 1
            kw = 2 if w >= 2 else 1
            if kh > 1 or kw > 1:
                x = F.max_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
            
            # --- MODIFIED: 使用正确的1D池化来缩减掩码（仅当时间维长度>=2时） ---
            if kw > 1:
                mask_for_pooling = time_mask.unsqueeze(1).float()
                pooled_mask = self.mask_pool(mask_for_pooling)
                time_mask = pooled_mask.squeeze(1)
            
        x = self.final_pool(x)
        output = x.view(x.size(0), -1) 
        
        return output
    

class TemporalEncoder(nn.Module):
    """
    A unified temporal encoder that supports both LSTM and Transformer architectures.
    
    This module processes sequential features and outputs a fixed-size representation.
    
    Args:
        mode (str): 'lstm' or 'transformer' - the temporal modeling approach
        input_dim (int): Dimension of input features at each timestep
        seq_len (int): Maximum sequence length
        
        # LSTM specific args:
        lstm_hidden (int): Hidden size for LSTM
        lstm_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        
        # Transformer specific args:
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer layers
        ff_dim (int): Feedforward dimension in transformer
        dropout (float): Dropout rate
    """
    
    def __init__(self,
                 mode: str,
                 input_dim: int,
                 seq_len: int,
                 # LSTM specific args (required if mode is 'lstm')
                 lstm_hidden: int = None,
                 lstm_layers: int = None,
                 bidirectional: bool = None,
                 # Transformer specific args (required if mode is 'transformer')
                 num_heads: int = None,
                 num_layers: int = None,
                 ff_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.mode = mode
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        if mode == 'lstm':
            # --- Strict configuration check for LSTM ---
            if not all([lstm_hidden, lstm_layers is not None, bidirectional is not None]):
                raise ValueError("`lstm_hidden`, `lstm_layers`, and `bidirectional` must be provided for LSTM mode.")
            
            self.bidirectional = bidirectional
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.output_dim = lstm_hidden * (2 if bidirectional else 1)
            
        elif mode == 'transformer':
            # --- Strict configuration check for Transformer ---
            if not all([num_heads, num_layers, ff_dim]):
                 raise ValueError("`num_heads`, `num_layers`, and `ff_dim` must be provided for Transformer mode.")

            # Learnable [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

            # Register positional encoding at init so it exists for strict state_dict loading
            # Shape: (1, seq_len + 1 [CLS], input_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, seq_len + 1, self.input_dim))
            self.dropout = nn.Dropout(dropout)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-LN for better stability
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output_dim = input_dim
            
        else:
            raise ValueError(f"Unknown temporal encoder mode: {mode}")
    
    def forward(self, features, mask=None):
        """
        Process sequential features and return a fixed-size representation.
        
        Args:
            features: (batch_size, seq_len, input_dim) 
            mask: (batch_size, seq_len) - 1 for valid positions, 0 for padding
            
        Returns:
            (batch_size, output_dim) - aggregated temporal representation
        """
        batch_size, seq_len, _ = features.shape
        
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=features.device)
            
        if self.mode == 'lstm':
            # LSTM processing with proper sequence packing
            lengths = mask.sum(dim=1).to(torch.int64)
            
            # Align sequences (assuming padding at beginning)
            aligned_features = torch.zeros_like(features)
            for i in range(batch_size):
                L = lengths[i]
                if L > 0:
                    aligned_features[i, :L] = features[i, -L:]
            
            # Pack sequences
            packed_input = pack_padded_sequence(
                aligned_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # LSTM forward
            _, (h_n, _) = self.lstm(packed_input)
            
            # Extract final hidden state
            if self.bidirectional:
                output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            else:
                output = h_n[-1,:,:]
                
        elif self.mode == 'transformer':
            # Transformer processing with [CLS] token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            transformer_input = torch.cat([cls_tokens, features], dim=1)
            
            # Add positional encoding (resize on-the-fly without replacing the registered parameter)
            current_seq_len = transformer_input.size(1)  # seq_len + 1 for CLS
            if self.pos_encoding.size(1) != current_seq_len:
                # Safely adapt by slicing or padding the existing encoding tensor
                if self.pos_encoding.size(1) > current_seq_len:
                    pos_enc = self.pos_encoding[:, :current_seq_len]
                else:
                    pad_len = current_seq_len - self.pos_encoding.size(1)
                    pad = self.pos_encoding[:, -1:].repeat(1, pad_len, 1)
                    pos_enc = torch.cat([self.pos_encoding, pad], dim=1)
            else:
                pos_enc = self.pos_encoding

            transformer_input = transformer_input + pos_enc
            transformer_input = self.dropout(transformer_input)
            
            # Create attention mask
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            transformer_mask = torch.cat([cls_mask, mask], dim=1)
            attention_mask = (transformer_mask == 0)
            
            # Apply transformer
            transformer_output = self.transformer(
                transformer_input, 
                src_key_padding_mask=attention_mask
            )
            
            # Extract [CLS] representation
            output = transformer_output[:, 0]
            
        return output


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

        # Build conv layers manually for residual connections
        self.conv_blocks = nn.ModuleList()
        self.se_layers = nn.ModuleList() if use_se else None
        self.residual_projections = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_c, k) in enumerate(zip(conv_channels, kernel_sizes)):
            # Create conv block
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_c, kernel_size=k, padding='same'),
                nn.BatchNorm2d(out_c)
            )
            self.conv_blocks.append(conv_block)

            if self.se_layers is not None:
                # timm 1.0.15 signature: SqueezeExcite(channels, rd_ratio=1/16, rd_channels=None, ...)
                rd_ratio = 1.0 / max(1, se_reduction)
                self.se_layers.append(TimmSE(channels=out_c, rd_ratio=rd_ratio))
            
            # Add residual projection if needed
            if use_residual and in_channels != out_c:
                self.residual_projections.append(nn.Conv2d(in_channels, out_c, kernel_size=1))
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
    
    def forward(self, tof_grids):
        """
        Forward pass for TOF 2D CNN with optional residual connections.
        
        Args:
            tof_grids: Tensor of shape (batch_size, num_sensors, 8, 8)
                      Each 8x8 grid represents one TOF sensor's depth map
        
        Returns:
            Tensor of shape (batch_size, last_conv_channels)
        """
        x = tof_grids

        # Apply optional pre-conv sensor gate
        if self.use_sensor_gate:
            if self.sensor_gate_adaptive:
                gate = self.sensor_gate(x)
                x = x * gate
            else:
                x = x * self.sensor_gate
        
        # Process through conv blocks with optional residual connections
        for i, conv_block in enumerate(self.conv_blocks):
            if self.use_residual:
                residual = x
            
            # Apply conv block
            x = conv_block(x)

            # Apply SE before residual addition (SENet-style)
            if self.se_layers is not None:
                x = self.se_layers[i](x)
            
            # Apply residual connection if enabled
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    # Project residual to match output dimensions
                    residual = self.residual_projections[i](residual)
                x = x + residual
            
            x = self.activation(x)
            
            # Apply pooling (except for last layer)
            if i < len(self.conv_blocks) - 1:
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
        # Strict requirements for temporal params
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
    
    def forward(self, tof_sequence, mask=None):
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
        
        # Apply 2D CNN to all timesteps
        spatial_features = self.spatial_cnn(tof_grids_flat)  # (batch_size * seq_len, out_features)
        
        # Reshape back to sequential format
        spatial_features = spatial_features.view(batch_size, seq_len, self.spatial_out_dim)  # (B, L, C)

        # --- Temporal feature aggregation ---
        temporal_features = self.temporal_encoder(spatial_features, mask)  # (B, output_dim)
        
        return temporal_features