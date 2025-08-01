import torch
import torch.nn as nn
import numpy as np
from . import MODELS
from torch.nn.utils.rnn import pack_padded_sequence


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
    
    def __init__(self, mode='transformer', input_dim=128, seq_len=100,
                 # LSTM args
                 lstm_hidden=256, lstm_layers=2, bidirectional=True,
                 # Transformer args  
                 num_heads=8, num_layers=2, ff_dim=512, dropout=0.1):
        super().__init__()
        
        self.mode = mode
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        if mode == 'lstm':
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
            # Learnable [CLS] token
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
            
            # Positional encoding for seq_len + 1 (includes [CLS] token)
            self.pos_encoding = nn.Parameter(torch.randn(1, seq_len + 1, input_dim))
            
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
            
            # Add positional encoding
            transformer_input = transformer_input + self.pos_encoding
            
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
    
    def __init__(self, input_channels=5, out_features=128, 
                 conv_channels=None, kernel_sizes=None):
        super(TOF2DCNN, self).__init__()
        
        self.input_channels = input_channels
        # Note: out_features parameter kept for compatibility, but actual output
        # will be determined by the last conv channel
        
        # Dynamic 2D conv stack
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 2]
        assert len(conv_channels) == len(kernel_sizes), "conv_channels and kernel_sizes length mismatch"

        layers = []
        in_channels = input_channels  # Initialize with input_channels parameter
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
        Forward pass for TOF 2D CNN.
        
        Args:
            tof_grids: Tensor of shape (batch_size, num_sensors, 8, 8)
                      Each 8x8 grid represents one TOF sensor's depth map
        
        Returns:
            Tensor of shape (batch_size, last_conv_channels)
        """
        # Process through 2D convolutions
        x = self.conv_layers(tof_grids)  # (batch, last_conv_channels, H, W)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, last_conv_channels, 1, 1)
        
        # Flatten to get feature vector
        x = x.view(x.size(0), -1)  # (batch, last_conv_channels)
        
        # COMMENTED OUT: Direct CNN features without projection
        # x = self.projection(x)  # (batch, out_features)
        
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



@MODELS.register_module()
class TemporalTOF2DCNN(nn.Module):
    """
    2-stage module for sequential TOF data, now with support for appended flags.

    Stage 1: `TOF2DCNN` extracts spatial features for every timestep.
    Stage 2: Temporal encoder (LSTM or Transformer) for temporal modeling.
    
    The input `tof_sequence` can optionally contain appended flag features.
    The model intelligently separates pixel data from flag data.
    """

    def __init__(self, input_channels=5, seq_len=100, out_features=128,
                 conv_channels=None, kernel_sizes=None,
                 # Temporal encoder selection
                 temporal_mode='transformer',
                 # LSTM parameters
                 lstm_hidden=256, lstm_layers=2, bidirectional=True,
                 # Transformer parameters
                 num_heads=8, num_layers=2, ff_dim=512, dropout=0.1):
        super(TemporalTOF2DCNN, self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len
        self.temporal_mode = temporal_mode
        # Assume one flag per sensor
        self.num_flags = self.input_channels
        
        # ------------- Spatial CNN applied per-frame -------------
        # CNN will see pixel maps plus per-sensor flag maps as extra channels
        total_in_channels = input_channels + self.num_flags
        self.spatial_cnn = TOF2DCNN(total_in_channels, out_features,
                                    conv_channels, kernel_sizes)
        
        # Get the actual output dimension from spatial CNN
        self.spatial_out_dim = self.spatial_cnn.out_features
        
        # ------------- Temporal Encoder ---------------------
        # The temporal encoder processes the concatenated spatial features and flags
        temporal_input_dim = self.spatial_out_dim
        
        self.temporal_encoder = TemporalEncoder(
            mode=temporal_mode,
            input_dim=temporal_input_dim,
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
        
    def forward(self, tof_sequence, mask=None):
        """
        Forward pass with unified temporal encoder and flag handling.
        
        Args:
            tof_sequence: (batch, seq, features) - TOF data, may include flags
            mask: (batch, seq) - 1 for valid positions, 0 for padding
        """
        batch_size, seq_len, total_features = tof_sequence.shape

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=tof_sequence.device)

        # --- Split pixel data and flags ---
        pixel_features = self.input_channels * 64
        tof_pixels = tof_sequence[:, :, :pixel_features]
        
        # Reshape pixels to images: (B*L, sensors, 8, 8)
        pixel_img = tof_pixels.contiguous().view(batch_size * seq_len, self.input_channels, 8, 8)

        # Broadcast flags into image channels and concatenate
        if self.num_flags > 0:
            tof_flags = tof_sequence[:, :, pixel_features:]
            # (B,L, num_flags) -> (B*L, num_flags,1,1) -> expand to 8x8
            flag_img = tof_flags.contiguous().view(batch_size * seq_len, self.num_flags, 1, 1).expand(-1, -1, 8, 8)
            cnn_input = torch.cat([pixel_img, flag_img], dim=1)  # channels = sensors + flags
        else:
            cnn_input = pixel_img
        
        # --- Spatial feature extraction ---
        spatial_features = self.spatial_cnn(cnn_input)  # (B*L, spatial_out_dim)
        spatial_features = spatial_features.view(batch_size, seq_len, self.spatial_out_dim)

        # No need to append flags again; they already influenced spatial features
        temporal_input = spatial_features
        
        # --- Temporal feature aggregation ---
        temporal_features = self.temporal_encoder(temporal_input, mask)
        
        return temporal_features
    
    def get_model_info(self):
        """Get model parameter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }
