import torch
import torch.nn as nn
import numpy as np

class TOF2DCNN(nn.Module):
    """
    2D CNN for processing Time-of-Flight (TOF) sensor grids.
    
    Each TOF sensor provides an 8x8 grid of depth values.
    This module processes multiple TOF sensors and extracts spatial features.
    
    Args:
        num_tof_sensors (int): Number of TOF sensors (typically 5)
        out_features (int): Output feature dimension after processing
    """
    
    def __init__(self, num_tof_sensors=5, out_features=128):
        super(TOF2DCNN, self).__init__()
        
        self.num_tof_sensors = num_tof_sensors
        self.out_features = out_features
        
        # 2D CNN layers for processing 8x8 grids
        # Input: (batch_size, num_sensors, 8, 8)
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(num_tof_sensors, 32, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 4x4 -> 4x4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4 -> 2x2
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=2, padding=0),  # 2x2 -> 1x1
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Global average pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final projection layer
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_features),
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
        x = self.conv_layers(tof_grids)  # (batch, 128, 1, 1)
        
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


class TemporalTOF2DCNN(nn.Module):
    """
    2D CNN for processing sequential TOF sensor data.
    
    This module processes TOF grids across time steps, applying 2D CNN
    to each timestep and then aggregating temporal information.
    
    Args:
        num_tof_sensors (int): Number of TOF sensors (typically 5)
        seq_len (int): Sequence length
        out_features (int): Output feature dimension
    """
    
    def __init__(self, num_tof_sensors=5, seq_len=100, out_features=128):
        super(TemporalTOF2DCNN, self).__init__()
        
        self.num_tof_sensors = num_tof_sensors
        self.seq_len = seq_len
        self.out_features = out_features
        
        # 2D CNN for processing individual timesteps
        self.spatial_cnn = TOF2DCNN(num_tof_sensors, out_features)
        
        # Temporal aggregation layer (could be LSTM, attention, or simple pooling)
        self.temporal_aggregation = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, tof_sequence):
        """
        Forward pass for temporal TOF processing.
        
        Args:
            tof_sequence: Tensor of shape (batch_size, seq_len, num_sensors * 64)
        
        Returns:
            Tensor of shape (batch_size, out_features)
        """
        batch_size, seq_len, _ = tof_sequence.shape
        
        # Reshape to grids: (batch_size, seq_len, num_sensors, 8, 8)
        tof_grids = reshape_tof_features(tof_sequence, self.num_tof_sensors)
        
        # Process each timestep through 2D CNN
        # Reshape to (batch_size * seq_len, num_sensors, 8, 8)
        tof_grids_flat = tof_grids.view(batch_size * seq_len, self.num_tof_sensors, 8, 8)
        
        # Apply 2D CNN to all timesteps
        spatial_features = self.spatial_cnn(tof_grids_flat)  # (batch_size * seq_len, out_features)
        
        # Reshape back to sequential format
        spatial_features = spatial_features.view(batch_size, seq_len, self.out_features)
        
        # Temporal aggregation (simple mean pooling for now)
        # You could replace this with LSTM or attention mechanism
        temporal_features = torch.mean(spatial_features, dim=1)  # (batch_size, out_features)
        
        # Final projection
        output = self.temporal_aggregation(temporal_features)
        
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