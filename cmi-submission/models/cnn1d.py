import torch
import torch.nn as nn
from . import MODELS  # Import registry

@MODELS.register_module()
class CNN1D(nn.Module):
    """
    Simple 1D CNN for gesture classification.
    
    REFACTORED: This class is now modular. It can function as a standalone
    classifier or as a feature extractor for a larger fusion model.
    """
    def __init__(self,
                 input_channels: int,
                 sequence_length: int = 100,
                 # legacy arg for standalone classifier
                 num_classes: int = 18,
                 # v3 architecture lists
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
            layers.append(nn.Conv1d(in_channels, out_c, kernel_size=k, padding=k//2))
            self.bn_layers.append(nn.BatchNorm1d(out_c))
            in_channels = out_c
        self.conv_layers = nn.ModuleList(layers)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.cnn_output_size = filters[-1]
    
    def forward(self, x):
        """
        Defines the full forward pass for using this model as a feature extractor.
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len) -> Conv1d的标准输入格式

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.bn_layers[i](x)
            x = torch.relu(x)
            
            # Apply pooling except on the last conv layer
            if i < len(self.conv_layers) - 1:
                x = self.pool(x)

        # Global pooling to collapse temporal dimension
        x = self.global_pool(x)  # (batch, channels, 1)

        # Flatten to (batch, feature_dim)
        x = x.view(x.size(0), -1)

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
