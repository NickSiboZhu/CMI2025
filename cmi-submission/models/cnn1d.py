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
        
        # ------------------------------------------------------------------
        # Dynamic conv stack -------------------------------------------------
        # ------------------------------------------------------------------
        if filters is None:
            filters = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [5, 5, 3]
        assert len(filters) == len(kernel_sizes), "filters and kernel_sizes length mismatch"

        layers = []
        ln_layers = []  # store LayerNorm to apply after transpose
        in_channels = input_channels
        for out_c, k in zip(filters, kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_c, kernel_size=k, padding=k//2))
            ln_layers.append(nn.LayerNorm(out_c))
            in_channels = out_c
        self.conv_layers = nn.ModuleList(layers)
        self.ln_layers = nn.ModuleList(ln_layers)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.cnn_output_size = filters[-1]
        
        # --- 2. Classifier Block ---
        # The original classifier head is kept for when this model is used standalone.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    # --- NEW: Helper method for the Fusion Model ---
    def get_cnn_output_size(self):
        """
        Helper function to allow the fusion model to know the output dimension
        of this CNN branch.
        """
        return self.cnn_output_size

    # --- NEW: Feature Extraction method ---
    def extract_features(self, x):
        """
        This method only performs feature extraction with LayerNorm.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
        
        Returns:
            A flattened feature tensor of shape (batch_size, cnn_output_size)
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)

        for conv, ln in zip(self.conv_layers, self.ln_layers):
            x = conv(x)
            x = x.transpose(1, 2)  # (batch, seq_len, channels) for LN
            x = ln(x)
            x = x.transpose(1, 2)
            x = torch.relu(x)
            # Apply pooling except maybe last layer to keep information; here pool after every conv except last
            if conv != self.conv_layers[-1]:
                x = self.pool(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 256, 1)
        
        # Flatten the output for concatenation in the fusion model
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        return x

    def forward(self, x):
        """
        Defines the full forward pass for using this model as a standalone classifier.
        """
        # 1. Extract features using our new method
        features = self.extract_features(x)
        
        # 2. Pass the features through the classifier head
        output = self.classifier(features)
        
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