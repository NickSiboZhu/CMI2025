import torch
import torch.nn as nn

# Import the refactored CNN and the new MLP class
from .cnn import Simple1DCNN
from .mlp import MLP

class FusionModel(nn.Module):
    """
    A multimodal fusion model for gesture recognition.

    This model combines a CNN branch for processing time-series sensor data
    with an MLP branch for processing static demographic data. The features
    from both branches are then fused and passed to a final classifier head.
    """
    def __init__(self, seq_input_channels: int, static_input_features: int, num_classes: int, sequence_length: int):
        """
        Initializes all components of the fusion model.
        
        Args:
            seq_input_channels (int): Number of input features for the sequential data (CNN branch).
            static_input_features (int): Number of input features for the static data (MLP branch).
            num_classes (int): The total number of output classes for classification.
            sequence_length (int): The length of the input time-series sequences.
        """
        super(FusionModel, self).__init__()

        # --- Branch 1: CNN for Sequential Feature Extraction ---
        self.cnn_branch = Simple1DCNN(seq_input_channels, num_classes, sequence_length)
        self.cnn_output_size = self.cnn_branch.get_cnn_output_size()

        # --- Branch 2: MLP for Static Feature Extraction ---
        self.mlp_output_size = 32
        self.mlp_branch = MLP(static_input_features, self.mlp_output_size)
        
        # --- Fusion and Classifier Head ---
        combined_feature_size = self.cnn_output_size + self.mlp_output_size # Should be 256 + 32 = 288
        
        # --- MODIFIED: Implemented the user's proposed "funnel" architecture ---
        # This structure gradually reduces dimensionality, forcing the model
        # to learn a compressed and robust representation before classification.
        self.classifier_head = nn.Sequential(
            # Apply BatchNorm to the concatenated features for stabilization
            nn.BatchNorm1d(combined_feature_size),

            # Layer 1: 288 -> 128
            nn.Linear(combined_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Layer 2: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 64 -> num_classes
            # We go directly to num_classes from 64, which is a common practice.
            # Adding another layer to 32 might be too aggressive.
            nn.Linear(64, num_classes)
        )

    def forward(self, seq_input: torch.Tensor, static_input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass logic of the fusion model.
        """
        # 1. Process sequential data through the CNN branch
        cnn_features = self.cnn_branch.extract_features(seq_input)
        
        # 2. Process static data through the MLP branch
        mlp_features = self.mlp_branch(static_input)
        
        # 3. Concatenate the features from both branches
        combined_features = torch.cat((cnn_features, mlp_features), dim=1)
        
        # 4. Pass the fused features through the final classifier head
        output = self.classifier_head(combined_features)
        
        return output

    def get_model_info(self) -> dict:
        """
        Helper function to return the model's total parameter count and approximate size.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024**2)  # Assuming float32 parameters
        return {'total_params': total_params, 'model_size_mb': model_size_mb}

