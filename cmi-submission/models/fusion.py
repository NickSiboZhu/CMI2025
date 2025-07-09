import torch
import torch.nn as nn

# Import the refactored CNN classes
from .cnn1d import Simple1DCNN
from .cnn2d import TemporalTOF2DCNN
from .mlp import MLP

class FusionModel(nn.Module):
    """
    A multimodal fusion model for gesture recognition.

    This model combines:
    1. A 1D CNN branch for processing sequential sensor data (IMU, THM, etc.)
    2. A 2D CNN branch for processing TOF sensor grids (spatial depth information)
    3. An MLP branch for processing static demographic data
    
    The features from all branches are fused and passed to a final classifier head.
    """
    def __init__(self, seq_input_channels: int, tof_input_channels: int, static_input_features: int, 
                 num_classes: int, sequence_length: int):
        """
        Initializes all components of the fusion model.
        
        Args:
            seq_input_channels (int): Number of input features for sequential data (IMU, THM, etc.)
            tof_input_channels (int): Number of TOF features (should be 5 * 64 = 320)
            static_input_features (int): Number of static demographic features
            num_classes (int): The total number of output classes for classification
            sequence_length (int): The length of the input time-series sequences
        """
        super(FusionModel, self).__init__()

        # --- Branch 1: 1D CNN for Sequential Feature Extraction (IMU, THM, etc.) ---
        self.cnn_1d_branch = Simple1DCNN(seq_input_channels, num_classes, sequence_length)
        self.cnn_1d_output_size = self.cnn_1d_branch.get_cnn_output_size()

        # --- Branch 2: 2D CNN for TOF Spatial Feature Extraction ---
        self.tof_2d_output_size = 128
        self.cnn_2d_branch = TemporalTOF2DCNN(
            num_tof_sensors=5, 
            seq_len=sequence_length, 
            out_features=self.tof_2d_output_size
        )
        
        # --- Branch 3: MLP for Static Feature Extraction ---
        self.mlp_output_size = 32
        self.mlp_branch = MLP(static_input_features, self.mlp_output_size)
        
        # --- Fusion and Classifier Head ---
        combined_feature_size = self.cnn_1d_output_size + self.tof_2d_output_size + self.mlp_output_size
        # Should be 256 + 128 + 32 = 416
        
        # Multimodal fusion classifier with funnel architecture
        self.classifier_head = nn.Sequential(
            # Apply BatchNorm to the concatenated features for stabilization
            nn.BatchNorm1d(combined_feature_size),

            # Layer 1: 416 -> 256
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            # Layer 2: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Layer 3: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 4: 64 -> num_classes
            nn.Linear(64, num_classes)
        )

    def forward(self, seq_input: torch.Tensor, tof_input: torch.Tensor, static_input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass logic of the multimodal fusion model.
        
        Args:
            seq_input: Sequential sensor data (batch_size, seq_len, seq_features)
            tof_input: TOF sensor data (batch_size, seq_len, 320) - 5 sensors × 64 pixels
            static_input: Static demographic data (batch_size, static_features)
        """
        # 1. Process sequential data through the 1D CNN branch
        cnn_1d_features = self.cnn_1d_branch.extract_features(seq_input)
        
        # 2. Process TOF data through the 2D CNN branch (only if TOF features exist)
        if tof_input.shape[2] > 0:  # Check if TOF features are present
            cnn_2d_features = self.cnn_2d_branch(tof_input)
        else:
            # Create zero tensor with correct shape for missing TOF features
            batch_size = seq_input.shape[0]
            cnn_2d_features = torch.zeros(batch_size, self.tof_2d_output_size, device=seq_input.device)
        
        # 3. Process static data through the MLP branch
        mlp_features = self.mlp_branch(static_input)
        
        # 4. Concatenate the features from all branches
        combined_features = torch.cat((cnn_1d_features, cnn_2d_features, mlp_features), dim=1)
        
        # 5. Pass the fused features through the final classifier head
        output = self.classifier_head(combined_features)
        
        return output

    def get_model_info(self) -> dict:
        """
        Helper function to return the model's total parameter count and approximate size.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024**2)  # Assuming float32 parameters
        
        # Get info from individual branches
        cnn_1d_info = self.cnn_1d_branch.get_model_info()
        cnn_2d_info = self.cnn_2d_branch.get_model_info()
        mlp_info = self.mlp_branch.get_model_info()
        
        return {
            'total_params': total_params, 
            'model_size_mb': model_size_mb,
            'cnn_1d_params': cnn_1d_info['total_params'],
            'cnn_2d_params': cnn_2d_info['total_params'],
            'mlp_params': mlp_info['total_params'],
            'fusion_head_params': total_params - cnn_1d_info['total_params'] - cnn_2d_info['total_params'] - mlp_info['total_params']
        }


# Backward compatibility wrapper for existing code
class FusionModelLegacy(nn.Module):
    """
    Legacy wrapper for the old 2-input FusionModel interface.
    Automatically separates TOF features from sequential features.
    """
    def __init__(self, seq_input_channels: int, static_input_features: int, num_classes: int, sequence_length: int):
        super(FusionModelLegacy, self).__init__()
        
        # Assume TOF features are the last 320 features (5 sensors × 64 pixels)
        tof_channels = 320
        non_tof_channels = seq_input_channels - tof_channels
        
        if non_tof_channels < 0:
            raise ValueError(f"seq_input_channels ({seq_input_channels}) must be >= 320 for TOF features")
        
        self.tof_channels = tof_channels
        self.non_tof_channels = non_tof_channels
        
        self.fusion_model = FusionModel(
            seq_input_channels=non_tof_channels,
            tof_input_channels=tof_channels,
            static_input_features=static_input_features,
            num_classes=num_classes,
            sequence_length=sequence_length
        )
    
    def forward(self, seq_input: torch.Tensor, static_input: torch.Tensor) -> torch.Tensor:
        """
        Legacy forward pass that automatically separates TOF features.
        """
        # Split sequential input into non-TOF and TOF features
        non_tof_input = seq_input[:, :, :self.non_tof_channels]
        tof_input = seq_input[:, :, self.non_tof_channels:]
        
        return self.fusion_model(non_tof_input, tof_input, static_input)
    
    def get_model_info(self):
        return self.fusion_model.get_model_info()

