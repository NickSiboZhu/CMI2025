import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    """
    Simple 1D CNN for gesture classification.
    
    REFACTORED: This class is now modular. It can function as a standalone
    classifier or as a feature extractor for a larger fusion model.
    """
    def __init__(self, input_channels, num_classes, sequence_length):
        super(Simple1DCNN, self).__init__()
        
        # --- 1. Feature Extractor Block ---
        # All convolutional layers are now grouped into a single sequential module.
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # Global pooling layer to handle variable lengths and create a fixed-size output.
            nn.AdaptiveMaxPool1d(1)
        )
        
        # The output size of the feature_extractor is the number of channels from the last conv layer.
        self.cnn_output_size = 256
        
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
        This method only performs feature extraction. It will be called by the
        main FusionModel.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
        
        Returns:
            A flattened feature tensor of shape (batch_size, cnn_output_size)
        """
        # Conv1d expects: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        # Pass through the convolutional layers
        x = self.feature_extractor(x) # Shape: (batch_size, 256, 1)
        
        # Flatten the output for concatenation in the fusion model
        x = x.view(x.size(0), -1) # Shape: (batch_size, 256)
        
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