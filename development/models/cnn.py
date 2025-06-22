import torch
import torch.nn as nn
from torch.utils.data import Dataset

class GestureDataset(Dataset):
    """
    Custom Dataset for gesture sequences
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Simple1DCNN(nn.Module):
    """
    Simple 1D CNN for gesture classification
    
    Args:
        input_channels (int): Number of input features
        num_classes (int): Number of output classes
        sequence_length (int): Length of input sequences
    """
    def __init__(self, input_channels, num_classes, sequence_length):
        super(Simple1DCNN, self).__init__()
        
        # Conv layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Dense layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, sequence_length, features)
        # Conv1d expects: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
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