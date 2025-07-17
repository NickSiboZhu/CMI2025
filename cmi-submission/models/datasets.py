import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """
    Dataset for multimodal gesture recognition with separated features.
    
    Handles three types of data:
    1. Non-TOF sequential features (IMU, THM, etc.)
    2. TOF sequential features (spatial depth grids)
    3. Static demographic features
    """
    
    def __init__(self, X_non_tof, X_tof, X_static, y):
        """
        Initialize multimodal dataset.
        
        Args:
            X_non_tof: Non-TOF sequential data (n_samples, seq_len, non_tof_features)
            X_tof: TOF sequential data (n_samples, seq_len, tof_features)
            X_static: Static demographic data (n_samples, static_features)
            y: Labels (n_samples,)
        """
        self.X_non_tof = torch.FloatTensor(X_non_tof)
        self.X_tof = torch.FloatTensor(X_tof)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.LongTensor(y)
        
        # Validate shapes
        assert len(self.X_non_tof) == len(self.X_tof) == len(self.X_static) == len(self.y), \
            "All data arrays must have the same number of samples"
    
    def __len__(self):
        return len(self.X_non_tof)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            tuple: ((non_tof_data, tof_data, static_data), label)
        """
        return (self.X_non_tof[idx], self.X_tof[idx], self.X_static[idx]), self.y[idx]


class LegacyDataset(Dataset):
    """
    Legacy dataset for backward compatibility with 2-input models.
    """
    
    def __init__(self, X_seq, X_static, y):
        """
        Initialize legacy dataset.
        
        Args:
            X_seq: Sequential data (n_samples, seq_len, features)
            X_static: Static data (n_samples, static_features)
            y: Labels (n_samples,)
        """
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            tuple: ((seq_data, static_data), label)
        """
        return (self.X_seq[idx], self.X_static[idx]), self.y[idx]
