import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """
    Dataset for multimodal gesture recognition with separated features.
    
    Handles four types of data:
    1. IMU sequential features (accelerometer, rotation)
    2. THM sequential features (thermopile sensors)
    3. TOF sequential features (spatial depth grids)
    4. Static demographic features
    """
    
    def __init__(self, X_imu, X_thm, X_tof, X_static, y):
        """
        Initialize multimodal dataset.
        
        Args:
            X_imu: IMU sequential data (n_samples, seq_len, imu_features)
            X_thm: THM sequential data (n_samples, seq_len, thm_features)
            X_tof: TOF sequential data (n_samples, seq_len, tof_features)
            X_static: Static demographic data (n_samples, static_features)
            y: Labels (n_samples,)
        """
        self.X_imu = torch.FloatTensor(X_imu)
        self.X_thm = torch.FloatTensor(X_thm)
        self.X_tof = torch.FloatTensor(X_tof)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.LongTensor(y)
        
        # Validate shapes
        n_samples = len(self.X_imu)
        assert len(self.X_thm) == n_samples, "THM data must have same number of samples"
        assert len(self.X_tof) == n_samples, "TOF data must have same number of samples"
        assert len(self.X_static) == n_samples, "Static data must have same number of samples"
        assert len(self.y) == n_samples, "Labels must have same number of samples"
    
    def __len__(self):
        return len(self.X_imu)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            tuple: ((imu_data, tof_data, static_data, thm_data), label)
        """
        return (self.X_imu[idx], self.X_tof[idx], self.X_static[idx], self.X_thm[idx]), self.y[idx]


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
