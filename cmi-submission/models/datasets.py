import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    """
    MODIFIED: Dataset for multimodal gesture recognition that supports sample weights.
    
    Handles four types of data and an optional weight for each sample:
    1. IMU sequential features (accelerometer, rotation)
    2. THM sequential features (thermopile sensors)
    3. TOF sequential features (spatial depth grids)
    4. Static demographic features
    5. Sample weights for weighted loss calculation
    """
    
    def __init__(self, X_imu, X_thm, X_tof, X_static, y, mask, class_weight_dict=None):
        """
        Initialize multimodal dataset.
        
        Args:
            X_imu: IMU sequential data (n_samples, seq_len, imu_features)
            X_thm: THM sequential data (n_samples, seq_len, thm_features)
            X_tof: TOF sequential data (n_samples, seq_len, tof_features)
            X_static: Static demographic data (n_samples, static_features)
            y: Labels (n_samples,)
            mask: Padding mask for sequential data (n_samples, seq_len)
            class_weight_dict (dict, optional): A dictionary mapping class_index to weight.
                                                 Used for creating sample-specific weights.
        """
        self.X_imu = torch.FloatTensor(X_imu)
        self.X_thm = torch.FloatTensor(X_thm)
        self.X_tof = torch.FloatTensor(X_tof)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.LongTensor(y)
        self.mask = torch.FloatTensor(mask)
        
        # Validate shapes
        n_samples = len(self.X_imu)
        assert len(self.X_thm) == n_samples, "THM data must have same number of samples"
        assert len(self.X_tof) == n_samples, "TOF data must have same number of samples"
        assert len(self.X_static) == n_samples, "Static data must have same number of samples"
        assert len(self.y) == n_samples, "Labels must have same number of samples"
        assert len(self.mask) == n_samples, "Mask must have same number of samples"

        # Create sample weights based on the class weight dictionary
        self.sample_weights = None
        if class_weight_dict:
            # For each label in y, find its corresponding weight from the dictionary
            weights = np.array([class_weight_dict.get(label_idx, 1.0) for label_idx in y])
            self.sample_weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.X_imu)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            tuple: ((imu_data, thm_data, tof_data, static_data, mask), label, sample_weight)
        """
        data_tuple = (self.X_imu[idx], self.X_thm[idx], self.X_tof[idx], self.X_static[idx], self.mask[idx])
        label = self.y[idx]
        
        if self.sample_weights is not None:
            weight = self.sample_weights[idx]
            return data_tuple, label, weight
        else:
            # Return a default weight of 1.0 if no weights are specified
            return data_tuple, label, 1.0
