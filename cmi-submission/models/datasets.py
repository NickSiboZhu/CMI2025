import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    """
    MODIFIED: Dataset for multimodal gesture recognition that supports sample weights.
    
    Handles three types of data and an optional weight for each sample:
    1. Non-TOF sequential features (IMU, THM, etc.)
    2. TOF sequential features (spatial depth grids)
    3. Static demographic features
    4. Sample weights for weighted loss calculation
    """
    
    def __init__(self, X_non_tof, X_tof, X_static, y, mask, class_weight_dict=None):
        """
        Initialize multimodal dataset.
        
        Args:
            X_non_tof: Non-TOF sequential data (n_samples, seq_len, non_tof_features)
            X_tof: TOF sequential data (n_samples, seq_len, tof_features)
            X_static: Static demographic data (n_samples, static_features)
            y: Labels (n_samples,)
            class_weight_dict (dict, optional): A dictionary mapping class_index to weight.
                                                 Used for creating sample-specific weights.
        """
        self.X_non_tof = torch.FloatTensor(X_non_tof)
        self.X_tof = torch.FloatTensor(X_tof)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.LongTensor(y)
        self.mask = torch.FloatTensor(mask)
        
        # Validate shapes
        assert len(self.X_non_tof) == len(self.X_tof) == len(self.X_static) == len(self.y), \
            "All data arrays must have the same number of samples"

        # ✨ NEW: Create sample weights based on the class weight dictionary
        self.sample_weights = None
        if class_weight_dict:
            # For each label in y, find its corresponding weight from the dictionary
            weights = np.array([class_weight_dict.get(label_idx, 1.0) for label_idx in y])
            self.sample_weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.X_non_tof)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: ((non_tof_data, tof_data, static_data, mask), label, sample_weight)
        """
        # ✨ 将mask添加到数据元组中
        data_tuple = (self.X_non_tof[idx], self.X_tof[idx], self.X_static[idx], self.mask[idx])
        label = self.y[idx]
        
        if self.sample_weights is not None:
            weight = self.sample_weights[idx]
            return data_tuple, label, weight
        else:
            return data_tuple, label, 1.0
