import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalDataset(Dataset):
    """
    MODIFIED: A memory-efficient multimodal dataset for gesture recognition.
    
    This version now handles five types of data and performs on-the-fly normalization
    for spectrograms using pre-computed dataset statistics.
    
    Key Changes:
    - Added support for spectrogram data (X_spec).
    - Added support for spectrogram statistics (spec_stats) for normalization.
    - Data is kept as NumPy arrays and converted to Tensors in `__getitem__` to
      significantly reduce memory usage.
    """
    
    def __init__(self, X_imu, X_thm, X_tof, X_spec, X_static, y, mask, class_weight_dict=None, spec_stats=None):
        """
        Initialize the memory-efficient multimodal dataset.
        
        Args:
            X_imu, X_thm, X_tof, X_spec, X_static: NumPy arrays for each data modality.
            y (np.array): Labels.
            mask (np.array): Padding mask for sequential data.
            class_weight_dict (dict, optional): Maps class_index to a weight.
            spec_stats (dict, optional): Contains 'mean' and 'std' for spectrogram normalization.
        """
        # --- MODIFIED: Store data as NumPy arrays to save memory ---
        self.X_imu = X_imu
        self.X_thm = X_thm
        self.X_tof = X_tof
        self.X_spec = X_spec      # NEW: Spectrogram data
        self.X_static = X_static
        self.y = y
        self.mask = mask if mask is not None else np.ones((len(y), X_imu.shape[1]), dtype=np.float32)
        
        # --- NEW: Store spectrogram statistics ---
        self.spec_stats = spec_stats
        if self.spec_stats:
            self.spec_mean = self.spec_stats['mean']
            self.spec_std = self.spec_stats['std']

        # Validate shapes
        n_samples = len(self.X_imu)
        assert len(self.X_thm) == n_samples, "THM data must have same number of samples"
        assert len(self.X_tof) == n_samples, "TOF data must have same number of samples"
        assert len(self.X_spec) == n_samples, "Spectrogram data must have same number of samples" # NEW
        assert len(self.X_static) == n_samples, "Static data must have same number of samples"
        assert len(self.y) == n_samples, "Labels must have same number of samples"
        assert len(self.mask) == n_samples, "Mask must have same number of samples"

        # Create sample weights as a NumPy array
        self.sample_weights = None
        if class_weight_dict:
            # For each label in y, find its corresponding weight from the dictionary (strict)
            try:
                weights = np.array([class_weight_dict[label_idx] for label_idx in y])
            except KeyError as e:
                raise KeyError(f"Missing weight for class index {e.args[0]} in class_weight_dict.")
            self.sample_weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.X_imu)
    
    def __getitem__(self, idx):
        """
        Get a single sample, perform on-the-fly normalization, and convert to Tensor.
        
        Returns:
            tuple: ((imu, thm, tof, spec, static, mask), label, weight)
        """
        # 1. Fetch data for the index as NumPy arrays
        imu_data = self.X_imu[idx]
        thm_data = self.X_thm[idx]
        tof_data = self.X_tof[idx]
        spec_data = self.X_spec[idx]
        static_data = self.X_static[idx]
        mask_data = self.mask[idx]
        label = self.y[idx]

        # 2. On-the-fly spectrogram normalization using pre-computed stats
        if self.spec_stats:
            eps = 1e-6
            spec_data = (spec_data - self.spec_mean) / (self.spec_std + eps)

        # 3. Convert all data to Tensors just before returning
        inputs = (
            torch.tensor(imu_data, dtype=torch.float32),
            torch.tensor(thm_data, dtype=torch.float32),
            torch.tensor(tof_data, dtype=torch.float32),
            torch.tensor(spec_data, dtype=torch.float32),
            torch.tensor(static_data, dtype=torch.float32),
            torch.tensor(mask_data, dtype=torch.float32)
        )
        target = torch.tensor(label, dtype=torch.long)
        
        # 4. Handle sample weight
        if self.sample_weights is not None:
            weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        else:
            # Strict: require weights to be provided by caller
            raise RuntimeError("Sample weights were not provided. Provide class_weight_dict when creating the dataset.")
        
        return inputs, target, weight
