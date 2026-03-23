import numpy as np
import torch
from torch.utils.data import Dataset

def apply_time_series_augmentations(imu_data, thm_data, tof_data, aug_params, rng):
    scale_low, scale_high = aug_params.get('scale_range', (0.9, 1.1))
    noise_low, noise_high = aug_params.get('noise_std_range', (0.01, 0.05))

    scale = rng.uniform(scale_low, scale_high)
    imu_data = imu_data * scale
    thm_data = thm_data * scale

    noise_std = rng.uniform(noise_low, noise_high)
    imu_data = imu_data + rng.normal(0, noise_std, imu_data.shape)
    thm_data = thm_data + rng.normal(0, noise_std, thm_data.shape)
    return imu_data, thm_data, tof_data


def apply_spec_augment(spec_data, aug_params, rng):
    C, F, T = spec_data.shape
    freq_param = int(aug_params.get('freq_mask_param', 5))
    n_f = int(aug_params.get('num_freq_masks', 2))
    time_param = int(aug_params.get('time_mask_param', 7))
    n_t = int(aug_params.get('num_time_masks', 2))

    # Filling with the sample mean avoids injecting a hard zero-valued edge.
    fill = spec_data.mean()

    max_f = min(freq_param, max(0, F - 1))
    if max_f > 0 and F > 1:
        for _ in range(n_f):
            w = rng.integers(1, max_f + 1)
            hi = F - w
            if hi <= 0: break
            s = rng.integers(0, hi + 1)
            spec_data[:, s:s + w, :] = fill

    max_t = min(time_param, max(0, T - 1))
    if max_t > 0 and T > 1:
        for _ in range(n_t):
            w = rng.integers(1, max_t + 1)
            hi = T - w
            if hi <= 0: break
            s = rng.integers(0, hi + 1)
            spec_data[:, :, s:s + w] = fill

    return spec_data




class MultimodalDataset(Dataset):
    """
    Memory-efficient multimodal dataset with optional on-the-fly augmentation.

    Arrays stay in NumPy form until ``__getitem__`` so a full K-fold run can keep
    all folds in memory at once without paying the tensor overhead upfront.
    """
    
    def __init__(self, X_imu, X_thm, X_tof, X_spec, X_static, y, mask, 
                 X_tof_channel_mask=None, X_thm_channel_mask=None, X_imu_channel_mask=None,
                 class_weight_dict=None, spec_stats=None, augment=False, aug_params=None):
        """Initialize the dataset from already prepared NumPy arrays."""
        self.X_imu = X_imu
        self.X_thm = X_thm
        self.X_tof = X_tof
        self.X_spec = X_spec
        self.X_static = X_static
        self.y = y
        self.mask = mask if mask is not None else np.ones((len(y), X_imu.shape[1]), dtype=np.float32)
        self.X_tof_channel_mask = X_tof_channel_mask
        self.X_thm_channel_mask = X_thm_channel_mask
        self.X_imu_channel_mask = X_imu_channel_mask
        
        self.spec_stats = spec_stats
        if self.spec_stats:
            self.spec_mean = self.spec_stats['mean']
            self.spec_std = self.spec_stats['std']

        self.augment = augment
        self.aug_params = aug_params if aug_params is not None else {
            'time_series_augment_prob': 0.5, 
            'spec_augment_prob': 0.5,        
            'noise_std_range': (0.01, 0.05),
            'scale_range': (0.9, 1.1),     
            # Defaults assume the current 11x17 spectrogram resolution.
            'freq_mask_param': 5,  
            'num_freq_masks': 2,   
            'time_mask_param': 7, 
            'num_time_masks': 2,   
        }
        
        n_samples = len(self.X_imu)
        assert all(len(arr) == n_samples for arr in [self.X_thm, self.X_tof, self.X_spec, self.X_static, self.y, self.mask]), \
            "All data arrays must have the same number of samples"

        self.sample_weights = None
        if class_weight_dict:
            try:
                weights = np.array([class_weight_dict[label_idx] for label_idx in y], dtype=np.float32)
                self.sample_weights = torch.from_numpy(weights)
            except KeyError as e:
                raise KeyError(f"Missing weight for class index {e.args[0]} in class_weight_dict.")
        else:
            self.sample_weights = torch.ones(n_samples, dtype=torch.float32)

    def __len__(self):
        return len(self.X_imu)

    def __getitem__(self, idx):
        # Augmentations mutate arrays in place, so each sample must work on copies.
        imu_data = self.X_imu[idx].copy()
        thm_data = self.X_thm[idx].copy()
        tof_data = self.X_tof[idx].copy()
        spec_data = self.X_spec[idx].copy()
        static_data = self.X_static[idx].copy()
        mask_data = self.mask[idx].copy()
        label = self.y[idx]

        rng = getattr(self, "_rng", None)
        if rng is None:
            # Keep augmentation functional even when DataLoader workers are not seeded.
            rng = np.random.default_rng()

        if self.augment and self.aug_params:
            # Time-series augmentation is intentionally left opt-in because it was not
            # part of the strongest public configuration.
            # if rng.random() < self.aug_params.get('time_series_augment_prob', 0.0):
            #     imu_data, thm_data, tof_data = apply_time_series_augmentations(
            #         imu_data, thm_data, tof_data, self.aug_params, rng
            #     )

            if rng.random() < self.aug_params.get('spec_augment_prob', 0.0):
                spec_data = apply_spec_augment(spec_data, self.aug_params, rng)

        if self.spec_stats:
            eps = 1e-6
            spec_data = (spec_data - self.spec_mean) / (self.spec_std + eps)

        inputs = (
            torch.tensor(imu_data, dtype=torch.float32),
            torch.tensor(thm_data, dtype=torch.float32),
            torch.tensor(tof_data, dtype=torch.float32),
            torch.tensor(spec_data, dtype=torch.float32),
            torch.tensor(static_data, dtype=torch.float32),
            torch.tensor(mask_data, dtype=torch.float32),
            torch.tensor(self.X_tof_channel_mask[idx] if self.X_tof_channel_mask is not None else np.ones((self.X_tof.shape[2]//64 if self.X_tof is not None else 5,), dtype=np.float32), dtype=torch.float32),
            torch.tensor(self.X_thm_channel_mask[idx] if self.X_thm_channel_mask is not None else np.ones((self.X_thm.shape[2] if self.X_thm is not None else 0,), dtype=np.float32), dtype=torch.float32),
            torch.tensor(self.X_imu_channel_mask[idx] if self.X_imu_channel_mask is not None else np.ones((self.X_imu.shape[2] if self.X_imu is not None else 0,), dtype=np.float32), dtype=torch.float32)
        )
        target = torch.tensor(label, dtype=torch.long)
        weight = self.sample_weights[idx]
        
        return inputs, target, weight
