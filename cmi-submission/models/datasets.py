import torch
from torch.utils.data import Dataset
import numpy as np


import numpy as np
import torch

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
    # spec_data: (C, F, T)
    C, F, T = spec_data.shape
    freq_param = int(aug_params.get('freq_mask_param', 5))
    n_f = int(aug_params.get('num_freq_masks', 2))
    time_param = int(aug_params.get('time_mask_param', 7))
    n_t = int(aug_params.get('num_time_masks', 2))

    fill = spec_data.mean()

    # 频率掩码
    max_f = min(freq_param, max(0, F - 1))
    if max_f > 0 and F > 1:
        for _ in range(n_f):
            w = rng.integers(1, max_f + 1)
            hi = F - w
            if hi <= 0: break
            s = rng.integers(0, hi + 1)
            spec_data[:, s:s + w, :] = fill

    # 时间掩码
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
    一个内存高效的多模态手势识别数据集类，集成了数据增强功能。
    
    特性:
    - 支持五种数据模态: IMU, THM, TOF, Spectrogram, Static。
    - 在 `__getitem__` 中将 NumPy 数组动态转换为张量以节省内存。
    - 对语谱图进行即时 (on-the-fly) 标准化。
    - [新增] 集成了针对时序和语谱图的数据增强，可通过参数进行精细控制。
    """
    
    def __init__(self, X_imu, X_thm, X_tof, X_spec, X_static, y, mask, 
                 class_weight_dict=None, spec_stats=None, augment=False, aug_params=None):
        """
        初始化数据集。
        
        Args:
            X_imu, X_thm, X_tof, X_spec, X_static: 各模态的 NumPy 数组。
            y (np.array): 标签。
            mask (np.array): 序列数据的填充掩码。
            class_weight_dict (dict, optional): 类别权重映射。
            spec_stats (dict, optional): 包含语谱图 'mean' 和 'std' 的字典，用于标准化。
            augment (bool, optional): 是否应用数据增强。应在训练时设为 True。
            aug_params (dict, optional): 包含所有增强超参数的字典。
        """
        # --- 存储原始数据 (NumPy格式以节省内存) ---
        self.X_imu = X_imu
        self.X_thm = X_thm
        self.X_tof = X_tof
        self.X_spec = X_spec
        self.X_static = X_static
        self.y = y
        self.mask = mask if mask is not None else np.ones((len(y), X_imu.shape[1]), dtype=np.float32)
        
        # --- 存储语谱图统计量 ---
        self.spec_stats = spec_stats
        if self.spec_stats:
            self.spec_mean = self.spec_stats['mean']
            self.spec_std = self.spec_stats['std']

        # --- 数据增强设置 ---
        self.augment = augment
        self.aug_params = aug_params if aug_params is not None else {
            # --- 全局开关 ---
            'time_series_augment_prob': 0.5, 
            'spec_augment_prob': 0.5,        
            
            # --- 时序增强参数 (无需修改) ---
            'noise_std_range': (0.01, 0.05),
            'scale_range': (0.9, 1.1),     

            # --- 语谱图增强参数 (修正后) ---
            # 频率轴总长为11，最大遮挡宽度设为5 (约45%)
            'freq_mask_param': 5,  
            'num_freq_masks': 2,   
            
            # 时间轴总长为17，最大遮挡宽度设为7 (约40%)
            'time_mask_param': 7, 
            'num_time_masks': 2,   
        }
        
        # --- 验证数据形状一致性 ---
        n_samples = len(self.X_imu)
        assert all(len(arr) == n_samples for arr in [self.X_thm, self.X_tof, self.X_spec, self.X_static, self.y, self.mask]), \
            "All data arrays must have the same number of samples"

        # --- 创建样本权重 ---
        self.sample_weights = None
        if class_weight_dict:
            try:
                weights = np.array([class_weight_dict[label_idx] for label_idx in y], dtype=np.float32)
                self.sample_weights = torch.from_numpy(weights)
            except KeyError as e:
                raise KeyError(f"Missing weight for class index {e.args[0]} in class_weight_dict.")
        else:
            # 如果不提供权重，则默认为1.0
            self.sample_weights = torch.ones(n_samples, dtype=torch.float32)

    def __len__(self):
        return len(self.X_imu)

    def __getitem__(self, idx):
        # 1. 获取原始数据，并使用 .copy() 以免在增强时修改原始数据
        imu_data = self.X_imu[idx].copy()
        thm_data = self.X_thm[idx].copy()
        tof_data = self.X_tof[idx].copy()
        spec_data = self.X_spec[idx].copy()
        static_data = self.X_static[idx].copy()
        mask_data = self.mask[idx].copy()
        label = self.y[idx]

        rng = getattr(self, "_rng", None)
        if rng is None:
            # 退化：没有 worker_init_fn 时也能工作
            rng = np.random.default_rng()

        # 2. <<< 应用数据增强 (仅在训练时) >>>
        if self.augment and self.aug_params:
            # 若需要时序增强，打开下一行
            # if rng.random() < self.aug_params.get('time_series_augment_prob', 0.0):
            #     imu_data, thm_data, tof_data = apply_time_series_augmentations(
            #         imu_data, thm_data, tof_data, self.aug_params, rng
            #     )

            if rng.random() < self.aug_params.get('spec_augment_prob', 0.0):
                spec_data = apply_spec_augment(spec_data, self.aug_params, rng)

        # 3. 对语谱图进行即时标准化
        if self.spec_stats:
            eps = 1e-6
            spec_data = (spec_data - self.spec_mean) / (self.spec_std + eps)

        # 4. 将所有数据转换为张量
        inputs = (
            torch.tensor(imu_data, dtype=torch.float32),
            torch.tensor(thm_data, dtype=torch.float32),
            torch.tensor(tof_data, dtype=torch.float32),
            torch.tensor(spec_data, dtype=torch.float32),
            torch.tensor(static_data, dtype=torch.float32),
            torch.tensor(mask_data, dtype=torch.float32)
        )
        target = torch.tensor(label, dtype=torch.long)
        weight = self.sample_weights[idx]
        
        return inputs, target, weight