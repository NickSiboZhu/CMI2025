import torch
from torch.utils.data import Dataset
import numpy as np

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

    def _apply_time_series_augmentations(self, imu_data, thm_data, tof_data):
        """应用时序数据增强"""
        # 幅度缩放
        scale = np.random.uniform(*self.aug_params.get('scale_range', (0.9, 1.1)))
        imu_data *= scale
        thm_data *= scale

        # 高斯噪声
        noise_std = np.random.uniform(*self.aug_params.get('noise_std_range', (0.01, 0.05)))
        imu_data += np.random.normal(0, noise_std, imu_data.shape)
        thm_data += np.random.normal(0, noise_std, thm_data.shape)

        return imu_data, thm_data, tof_data

    def _apply_spec_augment(self, spec_data):
        """应用 SpecAugment"""
        freq_mask_param = self.aug_params.get('freq_mask_param', 5)
        num_freq_masks = self.aug_params.get('num_freq_masks', 2)
        time_mask_param = self.aug_params.get('time_mask_param', 7)
        num_time_masks = self.aug_params.get('num_time_masks', 2)
        
        # 语谱图形状: (Channels, Freq_bins, Time_bins)
        _, num_freq_bins, num_time_bins = spec_data.shape
        
        # 使用语谱图自身的均值进行填充，这比用0填充更好
        mask_fill_value = spec_data.mean()

        # 1. 频率掩码
        # Clamp mask width to valid range based on actual bins
        max_freq_width = min(int(freq_mask_param), max(0, num_freq_bins - 1))
        if max_freq_width > 0 and num_freq_bins > 1:
            for _ in range(int(num_freq_masks)):
                # Choose a positive width in [1, max_freq_width]
                f = np.random.randint(1, max_freq_width + 1)
                # Start index in [0, num_freq_bins - f] inclusive
                start_high = num_freq_bins - f
                if start_high <= 0:
                    continue
                f0 = np.random.randint(0, start_high + 1)
                spec_data[:, f0:f0 + f, :] = mask_fill_value
            
        # 2. 时间掩码
        max_time_width = min(int(time_mask_param), max(0, num_time_bins - 1))
        if max_time_width > 0 and num_time_bins > 1:
            for _ in range(int(num_time_masks)):
                t = np.random.randint(1, max_time_width + 1)
                start_high = num_time_bins - t
                if start_high <= 0:
                    continue
                t0 = np.random.randint(0, start_high + 1)
                spec_data[:, :, t0:t0 + t] = mask_fill_value
            
        return spec_data

    def __getitem__(self, idx):
        # 1. 获取原始数据，并使用 .copy() 以免在增强时修改原始数据
        imu_data = self.X_imu[idx].copy()
        thm_data = self.X_thm[idx].copy()
        tof_data = self.X_tof[idx].copy()
        spec_data = self.X_spec[idx].copy()
        static_data = self.X_static[idx].copy()
        mask_data = self.mask[idx].copy()
        label = self.y[idx]

        # 2. <<< 应用数据增强 (仅在训练时) >>>
        if self.augment and self.aug_params:
            # 应用时序增强
            # if np.random.rand() < self.aug_params.get('time_series_augment_prob', 0.0):
            #     imu_data, thm_data, tof_data = self._apply_time_series_augmentations(imu_data, thm_data, tof_data)
            
            # 应用语谱图增强
            if np.random.rand() < self.aug_params.get('spec_augment_prob', 0.0):
                spec_data = self._apply_spec_augment(spec_data)

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