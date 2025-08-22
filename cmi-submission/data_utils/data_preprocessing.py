# data_preprocessing.py

import pandas as pd
import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
import pickle
import os
from .tof_utils import interpolate_tof
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from scipy.spatial.transform import Rotation as R
import warnings
from scipy import signal
from joblib import Parallel, delayed
import time

# --- 2. 优化后的语谱图生成函数 ---
def generate_spectrogram(ts_data, fs, nperseg, noverlap, max_length):
    """一个封装好的、使用动态参数的优化版函数"""
    if ts_data is None or len(ts_data) == 0:
        # 如果数据为空, 计算预期的形状并返回零矩阵
        freqs, time_bins, _ = signal.stft(np.zeros(max_length), fs=fs, nperseg=nperseg, noverlap=noverlap)
        spec_shape = (len(freqs), len(time_bins))
        return np.zeros(spec_shape, dtype=np.float32)
    
    f, t, Zxx = signal.stft(ts_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    log_spectrogram = np.log1p(np.abs(Zxx))
    return log_spectrogram.astype(np.float32)


# --- 3. 用于并行处理的工作函数 ---
def process_and_get_stats(group, spec_params, max_length):
    """处理单个group，并返回其统计数据"""
    # 从字典中解包参数
    fs = spec_params['fs']
    nperseg = spec_params['nperseg']
    noverlap = spec_params['noverlap']
    
    count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    spec_source_cols = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z']
    for col in spec_source_cols: # 假设 spec_source_cols 已定义
        signal_1d = group[col].values
        seq_len = len(signal_1d)
        
        if seq_len >= max_length:
            padded_signal = signal_1d[-max_length:]
        else:
            padded_signal = np.pad(signal_1d, (max_length - seq_len, 0), 'constant')
        
        # 调用使用动态参数的优化函数
        spec = generate_spectrogram(padded_signal, fs, nperseg, noverlap, max_length)
        
        count += spec.size
        total_sum += np.sum(spec)
        total_sum_sq += np.sum(spec**2)
            
    return count, total_sum, total_sum_sq

def _remove_gravity_from_acc_polars(group_df: pl.DataFrame) -> pl.DataFrame:
    """
    [Internal Helper] Polars-native version of remove_gravity_from_acc.
    Accepts and returns a Polars DataFrame.
    """
    acc_values = group_df.select(['acc_x', 'acc_y', 'acc_z']).to_numpy()
    quat_values = group_df.select(['rot_x', 'rot_y', 'rot_z', 'rot_w']).to_numpy()

    # Detect sequences where all quaternions are effectively identity [0,0,0,1]
    # This typically indicates originally-missing rot_* that were imputed.
    if quat_values.size == 0:
        all_missing_seq = False
    else:
        identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        per_row_identity = np.all(np.isclose(quat_values, identity_quat, atol=1e-6), axis=1)
        all_missing_seq = bool(per_row_identity.all())

    linear_accel = np.zeros_like(acc_values)
    rot_missing_col = np.zeros((acc_values.shape[0],), dtype=np.float32)

    if all_missing_seq:
        # Fallback: estimate gravity from accelerometer via EMA low-pass, then subtract
        # Sampling ~10 Hz → choose cutoff ~0.3 Hz
        dt = 0.1
        fc = 0.3
        tau = 1.0 / (2.0 * np.pi * fc)
        alpha = dt / (tau + dt)

        gravity = np.zeros_like(acc_values, dtype=np.float32)
        if acc_values.shape[0] > 0:
            gravity[0] = acc_values[0]
            for t in range(1, acc_values.shape[0]):
                gravity[t] = (1.0 - alpha) * gravity[t - 1] + alpha * acc_values[t]
        linear_accel = (acc_values - gravity).astype(np.float32)
        rot_missing_col[:] = 1.0
    else:
        gravity_world = np.array([0, 0, 9.81])
        for i in range(acc_values.shape[0]):
            if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
                linear_accel[i, :] = acc_values[i, :]
                continue
            try:
                rotation = R.from_quat(quat_values[i])
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
            except (ValueError, IndexError):
                linear_accel[i, :] = acc_values[i, :]

    return pl.DataFrame({
        'linear_acc_x': linear_accel[:, 0],
        'linear_acc_y': linear_accel[:, 1],
        'linear_acc_z': linear_accel[:, 2],
        'rot_missing': rot_missing_col,
    }).cast({'linear_acc_x': pl.Float32, 'linear_acc_y': pl.Float32, 'linear_acc_z': pl.Float32, 'rot_missing': pl.Float32})

def _calculate_angular_velocity_from_quat_polars(group_df: pl.DataFrame, time_delta=1/10) -> pl.DataFrame:
    """
    [Internal Helper] Polars-native version of calculate_angular_velocity.
    """
    quat_values = group_df.select(['rot_x', 'rot_y', 'rot_z', 'rot_w']).to_numpy()
    
    if len(quat_values) < 2:
        return pl.DataFrame({
            'angular_vel_x': np.zeros(len(quat_values)),
            'angular_vel_y': np.zeros(len(quat_values)),
            'angular_vel_z': np.zeros(len(quat_values)),
        }, schema={'angular_vel_x': pl.Float32, 'angular_vel_y': pl.Float32, 'angular_vel_z': pl.Float32})

    # If the entire sequence quaternions are identity (imputed), zero angular velocity
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if np.all(np.isclose(quat_values, identity_quat, atol=1e-6)):
        zeros = np.zeros((len(quat_values),), dtype=np.float32)
        return pl.DataFrame({
            'angular_vel_x': zeros,
            'angular_vel_y': zeros,
            'angular_vel_z': zeros,
        }, schema={'angular_vel_x': pl.Float32, 'angular_vel_y': pl.Float32, 'angular_vel_z': pl.Float32})

    q_t_plus_dt = R.from_quat(quat_values)
    q_t = R.from_quat(np.roll(quat_values, 1, axis=0))
    q_t.as_quat()[0] = q_t_plus_dt.as_quat()[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rot_delta = q_t.inv() * q_t_plus_dt
        
    angular_vel = rot_delta.as_rotvec() / time_delta
    angular_vel[0, :] = 0
    
    return pl.DataFrame({
        'angular_vel_x': angular_vel[:, 0],
        'angular_vel_y': angular_vel[:, 1],
        'angular_vel_z': angular_vel[:, 2],
    }).cast(pl.Float32)
    
def _calculate_angular_distance_polars(group_df: pl.DataFrame) -> pl.DataFrame:
    """
    [Internal Helper] Polars-native version of calculate_angular_distance.
    """
    quat_values = group_df.select(['rot_x', 'rot_y', 'rot_z', 'rot_w']).to_numpy()

    if len(quat_values) < 2:
        return pl.DataFrame({'angular_distance': np.zeros(len(quat_values))}, schema={'angular_distance': pl.Float64})
        
    # Identity-only sequences → zero angular distance
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    if np.all(np.isclose(quat_values, identity_quat, atol=1e-6)):
        zeros = np.zeros((len(quat_values),), dtype=np.float32)
        return pl.DataFrame({'angular_distance': zeros}).cast(pl.Float32)

    q2 = R.from_quat(quat_values)
    q1 = R.from_quat(np.roll(quat_values, 1, axis=0))
    q1.as_quat()[0] = q2.as_quat()[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dR = q1.inv() * q2
        
    angular_dist = np.linalg.norm(dR.as_rotvec(), axis=1)
    angular_dist[0] = 0
    
    return pl.DataFrame({'angular_distance': angular_dist}).cast(pl.Float32)


def feature_engineering(train_df: pd.DataFrame): 
    """ 
    Applies the full feature engineering pipeline.
    MODIFIED: Removed Jerk and Snap calculations as they are being replaced by spectrograms.
    """ 
    print("\nApplying advanced feature engineering (with Polars backend)...")

    # --- 步骤 1: 从 Pandas 转换为 Polars ---
    original_index = train_df.index
    pl_df = pl.from_pandas(train_df)
    cols_to_process = [c for c in pl_df.columns if c.startswith('acc_') or c.startswith('rot_')]

    # --- 步骤 2: 使用 Polars 表达式进行高性能计算 ---
    pl_df = pl_df.with_columns(
        pl.col(cols_to_process).interpolate().over('sequence_id').fill_null(0.0)
    ).with_columns(
        pl.when((pl.col('rot_x') == 0) & (pl.col('rot_y') == 0) & (pl.col('rot_z') == 0) & (pl.col('rot_w') == 0))
          .then(1.0)
          .otherwise(pl.col('rot_w'))
          .alias('rot_w')
    )

    print("Calculating engineered features (excluding jerk/snap)...")
    # 计算基础特征
    pl_df = pl_df.with_columns(
        (pl.col('acc_x')**2 + pl.col('acc_y')**2 + pl.col('acc_z')**2).sqrt().alias('acc_mag'),
        (2 * pl.col('rot_w').clip(-1, 1).arccos()).alias('rot_angle'),
    )

    # --- 步骤 3: 使用 map_groups 高效调用新的 Polars 辅助函数 ---
    linear_accel_results = pl_df.group_by('sequence_id', maintain_order=True).map_groups(_remove_gravity_from_acc_polars)
    pl_df = pl.concat([pl_df, linear_accel_results], how='horizontal')
    pl_df = pl_df.with_columns(
        (pl.col('linear_acc_x')**2 + pl.col('linear_acc_y')**2 + pl.col('linear_acc_z')**2).sqrt().alias('linear_acc_mag')
    )

    # --- NEW: Thermopile (THM) per-sensor and overall missingness flags (computed before THM interpolation) ---
    thm_cols = [c for c in pl_df.columns if c.startswith('thm_') and len(c.split('_')) == 2]
    if thm_cols:
        # Per-sequence, per-sensor all-null flags
        thm_missing_agg = pl_df.group_by('sequence_id', maintain_order=True).agg(
            [pl.col(col).is_null().all().alias(f"{col}_missing") for col in thm_cols]
        )
        # Overall THM missing: all sensors missing within the sequence
        sum_expr = None
        for col in thm_cols:
            expr = pl.col(f"{col}_missing").cast(pl.Int8)
            sum_expr = expr if sum_expr is None else (sum_expr + expr)
        thm_missing_agg = thm_missing_agg.with_columns(
            (sum_expr == pl.lit(len(thm_cols))).alias('thm_missing')
        )
        # Cast flags to Float32 for consistency
        cast_exprs = [pl.col(f"{col}_missing").cast(pl.Float32) for col in thm_cols] + [pl.col('thm_missing').cast(pl.Float32)]
        thm_missing_agg = thm_missing_agg.with_columns(cast_exprs)
        # Join back per-row
        pl_df = pl_df.join(thm_missing_agg, on='sequence_id', how='left')

    angular_vel_results = pl_df.group_by('sequence_id', maintain_order=True).map_groups(_calculate_angular_velocity_from_quat_polars)
    pl_df = pl.concat([pl_df, angular_vel_results], how='horizontal')

    angular_dist_results = pl_df.group_by('sequence_id', maintain_order=True).map_groups(_calculate_angular_distance_polars)
    pl_df = pl.concat([pl_df, angular_dist_results], how='horizontal')
    
    # --- 步骤 4: 定义最终特征列并进行最终清理 ---
    # --- MODIFIED: Removed jerk and snap features from the list ---
    final_feature_cols = [ 
        'rot_w', 'rot_x', 'rot_y', 'rot_z',  
        'acc_mag', 'rot_angle',
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z',  
        'linear_acc_mag',
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z',  
        'angular_distance'
    ] 
    # Exclude *_missing flags from time-series feature list
    tof_thm_cols = [
        c for c in pl_df.columns
        if (c.startswith('tof_') or c.startswith('thm_')) and not c.endswith('_missing')
    ] 
    final_feature_cols.extend(tof_thm_cols) 
    final_feature_cols = [c for c in final_feature_cols if c in pl_df.columns] 

    print("Cleaning up all NaNs generated during feature engineering...")
    pl_df = pl_df.with_columns(
        pl.col(final_feature_cols).interpolate().over('sequence_id').fill_null(0.0)
    )
    
    print(f"Generated {len(final_feature_cols)} features after engineering.")

    # --- 步骤 5: 从 Polars 转换回 Pandas ---
    final_pandas_df = pl_df.to_pandas()
    final_pandas_df.index = original_index

    return final_pandas_df, final_feature_cols

# Static feature columns (shared between training and inference)
STATIC_FEATURE_COLS = [
    'adult_child', 'age', 'sex', 'handedness', 'height_cm', 
    'shoulder_to_wrist_cm', 'elbow_to_wrist_cm',
    'rot_missing',
    # Per-sensor THM missing flags will be dynamically present if THM exists
    # We'll treat any *_missing columns as static by selection in downstream code
]

# Function to dynamically detect sensor configuration from data columns
def get_sensor_config(df_columns):
    """
    Dynamically detect sensor configuration from dataframe columns.
    
    Returns:
        dict: Configuration with num_thm_sensors, num_tof_sensors, tof_pixels_per_sensor
    """
    thm_sensors = set()
    tof_sensors = set()
    tof_pixels = set()
    
    for col in df_columns:
        # Detect THM sensors (pattern: thm_X)
        if col.startswith('thm_') and len(col.split('_')) == 2:
            try:
                sensor_id = int(col.split('_')[1])
                thm_sensors.add(sensor_id)
            except ValueError:
                pass
        
        # Detect TOF sensors and pixels (pattern: tof_X_vY)
        elif col.startswith('tof_') and '_v' in col:
            parts = col.split('_')
            if len(parts) == 3:
                try:
                    sensor_id = int(parts[1])
                    pixel_id = int(parts[2][1:])  # Remove 'v' prefix
                    tof_sensors.add(sensor_id)
                    tof_pixels.add(pixel_id)
                except ValueError:
                    pass
    
    return {
        'num_thm_sensors': len(thm_sensors),
        'num_tof_sensors': len(tof_sensors),
        'tof_pixels_per_sensor': len(tof_pixels),
        'thm_sensor_ids': sorted(thm_sensors),
        'tof_sensor_ids': sorted(tof_sensors)
    }

# Generate feature columns dynamically based on actual data
def generate_feature_columns(df_columns):
    """Generate THM and TOF feature column lists based on actual data columns."""
    config = get_sensor_config(df_columns)
    
    # Generate THM columns based on detected sensors
    thm_cols = []
    for sensor_id in config['thm_sensor_ids']:
        col = f"thm_{sensor_id}"
        if col in df_columns:
            thm_cols.append(col)
    
    # Generate TOF columns based on detected sensors and pixels
    tof_cols = []
    for sensor_id in config['tof_sensor_ids']:
        for pixel in range(config['tof_pixels_per_sensor']):
            col = f"tof_{sensor_id}_v{pixel}"
            if col in df_columns:
                tof_cols.append(col)
    
    return thm_cols, tof_cols

# Helper to locate shared weights directory inside cmi-submission
def _get_weights_dir():
    module_dir = os.path.dirname(os.path.abspath(__file__))  # …/cmi-submission/data_utils
    subm_root  = os.path.abspath(os.path.join(module_dir, '..'))      # …/cmi-submission
    weights_dir = os.path.join(subm_root, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir

def load_and_preprocess_data(variant: str = "full"):
    """
    Load training data and demographics, preprocess, and return a full DataFrame.
    MODIFIED: Applies advanced feature engineering after data loading.
    """
    print("Loading data...")
    
    # --- File path logic ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    dev_data_dir = os.path.join(project_root, 'development', 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    demographics_path = os.path.join(data_dir, 'train_demographics.csv')
    if not os.path.exists(train_path):
        alt_train = os.path.join(dev_data_dir, 'train.csv')
        if os.path.exists(alt_train):
            train_path = alt_train
            demographics_path = os.path.join(dev_data_dir, 'train_demographics.csv')
            print(f"⚠️  train.csv not found in {data_dir}. Falling back to {dev_data_dir}.")
    if not os.path.exists(train_path):
        raise FileNotFoundError("train.csv not found in either cmi-submission/data or development/data")
    
    # --- Data loading and merging ---
    train_df = pd.read_csv(train_path)
    demographics_df = pd.read_csv(demographics_path)
    train_df = train_df.merge(demographics_df, on='subject', how='left')
    print(f"Train data shape before FE: {train_df.shape}")

    if variant == "full":
        print("\nFiltering out sequences with no valid ToF or THM data...")
        
        # 找出所有以 'tof_' 或 'thm_' 开头的列 (这部分逻辑不变)
        tof_cols = [c for c in train_df.columns if c.startswith('tof_')]
        thm_cols = [c for c in train_df.columns if c.startswith('thm_')]
        all_sensor_cols = tof_cols + thm_cols
        
        if all_sensor_cols:
            original_seq_count = train_df['sequence_id'].nunique()
            # 1. 对所有传感器列创建一个布尔 DataFrame (True 代表非空值)
            #    然后使用 .any(axis=1) 横向检查每一行，只要行内有一个 True，结果就为 True。
            #    这一步会生成一个布尔 Series，长度与 train_df 的行数相同。
            has_valid_row = train_df[all_sensor_cols].notna().any(axis=1)
            
            # 2. 使用 .loc 基于上面的布尔 Series 快速定位到所有包含有效数据的行，
            #    并提取这些行的 'sequence_id'，最后用 .unique() 获取不重复的ID列表。
            #    这比逐个分组应用函数快几个数量级。
            full_quality_sids = train_df.loc[has_valid_row, 'sequence_id'].unique()
            
            # 3. 使用 .isin() 高效过滤。这是基于列表筛选行的最快方法。
            train_df = train_df[train_df['sequence_id'].isin(full_quality_sids)]
            print(f"  {original_seq_count} total sequences found.")
            print(f"  {len(full_quality_sids)} sequences have at least one valid ToF or THM reading and will be used.")
            print(f"  Filtered data shape: {train_df.shape}")
    
    # --- Label encoding ---
    label_encoder = LabelEncoder()
    train_df['gesture_encoded'] = label_encoder.fit_transform(train_df['gesture'])

    # --- Add TOF missing flags BEFORE interpolation ---
    if variant != "imu":
        train_df = add_tof_missing_flags(train_df)

    # --- Spatial interpolation for TOF sensors ---
    if variant != "imu":
        train_df = interpolate_tof(train_df)
    
    # --- *** NEW: APPLY ADVANCED FEATURE ENGINEERING *** ---
    train_df, feature_cols = feature_engineering(train_df)

    #  将静态特征列添加回总特征列表
    # 1. 找出数据中实际存在的静态列
    existing_static_cols = [c for c in STATIC_FEATURE_COLS if c in train_df.columns]
    
    # 2. 将它们添加到 feature_cols 列表中，并去重
    for col in existing_static_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    # 3. 动态添加所有 *_missing 静态标志列到特征列表（静态分支专用）
    dynamic_missing_flags = [c for c in train_df.columns if c.endswith('_missing')]
    for col in dynamic_missing_flags:
        if col not in feature_cols:
            feature_cols.append(col)
    
    # --- Filter features based on variant if necessary ---
    if variant == "imu":
        # 确保在imu模式下，特征列只包含IMU和人口统计学特征
        imu_engineered_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
        demographic_cols = [c for c in STATIC_FEATURE_COLS if c in train_df.columns]
        # 合并并去重
        feature_cols = sorted(list(set(imu_engineered_cols + demographic_cols)))
        
    print(f"Variant: {variant}. Final feature columns after filtering: {len(feature_cols)}")

    # Ensure chronological order so interpolation is meaningful
    train_df = train_df.sort_values(['sequence_id', 'sequence_counter'])
    
    # Return the full preprocessed DataFrame
    print("✅ Preprocessing complete. Returning full DataFrame.")
    return train_df, label_encoder, feature_cols

def pad_sequences(sequences, max_length: int):
    """
    Pad sequences to same length and generate a corresponding attention mask.
    MODIFIED: Creates float32 arrays to save memory and requires max_length.
    """
    if not sequences:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32) # CHANGED
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    print(f"Padding sequences to length: {max_length}")
    print("Strategy: Keep END of sequences, pad zeros at BEGINNING, generate mask")
    
    num_features = sequences[0].shape[1]
    # CHANGED: Specify dtype=np.float32 for both arrays
    padded_sequences = np.zeros((len(sequences), max_length, num_features), dtype=np.float32)
    masks = np.zeros((len(sequences), max_length), dtype=np.float32) 
    
    for i, seq in enumerate(sequences):
        # Ensure the sequence being placed is also float32
        seq_as_float32 = np.asarray(seq, dtype=np.float32) # ADDED this line
        seq_length = len(seq_as_float32)
        
        if seq_length >= max_length:
            padded_sequences[i, :, :] = seq_as_float32[-max_length:, :]
            masks[i, :] = 1.0
        else:
            start_idx = max_length - seq_length
            padded_sequences[i, start_idx:, :] = seq_as_float32
            masks[i, start_idx:] = 1.0
            
    return padded_sequences, masks

class TofScaler(BaseEstimator, TransformerMixin):
    """
    一个自定义的Scikit-learn转换器，用于处理ToF（Time-of-Flight）数据。
    
    它只对大于等于0的有效距离值进行Min-Max缩放，而忽略代表“无响应”的-1值。
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.scaler_ = None

    def fit(self, X, y=None):
        """
        从输入数据X中学习缩放参数。
        """
        X_np = np.asarray(X)
        values_to_fit = X_np[X_np != -1].reshape(-1, 1)
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range)
        if len(values_to_fit) > 0:
            self.scaler_.fit(values_to_fit)
        return self

    def transform(self, X):
        """
        使用学习到的参数转换数据X。
        """
        if self.scaler_ is None:
            raise RuntimeError("This TofScaler instance is not fitted yet.")
        
        X_np = np.asarray(X)
        X_transformed = X_np.copy().astype(float)
        mask = (X_np != -1)
        
        if np.any(mask):
            valid_data = X_np[mask].reshape(-1, 1)
            X_transformed[mask] = self.scaler_.transform(valid_data).flatten()
            
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features is required for get_feature_names_out.")
        return np.asarray(input_features, dtype=object)

class MaskedStandardAndMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    A custom scaler that:
    - Applies Z-score standardization to IMU/ROT/THM/engineered features while IGNORING invalid entries
      for ROT and THM using their corresponding missing flags.
    - Applies Min-Max scaling to TOF features with masking by per-sensor flags `tof_{sid}_missing`.
      TOF values for rows where the corresponding sensor flag == 1 are excluded from min/max statistics
      and assigned a sentinel value after scaling.
    - Leaves *_missing flags and other passthrough columns unchanged.

    Invalid definitions used during fit/transform:
    - ROT columns (prefix 'rot_'): invalid where 'rot_missing' == 1 (sequence-level flag)
    - THM sensor columns (pattern 'thm_{id}'): invalid where 'thm_{id}_missing' == 1 (sequence-level flag)

    Notes:
    - Feature order is preserved. get_feature_names_out() returns the input column order.
    - Outputs float32 arrays to match the rest of the pipeline.
    """

    def __init__(self, zscore_prefixes=None, demographic_cols=None, invalid_fill_value: float = -12.0, invalid_fill_value_tof: float = -1.0):
        self.zscore_prefixes = zscore_prefixes or ['acc_', 'rot_', 'thm_', 'linear_', 'angular_']
        self.demographic_cols = demographic_cols or ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
        self.invalid_fill_value = float(invalid_fill_value)
        self.invalid_fill_value_tof = float(invalid_fill_value_tof)
        self.feature_names_ = None
        self.zscore_cols_ = None
        self.tof_cols_ = None
        self.passthrough_cols_ = None
        self.mean_ = {}
        self.std_ = {}
        self.min_ = {}
        self.max_ = {}

    def _is_thm_sensor_col(self, col_name: str) -> bool:
        return col_name.startswith('thm_') and (len(col_name.split('_')) == 2)

    def _build_valid_mask(self, X_df: pd.DataFrame, col: str):
        # Default: all valid
        valid_mask = np.ones((len(X_df),), dtype=bool)
        if col.startswith('rot_'):
            if 'rot_missing' in X_df.columns:
                valid_mask = (X_df['rot_missing'].values == 0)
        elif self._is_thm_sensor_col(col):
            sensor_id = col.split('_')[1]
            flag_col = f"thm_{sensor_id}_missing"
            if flag_col in X_df.columns:
                valid_mask = (X_df[flag_col].values == 0)
        elif col.startswith('tof_') and '_v' in col and not col.endswith('_missing'):
            # Map tof_{sid}_v{pix} -> tof_{sid}_missing
            parts = col.split('_')
            if len(parts) >= 3:
                sensor_id = parts[1]
                flag_col = f"tof_{sensor_id}_missing"
                if flag_col in X_df.columns:
                    valid_mask = (X_df[flag_col].values == 0)
        return valid_mask

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MaskedStandardAndMinMaxScaler expects a pandas DataFrame.")

        self.feature_names_ = list(X.columns)

        # Identify columns
        all_missing_flags = [c for c in X.columns if c.endswith('_missing')]
        # z-score candidates: prefixes + demographics, excluding *_missing and tof_
        zscore_cols = [
            c for c in X.columns
            if (any(c.startswith(p) for p in self.zscore_prefixes) or c in self.demographic_cols)
            and not c.endswith('_missing')
            and not c.startswith('tof_')
        ]
        self.zscore_cols_ = zscore_cols
        self.tof_cols_ = [c for c in X.columns if c.startswith('tof_') and not c.endswith('_missing')]
        # passthrough is everything else
        self.passthrough_cols_ = [
            c for c in X.columns
            if c not in set(self.zscore_cols_).union(self.tof_cols_)
        ]

        # Compute Z-score stats with masking
        for col in self.zscore_cols_:
            values = X[col].values.astype(np.float64)
            valid_mask = self._build_valid_mask(X, col)
            valid_values = values[valid_mask]
            if valid_values.size == 0:
                mean_val, std_val = 0.0, 1.0
            else:
                mean_val = float(np.mean(valid_values))
                std_val = float(np.std(valid_values))
                if std_val == 0.0:
                    std_val = 1.0
            self.mean_[col] = mean_val
            self.std_[col] = std_val

        # Compute Min-Max stats for TOF (masked by per-sensor flags tof_{sid}_missing)
        for col in self.tof_cols_:
            values = X[col].values.astype(np.float64)
            valid_mask = self._build_valid_mask(X, col)
            valid_values = values[valid_mask]
            if valid_values.size == 0:
                self.min_[col], self.max_[col] = 0.0, 1.0
            else:
                min_val = float(np.min(valid_values))
                max_val = float(np.max(valid_values))
                if max_val == min_val:
                    max_val = min_val + 1.0
                self.min_[col], self.max_[col] = min_val, max_val

        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MaskedStandardAndMinMaxScaler expects a pandas DataFrame.")
        if self.feature_names_ is None:
            raise RuntimeError("This scaler instance is not fitted yet.")

        # Ensure same columns/order
        X = X[self.feature_names_]

        n_rows = len(X)
        n_cols = len(self.feature_names_)
        out = np.zeros((n_rows, n_cols), dtype=np.float32)

        for j, col in enumerate(self.feature_names_):
            col_vals = X[col].values.astype(np.float32)
            if col in self.zscore_cols_:
                mean_val = self.mean_.get(col, 0.0)
                std_val = self.std_.get(col, 1.0)
                scaled = (col_vals - mean_val) / std_val
                # Assign sentinel to invalid entries for ROT/THM so they are distinguishable from mean-zero
                valid_mask = self._build_valid_mask(X, col)
                scaled[~valid_mask] = self.invalid_fill_value
                out[:, j] = scaled.astype(np.float32)
            elif col in self.tof_cols_:
                min_val = self.min_.get(col, 0.0)
                max_val = self.max_.get(col, 1.0)
                denom = max_val - min_val
                if denom == 0.0:
                    scaled = np.zeros_like(col_vals, dtype=np.float32)
                else:
                    scaled = ((col_vals - min_val) / denom).astype(np.float32)
                # Set invalid rows (where sensor is missing) to sentinel
                valid_mask = self._build_valid_mask(X, col)
                scaled[~valid_mask] = self.invalid_fill_value_tof
                out[:, j] = scaled
            else:
                # Passthrough (e.g., *_missing flags)
                out[:, j] = col_vals

        return out.astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)

# --- NEW: Compute TOF missing flags BEFORE any interpolation/fill ---
def add_tof_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-sequence, per-sensor TOF missing flags named `tof_{sid}_missing`.
    A sensor is considered missing for a sequence if ALL its 64 pixel columns
    are NaN for ALL rows in that sequence.

    Must be called BEFORE any TOF interpolation/fill so flags reflect true
    missingness. Flags are cast to Float32 to be consistent with other *_missing.
    """
    if df is None or df.empty:
        return df

    config = get_sensor_config(df.columns)
    tof_sensor_ids = config['tof_sensor_ids']
    if not tof_sensor_ids:
        return df

    pl_df = pl.from_pandas(df)

    # Build per-row all-null indicators per sensor
    row_null_cols = []
    for sid in tof_sensor_ids:
        sensor_cols = [c for c in df.columns if c.startswith(f"tof_{sid}_v")]
        if not sensor_cols:
            continue
        row_flag_col = f"_row_all_null_tof_{sid}"
        row_null_cols.append(row_flag_col)
        pl_df = pl_df.with_columns(
            pl.all_horizontal([pl.col(c).is_null() for c in sensor_cols]).alias(row_flag_col)
        )

    if not row_null_cols:
        return df

    # Aggregate to sequence-level missing flags (True if all rows are all-null)
    agg_exprs = []
    for col in row_null_cols:
        # col pattern: _row_all_null_tof_{sid}
        sid = col.split('_')[-1]
        agg_exprs.append(pl.col(col).all().alias(f"tof_{sid}_missing"))

    flags_df = pl_df.group_by('sequence_id', maintain_order=True).agg(agg_exprs)

    # Cast flags to Float32 (0.0/1.0)
    cast_map = {c: pl.Float32 for c in flags_df.columns if c != 'sequence_id'}
    flags_df = flags_df.cast(cast_map)

    # Join back per-row and drop temps
    pl_df = pl_df.join(flags_df, on='sequence_id', how='left')
    pl_df = pl_df.drop(row_null_cols)

    return pl_df.to_pandas()

def normalize_features(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    根据指定的接口，使用统一的预处理器对训练集和验证集进行标准化。
    - 对IMU, 温度, 人口统计学特征以及所有新工程化的特征应用Z-score标准化。
    - 对ToF特征应用自定义的Min-Max规范化（-1保持不变）。
    - 返回标准化后的DataFrame以及一个单一、已拟合的scaler对象。
    """
    # 1. 识别需要Z-score标准化的特征
    zscore_prefixes = ['acc_', 'rot_', 'thm_', 'linear_', 'angular_']
    zscore_cols = [
        col for col in X_train.columns 
        if any(col.startswith(p) for p in zscore_prefixes) and not col.endswith('_failed')
    ]
    # Do not standardize binary indicators like *_missing
    all_missing_flags = [c for c in X_train.columns if c.endswith('_missing')]
    zscore_cols = [col for col in zscore_cols if col not in all_missing_flags]
    
    demographic_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    existing_demographic_cols = [col for col in demographic_cols if col in X_train.columns]
    
    for col in existing_demographic_cols:
        if col not in zscore_cols:
            zscore_cols.append(col)

    # 2. 构建自定义带掩码的缩放器（Z-score 忽略无效 ROT/THM；TOF 使用 Min-Max 与之前一致）
    scaler = MaskedStandardAndMinMaxScaler(
        zscore_prefixes=zscore_prefixes,
        demographic_cols=existing_demographic_cols,
    )

    # 3. 在训练数据上拟合scaler
    print(f"\nNormalizing features...")
    # 统计信息仅用于日志，实际列选择在自定义scaler内部完成
    print(f"Applying Z-score (masked) to ~{len(zscore_cols)} columns (prefix based, excluding *_missing & tof_).")
    print(f"Applying Min-Max to TOF columns (unchanged from previous training).")
    scaler.fit(X_train)

    # 4. 使用已拟合的scaler转换训练集和验证集
    X_train_transformed_np = scaler.transform(X_train).astype(np.float32)
    X_val_transformed_np = scaler.transform(X_val).astype(np.float32)

    # 5. 获取列名并重建DataFrame（顺序保持不变）
    feature_names = scaler.get_feature_names_out()
    X_train_normalized = pd.DataFrame(X_train_transformed_np, index=X_train.index, columns=feature_names)
    X_val_normalized = pd.DataFrame(X_val_transformed_np, index=X_val.index, columns=feature_names)
    
    return X_train_normalized, X_val_normalized, scaler

# --- NEW: Function to prepare base data (for hyperparameter search) ---
def prepare_base_data_kfold(variant: str = "full", n_splits: int = 5):
    """
    加载并预处理所有时域和静态数据，进行特征工程，标准化，并按K-Fold分割。
    此函数 *不* 生成频谱图，为超参数搜索优化。
    """
    # 1. 加载并进行基础预处理 (包括特征工程)
    start_time = time.time()
    all_data_df, label_encoder, all_feature_cols = load_and_preprocess_data(variant)
    print(f"Base data loading and FE took {time.time() - start_time:.2f} seconds.")

    # 2. 准备分层分组K折交叉验证
    labels_map_df = all_data_df[["sequence_id", "gesture_encoded", "subject"]].drop_duplicates().reset_index(drop=True)
    y = labels_map_df["gesture_encoded"].values
    subjects = labels_map_df["subject"].values
    unique_seq_ids = labels_map_df["sequence_id"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    base_fold_data = []

    print(f"\nPreparing {n_splits}-fold base splits (T-series & Static features)...")

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(unique_seq_ids)), y, groups=subjects)):
        print(f"\n--- Preparing Base Fold {fold_idx+1}/{n_splits} ---")
        train_sids = unique_seq_ids[train_idx]
        val_sids = unique_seq_ids[val_idx]

        train_df = all_data_df[all_data_df["sequence_id"].isin(train_sids)].copy()
        val_df = all_data_df[all_data_df["sequence_id"].isin(val_sids)].copy()

        y_train = labels_map_df[labels_map_df["sequence_id"].isin(train_sids)]["gesture_encoded"].values
        y_val = labels_map_df[labels_map_df["sequence_id"].isin(val_sids)]["gesture_encoded"].values

        # 3. 标准化时域特征
        X_train_norm, X_val_norm, scaler_fold = normalize_features(train_df[all_feature_cols], val_df[all_feature_cols])
        # Add sequence_id back for grouping
        X_train_norm["sequence_id"] = train_df["sequence_id"]
        X_val_norm["sequence_id"] = val_df["sequence_id"]

        base_fold_data.append({
            'X_train_norm': X_train_norm,
            'X_val_norm': X_val_norm,
            'y_train': y_train,
            'y_val': y_val,
            'train_sids': train_sids,
            'val_sids': val_sids,
            'scaler': scaler_fold,
            'val_idx': val_idx,
            'all_feature_cols': all_feature_cols
        })

    print("\n✅ Base K-fold data prepared.")
    return base_fold_data, label_encoder, y, unique_seq_ids

# --- NEW: Function to generate and attach spectrograms ---
def generate_and_attach_spectrograms(base_fold_data, spec_params, variant="full"):
    """
    接收基础K-Fold数据和频谱图参数，生成频谱图并将其附加到每个折叠中。
    [REVERTED] 恢复了原始的数据处理策略。
    """
    print(f"\nGenerating and attaching spectrograms with params: {spec_params}")
    fs = spec_params['fs']
    nperseg = spec_params['nperseg']
    noverlap = spec_params['noverlap']
    max_length = spec_params['max_length']

    # 动态识别各模态的列名
    sample_fold = base_fold_data[0]
    all_feature_cols = sample_fold['all_feature_cols']
    
    # Treat any *_missing flags as static features alongside STATIC_FEATURE_COLS
    static_cols = [c for c in all_feature_cols if (c in STATIC_FEATURE_COLS) or c.endswith('_missing')]
    thm_cols, tof_cols = generate_feature_columns(all_feature_cols)
    thm_cols = [c for c in thm_cols if c in all_feature_cols]
    tof_cols = [c for c in tof_cols if c in all_feature_cols]
    imu_cols = [c for c in all_feature_cols if c not in static_cols and c not in tof_cols and c not in thm_cols]
    spec_source_cols = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z']
    spec_source_cols = [c for c in spec_source_cols if c in all_feature_cols]
    print(f"Generating spectrograms from {len(spec_source_cols)} source signals.")

    final_fold_data = []

    for fold_idx, base_fold in enumerate(base_fold_data):
        print(f"\n--- Generating Spectrograms for Fold {fold_idx+1}/{len(base_fold_data)} ---")
        X_train_norm = base_fold['X_train_norm']
        X_val_norm = base_fold['X_val_norm']
        train_sids = base_fold['train_sids']
        val_sids = base_fold['val_sids']

        # 1. 计算该折叠训练集的全局频谱图统计量
        print("Calculating global spectrogram statistics...")
        groups = [group for _, group in X_train_norm.groupby('sequence_id')]
        results = Parallel(n_jobs=-1)(delayed(process_and_get_stats)(group, spec_params, max_length) for group in groups)
        
        total_count = sum(r[0] for r in results)
        global_sum = sum(r[1] for r in results)
        global_sum_sq = sum(r[2] for r in results)

        global_spec_mean = global_sum / total_count if total_count > 0 else 0.0
        global_spec_std = np.sqrt(global_sum_sq / total_count - global_spec_mean**2) if total_count > 0 else 1.0
        print(f"  Global Spec Mean: {global_spec_mean:.4f}, Global Spec Std: {global_spec_std:.4f}")
        spec_stats = {'mean': global_spec_mean, 'std': global_spec_std}

        # 2. 分离多模态数据, padding时域数据, 并生成频谱图
        # --- 处理训练集 ---
        grouped_train = X_train_norm.groupby('sequence_id')
        train_static_list, train_imu_list, train_thm_list, train_tof_list, train_spec_list = [], [], [], [], []

        for sid in train_sids:
            group = grouped_train.get_group(sid)
            train_static_list.append(group[static_cols].iloc[0].values)
            train_imu_list.append(group[imu_cols].values)
            train_thm_list.append(group[thm_cols].values) # [REVERTED]
            train_tof_list.append(group[tof_cols].values) # [REVERTED]
            
            sequence_spectrograms = []
            for col in spec_source_cols:
                signal_1d = group[col].values
                seq_len = len(signal_1d)
                padded_signal = signal_1d[-max_length:] if seq_len >= max_length else np.pad(signal_1d, (max_length - seq_len, 0), 'constant')
                spec = generate_spectrogram(padded_signal, fs, nperseg, noverlap, max_length)
                sequence_spectrograms.append(spec)
            train_spec_list.append(np.stack(sequence_spectrograms, axis=0))

        # --- [REVERTED] 恢复到原始的直接padding逻辑 ---
        X_train_static = np.array(train_static_list, dtype=np.float32)
        X_train_imu, train_mask = pad_sequences(train_imu_list, max_length=max_length)
        X_train_thm, _ = pad_sequences(train_thm_list, max_length=max_length)
        X_train_tof, _ = pad_sequences(train_tof_list, max_length=max_length)
        X_train_spec = np.array(train_spec_list, dtype=np.float32)


        # --- 处理验证集 ---
        grouped_val = X_val_norm.groupby('sequence_id')
        val_static_list, val_imu_list, val_thm_list, val_tof_list, val_spec_list = [], [], [], [], []

        for sid in val_sids:
            group = grouped_val.get_group(sid)
            val_static_list.append(group[static_cols].iloc[0].values)
            val_imu_list.append(group[imu_cols].values)
            val_thm_list.append(group[thm_cols].values) # [REVERTED]
            val_tof_list.append(group[tof_cols].values) # [REVERTED]

            sequence_spectrograms = []
            for col in spec_source_cols:
                signal_1d = group[col].values
                seq_len = len(signal_1d)
                padded_signal = signal_1d[-max_length:] if seq_len >= max_length else np.pad(signal_1d, (max_length - seq_len, 0), 'constant')
                spec = generate_spectrogram(padded_signal, fs, nperseg, noverlap, max_length)
                sequence_spectrograms.append(spec)
            val_spec_list.append(np.stack(sequence_spectrograms, axis=0))

        # --- [REVERTED] 恢复到原始的直接padding逻辑 ---
        X_val_static = np.array(val_static_list, dtype=np.float32)
        X_val_imu, val_mask = pad_sequences(val_imu_list, max_length=max_length)
        X_val_thm, _ = pad_sequences(val_thm_list, max_length=max_length)
        X_val_tof, _ = pad_sequences(val_tof_list, max_length=max_length)
        X_val_spec = np.array(val_spec_list, dtype=np.float32)

        # 3. 存储该折的所有数据
        final_fold = {
            'X_train_imu': X_train_imu, 'X_train_thm': X_train_thm, 'X_train_tof': X_train_tof,
            'X_train_spec': X_train_spec, 'X_train_static': X_train_static,
            'y_train': base_fold['y_train'], 'train_mask': train_mask,
            'X_val_imu': X_val_imu, 'X_val_thm': X_val_thm, 'X_val_tof': X_val_tof,
            'X_val_spec': X_val_spec, 'X_val_static': X_val_static,
            'y_val': base_fold['y_val'], 'val_mask': val_mask,
            'scaler': base_fold['scaler'],
            'spec_stats': spec_stats,
            'val_idx': base_fold['val_idx'],
        }
        final_fold_data.append(final_fold)

    return final_fold_data

# --- MODIFIED: Main data preparation entry point ---
def prepare_data_kfold_multimodal(show_stratification: bool=False, variant: str="full", n_splits: int=5, spec_params: dict = None):
    """
    为多模态架构准备K-Fold数据的主入口函数。
    此函数现在协调基础数据的准备和频谱图的生成。
    """
    full_start_time = time.time()
    # Strict: require spec_params from config and validate overlap
    if spec_params is None:
        raise ValueError("spec_params must be provided (fs, nperseg, max_length, and noverlap or noverlap_ratio).")

    if 'noverlap' in spec_params:
        if spec_params['noverlap'] >= spec_params['nperseg']:
            raise ValueError("spec_params['noverlap'] must be < spec_params['nperseg'].")
        normalized_spec_params = {**spec_params}
    elif 'noverlap_ratio' in spec_params:
        noverlap = int(spec_params['nperseg'] * spec_params['noverlap_ratio'])
        if noverlap >= spec_params['nperseg']:
            raise ValueError("Computed noverlap from noverlap_ratio must be < nperseg.")
        normalized_spec_params = {**spec_params, 'noverlap': noverlap}
    else:
        raise ValueError("spec_params must include either 'noverlap' or 'noverlap_ratio'.")
    
    # 2. 准备基础数据 (除频谱图外的所有内容)
    base_fold_data, label_encoder, y, unique_seq_ids = prepare_base_data_kfold(
        variant=variant, 
        n_splits=n_splits
    )

    # 3. 生成频谱图并附加到基础数据中
    fold_data = generate_and_attach_spectrograms(
        base_fold_data, 
        normalized_spec_params, 
        variant=variant
    )

    print(f"\n✅ Full K-fold data preparation took {time.time() - full_start_time:.2f} seconds.")
    return fold_data, label_encoder, y, unique_seq_ids


if __name__ == "__main__":
    print("Running data preprocessing script...")
    # 可选择 "full" 或 "imu" 变体进行测试
    VARIANT = "full" 
    MAX_LENGTH = 100 # Define max_length for the test run
    
    SPEC_PARAMS = {
        'fs': 10.0,
        'nperseg': 20,
        'noverlap_ratio': 0.75,
        'max_length': MAX_LENGTH,
    }
    fold_data, le, y_all, sids = prepare_data_kfold_multimodal(variant=VARIANT, spec_params=SPEC_PARAMS)
    
    print("\n--- Example: Data from Fold 1 ---")
    first_fold = fold_data[0]
    print(f"X_train_imu shape: {first_fold['X_train_imu'].shape}")
    print(f"X_train_thm shape: {first_fold['X_train_thm'].shape}")
    print(f"X_train_tof shape: {first_fold['X_train_tof'].shape}")
    print(f"X_train_spec shape: {first_fold['X_train_spec'].shape}")
    print(f"X_train_static shape: {first_fold['X_train_static'].shape}")
    print(f"y_train shape: {first_fold['y_train'].shape}")
    print(f"Scaler type: {type(first_fold['scaler'])}")
    print(f"Spec stats: {first_fold['spec_stats']}")
    
    print("\nData preprocessing script finished successfully!")