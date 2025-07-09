import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import pickle
import os
from .tof_utils import interpolate_tof
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.transform import Rotation as R
import warnings

# --- START: Advanced Feature Engineering (from analyzed code) ---

def remove_gravity_from_acc(acc_data, rot_data):
    """
    Removes the gravity component from accelerometer data using quaternion rotations.
    """
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values if isinstance(acc_data, pd.DataFrame) else acc_data
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values if isinstance(rot_data, pd.DataFrame) else rot_data
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81]) # Standard gravity vector
    
    for i in range(acc_values.shape[0]):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except (ValueError, IndexError):
            # Fallback to raw acceleration if quaternion is invalid
            linear_accel[i, :] = acc_values[i, :]
            
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/10):
    """
    Calculates angular velocity from quaternion data.
    NOTE: time_delta is set to 1/10 assuming 10Hz data.
    """
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values if isinstance(rot_data, pd.DataFrame) else rot_data
    angular_vel = np.zeros((quat_values.shape[0], 3))
    
    # Use diff to get rotation between consecutive timestamps
    q_t_plus_dt = R.from_quat(quat_values)
    q_t = R.from_quat(np.roll(quat_values, 1, axis=0))
    
    # First element has no previous, handle safely
    q_t.as_quat()[0] = q_t_plus_dt.as_quat()[0] 
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Suppress warnings about near-identity rotations
        # rot_delta = q_t_plus_dt * q_t.inv() # This is incorrect order
        rot_delta = q_t.inv() * q_t_plus_dt
    
    angular_vel = rot_delta.as_rotvec() / time_delta
    
    # Set the first element's velocity to zero as it cannot be calculated
    angular_vel[0, :] = 0
    
    return angular_vel

def calculate_angular_distance(rot_data):
    """
    Calculates the angular distance between consecutive quaternion rotations.
    """
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values if isinstance(rot_data, pd.DataFrame) else rot_data
    angular_dist = np.zeros(quat_values.shape[0])

    # A more efficient vectorized approach
    q2 = R.from_quat(quat_values)
    q1 = R.from_quat(np.roll(quat_values, 1, axis=0))
    
    # First element has no previous, handle safely
    q1.as_quat()[0] = q2.as_quat()[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # dR = q2 * q1.inv() # Incorrect
        dR = q1.inv() * q2
    
    angular_dist = np.linalg.norm(dR.as_rotvec(), axis=1)
    
    # Set the first element's distance to zero
    angular_dist[0] = 0
    
    return angular_dist

def feature_engineering(train_df: pd.DataFrame):
    """
    Applies the full feature engineering pipeline.
    MODIFIED: Includes a detailed logging step to inspect NaNs before the final fix.
    """
    print("\nApplying advanced feature engineering...")
    
    # Suppress potential SettingWithCopyWarning
    pd.options.mode.chained_assignment = None

    # --- Initial NaN handling for raw data ---
    cols_to_process = [c for c in train_df.columns if c.startswith('acc_') or c.startswith('rot_')]
    train_df[cols_to_process] = train_df.groupby('sequence_id')[cols_to_process].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    train_df[cols_to_process] = train_df[cols_to_process].fillna(0.0)
    rot_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    zero_norm_mask = (train_df[rot_cols] == 0).all(axis=1)
    if zero_norm_mask.any():
        train_df.loc[zero_norm_mask, 'rot_w'] = 1.0
    
    # --- Feature Calculation (this will create new NaNs from .diff()) ---
    print("Calculating engineered features...")
    train_df['acc_mag'] = np.sqrt(train_df['acc_x']**2 + train_df['acc_y']**2 + train_df['acc_z']**2)
    train_df['rot_angle'] = 2 * np.arccos(train_df['rot_w'].clip(-1, 1))
    train_df['acc_mag_jerk'] = train_df.groupby('sequence_id')['acc_mag'].diff()
    train_df['rot_angle_vel'] = train_df.groupby('sequence_id')['rot_angle'].diff()
    
    linear_accel_df = train_df.groupby('sequence_id').apply(
        lambda df: pd.DataFrame(remove_gravity_from_acc(df[['acc_x', 'acc_y', 'acc_z']], df[rot_cols]), 
                                columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=df.index), 
        include_groups=False
    ).droplevel(0)
    train_df = train_df.join(linear_accel_df)

    train_df['linear_acc_mag'] = np.sqrt(train_df['linear_acc_x']**2 + train_df['linear_acc_y']**2 + train_df['linear_acc_z']**2)
    train_df['linear_acc_mag_jerk'] = train_df.groupby('sequence_id')['linear_acc_mag'].diff()

    angular_velocity_df = train_df.groupby('sequence_id').apply(
        lambda df: pd.DataFrame(calculate_angular_velocity_from_quat(df[rot_cols]), 
                                columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=df.index), 
        include_groups=False
    ).droplevel(0)
    train_df = train_df.join(angular_velocity_df)
    
    train_df[['angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z']] = train_df.groupby('sequence_id')[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']].diff()
    train_df[['angular_snap_x', 'angular_snap_y', 'angular_snap_z']] = train_df.groupby('sequence_id')[['angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z']].diff()

    angular_distance_df = train_df.groupby('sequence_id').apply(
        lambda df: pd.DataFrame(calculate_angular_distance(df[rot_cols]), 
                                columns=['angular_distance'], index=df.index), 
        include_groups=False
    ).droplevel(0)
    train_df = train_df.join(angular_distance_df)

    # --- Define the final list of all features created ---
    final_feature_cols = [
        'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 
        'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel', 
        'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 
        'linear_acc_mag', 'linear_acc_mag_jerk', 
        'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 
        'angular_distance', 'angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z',
        'angular_snap_x', 'angular_snap_y', 'angular_snap_z'
    ]
    tof_thm_cols = [c for c in train_df.columns if c.startswith('tof_') or c.startswith('thm_')]
    final_feature_cols.extend(tof_thm_cols)
    final_feature_cols = [c for c in final_feature_cols if c in train_df.columns]

    # --- Final, comprehensive NaN cleanup ---
    print(f"Cleaning up all NaNs generated during feature engineering...")
    train_df[final_feature_cols] = train_df.groupby('sequence_id')[final_feature_cols].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    train_df[final_feature_cols] = train_df[final_feature_cols].fillna(0.0)
    
    print(f"Generated {len(final_feature_cols)} features after engineering.")
    
    # Restore pandas options
    pd.options.mode.chained_assignment = 'warn'
    
    return train_df, final_feature_cols

# Helper to locate shared weights directory inside cmi-submission
def _get_weights_dir():
    module_dir = os.path.dirname(os.path.abspath(__file__))  # …/cmi-submission/data_utils
    subm_root  = os.path.abspath(os.path.join(module_dir, '..'))     # …/cmi-submission
    weights_dir = os.path.join(subm_root, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir

def load_and_preprocess_data(variant: str = "full"):
    """
    Load training data and demographics, preprocess, and return a full DataFrame.
    MODIFIED: Applies advanced feature engineering after data loading.
    """
    print("Loading data...")
    
    # --- File path logic (as in your original code) ---
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
    
    # --- Data loading and merging (as in your original code) ---
    train_df = pd.read_csv(train_path)
    demographics_df = pd.read_csv(demographics_path)
    train_df = train_df.merge(demographics_df, on='subject', how='left')
    print(f"Train data shape before FE: {train_df.shape}")

    if variant == "full":
        print("\nFiltering out sequences with no valid ToF data...")
        
        # 1. 识别所有ToF列
        tof_cols = [c for c in train_df.columns if c.startswith('tof_')]
        
        if tof_cols:
            # 2. 对每个序列，检查是否存在任何有效读数。
            #    一个有效读数被定义为：既不等于-1，也不是NaN。
            sids_with_valid_tof = (
                train_df.groupby('sequence_id')[tof_cols]
                        .apply(lambda group_df: ((group_df != -1) & (group_df.notna())).any().any())
            )
            
            # 3. 获取需要保留的序列ID
            full_quality_sids = sids_with_valid_tof[sids_with_valid_tof].index
            
            original_seq_count = train_df['sequence_id'].nunique()
            
            # 4. 根据筛选出的ID过滤DataFrame
            train_df = train_df[train_df['sequence_id'].isin(full_quality_sids)]
            
            print(f"  {original_seq_count} total sequences found.")
            print(f"  {len(full_quality_sids)} sequences have at least one valid ToF reading and will be used.")
            print(f"  Filtered data shape: {train_df.shape}")
    
    # --- Label encoding (as in your original code) ---
    label_encoder = LabelEncoder()
    train_df['gesture_encoded'] = label_encoder.fit_transform(train_df['gesture'])

    if variant != "imu":
        train_df = interpolate_tof(train_df)

    
    # --- *** NEW: APPLY ADVANCED FEATURE ENGINEERING *** ---
    train_df, feature_cols = feature_engineering(train_df)

    
    # --- Filter features based on variant if necessary ---
    if variant == "imu":
        # Keep only IMU-related engineered features + original demographics
        imu_engineered_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
        demographic_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm', 'sex_M']
        # Filter feature_cols to only contain IMU and available demographics
        feature_cols = imu_engineered_cols + [c for c in demographic_cols if c in train_df.columns]
        
    print(f"Variant: {variant}. Final feature columns after filtering: {len(feature_cols)}")

    # Ensure chronological order so interpolation is meaningful
    train_df = train_df.sort_values(['sequence_id', 'sequence_counter'])
    
    # Return the full preprocessed DataFrame
    print("✅ Preprocessing complete. Returning full DataFrame.")
    return train_df, label_encoder, feature_cols

def pad_sequences(sequences, max_length=None):
    """
    Pad sequences to same length, keeping the END of each sequence (most critical part)
    
    Strategy:
    - Keep the LAST max_length timesteps of each sequence
    - Pad with zeros at the BEGINNING if sequence is shorter
    - Truncate from the BEGINNING if sequence is longer
    
    This preserves the end of gestures which contains the most discriminative information.
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    print(f"Padding sequences to length: {max_length}")
    print("Strategy: Keep END of sequences, pad zeros at BEGINNING")
    
    num_features = sequences[0].shape[1]
    padded_sequences = np.zeros((len(sequences), max_length, num_features))
    
    for i, seq in enumerate(sequences):
        seq_length = len(seq)
        
        if seq_length >= max_length:
            # Take the LAST max_length timesteps (truncate from beginning)
            padded_sequences[i, :, :] = seq[-max_length:, :]
        else:
            # Pad with zeros at the BEGINNING, keep original sequence at the END
            start_idx = max_length - seq_length
            padded_sequences[i, start_idx:, :] = seq
    
    return padded_sequences

class TofScaler(BaseEstimator, TransformerMixin):
    """
    一个自定义的Scikit-learn转换器，用于处理ToF（Time-of-Flight）数据。
    
    它只对大于等于0的有效距离值进行Min-Max缩放，而忽略代表“无响应”的-1值。
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        # 在fit方法被调用前，scaler_是None
        self.scaler_ = None

    def fit(self, X, y=None):
        """
        从输入数据X中学习缩放参数。
        X可以是一个numpy数组或pandas DataFrame。
        """
        # 将输入转换为numpy数组以进行高效处理
        X_np = np.asarray(X)
        
        # 提取所有非-1的值来拟合内部的MinMaxScaler
        values_to_fit = X_np[X_np != -1].reshape(-1, 1)
        
        # 初始化并拟合内部的scaler
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range)
        if len(values_to_fit) > 0:
            self.scaler_.fit(values_to_fit)
            
        return self

    def transform(self, X):
        """
        使用学习到的参数转换数据X。
        """
        # 确保fit方法已经被调用
        if self.scaler_ is None:
            raise RuntimeError("This TofScaler instance is not fitted yet. Call 'fit' with appropriate data before using 'transform'.")
        
        X_np = np.asarray(X)
        # 创建一个浮点类型的副本以存储转换后的值
        X_transformed = X_np.copy().astype(float)
        
        # 创建一个布尔掩码，标记所有非-1的值
        mask = (X_np != -1)
        
        # 如果存在任何有效数据，则进行转换
        if np.any(mask):
            valid_data = X_np[mask].reshape(-1, 1)
            # 将转换后的值放回原位置
            X_transformed[mask] = self.scaler_.transform(valid_data).flatten()
            
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """
        返回转换后的特征名称。
        因为此缩放器不改变特征的数量或名称，所以直接返回输入的特征名。
        """
        if input_features is None:
            # 如果scikit-learn版本较旧，可能需要处理这种情况
            raise ValueError("input_features is required for get_feature_names_out.")
        return np.asarray(input_features, dtype=object)

def normalize_features(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    根据指定的接口，使用统一的预处理器对训练集和验证集进行标准化。
    MODIFIED: Now correctly handles all engineered features.

    - 对IMU, 温度, 人口统计学特征以及所有新工程化的特征应用Z-score标准化。
    - 对ToF特征应用自定义的Min-Max规范化（-1保持不变）。
    - 返回标准化后的DataFrame以及一个单一、已拟合的scaler对象。

    参数:
        X_train (pd.DataFrame): 训练数据集。
        X_val (pd.DataFrame): 验证数据集。

    返回:
        tuple: 包含三个元素的元组:
            - X_train_normalized (pd.DataFrame): 标准化后的训练数据。
            - X_val_normalized (pd.DataFrame): 标准化后的验证数据。
            - scaler (ColumnTransformer): 一个已在X_train上拟合好的、统一的预处理器对象。
    """
    # 1. 识别需要Z-score标准化的特征
    # *** MODIFIED: Added 'linear_' and 'angular_' prefixes to the list ***
    zscore_prefixes = ['acc_', 'rot_', 'thm_', 'linear_', 'angular_']
    zscore_cols = [col for col in X_train.columns if any(col.startswith(p) for p in zscore_prefixes)]
    
    # 除了分类特征以外的demographic
    demographic_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    existing_demographic_cols = [col for col in demographic_cols if col in X_train.columns]
    
    # Add demographic columns if they are not already included
    for col in existing_demographic_cols:
        if col not in zscore_cols:
            zscore_cols.append(col)

    # 2. 识别需要自定义ToF缩放的特征
    tof_cols = [col for col in X_train.columns if col.startswith('tof_')]
    
    # 3. 创建单一的、统一的预处理器 (ColumnTransformer)
    transformer_list = []
    
    # Ensure zscore_cols are present in the dataframe before adding the transformer
    final_zscore_cols = [c for c in zscore_cols if c in X_train.columns]
    if final_zscore_cols:
        transformer_list.append(('zscore', StandardScaler(), final_zscore_cols))
        
    final_tof_cols = [c for c in tof_cols if c in X_train.columns]
    if final_tof_cols:
        transformer_list.append(('tof_scaler', TofScaler(), final_tof_cols))

    # remainder='passthrough' 会保留所有未指定的列 (e.g., sex_M which is one-hot encoded)
    scaler = ColumnTransformer(
        transformers=transformer_list,
        remainder='passthrough',
        verbose_feature_names_out=False # 输出的列名更简洁
    )

    # 4. 在训练数据上拟合scaler
    print(f"\nNormalizing features...")
    print(f"Applying Z-score to {len(final_zscore_cols)} columns.")
    print(f"Applying custom ToF scale to {len(final_tof_cols)} columns.")
    
    scaler.fit(X_train)

    # 5. 使用已拟合的scaler转换训练集和验证集
    X_train_transformed_np = scaler.transform(X_train)
    X_val_transformed_np = scaler.transform(X_val)

    # 6. 获取新顺序的列名并重建DataFrame
    feature_names = scaler.get_feature_names_out()
    
    X_train_normalized = pd.DataFrame(X_train_transformed_np, index=X_train.index, columns=feature_names)
    X_val_normalized = pd.DataFrame(X_val_transformed_np, index=X_val.index, columns=feature_names)
    
    # 7. 返回符合接口要求的结果
    return X_train_normalized, X_val_normalized, scaler

# 未维护，不能直接使用
def prepare_data_single_split(variant: str = "full"):
    """
    Prepare data with single train/val split (for quick testing)
    """
    sequences, labels, sequence_ids, subjects, label_encoder, feature_cols = load_and_preprocess_data(variant)
    
    seq_lengths = [len(seq) for seq in sequences]
    max_length = 100  # Fixed length - keeps the critical END part of gestures
    print(f"Using max_length: {max_length} (fixed length, keeps end of gestures)")
    print(f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")
    
    X = pad_sequences(sequences, max_length)
    y = np.array(labels)
    subjects = np.array(subjects)
    
    # Single split using first fold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(X, y, groups=subjects))
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    X_train, X_val, scaler = normalize_features(X_train, X_val)
    
    # Create outputs directory (repo_root/outputs)
    weights_dir = _get_weights_dir()
    le_path = os.path.join(weights_dir, f'label_encoder_{variant}.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return X_train, X_val, y_train, y_val, label_encoder

def prepare_data_kfold(show_stratification=False, variant: str = "full"):
    """
    Prepare data for 5-fold cross-validation in the correct order:
    Split -> Normalize -> Pad
    """
    # 1. Load data to get a single, clean DataFrame
    all_data_df, label_encoder, feature_cols = load_and_preprocess_data(variant)

    weights_dir = _get_weights_dir()
    le_path = os.path.join(weights_dir, f'label_encoder_{variant}.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # 2. Create a map for stratification (one row per sequence)
    labels_map_df = all_data_df[['sequence_id', 'gesture_encoded', 'subject']].drop_duplicates().reset_index(drop=True)
    y = labels_map_df['gesture_encoded'].values
    subjects = labels_map_df['subject'].values
    unique_sequence_ids = labels_map_df['sequence_id'].values

    # 3. K-Fold setup and splitting (on sequence_id)
    print("\nPreparing 5-fold cross-validation splits...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_data = []
    
    for fold_idx, (train_seq_indices, val_seq_indices) in enumerate(sgkf.split(np.zeros(len(unique_sequence_ids)), y, groups=subjects)):
        
        train_sids = unique_sequence_ids[train_seq_indices]
        val_sids = unique_sequence_ids[val_seq_indices]

        # 4. Create train/val DataFrames based on split sequence IDs
        X_train_fold_df = all_data_df[all_data_df['sequence_id'].isin(train_sids)]
        X_val_fold_df = all_data_df[all_data_df['sequence_id'].isin(val_sids)]
        
        y_train_fold = labels_map_df[labels_map_df['sequence_id'].isin(train_sids)]['gesture_encoded'].values
        y_val_fold = labels_map_df[labels_map_df['sequence_id'].isin(val_sids)]['gesture_encoded'].values
        
        # Get subjects for this fold
        train_subjects_fold = subjects[train_seq_indices]
        val_subjects_fold = subjects[val_seq_indices]

        # 5. Normalize features using the DataFrames (will not error)
        X_train_norm_df, X_val_norm_df, scaler_fold = normalize_features(
            X_train_fold_df[feature_cols], 
            X_val_fold_df[feature_cols]
        )

        # 6. Group data into sequences and pad AFTER normalization
        # We need to align the normalized data with its original sequence_id
        # A safe way is to add sequence_id back to the normalized df before grouping
        X_train_norm_df['sequence_id'] = X_train_fold_df['sequence_id']
        X_val_norm_df['sequence_id'] = X_val_fold_df['sequence_id']
        
        train_sequences_list = [group[feature_cols].values for _, group in X_train_norm_df.groupby('sequence_id')]
        val_sequences_list = [group[feature_cols].values for _, group in X_val_norm_df.groupby('sequence_id')]

        # Padding
        max_length = 100
        X_train_padded = pad_sequences(train_sequences_list, max_length=max_length)
        X_val_padded = pad_sequences(val_sequences_list, max_length=max_length)

        print(f"\n--- Fold {fold_idx + 1} ---")
        print(f"Train shape (padded): {X_train_padded.shape}")
        print(f"Val shape (padded): {X_val_padded.shape}")
        
        # 7. Store fold data
        fold_data.append({
            'fold_idx': fold_idx,
            'X_train': X_train_padded,
            'X_val': X_val_padded,
            'y_train': y_train_fold,
            'y_val': y_val_fold,
            'scaler': scaler_fold,
            'train_subjects': train_subjects_fold,
            'val_subjects': val_subjects_fold,
            'train_idx': train_seq_indices,
            'val_idx': val_seq_indices,
        })
        
    print(f"\n✅ Prepared {len(fold_data)} folds for cross-validation")
    return fold_data, label_encoder, y, unique_sequence_ids


# Backward compatibility
def prepare_data(variant: str = "full"):
    """
    Backward-compatibility wrapper: returns single split for requested variant
    """
    return prepare_data_single_split(variant)

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, label_encoder = prepare_data()
    print("Data preprocessing completed!") 