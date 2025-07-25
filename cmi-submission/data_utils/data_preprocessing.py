import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
import pickle
import os
from .tof_utils import interpolate_tof
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from scipy.spatial.transform import Rotation as R
import warnings

# --- START: Advanced Feature Engineering (from your branch) ---

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

# Static feature columns (shared between training and inference)
STATIC_FEATURE_COLS = [
    'adult_child', 'age', 'sex', 'handedness', 'height_cm', 
    'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
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
        print("\nFiltering out sequences with no valid ToF data...")
        tof_cols = [c for c in train_df.columns if c.startswith('tof_')]
        if tof_cols:
            sids_with_valid_tof = (
                train_df.groupby('sequence_id')[tof_cols]
                        .apply(lambda group_df: ((group_df != -1) & (group_df.notna())).any().any())
            )
            full_quality_sids = sids_with_valid_tof[sids_with_valid_tof].index
            original_seq_count = train_df['sequence_id'].nunique()
            train_df = train_df[train_df['sequence_id'].isin(full_quality_sids)]
            print(f"  {original_seq_count} total sequences found.")
            print(f"  {len(full_quality_sids)} sequences have at least one valid ToF reading and will be used.")
            print(f"  Filtered data shape: {train_df.shape}")
    
    # --- Label encoding ---
    label_encoder = LabelEncoder()
    train_df['gesture_encoded'] = label_encoder.fit_transform(train_df['gesture'])

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

def pad_sequences(sequences, max_length=None):
    """
    Pad sequences to same length, keeping the END of each sequence (most critical part)
    
    Strategy:
    - Keep the LAST max_length timesteps of each sequence
    - Pad with zeros at the BEGINNING if sequence is shorter
    - Truncate from the BEGINNING if sequence is longer
    
    This preserves the end of gestures which contains the most discriminative information.
    """
    if not sequences:
        return np.array([])
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

def normalize_features(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    根据指定的接口，使用统一的预处理器对训练集和验证集进行标准化。
    - 对IMU, 温度, 人口统计学特征以及所有新工程化的特征应用Z-score标准化。
    - 对ToF特征应用自定义的Min-Max规范化（-1保持不变）。
    - 返回标准化后的DataFrame以及一个单一、已拟合的scaler对象。
    """
    # 1. 识别需要Z-score标准化的特征
    zscore_prefixes = ['acc_', 'rot_', 'thm_', 'linear_', 'angular_']
    zscore_cols = [col for col in X_train.columns if any(col.startswith(p) for p in zscore_prefixes)]
    
    demographic_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    existing_demographic_cols = [col for col in demographic_cols if col in X_train.columns]
    
    for col in existing_demographic_cols:
        if col not in zscore_cols:
            zscore_cols.append(col)

    # 2. 识别需要自定义ToF缩放的特征
    tof_cols = [col for col in X_train.columns if col.startswith('tof_')]
    
    # 3. 创建单一的、统一的预处理器 (ColumnTransformer)
    transformer_list = []
    
    final_zscore_cols = [c for c in zscore_cols if c in X_train.columns]
    if final_zscore_cols:
        transformer_list.append(('zscore', StandardScaler(), final_zscore_cols))
        
    final_tof_cols = [c for c in tof_cols if c in X_train.columns]
    if final_tof_cols:
        transformer_list.append(('tof_scaler', TofScaler(), final_tof_cols))

    scaler = ColumnTransformer(
        transformers=transformer_list,
        remainder='passthrough',
        verbose_feature_names_out=False
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
    
    return X_train_normalized, X_val_normalized, scaler


def prepare_data_kfold_multimodal(show_stratification: bool=False, variant: str="full", n_splits: int=5):
    """
    Prepare K-Fold data for multimodal architecture *with THM separated by default*.
    The pipeline: Split → Normalize → Pad, and returns numpy arrays ready for the
    `MultimodalDataset` (IMU + THM + TOF + STATIC).
    """
    # 1. Load and preprocess full data
    all_data_df, label_encoder, all_feature_cols = load_and_preprocess_data(variant)

    # 2. Identify columns for each modality dynamically
    static_cols = [c for c in STATIC_FEATURE_COLS if c in all_data_df.columns]
    
    # Dynamically detect THM and TOF columns from actual data
    thm_cols, tof_cols = generate_feature_columns(all_data_df.columns)
    thm_cols = [c for c in thm_cols if c in all_feature_cols]  # Only keep engineered features
    tof_cols = [c for c in tof_cols if c in all_feature_cols]
    
    imu_cols = [c for c in all_feature_cols if c not in static_cols and c not in tof_cols and c not in thm_cols]

    # 3. Prepare stratified group k-fold
    labels_map_df    = all_data_df[["sequence_id", "gesture_encoded", "subject"]].drop_duplicates().reset_index(drop=True)
    y                = labels_map_df["gesture_encoded"].values
    subjects         = labels_map_df["subject"].values
    unique_seq_ids   = labels_map_df["sequence_id"].values

    sgkf        = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_data   = []

    print(f"\nPreparing {n_splits}-fold splits with THM branch separated …")

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(unique_seq_ids)), y, groups=subjects)):
        print(f"\n--- Fold {fold_idx+1}/{n_splits} ---")
        train_sids = unique_seq_ids[train_idx]
        val_sids   = unique_seq_ids[val_idx]

        train_df = all_data_df[all_data_df["sequence_id"].isin(train_sids)].copy()
        val_df   = all_data_df[all_data_df["sequence_id"].isin(val_sids)].copy()

        y_train = labels_map_df[labels_map_df["sequence_id"].isin(train_sids)]["gesture_encoded"].values
        y_val   = labels_map_df[labels_map_df["sequence_id"].isin(val_sids)]["gesture_encoded"].values

        # 4. Normalization (ColumnTransformer) BEFORE padding
        X_train_norm, X_val_norm, scaler_fold = normalize_features(train_df[all_feature_cols], val_df[all_feature_cols])
        X_train_norm["sequence_id"] = train_df["sequence_id"]
        X_val_norm["sequence_id"]   = val_df["sequence_id"]

        # 5. Split into modalities + pad to fixed seq_len
        max_len = 100

        def _split_pad(df_norm, sids):
            static_list, imu_list, thm_list, tof_list = [], [], [], []
            for sid in sids:
                grp = df_norm[df_norm["sequence_id"] == sid]
                static_list.append(grp[static_cols].iloc[0].values)
                imu_list.append(grp[imu_cols].values)
                thm_list.append(grp[thm_cols].values)
                tof_list.append(grp[tof_cols].values)
            X_static = np.array(static_list)
            X_imu    = pad_sequences(imu_list, max_length=max_len)
            X_thm    = pad_sequences(thm_list, max_length=max_len)
            X_tof    = pad_sequences(tof_list, max_length=max_len)
            return X_imu, X_thm, X_tof, X_static

        X_train_imu, X_train_thm, X_train_tof, X_train_static = _split_pad(X_train_norm, train_sids)
        X_val_imu,   X_val_thm,   X_val_tof,   X_val_static   = _split_pad(X_val_norm,   val_sids)

        print(f"  Train samples: {len(train_sids)} | Val samples: {len(val_sids)}")
        print(f"  IMU shape: {X_train_imu.shape}, THM shape: {X_train_thm.shape}, TOF shape: {X_train_tof.shape}")

        fold_data.append({
            "fold_idx": fold_idx,
            "X_train_imu": X_train_imu,
            "X_train_thm": X_train_thm,
            "X_train_tof": X_train_tof,
            "X_train_static": X_train_static,
            "y_train": y_train,
            "X_val_imu": X_val_imu,
            "X_val_thm": X_val_thm,
            "X_val_tof": X_val_tof,
            "X_val_static": X_val_static,
            "y_val": y_val,
            "scaler": scaler_fold,
            "val_idx": val_idx,
        })

    print("\n✅ K-fold preparation done. Returning data …")
    return fold_data, label_encoder, y, unique_seq_ids


if __name__ == "__main__":
    print("Running data preprocessing script...")
    # 可选择 "full" 或 "imu" 变体进行测试
    VARIANT = "full" 
    
    fold_data, le, y_all, sids = prepare_data_kfold_multimodal(variant=VARIANT)
    
    print("\n--- Example: Data from Fold 1 ---")
    first_fold = fold_data[0]
    print(f"X_train_non_tof shape: {first_fold['X_train_non_tof'].shape}")
    print(f"X_train_tof shape: {first_fold['X_train_tof'].shape}")
    print(f"X_train_static shape: {first_fold['X_train_static'].shape}")
    print(f"y_train shape: {first_fold['y_train'].shape}")
    print(f"Scaler type: {type(first_fold['scaler'])}")
    
    print("\nData preprocessing script finished successfully!")