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
    print(f"Train data shape: {train_df.shape}")
    
    # --- Label encoding (as in your original code) ---
    label_encoder = LabelEncoder()
    train_df['gesture_encoded'] = label_encoder.fit_transform(train_df['gesture'])
    
    # --- Feature column identification (as in your original code) ---
    metadata_cols = ['row_id', 'sequence_id', 'sequence_type', 'sequence_counter',
                     'subject', 'orientation', 'behavior', 'phase', 'gesture', 'gesture_encoded']
    feature_cols = [col for col in train_df.columns if col not in metadata_cols]
    if variant == "imu":
        feature_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
    print(f"Variant: {variant}. Feature columns after filtering: {len(feature_cols)}")

    # --- Missing value handling ---
    print("\nHandling missing values (interpolating per sequence)...")
    if variant != "imu":
        # Restored call to your TOF interpolation function
        train_df = interpolate_tof(train_df)

    # Ensure chronological order so interpolation is meaningful
    train_df = train_df.sort_values(['sequence_id', 'sequence_counter'])

    # Linear interpolation forward & backward **within each sequence**
    train_df[feature_cols] = (
        train_df.groupby('sequence_id')[feature_cols]
                .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )
    # Fallback: column-wise median for any value still NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median())
    # Final safety net: replace any remaining NaN with 0
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    
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

    - 对IMU, 温度和人口统计学特征应用Z-score标准化。
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
    # 加速度，旋转和温度特征
    zscore_cols = [col for col in X_train.columns if 
                   col.startswith('acc_') or 
                   col.startswith('rot_') or 
                   col.startswith('thm_')]
    # 除了分类特征以外的demographic
    demographic_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    existing_demographic_cols = [col for col in demographic_cols if col in X_train.columns]
    zscore_cols.extend(existing_demographic_cols)

    # 2. 识别需要自定义ToF缩放的特征
    tof_cols = [col for col in X_train.columns if col.startswith('tof_')]
    
    # 3. 创建单一的、统一的预处理器 (ColumnTransformer)
    transformer_list = []
    if zscore_cols:
        transformer_list.append(('zscore', StandardScaler(), zscore_cols))
    if tof_cols:
        transformer_list.append(('tof_scaler', TofScaler(), tof_cols))

    # remainder='passthrough' 会保留所有未指定的列
    scaler = ColumnTransformer(
        transformers=transformer_list,
        remainder='passthrough',
        verbose_feature_names_out=False # 输出的列名更简洁
    )

    # 4. 在训练数据上拟合scaler
    scaler.fit(X_train)

    # 5. 使用已拟合的scaler转换训练集和验证集
    # .transform的输出是numpy数组，我们需要将其转换回DataFrame
    X_train_transformed_np = scaler.transform(X_train)
    X_val_transformed_np = scaler.transform(X_val)

    # 6. 获取新顺序的列名并重建DataFrame
    # scaler.get_feature_names_out() 会返回转换后正确的列名和顺序
    try:
        feature_names = scaler.get_feature_names_out()
    except Exception:
        print("⚠️ Warning: get_feature_names_out() failed. Using original column names.")
        # 兼容旧版本scikit-learn
        feature_names = X_train.columns # 这是一个简化的回退，顺序可能不完全匹配
    
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