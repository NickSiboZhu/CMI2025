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
    Load training data and demographics, preprocess for 1D CNN.

    Args:
        variant (str): Sensor variant to build. "full" keeps all features; 
                        "imu" drops thermopile (thm_*) and time-of-flight (tof__*) columns so
                        that a model trained on IMU-only data can generalise to test sequences
                        where those sensors are absent.
    """
    print("Loading data...")
    
    # Get the directory of this file and construct absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir  # primary location (cmi-submission/data)

    # Fallback: look in development/data/ if CSVs not found here (training environment)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))  # repo root
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
    
    train_df = pd.read_csv(train_path)
    demographics_df = pd.read_csv(demographics_path)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Demographics shape: {demographics_df.shape}")
    
    # Merge with demographics
    train_df = train_df.merge(demographics_df, on='subject', how='left')
    print(f"After merge shape: {train_df.shape}")
    
    # Check unique gestures
    print(f"\nUnique gestures: {train_df['gesture'].nunique()}")
    print("Gesture distribution:")
    print(train_df['gesture'].value_counts())
    
    # Encode gesture labels
    label_encoder = LabelEncoder()
    train_df['gesture_encoded'] = label_encoder.fit_transform(train_df['gesture'])
    
    # Get feature columns (exclude metadata and target)
    metadata_cols = ['row_id',  'sequence_id', 'sequence_type', 'sequence_counter',
                     'subject', 'orientation', 'behavior', 'phase', 'gesture', 'gesture_encoded']
    feature_cols = [col for col in train_df.columns if col not in metadata_cols]

    # Optionally drop THM / TOF sensors for IMU-only variant
    if variant == "imu":
        feature_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]

    print(f"Variant: {variant}. Feature columns after filtering: {len(feature_cols)}")
    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")
    
    # Handle missing values – interpolate inside each sequence along the time axis
    print("\nHandling missing values (interpolating per sequence)...")

    # Convert sentinel -1.0 to NaN first
    # train_df[feature_cols] = train_df[feature_cols].replace(-1.0, np.nan)

    # ------------------------------------------------------------
    # Spatial interpolation for each TOF sensor (8×8 grid)
    # ------------------------------------------------------------
    # Skip TOF interpolation entirely for IMU-only variant to save time
    if variant != "imu":
        train_df = interpolate_tof(train_df)

    # Ensure chronological order so interpolation is meaningful
    train_df = train_df.sort_values(['sequence_id', 'sequence_counter'])

    # Linear interpolation forward & backward **within each sequence**
    train_df[feature_cols] = (
        train_df.groupby('sequence_id')[feature_cols]
                .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )

    # Fallback: column-wise median for any value still NaN (e.g., entire column NaN)
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median())

    # Final safety net: replace any remaining NaN with 0
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    
    # Group by sequence to create samples
    print("Grouping by sequence...")
    sequences = []
    labels = []
    sequence_ids = []
    subjects = []  # Track subject for each sequence
    
    for seq_id, group in train_df.groupby('sequence_id'):
        # Sort by sequence_counter to maintain temporal order
        group = group.sort_values('sequence_counter')
        
        # Extract features and label
        sequence_features = group[feature_cols].values
        gesture_label = group['gesture_encoded'].iloc[0]  # Same for all steps in sequence
        subject_id = group['subject'].iloc[0]  # Same for all steps in sequence
        
        sequences.append(sequence_features)
        labels.append(gesture_label)
        sequence_ids.append(seq_id)
        subjects.append(subject_id)
    
    print(f"Created {len(sequences)} sequences")
    print(f"Number of unique subjects: {len(set(subjects))}")
    
    # Check sequence lengths
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Sequence length stats:")
    print(f"  Min: {min(seq_lengths)}")
    print(f"  Max: {max(seq_lengths)}")
    print(f"  Mean: {np.mean(seq_lengths):.1f}")
    print(f"  Median: {np.median(seq_lengths)}")
    
    return sequences, labels, sequence_ids, subjects, label_encoder, feature_cols

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
        # 兼容旧版本scikit-learn
        feature_names = X_train.columns # 这是一个简化的回退，顺序可能不完全匹配
    
    X_train_normalized = pd.DataFrame(X_train_transformed_np, index=X_train.index, columns=feature_names)
    X_val_normalized = pd.DataFrame(X_val_transformed_np, index=X_val.index, columns=feature_names)
    
    # 7. 返回符合接口要求的结果
    return X_train_normalized, X_val_normalized, scaler

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
    Prepare data for 5-fold cross-validation
    Returns all folds for training 5 different models
    
    Args:
        show_stratification (bool): If True, show detailed stratification analysis
    """
    # Load and preprocess
    sequences, labels, sequence_ids, subjects, label_encoder, feature_cols = load_and_preprocess_data(variant)
    
    # Pad sequences (use fixed length of 100, keeping the critical END part)
    seq_lengths = [len(seq) for seq in sequences]
    max_length = 100  # Fixed length - keeps the critical END part of gestures
    print(f"Using max_length: {max_length} (fixed length, keeps end of gestures)")
    print(f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")
    
    X = pad_sequences(sequences, max_length)
    y = np.array(labels)
    subjects = np.array(subjects)
    
    print(f"Final data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Number of unique subjects: {len(np.unique(subjects))}")
    
    # 5-Fold Cross-Validation Setup
    print("\nPreparing 5-fold cross-validation splits...")
    print("This will create 5 different train/val combinations for 5 models")
    
    if show_stratification:
        print(f"\n✨ Stratification happens here:")
        print(f"   sgkf.split(X, y, groups=subjects)")
        print(f"              ↑  ↑      ↑")
        print(f"              |  |      └─ Groups (subjects)")  
        print(f"              |  └─ Stratified (gesture labels)")
        print(f"              └─ Features")
    
    # ✨ This is where both GROUP and STRATIFIED constraints are applied
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Prepare all 5 folds
    fold_data = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=subjects)):
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # Split data
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        train_subjects_fold = subjects[train_idx]
        val_subjects_fold = subjects[val_idx]
        
        # Verify no subject overlap
        train_subjects_set = set(train_subjects_fold)
        val_subjects_set = set(val_subjects_fold)
        overlap = train_subjects_set.intersection(val_subjects_set)
        
        print(f"Train subjects: {len(train_subjects_set)}")
        print(f"Val subjects: {len(val_subjects_set)}")
        print(f"Subject overlap: {len(overlap)} (should be 0)")
        
        if len(overlap) > 0:
            print(f"WARNING: Overlapping subjects found: {overlap}")
        else:
            print("✓ No subject overlap")
        
        # Check class distribution
        train_dist = np.bincount(y_train_fold)
        val_dist = np.bincount(y_val_fold)
        
        if show_stratification:
            # Show detailed stratification analysis
            train_pct = train_dist / len(y_train_fold) * 100
            val_pct = val_dist / len(y_val_fold) * 100
            print(f"Stratification check:")
            for i, class_name in enumerate(label_encoder.classes_):
                if i < len(train_pct) and i < len(val_pct):
                    print(f"  {class_name}: Train {train_pct[i]:.1f}%, Val {val_pct[i]:.1f}%")
        else:
            # Simple output
            print(f"Train class distribution: {train_dist}")
            print(f"Val class distribution: {val_dist}")
        
        # Normalize features for this fold
        X_train_norm, X_val_norm, scaler_fold = normalize_features(X_train_fold, X_val_fold)
    
        print(f"Train shape: {X_train_norm.shape}")
        print(f"Val shape: {X_val_norm.shape}")
        
        # Store fold data (include original indices for OOF reconstruction)
        fold_data.append({
            'fold_idx': fold_idx,
            'X_train': X_train_norm,
            'X_val': X_val_norm,
            'y_train': y_train_fold,
            'y_val': y_val_fold,
            'scaler': scaler_fold,
            'train_subjects': train_subjects_fold,
            'val_subjects': val_subjects_fold,
            'train_idx': train_idx,
            'val_idx': val_idx,
        })
    
    # Save preprocessing objects (using fold 0's scaler as default)
    weights_dir = _get_weights_dir()
    le_path = os.path.join(weights_dir, f'label_encoder_{variant}.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n✅ Prepared {len(fold_data)} folds for cross-validation")
    print("Each fold will train a separate model")
    
    # Return additional arrays for OOF handling
    return fold_data, label_encoder, y, sequence_ids

# Backward compatibility
def prepare_data(variant: str = "full"):
    """
    Backward-compatibility wrapper: returns single split for requested variant
    """
    return prepare_data_single_split(variant)

if __name__ == "__main__":
    X_train, X_val, y_train, y_val, label_encoder = prepare_data()
    print("Data preprocessing completed!") 