import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import pickle
import os
from .tof_utils import interpolate_tof

# Static feature columns (shared between training and inference)
STATIC_FEATURE_COLS = [
    'adult_child', 'age', 'sex', 'handedness', 'height_cm', 
    'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
]

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

    # --- MODIFIED: Define and separate static vs. sequential features ---
    static_feature_cols = STATIC_FEATURE_COLS
    
    # Get feature columns (exclude metadata and target)
    metadata_cols = ['row_id',  'sequence_id', 'sequence_type', 'sequence_counter',
                     'subject', 'orientation', 'behavior', 'phase', 'gesture', 'gesture_encoded']
                     
    seq_feature_cols = [col for col in train_df.columns if col not in metadata_cols and col not in static_feature_cols]

    # Optionally drop THM / TOF sensors for IMU-only variant
    if variant == "imu":
        seq_feature_cols = [c for c in seq_feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]

    print(f"Variant: {variant}. Feature columns after filtering: {len(seq_feature_cols)}")
    print(f"\nFeature columns: {len(seq_feature_cols)}")
    print(f"Sample features: {seq_feature_cols[:10]}")
    
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
    train_df[seq_feature_cols] = (
        train_df.groupby('sequence_id')[seq_feature_cols]
                .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )

    # Fallback: column-wise median for any value still NaN (e.g., entire column NaN)
    train_df[seq_feature_cols] = train_df[seq_feature_cols].fillna(train_df[seq_feature_cols].median())

    # Final safety net: replace any remaining NaN with 0
    train_df[seq_feature_cols] = train_df[seq_feature_cols].fillna(0)
    
    # Group by sequence to create samples
    print("Grouping by sequence...")
    sequences = []
    labels = []
    sequence_ids = []
    subjects = []  # Track subject for each sequence
    static_features_list = [] # <-- 新增
    
    for seq_id, group in train_df.groupby('sequence_id'):
        # Sort by sequence_counter to maintain temporal order
        group = group.sort_values('sequence_counter')
        
        # Extract features and label
        sequence_features = group[seq_feature_cols].values
        static_features_list.append(group[static_feature_cols].iloc[0].values) # <-- 新增
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

    # 将静态特征转换为numpy数组
    static_features_arr = np.array(static_features_list)
    
    return sequences, static_features_arr, labels, sequence_ids, subjects, label_encoder, seq_feature_cols

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

def normalize_sequential_features(X_train, X_val):
    """Normalizes the 3D sequential data."""
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train_reshaped)
    
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_val_normalized = scaler.transform(X_val_reshaped)
    
    X_train_normalized = X_train_normalized.reshape(n_samples, n_timesteps, n_features)
    X_val_normalized = X_val_normalized.reshape(X_val.shape[0], n_timesteps, n_features)
    
    return X_train_normalized, X_val_normalized, scaler

# --- NEW: Separate normalization function for static features ---
def normalize_static_features(X_train_static, X_val_static):
    """Normalizes the 2D static data."""
    scaler = StandardScaler()
    X_train_static_norm = scaler.fit_transform(X_train_static)
    X_val_static_norm = scaler.transform(X_val_static)
    return X_train_static_norm, X_val_static_norm, scaler



def prepare_data_kfold(show_stratification=False, variant: str = "full"):
    """
    Prepare data for 5-fold cross-validation
    Returns all folds for training 5 different models
    
    Args:
        show_stratification (bool): If True, show detailed stratification analysis
    """
    # Load and preprocess
    # --- MODIFIED: Unpack new static features array ---
    sequences, X_static, labels, sequence_ids, subjects, label_encoder, _ = load_and_preprocess_data(variant)
    
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
        # --- MODIFIED: Split both sequential and static data ---
        X_train_seq_fold, X_val_seq_fold = X[train_idx], X[val_idx]
        X_train_static_fold, X_val_static_fold = X_static[train_idx], X_static[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # --- MODIFIED: Normalize both data types separately ---
        X_train_seq_norm, X_val_seq_norm, seq_scaler = normalize_sequential_features(X_train_seq_fold, X_val_seq_fold)
        X_train_static_norm, X_val_static_norm, static_scaler = normalize_static_features(X_train_static_fold, X_val_static_fold)
        
        print(f"Train shapes: Seq={X_train_seq_norm.shape}, Static={X_train_static_norm.shape}")
        print(f"Val shapes:   Seq={X_val_seq_norm.shape}, Static={X_val_static_norm.shape}")
        
        # Verify no subject overlap
        train_subjects_fold = subjects[train_idx]
        val_subjects_fold = subjects[val_idx]
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
        
        # Store fold data (include original indices for OOF reconstruction)
        # --- MODIFIED: Store all data components for the fold ---
        fold_data.append({
            'X_train_seq': X_train_seq_norm,
            'X_train_static': X_train_static_norm,
            'y_train': y_train_fold,
            'X_val_seq': X_val_seq_norm,
            'X_val_static': X_val_static_norm,
            'y_val': y_val_fold,
            'seq_scaler': seq_scaler,
            'static_scaler': static_scaler,
            'val_idx': val_idx,
            'train_subjects': subjects[train_idx],
            'val_subjects': subjects[val_idx],
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