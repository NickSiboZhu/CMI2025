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

# TOF feature columns (5 sensors × 64 pixels each)
TOF_FEATURE_COLS = [f"tof_{sensor}_v{pixel}" for sensor in range(1, 6) for pixel in range(64)]

def separate_tof_features(X_seq, seq_feature_cols):
    """
    Separate TOF features from other sequential features.
    
    Args:
        X_seq: Sequential data array of shape (n_samples, seq_len, n_features)
        seq_feature_cols: List of feature column names
    
    Returns:
        X_non_tof: Non-TOF sequential features
        X_tof: TOF features reshaped for 2D CNN processing
        non_tof_cols: List of non-TOF feature column names
        tof_cols: List of TOF feature column names
    """
    # Identify TOF and non-TOF columns
    tof_cols = [col for col in seq_feature_cols if col.startswith('tof_')]
    non_tof_cols = [col for col in seq_feature_cols if not col.startswith('tof_')]
    
    # Get indices for TOF and non-TOF features
    tof_indices = [i for i, col in enumerate(seq_feature_cols) if col.startswith('tof_')]
    non_tof_indices = [i for i, col in enumerate(seq_feature_cols) if not col.startswith('tof_')]
    
    # Separate the features
    X_non_tof = X_seq[:, :, non_tof_indices] if non_tof_indices else np.empty((X_seq.shape[0], X_seq.shape[1], 0))
    X_tof = X_seq[:, :, tof_indices] if tof_indices else np.empty((X_seq.shape[0], X_seq.shape[1], 0))
    
    print(f"Separated features:")
    print(f"  Non-TOF features: {len(non_tof_cols)} columns")
    print(f"  TOF features: {len(tof_cols)} columns")
    print(f"  Non-TOF shape: {X_non_tof.shape}")
    print(f"  TOF shape: {X_tof.shape}")
    
    return X_non_tof, X_tof, non_tof_cols, tof_cols

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

def normalize_tof_features(X_train_tof, X_val_tof):
    """
    Normalize TOF features preserving 8x8 spatial structure.
    Each 8x8 grid (64 pixels per sensor) is normalized independently.
    
    Args:
        X_train_tof: (n_samples, seq_len, 320) -> 5 sensors × 64 pixels each
        X_val_tof: (n_samples, seq_len, 320)
    
    Returns:
        Normalized arrays and per-sensor scalers
    """
    if X_train_tof.shape[2] == 0:
        return X_train_tof, X_val_tof, None
    
    print("Normalizing TOF features with spatial awareness (per 8x8 grid)...")
    
    n_samples, seq_len, n_tof_features = X_train_tof.shape
    n_sensors = n_tof_features // 64  # Should be 5 sensors
    
    assert n_tof_features == n_sensors * 64, f"Expected {n_sensors * 64} TOF features, got {n_tof_features}"
    
    # Reshape to separate sensors: (n_samples, seq_len, n_sensors, 8, 8)
    X_train_spatial = X_train_tof.reshape(n_samples, seq_len, n_sensors, 8, 8)
    X_val_spatial = X_val_tof.reshape(X_val_tof.shape[0], seq_len, n_sensors, 8, 8)
    
    # Store per-sensor scalers
    sensor_scalers = []
    X_train_normalized = np.zeros_like(X_train_spatial)
    X_val_normalized = np.zeros_like(X_val_spatial)
    
    for sensor_idx in range(n_sensors):
        # Extract one sensor's data: (n_samples, seq_len, 8, 8)
        train_sensor = X_train_spatial[:, :, sensor_idx, :, :]
        val_sensor = X_val_spatial[:, :, sensor_idx, :, :]
        
        # Reshape for normalization: (n_samples * seq_len, 64)
        train_flat = train_sensor.reshape(-1, 64)
        val_flat = val_sensor.reshape(-1, 64)
        
        # Normalize per-grid (each row is one 8x8 grid)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_norm_flat = scaler.fit_transform(train_flat)
        val_norm_flat = scaler.transform(val_flat)
        
        # Reshape back to spatial: (n_samples, seq_len, 8, 8)
        X_train_normalized[:, :, sensor_idx, :, :] = train_norm_flat.reshape(n_samples, seq_len, 8, 8)
        X_val_normalized[:, :, sensor_idx, :, :] = val_norm_flat.reshape(X_val_tof.shape[0], seq_len, 8, 8)
        
        sensor_scalers.append(scaler)
    
    # Flatten back to original shape: (n_samples, seq_len, 320)
    X_train_final = X_train_normalized.reshape(n_samples, seq_len, n_tof_features)
    X_val_final = X_val_normalized.reshape(X_val_tof.shape[0], seq_len, n_tof_features)
    
    print(f"✅ Normalized {n_sensors} TOF sensors independently (preserving 8x8 spatial structure)")
    
    return X_train_final, X_val_final, sensor_scalers


def prepare_data_kfold_multimodal(show_stratification=False, variant: str = "full"):
    """
    Prepare data for 5-fold cross-validation with multimodal architecture.
    Separates TOF features for 2D CNN processing.
    
    Args:
        show_stratification (bool): If True, show detailed stratification analysis
        variant (str): "full" or "imu" variant
    
    Returns:
        fold_data: List of fold dictionaries with separated features
        label_encoder: Fitted label encoder
        y: All labels
        sequence_ids: All sequence IDs
    """
    # Load and preprocess
    sequences, X_static, labels, sequence_ids, subjects, label_encoder, seq_feature_cols = load_and_preprocess_data(variant)
    
    # Pad sequences
    seq_lengths = [len(seq) for seq in sequences]
    max_length = 100
    print(f"Using max_length: {max_length} (fixed length, keeps end of gestures)")
    print(f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")
    
    X_seq = pad_sequences(sequences, max_length)
    y = np.array(labels)
    subjects = np.array(subjects)
    
    # Separate TOF and non-TOF features
    X_non_tof, X_tof, non_tof_cols, tof_cols = separate_tof_features(X_seq, seq_feature_cols)
    
    print(f"Final data shapes:")
    print(f"  Non-TOF sequential: {X_non_tof.shape}")
    print(f"  TOF sequential: {X_tof.shape}")
    print(f"  Static: {X_static.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Number of classes: {len(np.unique(y))}")
    print(f"  Number of unique subjects: {len(np.unique(subjects))}")
    
    # 5-Fold Cross-Validation Setup
    print("\nPreparing 5-fold cross-validation splits for multimodal architecture...")
    
    if show_stratification:
        print(f"\n✨ Stratification happens here:")
        print(f"   sgkf.split(X_seq, y, groups=subjects)")
        print(f"              ↑    ↑      ↑")
        print(f"              |    |      └─ Groups (subjects)")  
        print(f"              |    └─ Stratified (gesture labels)")
        print(f"              └─ Features")
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Prepare all 5 folds
    fold_data = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_seq, y, groups=subjects)):
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # Split all data types
        X_train_non_tof_fold = X_non_tof[train_idx]
        X_val_non_tof_fold = X_non_tof[val_idx]
        X_train_tof_fold = X_tof[train_idx]
        X_val_tof_fold = X_tof[val_idx]
        X_train_static_fold = X_static[train_idx]
        X_val_static_fold = X_static[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Normalize features separately
        # 1. Non-TOF sequential features
        if X_train_non_tof_fold.shape[2] > 0:  # Only if we have non-TOF features
            X_train_non_tof_norm, X_val_non_tof_norm, non_tof_scaler = normalize_sequential_features(
                X_train_non_tof_fold, X_val_non_tof_fold
            )
        else:
            X_train_non_tof_norm = X_train_non_tof_fold
            X_val_non_tof_norm = X_val_non_tof_fold
            non_tof_scaler = None
        
        # 2. TOF sequential features
        if X_train_tof_fold.shape[2] > 0:  # Only if we have TOF features
            X_train_tof_norm, X_val_tof_norm, tof_scaler = normalize_tof_features(
                X_train_tof_fold, X_val_tof_fold
            )
        else:
            X_train_tof_norm = X_train_tof_fold
            X_val_tof_norm = X_val_tof_fold
            tof_scaler = None
        
        # 3. Static features
        X_train_static_norm, X_val_static_norm, static_scaler = normalize_static_features(
            X_train_static_fold, X_val_static_fold
        )
        
        print(f"Train shapes:")
        print(f"  Non-TOF seq: {X_train_non_tof_norm.shape}")
        print(f"  TOF seq: {X_train_tof_norm.shape}")
        print(f"  Static: {X_train_static_norm.shape}")
        print(f"Val shapes:")
        print(f"  Non-TOF seq: {X_val_non_tof_norm.shape}")
        print(f"  TOF seq: {X_val_tof_norm.shape}")
        print(f"  Static: {X_val_static_norm.shape}")
        
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
            train_pct = train_dist / len(y_train_fold) * 100
            val_pct = val_dist / len(y_val_fold) * 100
            print(f"Stratification check:")
            for i, class_name in enumerate(label_encoder.classes_):
                if i < len(train_pct) and i < len(val_pct):
                    print(f"  {class_name}: Train {train_pct[i]:.1f}%, Val {val_pct[i]:.1f}%")
        else:
            print(f"Train class distribution: {train_dist}")
            print(f"Val class distribution: {val_dist}")
        
        # Store fold data
        fold_data.append({
            'X_train_non_tof': X_train_non_tof_norm,
            'X_train_tof': X_train_tof_norm,
            'X_train_static': X_train_static_norm,
            'y_train': y_train_fold,
            'X_val_non_tof': X_val_non_tof_norm,
            'X_val_tof': X_val_tof_norm,
            'X_val_static': X_val_static_norm,
            'y_val': y_val_fold,
            'non_tof_scaler': non_tof_scaler,
            'tof_scaler': tof_scaler,
            'static_scaler': static_scaler,
            'val_idx': val_idx,
            'train_subjects': subjects[train_idx],
            'val_subjects': subjects[val_idx],
            'non_tof_cols': non_tof_cols,
            'tof_cols': tof_cols,
        })
    
    # Save preprocessing objects
    weights_dir = _get_weights_dir()
    le_path = os.path.join(weights_dir, f'label_encoder_{variant}.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n✅ Prepared {len(fold_data)} folds for multimodal cross-validation")
    print("Each fold will train a separate multimodal model")
    
    return fold_data, label_encoder, y, sequence_ids

def preprocess_single_sequence_multimodal(seq_pl, demog_pl):
    """
    Convert one sequence to separated sequential & static numpy arrays for multimodal inference.
    
    Returns (variant, non_tof_arr[L,F], tof_arr[L,F], static_arr[features])
    """
    import pandas as pd
    import polars as pl
    
    seq_df = seq_pl.to_pandas()
    if not demog_pl.is_empty():
        seq_df = seq_df.merge(demog_pl.to_pandas(), on="subject", how="left")

    # Determine variant (same logic as before)
    thm_cols = [c for c in seq_df.columns if c.startswith("thm_")]
    tof_cols = [c for c in seq_df.columns if c.startswith("tof_")]

    # If no THM/TOF columns exist at all, use IMU (defensive fallback)
    if not thm_cols and not tof_cols:
        variant = "imu"
    else:
        # Check if ALL rows in THM columns are null/-1
        thm_all_missing = True
        if thm_cols:
            thm_df = seq_df[thm_cols].replace(-1.0, np.nan)
            thm_all_missing = not thm_df.notna().values.any()
        
        # Check if ALL rows in TOF columns are null/-1  
        tof_all_missing = True
        if tof_cols:
            tof_df = seq_df[tof_cols].replace(-1.0, np.nan)
            tof_all_missing = not tof_df.notna().values.any()
        
        # Use IMU model if either sensor type is completely missing across all rows
        if (thm_cols and thm_all_missing) or (tof_cols and tof_all_missing):
            variant = "imu"
        else:
            variant = "full"

    # Define metadata columns
    metadata_cols = [
        "row_id", "sequence_id", "sequence_type", "sequence_counter",
        "subject", "orientation", "behavior", "phase",
    ]
    
    # Sequential features: exclude metadata AND static features
    seq_feat_cols = [c for c in seq_df.columns if c not in metadata_cols and c not in STATIC_FEATURE_COLS]

    if variant == "imu":
        seq_feat_cols = [c for c in seq_feat_cols if not (c.startswith("thm_") or c.startswith("tof_"))]

    # 2-D interpolation for TOF sensor grids (per row) – skip if IMU variant
    if variant != "imu":
        seq_df = interpolate_tof(seq_df)

    # Ensure chronological order before temporal interpolation
    seq_df = seq_df.sort_values("sequence_counter")

    # Linear interpolation forward/backward along the time axis
    seq_df[seq_feat_cols] = seq_df[seq_feat_cols].interpolate(method="linear", limit_direction="both")

    # Fallback to median for any column still containing NaN, then 0 as a last resort
    seq_df[seq_feat_cols] = seq_df[seq_feat_cols].fillna(seq_df[seq_feat_cols].median())
    seq_df[seq_feat_cols] = seq_df[seq_feat_cols].fillna(0)

    # Build padded sequential tensor (unscaled)
    seq_arr = seq_df.sort_values("sequence_counter")[seq_feat_cols].to_numpy()
    seq_arr = pad_sequences([seq_arr], max_length=100)[0]  # (L, F)
    seq_arr = seq_arr.astype(np.float32)

    # Separate TOF and non-TOF features
    non_tof_arr, tof_arr, _, _ = separate_tof_features(
        seq_arr.reshape(1, seq_arr.shape[0], seq_arr.shape[1]), 
        seq_feat_cols
    )
    
    # Remove batch dimension
    non_tof_arr = non_tof_arr[0]  # (L, non_tof_features)
    tof_arr = tof_arr[0]          # (L, tof_features)

    # Static demographic features (order must match training)
    if STATIC_FEATURE_COLS[0] in seq_df.columns:
        static_vec = seq_df.iloc[0][STATIC_FEATURE_COLS].to_numpy()
    else:
        # If not merged correctly, fall back to zeros
        static_vec = np.zeros(len(STATIC_FEATURE_COLS), dtype=np.float32)

    static_vec = static_vec.astype(np.float32)

    return variant, non_tof_arr, tof_arr, static_vec