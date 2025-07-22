#!/usr/bin/env python3
"""
Gesture Recognition Training Script (Config-Driven)

This script handles the complete training pipeline for gesture recognition using
multimodal fusion with 5-fold cross-validation and group-aware splitting. It uses the
official Kaggle competition metric for model selection and evaluation.

Usage:
    python train.py <config_file>                           # Config-driven training (required)
    python train.py ../cmi-submission/configs/multimodality_model_v1_full_config.py  # Example with config
    python train.py <config_file> --stratification          # Show stratification details
    
Configuration:
    All training parameters are specified in the config file:
    - Model architecture (MultimodalityModel with registry system)
    - Training settings (epochs, learning rate, loss function)
    - Data settings (variant, batch size)
    - Environment settings (GPU selection)
    
Note: Models use LayerNorm for sequential data (better for sensor sequences)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import time
import json
import shutil
import argparse
import sys
import os
import pickle
import pandas as pd
import importlib.util, runpy

# --- Helper to load python config file ---

def load_py_config(config_path):
    """Dynamically load a Python config file as a module and return the namespace as an object with attribute access."""
    cfg_dict = runpy.run_path(config_path)
    class _CfgObj(dict):
        def __getattr__(self, item):
            return self[item]
        __setattr__ = dict.__setitem__
    return _CfgObj(cfg_dict)

# ----------------------------------------------------------------------
# Ensure shared code in cmi-submission/ is the one we import everywhere
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))             # …/development
SUBM_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cmi-submission'))   # .. means go up one level

# Pre-pend so it has priority over the local development/ path.
if SUBM_DIR not in sys.path:
    sys.path.insert(0, SUBM_DIR)

# from transformers import get_cosine_schedule_with_warmup
from utils.scheduler import get_cosine_schedule_with_warmup
from utils.focal_loss import FocalLoss
from utils.registry import build_from_cfg
from models import MODELS

# Import our model and data preprocessing
from models.multimodality import MultimodalityModel
from models.datasets import MultimodalDataset
from data_utils.data_preprocessing import prepare_data_kfold_multimodal

# Directory holding all models, scalers, summaries
WEIGHT_DIR = os.path.join(SUBM_DIR, 'weights')
os.makedirs(WEIGHT_DIR, exist_ok=True)

def calculate_composite_weights_18_class(y_18_class_series: pd.Series, 
                                           label_encoder_18_class: LabelEncoder, 
                                           target_gesture_names: list):
    """
    为18分类模型计算自定义复合权重字典 {class_index: weight}。
    """
    # 1. 根据新的推导计算类别重要性
    # BFRB类别的重要性: 0.5*(1/16) + 0.5*(1/9)
    IMP_BFRB = 25 / 288
    # 单个NON-BFRB类别的重要性: 0.5*(1/20) + 0.5*(1/90)
    IMP_NON_BFRB_INDIVIDUAL = 11 / 360
    
    class_counts = y_18_class_series.value_counts()
    
    raw_weights = {}
    for name in label_encoder_18_class.classes_:
        count = class_counts.get(name, 1)  # 避免除以零
        if name in target_gesture_names:
            # 这是一个BFRB (target) 类别
            raw_weights[name] = IMP_BFRB / count
        else:
            # 这是一个NON-BFRB (non-target) 类别
            raw_weights[name] = IMP_NON_BFRB_INDIVIDUAL / count
            
    # 标准化权重 (使平均值为1)
    total_raw_weight = sum(raw_weights.values())
    num_classes = len(raw_weights)
    avg_raw_weight = total_raw_weight / num_classes if num_classes > 0 else 1.0
    avg_raw_weight = avg_raw_weight if avg_raw_weight > 1e-9 else 1.0
    
    normalized_weights = {name: w / avg_raw_weight for name, w in raw_weights.items()}
    
    # 创建最终的 class_weight 字典 {class_index: weight}
    class_weight_dict = {
        idx: normalized_weights.get(name, 1.0)
        for idx, name in enumerate(label_encoder_18_class.classes_)
    }
    print(class_weight_dict)

    return class_weight_dict

# --- NEW: Competition Metric Configuration ---
# Define BFRB vs Non-BFRB categories based on gesture names rather than indices
# This is much more robust and maintainable
BFRB_GESTURES = {
    'Above ear - pull hair',
    'Forehead - pull hairline', 
    'Forehead - scratch',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Neck - pinch skin',
    'Neck - scratch',
    'Cheek - pinch skin'
}

NON_BFRB_GESTURES = {
    'Drink from bottle/cup',
    'Glasses on/off',
    'Pull air toward your face',
    'Pinch knee/leg skin',
    'Scratch knee/leg skin',
    'Write name on leg',
    'Text on phone',
    'Feel around in tray and pull out an object',
    'Write name in air',
    'Wave hello'
}

def competition_metric(y_true, y_pred, label_encoder):
    """
    Calculates the official Kaggle competition metric using gesture names.
    It's the average of two F1 scores:
    1. Binary F1 score (BFRB vs. non-BFRB).
    2. Macro F1 score for gestures (8 BFRB classes + 1 combined non-BFRB class).
    
    Args:
        y_true (list or np.array): Ground truth label indices.
        y_pred (list or np.array): Predicted label indices.
        label_encoder: LabelEncoder to convert indices back to gesture names.
        
    Returns:
        float: The final competition score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert indices back to gesture names
    true_gestures = label_encoder.inverse_transform(y_true)
    pred_gestures = label_encoder.inverse_transform(y_pred)

    # --- Part 1: Binary F1 Score (BFRB vs Non-BFRB) ---
    y_true_binary = np.array([1 if gesture in BFRB_GESTURES else 0 for gesture in true_gestures])
    y_pred_binary = np.array([1 if gesture in BFRB_GESTURES else 0 for gesture in pred_gestures])
    binary_f1 = f1_score(y_true_binary, y_pred_binary, pos_label=1, zero_division=0)

    # --- Part 2: Macro F1 Score on Gestures ---
    # Create mapping for macro F1: each BFRB gesture gets its own class (0-7), 
    # all non-BFRB gestures get mapped to class 8
    bfrb_list = sorted(list(BFRB_GESTURES))  # Consistent ordering
    bfrb_to_idx = {gesture: i for i, gesture in enumerate(bfrb_list)}
    non_target_class_id = len(bfrb_list)  # 8
    
    # Map gestures to indices for macro F1 calculation
    y_true_multi = np.array([
        bfrb_to_idx[gesture] if gesture in BFRB_GESTURES else non_target_class_id 
        for gesture in true_gestures
    ])
    y_pred_multi = np.array([
        bfrb_to_idx[gesture] if gesture in BFRB_GESTURES else non_target_class_id 
        for gesture in pred_gestures
    ])
    
    # Calculate macro F1 over all 9 classes (8 BFRB + 1 non-BFRB)
    all_classes = list(range(len(bfrb_list) + 1))  # 0-8
    macro_f1 = f1_score(y_true_multi, y_pred_multi, average='macro', labels=all_classes, zero_division=0)

    # --- Final Score ---
    final_score = (binary_f1 + macro_f1) / 2.0
    return final_score


def setup_device(gpu_id=None):
    """
    Setup computation device with intelligent GPU selection
    
    Args:
        gpu_id (int, optional): Specific GPU ID to use
        
    Returns:
        torch.device: Selected device
    """
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory // 1024**3} GB)")
    
    # Select GPU
    if gpu_id is not None:
        if 0 <= gpu_id < torch.cuda.device_count():
            selected_gpu = gpu_id
            print(f"Using specified GPU {gpu_id}")
        else:
            print(f"Invalid GPU ID {gpu_id}, auto-selecting...")
            selected_gpu = None
    else:
        selected_gpu = None
    
    # Auto-select GPU with most free memory
    if selected_gpu is None:
        best_gpu = 0
        max_free = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            free = props.total_memory - allocated
            
            if free > max_free:
                max_free = free
                best_gpu = i
        
        selected_gpu = best_gpu
        print(f"🎯 Auto-selected GPU {selected_gpu} (most free memory: {max_free // 1024**3} GB)")
    
    # Setup selected GPU
    device = torch.device(f'cuda:{selected_gpu}')
    torch.cuda.set_device(selected_gpu)
    torch.cuda.empty_cache()
    
    # Display memory info
    props = torch.cuda.get_device_properties(selected_gpu)
    allocated = torch.cuda.memory_allocated(selected_gpu)
    free = props.total_memory - allocated
    
    print(f"\n🎯 Using GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")
    print(f"GPU {selected_gpu} Memory:")
    print(f"   Total: {props.total_memory // 1024**3} GB")
    print(f"   Used: {allocated // 1024**2} MB")
    print(f"   Free: {free // 1024**3} GB")
    
    return device


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Train for one epoch.
    MODIFIED: Now handles sample weights for weighted loss calculation.
    """
    model.train() 
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # MODIFIED: Unpack sample_weights from the dataloader
    for (non_tof_data, tof_data, static_data), target, sample_weights in dataloader:
        non_tof_data = non_tof_data.to(device)
        tof_data = tof_data.to(device)
        static_data = static_data.to(device)
        target = target.to(device)
        # NEW: Move sample weights to the device
        sample_weights = sample_weights.to(device)
        
        optimizer.zero_grad()
        output = model(non_tof_data, tof_data, static_data)
        
        # NEW: Manual weighted loss calculation
        # The criterion must be initialized with reduction='none'
        per_sample_loss = criterion(output, target)
        weighted_loss = per_sample_loss * sample_weights
        loss = weighted_loss.mean() # Get the mean of weighted losses
        
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = 100. * (all_preds == all_targets).sum() / len(all_targets)
    
    return avg_loss, accuracy, all_preds, all_targets


def validate_and_evaluate_epoch(model, dataloader, device, label_encoder):
    """
    Validate for one epoch and compute all relevant metrics.
    MODIFIED: Uses its own standard CrossEntropyLoss for validation loss to avoid
              being affected by weighted loss from training.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    # NEW: Instantiate its own criterion for unweighted loss calculation
    val_criterion = nn.CrossEntropyLoss().to(device)
    
    with torch.no_grad():
        # MODIFIED: Unpack the dummy weight value, but don't use it
        for (non_tof_data, tof_data, static_data), target, _ in dataloader:
            non_tof_data = non_tof_data.to(device)
            tof_data = tof_data.to(device)
            static_data = static_data.to(device)
            target = target.to(device)
            output = model(non_tof_data, tof_data, static_data)
            
            # MODIFIED: Calculate loss using the local, unweighted criterion
            loss = val_criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = 100. * (all_preds == all_targets).sum() / len(all_targets)
    comp_score = competition_metric(all_targets, all_preds, label_encoder)
    
    return avg_loss, accuracy, comp_score, all_preds, all_targets


def train_model(model, train_loader, val_loader, label_encoder, epochs=50, start_lr=0.001, patience=15, device='cpu', variant: str = 'full', fold_tag: str = '', criterion=None):
    """Train the model with validation, using competition metric for model selection."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=1e-2)
    
    # Setup cosine schedule with warmup (10% warmup steps)
    total_training_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_competition_score': [],
        'val_loss': [], 'val_accuracy': [], 'val_competition_score': [],
        'learning_rate': []
    }
    
    best_val_score = 0
    patience = patience
    patience_counter = 0
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train and get predictions for score calculation
        train_loss, train_acc, train_preds, train_targets = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        train_score = competition_metric(train_targets, train_preds, label_encoder)
        
        # Validate
        val_loss, val_acc, val_score, _, _ = validate_and_evaluate_epoch(model, val_loader, device, label_encoder)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model based on competition score
        if val_score > best_val_score:
            best_val_score = val_score
            ckpt_name = os.path.join(WEIGHT_DIR, f'best_model_tmp{("_" + variant) if variant else ""}{fold_tag}.pth')
            torch.save(model.state_dict(), ckpt_name)
            patience_counter = 0
            print(f"   🚀 New best val score: {best_val_score:.4f}. Model saved.")
        else:
            patience_counter += 1
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['train_competition_score'].append(train_score)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_competition_score'].append(val_score)
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): '
              f'Train Loss: {train_loss:.4f}, Train Score: {train_score:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f} | '
              f'LR: {current_lr:.6f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation score did not improve for {patience} epochs.')
            break
    
    # Load best model
    ckpt_name = os.path.join(WEIGHT_DIR, f'best_model_tmp{("_" + variant) if variant else ""}{fold_tag}.pth')
    model.load_state_dict(torch.load(ckpt_name))

    # delete temporary best model file
    if os.path.exists(ckpt_name):
        os.remove(ckpt_name)
        print(f"Temporary best model file '{ckpt_name}' deleted.")
    
    return history



def train_kfold_models(epochs=50, start_lr=0.001, batch_size=32, patience=15, show_stratification=False, device=None, variant: str = 'full', loss_function='ce', focal_gamma=2.0, focal_alpha=1.0, model_cfg: dict = None):
    """Train 5 models using 5-fold cross-validation"""
    print("="*60)
    print("TRAINING 5 MODELS WITH 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Setup device
    if device is None:
        device = setup_device()
    
    # Prepare all folds and get full labels & sequence IDs for OOF reconstruction
    fold_data, label_encoder, y_all, sequence_ids_all = prepare_data_kfold_multimodal(show_stratification=show_stratification, variant=variant)
    num_samples = len(y_all)
    oof_preds = np.full(num_samples, -1, dtype=int)
    oof_targets = y_all.copy()
    
    # Dynamically inject feature dimensions from actual data
    sample_non_tof = fold_data[0]['X_train_non_tof']
    sample_static = fold_data[0]['X_train_static']
    
    non_tof_channels = sample_non_tof.shape[2]
    static_features = sample_static.shape[1]
    
    print(f"Auto-detected dimensions:")
    print(f"  Non-TOF sequential channels: {non_tof_channels}")
    print(f"  Static features: {static_features}")
    
    # Inject into model config
    model_cfg['cnn_branch_cfg']['input_channels'] = non_tof_channels
    model_cfg['mlp_branch_cfg']['input_features'] = static_features
    
    # Model parameters (same for all folds)
    # Display model configuration from config file
    print(f"\nModel configuration from config file:")
    for k, v in (model_cfg or {}).items():
        if k != 'type':
            print(f"  {k}: {v}")

    # Configure loss function and normalization
    print(f"\nTraining Configuration:")
    if loss_function == 'focal':
        print(f"  Loss Function: Focal Loss (gamma={focal_gamma}, alpha={focal_alpha})")
    else:
        print(f"  Loss Function: Cross Entropy Loss")
    print(f"  Normalization: LayerNorm (better for sequential sensor data)")
    print(f"    - 1D CNN: LayerNorm for temporal features")
    print(f"    - 2D CNN: BatchNorm for spatial features") 
    print(f"    - Fusion Head: LayerNorm for combined features")
    
    # Store results for all folds
    fold_results = []
    fold_models = []
    fold_histories = []
    
    # Train each fold
    for fold_idx in range(5):
        print(f"\n" + "="*60)
        print(f"TRAINING FOLD {fold_idx + 1}/5")
        print("="*60)
        
        # Get fold data
        fold = fold_data[fold_idx]

        # Create a pandas Series with gesture names for the weight function
        y_train_series = pd.Series(label_encoder.inverse_transform(fold['y_train']))
        
        # The weight function returns a {class_index: weight} dictionary
        class_weight_dict = calculate_composite_weights_18_class(
            y_18_class_series=y_train_series,
            label_encoder_18_class=label_encoder,
            target_gesture_names=list(BFRB_GESTURES)
        )
        print("Sample weights calculated.")
        
        # Create multimodal datasets and dataloaders
        # MODIFIED: Pass the class_weight_dict to the training dataset
        train_dataset = MultimodalDataset(
            fold['X_train_non_tof'], 
            fold['X_train_tof'], 
            fold['X_train_static'], 
            fold['y_train'],
            class_weight_dict=class_weight_dict # Pass weights here
        )
        # Validation dataset does not need weights
        val_dataset = MultimodalDataset(
            fold['X_val_non_tof'], 
            fold['X_val_tof'], 
            fold['X_val_static'], 
            fold['y_val']
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Build model for this fold
        print(f"\nBuilding model for fold {fold_idx + 1}...")
        model = build_from_cfg(model_cfg, MODELS)
        
        # Configure loss function for this fold
        if loss_function == 'focal':
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none')
            print(f"  Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}, reduction='none')")
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')
            print(f"  Using Cross Entropy Loss (reduction='none')")
        
        model = model.to(device)
        # 编译模型加速训练，需要torch2.x版本
        model = torch.compile(model)
        criterion = criterion.to(device)
        print(f"Model and criterion loaded to {device}")
        
        # Train model
        print(f"\nTraining fold {fold_idx + 1}...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            label_encoder=label_encoder,
            epochs=epochs,
            start_lr=start_lr,
            patience=patience,
            device=device,
            variant=variant,
            fold_tag=f'_{fold_idx+1}',
            criterion=criterion,
        )
        
        # Evaluate model and capture predictions for OOF
        print(f"\nEvaluating fold {fold_idx + 1}...")
        _, _, _, all_preds_val, _ = validate_and_evaluate_epoch(model, val_loader, device, label_encoder)
        
        # Store into OOF array using original indices
        oof_preds[fold['val_idx']] = all_preds_val
        
        # Calculate final validation score
        best_val_score = max(history['val_competition_score'])
        
        # Save model for this fold
        model_filename = os.path.join(WEIGHT_DIR, f'model_fold_{fold_idx + 1}_{variant}.pth')
        checkpoint = {
            'state_dict': model.state_dict(),
            'model_cfg': model_cfg
        }
        torch.save(checkpoint, model_filename)
        print(f"Model saved as '{model_filename}'")
        
        # Store results
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_score': best_val_score,
            'model_filename': model_filename
        })
        
        fold_models.append(model)
        fold_histories.append(history)
        
        print(f"Fold {fold_idx + 1} - Best Val Score: {best_val_score:.4f}")
    
    # Summary of all folds
    print(f"\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    best_scores = [result['best_val_score'] for result in fold_results]
    
    print(f"Best Validation Scores per Fold: {[f'{score:.4f}' for score in best_scores]}")
    print(f"")
    print(f"Mean Best Score: {np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}")
    
    # Find best fold
    best_fold_idx = np.argmax(best_scores)
    print(f"\nBest performing fold: Fold {best_fold_idx + 1} (Score: {best_scores[best_fold_idx]:.4f})")
    
    # Save summary
    summary = {
        'fold_results': fold_results,
        'mean_best_score': float(np.mean(best_scores)),
        'std_best_score': float(np.std(best_scores)),
        'best_fold': int(best_fold_idx + 1),
        'best_fold_score': float(best_scores[best_fold_idx])
    }
    
    with open(os.path.join(WEIGHT_DIR, f'kfold_summary_{variant}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to 'development/outputs/kfold_summary_{variant}.json'")
    
    # Compute overall OOF competition score
    oof_comp_score = competition_metric(oof_targets, oof_preds, label_encoder)
    print(f"\n🏆 Overall OOF Competition Score: {oof_comp_score:.4f}")

    # Save OOF predictions to CSV
    oof_df = pd.DataFrame({
        'sequence_id': sequence_ids_all,
        'gesture_true': label_encoder.inverse_transform(oof_targets),
        'gesture_pred': label_encoder.inverse_transform(oof_preds),
    })
    oof_path = os.path.join(WEIGHT_DIR, f'oof_predictions_{variant}.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to '{oof_path}'")
    
    return fold_models, fold_histories, fold_results


def main():
    """Main training function - config file required"""
    parser = argparse.ArgumentParser(description="Gesture Recognition Training (Config Required)")
    parser.add_argument('--config', required=True, help='Path to python config file for training')
    parser.add_argument('--stratification', action='store_true', help='Show stratification details')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    cfg = load_py_config(args.config)
    
    # Extract parameters from config
    epochs = cfg.training.get('epochs', 50)
    start_lr = cfg.training.get('start_lr', 0.001)
    variant = cfg.data.get('variant', 'full')
    batch_size = cfg.data.get('batch_size', 32)
    patience = cfg.training.get('patience', 15)
    loss_function = 'focal' if cfg.training.get('loss', {}).get('type') == 'FocalLoss' else 'ce'
    focal_gamma = cfg.training.get('loss', {}).get('gamma', 2.0)
    focal_alpha = cfg.training.get('loss', {}).get('alpha', 1.0)
    gpu_id = cfg.environment.get('gpu_id')
    
    print(f"\nTraining Configuration from {args.config}:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {start_lr}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Patience: {patience}")
    print(f"  Variant: {variant}")
    print(f"  Loss Function: {loss_function}")
    if loss_function == 'focal':
        print(f"  Focal Gamma: {focal_gamma}")
        print(f"  Focal Alpha: {focal_alpha}")
    print(f"  GPU ID: {gpu_id}")
    
    # Setup device
    device = setup_device(gpu_id)
    
    # Train models using config
    fold_models, fold_histories, fold_results = train_kfold_models(
        epochs=epochs,
        start_lr=start_lr,
        batch_size=batch_size,
        patience=patience,
        show_stratification=args.stratification,
        device=device,
        variant=variant,
        loss_function=loss_function,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        model_cfg=cfg.model,
    )
    
    print("✅ Config-driven training completed!")
    return fold_models, fold_histories, fold_results


if __name__ == "__main__":
    results = main()