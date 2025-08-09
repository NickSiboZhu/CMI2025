# train.py

# ... (所有 imports 保持不变) ...
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch._dynamo

# 加上这行，可以在支持的硬件上提升性能
torch.set_float32_matmul_precision('high')

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
from torch import amp  

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
from utils.scheduler import get_cosine_schedule_with_warmup, WarmupAndReduceLROnPlateau
from utils.focal_loss import FocalLoss
from utils.registry import build_from_cfg
from models import MODELS

# --- MODIFIED: Import the new Hybrid model and updated dataset class ---
from models.multimodality import MultimodalityModel
from models.datasets import MultimodalDataset
from data_utils.data_preprocessing import prepare_data_kfold_multimodal

# Directory holding all models, scalers, summaries
WEIGHT_DIR = os.path.join(SUBM_DIR, 'weights')
os.makedirs(WEIGHT_DIR, exist_ok=True)
# ... (calculate_composite_weights_18_class 和 competition_metric 保持不变) ...
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
            # Query global (system-wide) free memory on the GPU using cudaMemGetInfo
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(i)
            except AttributeError:
                # Fallback for older PyTorch versions – use per-process reserved memory
                torch.cuda.set_device(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory
                free_mem = total_mem - torch.cuda.memory_reserved(i)

            if free_mem > max_free:
                max_free = free_mem
                best_gpu = i
        
        selected_gpu = best_gpu
        print(f"🎯 Auto-selected GPU {selected_gpu} (most free memory: {max_free // 1024**3} GB)")
    
    # Setup selected GPU
    device = torch.device(f'cuda:{selected_gpu}')
    torch.cuda.set_device(selected_gpu)
    torch.cuda.empty_cache()
    
    # Display memory info
    props = torch.cuda.get_device_properties(selected_gpu)

    # Use global memory stats for the final report as well
    try:
        free, total_mem = torch.cuda.mem_get_info(selected_gpu)
        used = total_mem - free
    except AttributeError:
        used = torch.cuda.memory_allocated(selected_gpu)
        total_mem = props.total_memory
        free = total_mem - used

    print(f"\n🎯 Using GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")
    print(f"GPU {selected_gpu} Memory:")
    print(f"   Total: {total_mem // 1024**3} GB")
    print(f"   Used: {used // 1024**2} MB")
    print(f"   Free: {free // 1024**3} GB")
    
    return device

# --- MODIFIED: Function signatures to accept `spec_data` ---
def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp=True, scheduler=None, mixup_enabled=False, mixup_alpha=0.4):
    model.train() 
    total_loss = 0
    all_preds, all_targets = [], []
    
    # --- MODIFIED: Unpack spec_data from dataloader ---
    for (imu_data, thm_data, tof_data, spec_data, static_data, mask), target, sample_weights in dataloader:
        imu_data, thm_data, tof_data = imu_data.to(device), thm_data.to(device), tof_data.to(device)
        spec_data, static_data, mask = spec_data.to(device), static_data.to(device), mask.to(device)
        target, sample_weights = target.to(device), sample_weights.to(device)
        
        optimizer.zero_grad()

        with amp.autocast(device_type=device.type, enabled=use_amp):
            if mixup_enabled and mixup_alpha > 0.0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                rand_index = torch.randperm(imu_data.size(0)).to(device)
                
                # --- MODIFIED: Mix spec_data as well ---
                imu_data = lam * imu_data + (1 - lam) * imu_data[rand_index]
                thm_data = lam * thm_data + (1 - lam) * thm_data[rand_index]
                tof_data = lam * tof_data + (1 - lam) * tof_data[rand_index]
                spec_data = lam * spec_data + (1 - lam) * spec_data[rand_index]
                static_data = lam * static_data + (1 - lam) * static_data[rand_index]
                mask = torch.max(mask, mask[rand_index])

                target_a, target_b = target, target[rand_index]
                weights_a, weights_b = sample_weights, sample_weights[rand_index]

                # --- MODIFIED: Pass spec_data to the model ---
                output = model(imu_data, thm_data, tof_data, spec_data, static_data, mask=mask)
                
                loss_a = criterion(output, target_a) * weights_a
                loss_b = criterion(output, target_b) * weights_b
                loss = (lam * loss_a + (1 - lam) * loss_b).mean()

            else:
                # --- MODIFIED: Pass spec_data to the model ---
                output = model(imu_data, thm_data, tof_data, spec_data, static_data, mask=mask)
                loss = (criterion(output, target) * sample_weights).mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None and not isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, WarmupAndReduceLROnPlateau)):
            scheduler.step()
        
        total_loss += loss.item()
        all_preds.extend(output.argmax(dim=1).cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    return total_loss / len(dataloader), 100. * (np.array(all_preds) == np.array(all_targets)).mean(), np.array(all_preds), np.array(all_targets)

def validate_and_evaluate_epoch(model, dataloader, device, label_encoder, use_amp=True):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    val_criterion = nn.CrossEntropyLoss().to(device)
    
    with torch.no_grad():
        # --- MODIFIED: Unpack spec_data from dataloader ---
        for (imu_data, thm_data, tof_data, spec_data, static_data, mask), target, _ in dataloader:
            imu_data, thm_data, tof_data = imu_data.to(device), thm_data.to(device), tof_data.to(device)
            spec_data, static_data, mask = spec_data.to(device), static_data.to(device), mask.to(device)
            target = target.to(device)
            
            with amp.autocast(device_type=device.type, enabled=use_amp):
                # --- MODIFIED: Pass spec_data to the model ---
                output = model(imu_data, thm_data, tof_data, spec_data, static_data, mask=mask)
                loss = val_criterion(output, target)
            
            total_loss += loss.item()
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    accuracy = 100. * (all_preds == all_targets).mean()
    comp_score = competition_metric(all_targets, all_preds, label_encoder)
    
    return avg_loss, accuracy, comp_score, all_preds, all_targets


def train_model(model, train_loader, val_loader, label_encoder, epochs=50, patience=15, weight_decay=1e-2, device='cpu', use_amp=True, variant: str = 'full', fold_tag: str = '', criterion=None, mixup_enabled=False, mixup_alpha=0.4, scheduler_cfg=None, output_dir: str = WEIGHT_DIR):
    """Train the model with validation, using competition metric for model selection."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # --- NEW: Optimizer setup with specific layer learning rates ---
    layer_lrs = scheduler_cfg.get('layer_lrs') if scheduler_cfg else None
    
    if layer_lrs:
        print("Using discriminative learning rates per layer.")
        param_groups = []
        
        # --- MODIFIED: Add 'spec' branch to the map ---
        branch_map = {
            'imu': model.imu_branch,
            'thm': getattr(model, 'thm_branch', None),
            'tof': getattr(model, 'tof_branch', None),
            'spec': getattr(model, 'spec_branch', None), # NEW
            'mlp': model.mlp_branch,
            'fusion': model.classifier_head
        }
        
        # Create parameter groups for active branches
        for name, branch in branch_map.items():
            if branch is not None:
                if name in layer_lrs:
                    param_groups.append({
                        'name': name,
                        'params': branch.parameters(),
                        'lr': layer_lrs[name]
                    })
                else:
                    # This is a safeguard. If a branch exists but its LR is not specified, raise an error.
                    raise ValueError(f"Learning rate for active branch '{name}' not specified in 'layer_lrs'.")
        
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)

    else:
        # Fallback for backward compatibility or simple training runs
        print("Using a single learning rate for the entire model.")
        start_lr = scheduler_cfg.get('start_lr', 1e-3) if scheduler_cfg else 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=weight_decay)

    scaler = amp.GradScaler(device=device.type, enabled=use_amp)
    print(f"Automatic Mixed Precision (AMP): {'Enabled' if scaler.is_enabled() else 'Disabled'}")

    # --- NEW: Dynamic Scheduler Setup ---
    if scheduler_cfg is None: scheduler_cfg = {'type': 'cosine'} # Default to cosine
    scheduler = None
    if scheduler_cfg['type'] == 'cosine':
        total_training_steps = epochs * len(train_loader)
        warmup_ratio = scheduler_cfg.get('warmup_ratio', 0.1)
        warmup_steps = int(warmup_ratio * total_training_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
        print(f"Using Cosine Annealing scheduler with warmup ratio: {warmup_ratio} ({warmup_steps} steps).")
    elif scheduler_cfg['type'] == 'reduce_on_plateau':
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_cfg.get('factor', 0.2), patience=scheduler_cfg.get('patience', 10), min_lr=scheduler_cfg.get('min_lr', 1e-6), verbose=True)
        warmup_ratio = scheduler_cfg.get('warmup_ratio', 0.0)
        warmup_epochs = int(warmup_ratio * epochs) if warmup_ratio > 0 else 0
        if warmup_epochs > 0:
            scheduler = WarmupAndReduceLROnPlateau(optimizer, warmup_epochs, plateau_scheduler)
            print(f"Using ReduceLROnPlateau with a {warmup_epochs}-epoch linear warmup.")
        else:
            scheduler = plateau_scheduler
            print(f"Using standard ReduceLROnPlateau scheduler.")
    
    history = {'train_loss': [], 'train_accuracy': [], 'train_competition_score': [], 'val_loss': [], 'val_accuracy': [], 'val_competition_score': [], 'learning_rate': []}
    best_val_score = 0
    patience_counter = 0
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc, train_preds, train_targets = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, scheduler, mixup_enabled=mixup_enabled, mixup_alpha=mixup_alpha)
        train_score = competition_metric(train_targets, train_preds, label_encoder)
        val_loss, val_acc, val_score, _, _ = validate_and_evaluate_epoch(model, val_loader, device, label_encoder, use_amp)
        
        if scheduler is not None:
             if isinstance(scheduler, (WarmupAndReduceLROnPlateau, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                 scheduler.step(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            ckpt_name = os.path.join(output_dir, f'best_model_tmp{("_" + variant) if variant else ""}{fold_tag}.pth')
            torch.save(model.state_dict(), ckpt_name)
            patience_counter = 0
            print(f"   🚀 New best val score: {best_val_score:.4f}. Model saved.")
        else:
            patience_counter += 1
        
        history['train_loss'].append(train_loss); history['train_accuracy'].append(train_acc); history['train_competition_score'].append(train_score)
        history['val_loss'].append(val_loss); history['val_accuracy'].append(val_acc); history['val_competition_score'].append(val_score)
        history['learning_rate'].append([g['lr'] for g in optimizer.param_groups])
        
        epoch_time = time.time() - start_time
        
        if 'name' not in optimizer.param_groups[0]: lr_info = f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        else: lr_info = "LRs: " + ", ".join([f"{group['name']}={group['lr']:.2e}" for group in optimizer.param_groups])
        
        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): Train Loss: {train_loss:.4f}, Train Score: {train_score:.4f} | Val Loss: {val_loss:.4f}, Val Score: {val_score:.4f} | {lr_info}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} as validation score did not improve for {patience} epochs.')
            break
    
    ckpt_name = os.path.join(output_dir, f'best_model_tmp{("_" + variant) if variant else ""}{fold_tag}.pth')
    if os.path.exists(ckpt_name):
        model.load_state_dict(torch.load(ckpt_name))
        os.remove(ckpt_name)
        print(f"Loaded best model and deleted temporary file '{ckpt_name}'.")
    
    return history

def train_kfold_models(epochs=50, weight_decay=1e-2, batch_size=32, patience=15, show_stratification=False, use_amp=True, device=None, variant: str = 'full', loss_function='ce', focal_gamma=2.0, focal_alpha=1.0, model_cfg: dict = None, mixup_enabled=False, mixup_alpha=0.4, scheduler_cfg=None, spec_params: dict = None, output_dir: str = 'weights'):
    """
    使用5折交叉验证训练5个模型。
    
    此版本功能完整，包括:
    - 传递STFT参数以动态生成频谱图。
    - 为每个折叠创建并保存模型、scaler以及频谱图统计文件(spec_stats)。
    """
    print("="*60)
    print("TRAINING 5 MODELS WITH 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    if device is None:
        device = setup_device()
    
    # 准备K折数据，并传递spec_params
    fold_data, label_encoder, y_all, sequence_ids_all = prepare_data_kfold_multimodal(
        show_stratification=show_stratification, 
        variant=variant,
    )
    
    num_samples = len(y_all)
    oof_preds = np.full(num_samples, -1, dtype=int)
    oof_targets = y_all.copy()
    
    # 动态注入特征维度
    sample_imu = fold_data[0]['X_train_imu']
    sample_thm = fold_data[0]['X_train_thm']
    sample_tof = fold_data[0]['X_train_tof']
    sample_static = fold_data[0]['X_train_static']
    sample_spec = fold_data[0]['X_train_spec']

    model_cfg['imu_branch_cfg']['input_channels'] = sample_imu.shape[2]
    if 'thm_branch_cfg' in model_cfg:
        model_cfg['thm_branch_cfg']['input_channels'] = sample_thm.shape[2]
    if 'tof_branch_cfg' in model_cfg:
        model_cfg['tof_branch_cfg']['input_channels'] = sample_tof.shape[2] // 64
    if 'spec_branch_cfg' in model_cfg:
        model_cfg['spec_branch_cfg']['in_channels'] = sample_spec.shape[1]
    model_cfg['mlp_branch_cfg']['input_features'] = sample_static.shape[1]
    
    print(f"\nModel configuration from config file:")
    for k, v in (model_cfg or {}).items():
        if k != 'type': print(f"  {k}: {v}")
    
    fold_results = []
    fold_models = []
    fold_histories = []
    
    # 训练每个折叠
    for fold_idx in range(5):
        print(f"\n" + "="*60)
        print(f"TRAINING FOLD {fold_idx + 1}/5")
        print("="*60)
        
        fold = fold_data[fold_idx]
        spec_stats = fold['spec_stats']
        
        y_train_series = pd.Series(label_encoder.inverse_transform(fold['y_train']))
        class_weight_dict = calculate_composite_weights_18_class(y_train_series, label_encoder, list(BFRB_GESTURES))
        
        # 创建数据集，并传入spec_stats
        train_dataset = MultimodalDataset(
            fold['X_train_imu'], fold['X_train_thm'], fold['X_train_tof'], fold['X_train_spec'],
            fold['X_train_static'], fold['y_train'], mask=fold['train_mask'],
            class_weight_dict=class_weight_dict, spec_stats=spec_stats
        )
        val_dataset = MultimodalDataset(
            fold['X_val_imu'], fold['X_val_thm'], fold['X_val_tof'], fold['X_val_spec'],
            fold['X_val_static'], fold['y_val'], mask=fold['val_mask'],
            spec_stats=spec_stats
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        
        print(f"\nBuilding model for fold {fold_idx + 1}...")
        model = build_from_cfg(model_cfg, MODELS)
        
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='none') if loss_function == 'focal' else nn.CrossEntropyLoss(reduction='none')
        
        model = model.to(device)
        criterion = criterion.to(device)
        print(f"Model and criterion loaded to {device}")
        
        print(f"\nTraining fold {fold_idx + 1}...")
        history = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader, label_encoder=label_encoder, 
            epochs=epochs, patience=patience, weight_decay=weight_decay, use_amp=use_amp, 
            device=device, variant=variant, fold_tag=f'_{fold_idx+1}', criterion=criterion, 
            mixup_enabled=mixup_enabled, mixup_alpha=mixup_alpha, scheduler_cfg=scheduler_cfg, 
            output_dir=output_dir
        )
        
        print(f"\nEvaluating fold {fold_idx + 1}...")
        _, _, _, all_preds_val, _ = validate_and_evaluate_epoch(model, val_loader, device, label_encoder)
        oof_preds[fold['val_idx']] = all_preds_val
        best_val_score = max(history['val_competition_score'])
        
        # 保存模型
        model_filename = os.path.join(output_dir, f'model_fold_{fold_idx + 1}_{variant}.pth')
        torch.save({'state_dict': model.state_dict(), 'model_cfg': model_cfg}, model_filename)
        print(f"Model saved as '{model_filename}'")

        # 保存 scaler
        scaler_fold = fold['scaler']
        scaler_filename = os.path.join(output_dir, f'scaler_fold_{fold_idx + 1}_{variant}.pkl')
        with open(scaler_filename, 'wb') as sf:
            pickle.dump(scaler_fold, sf)
        print(f"Scaler saved as '{scaler_filename}'")
        
        # 保存该折叠的频谱图统计量 (spec_stats)
        spec_stats_filename = os.path.join(output_dir, f'spec_stats_fold_{fold_idx + 1}_{variant}.pkl')
        with open(spec_stats_filename, 'wb') as f:
            pickle.dump(spec_stats, f)
        print(f"Spectrogram stats saved as '{spec_stats_filename}'")
        
        fold_results.append({'fold': fold_idx + 1, 'best_val_score': best_val_score})
        fold_models.append(model)
        fold_histories.append(history)
        print(f"Fold {fold_idx + 1} - Best Val Score: {best_val_score:.4f}")
    
    # 保存全局标签编码器
    le_filename = os.path.join(output_dir, f'label_encoder_{variant}.pkl')
    with open(le_filename, 'wb') as lf:
        pickle.dump(label_encoder, lf)
    print(f"Label encoder saved as '{le_filename}'")
    
    # 打印和保存所有折叠的总结
    print(f"\n" + "="*60 + "\n5-FOLD CROSS-VALIDATION SUMMARY\n" + "="*60)
    best_scores = [result['best_val_score'] for result in fold_results]
    print(f"Best Validation Scores per Fold: {[f'{score:.4f}' for score in best_scores]}")
    print(f"\nMean Best Score: {np.mean(best_scores):.4f} ± {np.std(best_scores):.4f}")
    
    best_fold_idx = np.argmax(best_scores)
    print(f"\nBest performing fold: Fold {best_fold_idx + 1} (Score: {best_scores[best_fold_idx]:.4f})")
    
    summary = {
        'fold_results': fold_results,
        'mean_best_score': float(np.mean(best_scores)),
        'std_best_score': float(np.std(best_scores)),
        'best_fold': int(best_fold_idx + 1),
        'best_fold_score': float(best_scores[best_fold_idx])
    }
    with open(os.path.join(output_dir, f'kfold_summary_{variant}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to '{os.path.join(output_dir, f'kfold_summary_{variant}.json')}'")
    
    # 计算并保存OOF结果
    oof_comp_score = competition_metric(oof_targets, oof_preds, label_encoder)
    print(f"\n🏆 Overall OOF Competition Score: {oof_comp_score:.4f}")
    oof_df = pd.DataFrame({
        'sequence_id': sequence_ids_all,
        'gesture_true': label_encoder.inverse_transform(oof_targets),
        'gesture_pred': label_encoder.inverse_transform(oof_preds),
    })
    oof_path = os.path.join(output_dir, f'oof_predictions_{variant}.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to '{oof_path}'")
    
    return oof_comp_score, fold_models, fold_histories, fold_results

# ... (main function remains largely the same, just calling the modified functions) ...
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
    patience = cfg.training.get('patience', 15)
    weight_decay = cfg.training.get('weight_decay', 1e-2)
    batch_size = cfg.data.get('batch_size', 32)
    use_amp = cfg.training.get('use_amp', True)
    variant = cfg.data.get('variant', 'full')
    loss_function = 'focal' if cfg.training.get('loss', {}).get('type') == 'FocalLoss' else 'ce'
    focal_gamma = cfg.training.get('loss', {}).get('gamma', 2.0)
    focal_alpha = cfg.training.get('loss', {}).get('alpha', 1.0)
    gpu_id = cfg.environment.get('gpu_id')
    mixup_enabled = cfg.training.get('mixup_enabled', True)
    mixup_alpha = cfg.training.get('mixup_alpha', 0.4)
    # --- NEW: Read scheduler config from file ---
    scheduler_cfg = cfg.training.get('scheduler_cfg', {'type': 'cosine'})

    
    print(f"\nTraining Configuration from {args.config}:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  Use AMP: {use_amp}")
    print(f"  Patience: {patience}")
    print(f"  Variant: {variant}")
    print(f"  Mixup Enabled: {mixup_enabled}")
    if mixup_enabled:
        print(f"  Mixup Alpha: {mixup_alpha}")
    print(f"  Loss Function: {loss_function}")
    if loss_function == 'focal':
        print(f"  Focal Gamma: {focal_gamma}")
        print(f"  Focal Alpha: {focal_alpha}")
    print(f"  GPU ID: {gpu_id}")
    # --- NEW: Print scheduler config ---
    print(f"  Scheduler Config: {scheduler_cfg}")
    
    # Print layer-specific learning rates if configured
    layer_lrs = scheduler_cfg.get('layer_lrs') if scheduler_cfg else None
    if layer_lrs is not None:
        print(f"  Layer-Specific Learning Rates:")
        for layer_name, lr_value in layer_lrs.items():
            print(f"    - {layer_name.upper()}: {lr_value:.2e}")
    
    # Setup device
    device = setup_device(gpu_id)
    
    # Train models using config, using the default output directory
    oof_score, fold_models, fold_histories, fold_results = train_kfold_models(
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        weight_decay=weight_decay,
        use_amp=use_amp,
        show_stratification=args.stratification,
        device=device,
        variant=variant,
        loss_function=loss_function,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        model_cfg=cfg.model,
        mixup_enabled=mixup_enabled,
        mixup_alpha=mixup_alpha,
        scheduler_cfg=scheduler_cfg,
        output_dir=WEIGHT_DIR  # Explicitly pass the default
    )
    
    print("✅ Config-driven training completed!")
    return fold_models, fold_histories, fold_results


if __name__ == "__main__":
    results = main()