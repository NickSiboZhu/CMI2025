#!/usr/bin/env python3
"""
Gesture Recognition Training Script (with Competition Metric)

This script handles the complete training pipeline for gesture recognition using
1D CNN with 5-fold cross-validation and group-aware splitting. It uses the
official Kaggle competition metric for model selection and evaluation.

Usage:
    python train.py                     # Full 5-fold cross-validation
    python train.py --single            # Single train/val split
    python train.py --stratification    # Show stratification details
    python train.py --gpu 1             # Use specific GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import time
import json
import shutil
import argparse
import sys
import os
import pickle
import pandas as pd
# ----------------------------------------------------------------------
# Ensure shared code in cmi-submission/ is the one we import everywhere
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))             # …/development
SUBM_DIR    = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cmi-submission'))

# Pre-pend so it has priority over the local development/ path.
if SUBM_DIR not in sys.path:
    sys.path.insert(0, SUBM_DIR)

# from transformers import get_cosine_schedule_with_warmup
from utils.scheduler import get_cosine_schedule_with_warmup

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our model and data preprocessing
from models.cnn import Simple1DCNN, GestureDataset
from data_utils.data_preprocessing import (
    prepare_data_kfold,
    prepare_data_single_split,
)

# Directory holding all models, scalers, summaries
WEIGHT_DIR = os.path.join(SUBM_DIR, 'weights')
os.makedirs(WEIGHT_DIR, exist_ok=True)

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
    MODIFIED: Now also returns predictions and targets for metric calculation.
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad() # reset gradients
        output = model(data) # forward pass
        loss = criterion(output, target) # compute loss
        loss.backward() #  compute new gradients for each parameter
        optimizer.step() # update weights
        
        if scheduler is not None:
            scheduler.step() # update learning rate
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = 100. * (all_preds == all_targets).sum() / len(all_targets)
    
    return avg_loss, accuracy, all_preds, all_targets


def validate_and_evaluate_epoch(model, dataloader, criterion, device, label_encoder):
    """
    Validate for one epoch and compute all relevant metrics.
    
    Returns:
        avg_loss (float): Average validation loss.
        accuracy (float): Validation accuracy.
        comp_score (float): Competition metric score.
        all_preds (list): All predictions for the epoch.
        all_targets (list): All ground truth targets for the epoch.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
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


def train_model(model, train_loader, val_loader, label_encoder, epochs=50, start_lr=0.001, device='cpu', variant: str = 'full', fold_tag: str = ''):
    """Train the model with validation, using competition metric for model selection."""
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
    patience = 15
    patience_counter = 0
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train and get predictions for score calculation
        train_loss, train_acc, train_preds, train_targets = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        train_score = competition_metric(train_targets, train_preds, label_encoder)
        
        # Validate
        val_loss, val_acc, val_score, _, _ = validate_and_evaluate_epoch(model, val_loader, criterion, device, label_encoder)
        
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
    
    return history


def evaluate_model(model, dataloader, label_encoder, device='cpu'):
    """Evaluate model and print detailed metrics including competition score."""
    model.eval()
    
    # Use the validation function to get all metrics
    _, _, comp_score, all_preds, all_targets = validate_and_evaluate_epoch(model, dataloader, nn.CrossEntropyLoss(), device, label_encoder)

    print(f"\n🏆 Final Competition Score on Validation Set: {comp_score:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # Accuracy by class
    print("\nAccuracy by class:")
    for i, class_name in enumerate(label_encoder.classes_):
        mask = np.array(all_targets) == i
        if np.sum(mask) > 0:
            acc = np.mean(np.array(all_preds)[mask] == np.array(all_targets)[mask])
            print(f"{class_name}: {acc:.3f} ({np.sum(mask)} samples)")
    
    return all_preds, all_targets


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history, including the new competition score."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    # Competition Score
    ax1.plot(history['train_competition_score'], label='Train Score', color='blue')
    ax1.plot(history['val_competition_score'], label='Val Score', color='orange')
    ax1.set_title('Competition Score')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Loss
    ax3.plot(history['train_loss'], label='Train Loss')
    ax3.plot(history['val_loss'], label='Val Loss')
    ax3.set_title('Model Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history plot saved to {save_path}")


def train_single_model(epochs=50, start_lr=0.001, device=None, variant: str = 'full'):
    """Train a single model with single train/val split"""
    print("🎯 TRAINING SINGLE MODEL")
    print("="*60)
    
    # Setup device
    if device is None:
        device = setup_device()
    
    # Prepare data
    print("Preparing data for single model training...")
    X_train, X_val, y_train, y_val, label_encoder = prepare_data_single_split(variant)
    
    # Model parameters
    input_channels = X_train.shape[2]
    sequence_length = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    
    print(f"\nModel configuration:")
    print(f"Input shape: {X_train.shape}")
    print(f"Input channels (features): {input_channels}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {num_classes}")
    
    # Create datasets and dataloaders
    train_dataset = GestureDataset(X_train, y_train)
    val_dataset = GestureDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Build model
    print("\nBuilding model...")
    model = Simple1DCNN(input_channels, num_classes, sequence_length)
    model = model.to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\nModel Summary:")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Model size: {model_info['model_size_mb']:.1f} MB")
    
    # Train model
    print("\nTraining model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_encoder=label_encoder,
        epochs=epochs,
        start_lr=start_lr,
        device=device,
        variant=variant,
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred, y_true = evaluate_model(model, val_loader, label_encoder, device)
    
    # Plot training history
    plot_training_history(history, os.path.join(WEIGHT_DIR, f'single_model_history_{variant}.png'))
    
    # Save final model
    model_path = os.path.join(WEIGHT_DIR, f'single_model_{variant}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved as '{model_path}'")
    
    return model, history


def train_kfold_models(epochs=50, start_lr=0.001, show_stratification=False, device=None, variant: str = 'full'):
    """Train 5 models using 5-fold cross-validation"""
    print("="*60)
    print("TRAINING 5 MODELS WITH 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Setup device
    if device is None:
        device = setup_device()
    
    # Prepare all folds and get full labels & sequence IDs for OOF reconstruction
    fold_data, label_encoder, y_all, sequence_ids_all = prepare_data_kfold(show_stratification=show_stratification, variant=variant)
    num_samples = len(y_all)
    oof_preds = np.full(num_samples, -1, dtype=int)
    oof_targets = y_all.copy()
    
    # Model parameters (same for all folds)
    input_channels = fold_data[0]['X_train'].shape[2]
    sequence_length = fold_data[0]['X_train'].shape[1]
    num_classes = len(label_encoder.classes_)
    
    print(f"\nModel configuration:")
    print(f"Input channels (features): {input_channels}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {list(label_encoder.classes_)}")
    
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
        X_train, X_val = fold['X_train'], fold['X_val']
        y_train, y_val = fold['y_train'], fold['y_val']
        
        print(f"Fold {fold_idx + 1} - Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Create datasets and dataloaders
        train_dataset = GestureDataset(X_train, y_train)
        val_dataset = GestureDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Build model for this fold
        print(f"\nBuilding model for fold {fold_idx + 1}...")
        model = Simple1DCNN(input_channels, num_classes, sequence_length)
        
        try:
            model = model.to(device)
            print(f"Model loaded to {device}")
            
            # Train model
            print(f"\nTraining fold {fold_idx + 1}...")
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                label_encoder=label_encoder,
                epochs=epochs,
                start_lr=start_lr,
                device=device,
                variant=variant,
                fold_tag=f'_{fold_idx+1}',
            )
            
        except RuntimeError as e:
            if "cuDNN" in str(e) or "CUDA out of memory" in str(e):
                print(f"⚠️  GPU Error: {e}")
                print("🔄 Falling back to CPU...")
                
                # Clear GPU cache and move to CPU
                torch.cuda.empty_cache()
                device_fallback = torch.device('cpu')
                model = model.to(device_fallback)
                
                # Recreate data loaders with smaller batch size for CPU
                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
                
                print(f"Retrying training on CPU with batch_size=8...")
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    label_encoder=label_encoder,
                    epochs=epochs,
                    start_lr=start_lr,
                    device=device_fallback,
                    variant=variant,
                    fold_tag=f'_{fold_idx+1}',
                )
                device = device_fallback  # Update device for evaluation
            else:
                raise e
        
        # Evaluate model and capture predictions for OOF
        print(f"\nEvaluating fold {fold_idx + 1}...")
        _, _, _, all_preds_val, _ = validate_and_evaluate_epoch(model, val_loader, nn.CrossEntropyLoss(), device, label_encoder)
        
        # Store into OOF array using original indices
        oof_preds[fold['val_idx']] = all_preds_val
        
        # Calculate final validation score
        best_val_score = max(history['val_competition_score'])
        
        # Save model for this fold
        model_filename = os.path.join(WEIGHT_DIR, f'model_fold_{fold_idx + 1}_{variant}.pth')
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as '{model_filename}'")
        
        # Save the scaler corresponding to this fold for inference pairing
        scaler_fold = fold['scaler']
        
        # Create outputs directory if it doesn't exist
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        
        scaler_filename = os.path.join(WEIGHT_DIR, f'scaler_fold_{fold_idx + 1}_{variant}.pkl')
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler_fold, f)
        print(f"Scaler saved as '{scaler_filename}'")
        
        # Store results
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_score': best_val_score,
            'model_filename': model_filename,
            'train_subjects': len(set(fold['train_subjects'])),
            'val_subjects': len(set(fold['val_subjects']))
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
    
    # Copy best model as the main model
    best_model_filename = fold_results[best_fold_idx]['model_filename']
    dest_best = os.path.join(WEIGHT_DIR, f'best_model_{variant}.pth')
    shutil.copy(best_model_filename, dest_best)
    print(f"Best model copied to '{dest_best}'")
    
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
    """Main training function with command line interface"""
    parser = argparse.ArgumentParser(description="Gesture Recognition Training")
    parser.add_argument('--single', action='store_true', help='Train single model instead of 5-fold CV')
    parser.add_argument('--stratification', action='store_true', help='Show stratification details')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--start_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--gpu', type=int, help='Specific GPU ID to use')
    parser.add_argument('--variant', choices=['full', 'imu'], default='full', help='Sensor variant to train (full or imu-only)')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device(args.gpu)
    
    # Train model(s)
    if args.single:
        model, history = train_single_model(
            epochs=args.epochs, 
            start_lr=args.start_lr, 
            device=device,
            variant=args.variant,
        )
        print("✅ Single model training completed!")
        return model, history
    else:
        fold_models, fold_histories, fold_results = train_kfold_models(
            epochs=args.epochs, 
            start_lr=args.start_lr, 
            show_stratification=args.stratification,
            device=device,
            variant=args.variant,
        )
        print("✅ 5-fold cross-validation training completed!")
        return fold_models, fold_histories, fold_results


if __name__ == "__main__":
    results = main()