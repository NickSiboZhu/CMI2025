#!/usr/bin/env python3
"""
Gesture Recognition Training Script

This script handles the complete training pipeline for gesture recognition using
1D CNN with 5-fold cross-validation and group-aware splitting.

Usage:
    python train.py                    # Full 5-fold cross-validation
    python train.py --single           # Single train/val split
    python train.py --stratification   # Show stratification details
    python train.py --gpu 1            # Use specific GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
import json
import shutil
import argparse
import sys
import os
import pickle
# from transformers import get_cosine_schedule_with_warmup
from utils.scheduler import get_cosine_schedule_with_warmup

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our model and data preprocessing
from models.cnn import Simple1DCNN, GestureDataset
from data.data_preprocessing import (
    prepare_data_kfold,
    prepare_data_single_split,
)


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
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory // 1024**3} GB)")
    
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
    print(f"  Total: {props.total_memory // 1024**3} GB")
    print(f"  Used: {allocated // 1024**2} MB")
    print(f"  Free: {free // 1024**3} GB")
    
    return device


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001, device='cpu', variant: str = 'full', fold_tag: str = ''):
    """Train the model with validation"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # Setup cosine schedule with warmup (10% warmup steps)
    total_training_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # Track learning rate for each epoch so we can inspect the schedule later
    lr_values = []
    
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train (scheduler is stepped inside train_epoch)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Current learning rate (after scheduler step). Assumes single param group.
        current_lr = optimizer.param_groups[0]['lr']
        lr_values.append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_name = f'development/outputs/best_model_tmp{("_" + variant) if variant else ""}{fold_tag}.pth'
            torch.save(model.state_dict(), ckpt_name)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s): '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
              f'LR: {current_lr:.6f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    ckpt_name = f'development/outputs/best_model_tmp{("_" + variant) if variant else ""}{fold_tag}.pth'
    model.load_state_dict(torch.load(ckpt_name))
    
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'learning_rate': lr_values,
    }
    
    return history


def evaluate_model(model, dataloader, label_encoder, device='cpu'):
    """Evaluate model and print detailed metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=label_encoder.classes_
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
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history['train_accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    
    # Loss
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history plot saved to {save_path}")


def train_single_model(epochs=50, learning_rate=0.001, device=None, variant: str = 'full'):
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
    history = train_model(model, train_loader, val_loader, epochs, learning_rate, device, variant)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred, y_true = evaluate_model(model, val_loader, label_encoder, device)
    
    # Plot training history
    plot_training_history(history, f'development/outputs/single_model_history_{variant}.png')
    
    # Save final model
    model_path = f'development/outputs/single_model_{variant}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved as '{model_path}'")
    
    return model, history


def train_kfold_models(epochs=50, learning_rate=0.001, show_stratification=False, device=None, variant: str = 'full'):
    """Train 5 models using 5-fold cross-validation"""
    print("="*60)
    print("TRAINING 5 MODELS WITH 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Setup device
    if device is None:
        device = setup_device()
    
    # Prepare all folds
    fold_data, label_encoder = prepare_data_kfold(show_stratification=show_stratification, variant=variant)
    
    # Model parameters (same for all folds)
    input_channels = fold_data[0]['X_train'].shape[2]
    sequence_length = fold_data[0]['X_train'].shape[1]
    num_classes = len(label_encoder.classes_)
    
    print(f"\nModel configuration:")
    print(f"Input channels (features): {input_channels}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {label_encoder.classes_}")
    
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
                model, train_loader, val_loader, 
                epochs=epochs, learning_rate=learning_rate, device=device, variant=variant, fold_tag=f'_{fold_idx+1}'
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
                    model, train_loader, val_loader, 
                    epochs=epochs, learning_rate=learning_rate, device=device_fallback, variant=variant, fold_tag=f'_{fold_idx+1}'
                )
                device = device_fallback  # Update device for evaluation
            else:
                raise e
        
        # Evaluate model
        print(f"\nEvaluating fold {fold_idx + 1}...")
        y_pred, y_true = evaluate_model(model, val_loader, label_encoder, device)
        
        # Calculate final validation accuracy
        final_val_acc = history['val_accuracy'][-1]
        best_val_acc = max(history['val_accuracy'])
        
        # Save model for this fold
        model_filename = f'development/outputs/model_fold_{fold_idx + 1}_{variant}.pth'
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as '{model_filename}'")
        
        # Save the scaler corresponding to this fold for inference pairing
        scaler_fold = fold['scaler']
        
        # Create outputs directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        
        scaler_filename = os.path.join(outputs_dir, f'scaler_fold_{fold_idx + 1}_{variant}.pkl')
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler_fold, f)
        print(f"Scaler saved as '{scaler_filename}'")
        
        # Store results
        fold_results.append({
            'fold': fold_idx + 1,
            'final_val_acc': final_val_acc,
            'best_val_acc': best_val_acc,
            'model_filename': model_filename,
            'train_subjects': len(set(fold['train_subjects'])),
            'val_subjects': len(set(fold['val_subjects']))
        })
        
        fold_models.append(model)
        fold_histories.append(history)
        
        print(f"Fold {fold_idx + 1} - Final Val Acc: {final_val_acc:.2f}%, Best Val Acc: {best_val_acc:.2f}%")
    
    # Summary of all folds
    print(f"\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    final_accs = [result['final_val_acc'] for result in fold_results]
    best_accs = [result['best_val_acc'] for result in fold_results]
    
    print(f"Final Validation Accuracies: {[f'{acc:.2f}%' for acc in final_accs]}")
    print(f"Best Validation Accuracies:  {[f'{acc:.2f}%' for acc in best_accs]}")
    print(f"")
    print(f"Final Acc - Mean: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%")
    print(f"Best Acc  - Mean: {np.mean(best_accs):.2f}% ± {np.std(best_accs):.2f}%")
    
    # Find best fold
    best_fold_idx = np.argmax(best_accs)
    print(f"\nBest performing fold: Fold {best_fold_idx + 1} ({best_accs[best_fold_idx]:.2f}%)")
    
    # Copy best model as the main model
    best_model_filename = fold_results[best_fold_idx]['model_filename']
    dest_best = f'development/outputs/best_model_{variant}.pth'
    shutil.copy(best_model_filename, dest_best)
    print(f"Best model copied to '{dest_best}'")
    
    # Save summary
    summary = {
        'fold_results': fold_results,
        'mean_final_acc': float(np.mean(final_accs)),
        'std_final_acc': float(np.std(final_accs)),
        'mean_best_acc': float(np.mean(best_accs)),
        'std_best_acc': float(np.std(best_accs)),
        'best_fold': int(best_fold_idx + 1),
        'best_fold_acc': float(best_accs[best_fold_idx])
    }
    
    with open(f'development/outputs/kfold_summary_{variant}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to 'development/outputs/kfold_summary_{variant}.json'")
    
    return fold_models, fold_histories, fold_results


def main():
    """Main training function with command line interface"""
    parser = argparse.ArgumentParser(description="Gesture Recognition Training")
    parser.add_argument('--single', action='store_true', help='Train single model instead of 5-fold CV')
    parser.add_argument('--stratification', action='store_true', help='Show stratification details')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpu', type=int, help='Specific GPU ID to use')
    parser.add_argument('--variant', choices=['full', 'imu'], default='full', help='Sensor variant to train (full or imu-only)')
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device(args.gpu)
    
    # Train model(s)
    if args.single:
        model, history = train_single_model(
            epochs=args.epochs, 
            learning_rate=args.lr, 
            device=device,
            variant=args.variant,
        )
        print("✅ Single model training completed!")
        return model, history
    else:
        fold_models, fold_histories, fold_results = train_kfold_models(
            epochs=args.epochs, 
            learning_rate=args.lr, 
            show_stratification=args.stratification,
            device=device,
            variant=args.variant,
        )
        print("✅ 5-fold cross-validation training completed!")
        return fold_models, fold_histories, fold_results


if __name__ == "__main__":
    results = main() 