#!/usr/bin/env python3
"""
Advanced Hyperparameter Tuning with Optuna

Uses Bayesian optimization to intelligently search for the best hyperparameters.
Much more efficient than grid search!

Install: pip install optuna
"""

import optuna
import os
import sys
import tempfile
import json
import subprocess
from datetime import datetime

# Add cmi-submission to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'cmi-submission'))
if SUBM_DIR not in sys.path:
    sys.path.insert(0, SUBM_DIR)

def create_config_from_trial(trial):
    """Create a config based on Optuna trial suggestions"""
    
    # Define search spaces for each component
    config = {
        # 1D CNN architecture
        'cnn_num_layers': trial.suggest_int('cnn_num_layers', 2, 4),
        'cnn_base_filters': trial.suggest_categorical('cnn_base_filters', [32, 64, 128]),
        'cnn_filter_multiplier': trial.suggest_categorical('cnn_filter_multiplier', [1.5, 2.0, 2.5]),
        
        # MLP architecture  
        'mlp_num_layers': trial.suggest_int('mlp_num_layers', 1, 3),
        'mlp_base_size': trial.suggest_categorical('mlp_base_size', [64, 128, 256]),
        'mlp_output': trial.suggest_categorical('mlp_output', [32, 64, 128]),
        
        # TOF architecture
        'tof_output': trial.suggest_categorical('tof_output', [64, 128, 256]),
        'tof_base_channels': trial.suggest_categorical('tof_base_channels', [16, 32, 64]),
        
        # Fusion
        'fusion_type': trial.suggest_categorical('fusion_type', ['FusionHead', 'AttentionFusionHead']),
        'fusion_num_layers': trial.suggest_int('fusion_num_layers', 1, 3),
        'fusion_base_size': trial.suggest_categorical('fusion_base_size', [256, 512, 1024]),
        
        # Training hyperparameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    }
    
    return config

def build_architecture_lists(config):
    """Build the actual architecture lists from trial config"""
    
    # Build CNN filters list
    cnn_filters = []
    base = config['cnn_base_filters']
    multiplier = config['cnn_filter_multiplier']
    for i in range(config['cnn_num_layers']):
        cnn_filters.append(int(base * (multiplier ** i)))
    
    # Build CNN kernels (decreasing size)
    cnn_kernels = [7, 5, 3, 3][:config['cnn_num_layers']]
    
    # Build MLP hidden dims
    mlp_hidden = []
    base = config['mlp_base_size']
    for i in range(config['mlp_num_layers']):
        # Slight increase then decrease pattern
        if i == 0:
            mlp_hidden.append(base)
        else:
            mlp_hidden.append(int(base * 1.5) if i == 1 else int(base * 0.75))
    
    # Build TOF channels
    tof_base = config['tof_base_channels']
    tof_channels = [tof_base, tof_base * 2, tof_base * 4]
    
    # Build fusion hidden dims
    fusion_base = config['fusion_base_size']
    fusion_hidden = []
    for i in range(config['fusion_num_layers']):
        fusion_hidden.append(int(fusion_base / (2 ** i)))
    fusion_dropout = [config['dropout']] * config['fusion_num_layers']
    
    return {
        'cnn_filters': cnn_filters,
        'cnn_kernels': cnn_kernels,
        'mlp_hidden': mlp_hidden,
        'tof_channels': tof_channels,
        'fusion_hidden': fusion_hidden,
        'fusion_dropout': fusion_dropout,
    }

def create_temp_config(trial_config, architecture):
    """Create temporary config file"""
    
    config_content = f'''# Optuna trial config
data = dict(
    variant='full',
    max_length=100,
    batch_size={trial_config['batch_size']},
    seq_input_channels=12,
    static_input_features=7
)

model = dict(
    type='MultimodalityModel',
    num_classes=18,
    
    cnn_branch_cfg=dict(
        type='CNN1D',
        input_channels=data['seq_input_channels'],
        sequence_length=data['max_length'],
        filters={architecture['cnn_filters']},
        kernel_sizes={architecture['cnn_kernels']}
    ),
    
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=data['static_input_features'],
        hidden_dims={architecture['mlp_hidden']},
        output_dim={trial_config['mlp_output']},
        dropout_rate={trial_config['dropout']}
    ),
    
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        num_tof_sensors=5,
        seq_len=data['max_length'],
        out_features={trial_config['tof_output']},
        conv_channels={architecture['tof_channels']},
        kernel_sizes=[3, 3, 2]
    ),
    
    fusion_head_cfg=dict(
        type='{trial_config["fusion_type"]}',
        hidden_dims={architecture['fusion_hidden']},
        dropout_rates={architecture['fusion_dropout']}
    )
)

training = dict(
    epochs=20,  # Shorter for hyperparameter search
    patience=8,
    optimizer=dict(type='AdamW', lr={trial_config['learning_rate']}, weight_decay=0.01),
    loss=dict(type='FocalLoss', gamma=2.0, alpha=0.25),
    scheduler=dict(type='CosineAnnealingWarmRestarts', warmup_ratio=0.1)
)

environment = dict(gpu_id=None, seed=42)
'''
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(config_content)
        return f.name

def objective(trial):
    """Optuna objective function"""
    
    # Get trial configuration
    trial_config = create_config_from_trial(trial)
    architecture = build_architecture_lists(trial_config)
    
    # Create temporary config file
    config_path = create_temp_config(trial_config, architecture)
    
    try:
        # Run training
        cmd = f"python development/train.py --config {config_path}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            print(f"Trial failed: {result.stderr}")
            return float('inf')  # Return worst possible score
        
        # Parse competition score from output
        # You'll need to modify this based on your actual output format
        output_lines = result.stdout.split('\n')
        final_score = None
        
        for line in output_lines:
            if "Final competition score" in line or "competition_score" in line:
                # Extract score (adjust based on your output format)
                try:
                    score = float(line.split(':')[-1].strip())
                    final_score = score
                except:
                    continue
        
        if final_score is None:
            print("Could not parse competition score from output")
            return 0.0  # Default score if parsing fails
        
        return final_score  # Optuna maximizes by default
        
    except subprocess.TimeoutExpired:
        print(f"Trial timed out")
        return float('inf')
    except Exception as e:
        print(f"Trial error: {e}")
        return float('inf')
    finally:
        # Clean up temp file
        if os.path.exists(config_path):
            os.remove(config_path)

def main():
    """Run hyperparameter optimization"""
    
    print("🔍 Starting Hyperparameter Optimization with Optuna")
    print("=" * 60)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Maximize competition score
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"cmi_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Run optimization
    n_trials = 50  # Adjust based on your computational budget
    print(f"Running {n_trials} trials...")
    
    study.optimize(objective, n_trials=n_trials)
    
    # Results
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_file = f"hyperparameter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_score': study.best_value,
            'best_params': study.best_params,
            'all_trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                }
                for trial in study.trials
            ]
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # Generate best config file
    best_config = create_config_from_trial(study.best_trial)
    best_architecture = build_architecture_lists(best_config)
    best_config_path = create_temp_config(best_config, best_architecture)
    
    # Rename to permanent file
    final_config_path = f"best_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    os.rename(best_config_path, final_config_path)
    
    print(f"🎯 Best config saved to: {final_config_path}")
    print(f"\nTo use: python development/train.py --config {final_config_path}")

if __name__ == "__main__":
    main() 