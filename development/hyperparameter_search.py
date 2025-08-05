"""
# hyperparameter_search.py
进行超参数搜索。会在本地产生一个db文件记录搜索结果，可在一次搜索后加载其自动搜索。
它会自动读取配置文件中的 'variant' ('imu' 或 'full')
"""
# ----------------- 用户配置 -----------------
# 脚本将自动从该文件中读取 'variant' 并调整行为
CONFIG_FILE_PATH = r'cmi-submission/configs/multimodality_model_v3_full_config.py'
# 你想要运行的试验次数
N_TRIALS = 100
# 初始随机搜索的次数
N_STARTUP_TRIALS = 30
DB_PATH = 'sqlite:///full-v1.db'
# ----------------------------------------------------

import optuna
from optuna.samplers import TPESampler
import train  # 导入你修改后的 train.py
import torch
import copy
import sys
import os
import shutil
import json
import contextlib

# --- Path Definitions ---
# The final best models will be saved directly in the main weights directory
FINAL_WEIGHTS_DIR = train.WEIGHT_DIR
# Intermediate artifacts for each trial will be stored here and cleaned up later
TRIAL_ARTIFACTS_DIR = os.path.join(FINAL_WEIGHTS_DIR, 'trial_artifacts')
LOG_DIR = os.path.join('logs')
os.makedirs(TRIAL_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

@contextlib.contextmanager
def manage_output(log_path=None):
    """
    A context manager to redirect stdout and stderr.
    If log_path is provided, output is written to that file.
    Otherwise, it is suppressed.
    """
    if log_path:
        target = open(log_path, 'w', encoding='utf-8')
    else:
        target = open(os.devnull, 'w', encoding='utf-8')
    
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = target, target
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        target.close()

def save_best_model_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """
    Optuna 回调函数，用于在找到新的最佳试验时保存模型文件。
    """
    if study.best_trial.number == trial.number:
        print(f"\nNew best trial found: #{trial.number} with score {trial.value:.4f}. Saving models.")
        
        # --- NEW LOGIC: Copy from trial artifacts to the final weights directory ---
        # 1. Copy all 5 model folds
        for i in range(1, 6):
            src_path = os.path.join(TRIAL_ARTIFACTS_DIR, f'model_fold_{i}_{variant}.pth')
            dst_path = os.path.join(FINAL_WEIGHTS_DIR, f'model_fold_{i}_{variant}.pth')
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

        # 2. Copy all 5 scaler folds
        for i in range(1, 6):
            src_path = os.path.join(TRIAL_ARTIFACTS_DIR, f'scaler_fold_{i}_{variant}.pkl')
            dst_path = os.path.join(FINAL_WEIGHTS_DIR, f'scaler_fold_{i}_{variant}.pkl')
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

        # 3. Copy the global label encoder
        src_path = os.path.join(TRIAL_ARTIFACTS_DIR, f'label_encoder_{variant}.pkl')
        dst_path = os.path.join(FINAL_WEIGHTS_DIR, f'label_encoder_{variant}.pkl')
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

        # 4. Save the best hyperparameters to a JSON file in the final directory
        best_params_path = os.path.join(FINAL_WEIGHTS_DIR, 'best_params.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(trial.params, f, indent=4)
            
        # 5. Copy the best trial's log file to the final directory
        log_dir = os.path.join(LOG_DIR, study.study_name)
        src_log_path = os.path.join(log_dir, f'trial_{trial.number}.log')
        dst_log_path = os.path.join(FINAL_WEIGHTS_DIR, 'training_log.log')
        if os.path.exists(src_log_path):
            shutil.copy(src_log_path, dst_log_path)
        
        print(f"Best models and parameters for trial #{trial.number} saved to {FINAL_WEIGHTS_DIR}")


def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna 的目标函数。
    """
    cfg = train.load_py_config(CONFIG_FILE_PATH)
    variant = cfg.data.get('variant')
    
    # 将 variant 信息存入 trial，方便 callback 函数使用
    trial.set_user_attr('variant', variant)
    
    print(f"\nStarting Trial {trial.number} for '{variant}' branch")

    # ===================================================================
    #                  1. 定义通用超参数的搜索空间
    # ===================================================================

    cfg.training['epochs'] = trial.suggest_int('epochs', 20, 150)
    cfg.training['patience'] = trial.suggest_int('patience', 10, 30, step=5)
    cfg.training['start_lr'] = trial.suggest_float('start_lr', 1e-4, 1e-2, log=True)
    cfg.training['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    # --- Mixup Augmentation ---
    cfg.training['mixup_enabled'] = trial.suggest_categorical('mixup_enabled', [True, False])
    if cfg.training['mixup_enabled']:
        # If mixup is on, search for a meaningful alpha
        cfg.training['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.1, 0.5)
    else:
        # If mixup is off, alpha is irrelevant but should be 0
        cfg.training['mixup_alpha'] = 0.0

    # --- NEW: Scheduler Choice ---
    scheduler_type = trial.suggest_categorical('scheduler_type', ['cosine', 'reduce_on_plateau'])
    scheduler_cfg = {'type': scheduler_type}
    if scheduler_type == 'reduce_on_plateau':
        scheduler_cfg['factor'] = trial.suggest_float('lr_reduce_factor', 0.2, 0.8, step=0.1)
        scheduler_cfg['patience'] = trial.suggest_int('lr_patience', 5, 15, step=2)
        scheduler_cfg['min_lr'] = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)
        scheduler_cfg['warmup_ratio'] = trial.suggest_float('warmup_ratio_plateau', 0.0, 0.2) # Warmup for plateau
    elif scheduler_type == 'cosine':
        scheduler_cfg['warmup_ratio'] = trial.suggest_float('warmup_ratio_cosine', 0.0, 0.2)
    cfg.training['scheduler_cfg'] = scheduler_cfg
    # --- End of Scheduler Choice ---

    # --- Discriminative LR multipliers ---
    lr_mult = {}
    lr_mult['imu'] = trial.suggest_float('lr_mult_imu', 0.5, 1.5)  # Tune IMU LR multiplier
    lr_mult['mlp'] = trial.suggest_float('lr_mult_mlp', 1.0, 3.0)
    if variant == 'full':
        lr_mult['thm'] = trial.suggest_float('lr_mult_thm', 0.5, 1.5)
        lr_mult['tof'] = trial.suggest_float('lr_mult_tof', 0.1, 1.0)
    lr_mult['fusion'] = trial.suggest_float('lr_mult_fusion', 0.1, 2.0)
    scheduler_cfg['lr_multipliers'] = lr_mult


    num_imu_layers = trial.suggest_int('num_imu_layers', 2, 4)
    imu_filters = []
    imu_kernel_sizes = []
    for i in range(num_imu_layers):
        imu_filters.append(trial.suggest_int(f'imu_filter_{i}', 32, 1024, step=16))
        imu_kernel_sizes.append(trial.suggest_categorical(f'imu_kernel_{i}', [3, 5, 7, 9, 11]))
    cfg.model['imu_branch_cfg']['filters'] = imu_filters
    cfg.model['imu_branch_cfg']['kernel_sizes'] = imu_kernel_sizes
    
    num_mlp_hidden_layers = trial.suggest_int('num_mlp_hidden_layers', 1, 3)
    mlp_hidden_dims = []
    for i in range(num_mlp_hidden_layers):
        mlp_hidden_dims.append(trial.suggest_int(f'mlp_hidden_dim_{i}', 16, 128, step=16))
    cfg.model['mlp_branch_cfg']['hidden_dims'] = mlp_hidden_dims
    cfg.model['mlp_branch_cfg']['output_dim'] = trial.suggest_int('mlp_output_dim', 16, 128, step=16)
    cfg.model['mlp_branch_cfg']['dropout_rate'] = trial.suggest_float('mlp_dropout_rate', 0.1, 0.5)

    # ===================================================================
    #          2. 为 "full" 分支定义额外的超参数搜索空间
    # ===================================================================
    if variant == 'full':
        num_thm_layers = trial.suggest_int('num_thm_layers', 1, 4)
        thm_filters = []
        thm_kernel_sizes = []
        for i in range(num_thm_layers):
            thm_filters.append(trial.suggest_int(f'thm_filter_{i}', 16, 128, step=16))
            thm_kernel_sizes.append(trial.suggest_categorical(f'thm_kernel_{i}', [3, 5, 7]))
        cfg.model['thm_branch_cfg']['filters'] = thm_filters
        cfg.model['thm_branch_cfg']['kernel_sizes'] = thm_kernel_sizes

        tof_temporal_mode = trial.suggest_categorical('tof_temporal_mode', ['lstm', 'transformer'])
        cfg.model['tof_branch_cfg']['temporal_mode'] = tof_temporal_mode
        
        # --- Safe TOF CNN architecture search ---
        # Restrict to 2 or 3 layers. 4 layers is too deep for 8×8 input with hard-coded pooling.
        num_tof_cnn_layers = trial.suggest_categorical('num_tof_cnn_layers', [2, 3])

        if num_tof_cnn_layers == 2:
            # Any kernel combination is safe for 2 layers.
            tof_kernel_sizes = [
                trial.suggest_categorical('tof_kernel_0', [2, 3]),
                trial.suggest_categorical('tof_kernel_1', [2, 3]),
            ]
        else:  # 3 layers → use a verified safe progression 3→3→2 to avoid 1×1 before final conv
            tof_kernel_sizes = [3, 3, 2]

        # Channel width search (use lower max channels for deeper nets to save memory)
        tof_conv_channels = []
        max_ch = 256 if num_tof_cnn_layers == 2 else 128
        for i in range(num_tof_cnn_layers):
            tof_conv_channels.append(trial.suggest_int(f'tof_conv_channel_{i}', 32, max_ch, step=32))

        cfg.model['tof_branch_cfg']['conv_channels'] = tof_conv_channels
        cfg.model['tof_branch_cfg']['kernel_sizes'] = tof_kernel_sizes
        # --- End safe TOF CNN search ---
        
        if tof_temporal_mode == 'transformer':
            cfg.model['tof_branch_cfg']['num_heads'] = trial.suggest_categorical('tof_num_heads', [4, 8])
            cfg.model['tof_branch_cfg']['num_layers'] = trial.suggest_int('tof_num_layers', 1, 3)
            cfg.model['tof_branch_cfg']['ff_dim'] = trial.suggest_int('tof_ff_dim', 256, 1024, step=128)
            cfg.model['tof_branch_cfg']['dropout'] = trial.suggest_float('tof_dropout', 0.1, 0.4)
        elif tof_temporal_mode == 'lstm':
            cfg.model['tof_branch_cfg']['lstm_hidden'] = trial.suggest_int('tof_lstm_hidden', 64, 256, step=32)
            cfg.model['tof_branch_cfg']['lstm_layers'] = trial.suggest_int('tof_lstm_layers', 1, 2)
            cfg.model['tof_branch_cfg']['bidirectional'] = trial.suggest_categorical('tof_bidirectional', [True, False])

        # ===================================================================
    #          3. 定义 Fusion Head 的搜索空间 (MLP-only)
    # ===================================================================
    # Force MLP fusion and remove Transformer from the search space
    fusion_type = 'mlp'
    
    for key in ['hidden_dims', 'dropout_rates', 'branch_dims', 'embed_dim', 'num_heads', 'depth', 'dropout']:
        cfg.model['fusion_head_cfg'].pop(key, None)

    cfg.model['fusion_head_cfg']['type'] = 'FusionHead'
    num_fusion_layers = trial.suggest_int('mlp_fusion_layers', 1, 3)
    fusion_hidden_dims = []
    fusion_dropout_rates = []
    for i in range(num_fusion_layers):
        fusion_hidden_dims.append(trial.suggest_int(f'mlp_fusion_hidden_dim_{i}', 32, 512, step=16))
        fusion_dropout_rates.append(trial.suggest_float(f'mlp_fusion_dropout_{i}', 0.1, 0.5))
    cfg.model['fusion_head_cfg']['hidden_dims'] = fusion_hidden_dims
    cfg.model['fusion_head_cfg']['dropout_rates'] = fusion_dropout_rates
    
    try:
        # ===================================================================
        #                          4. 运行训练流程
        # ===================================================================
        # Create log directory for the study
        log_dir = os.path.join(LOG_DIR, study.study_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'trial_{trial.number}.log')

        # Write hyperparameters to the top of the log file
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"TRIAL {trial.number} PARAMETERS\n")
            f.write("="*60 + "\n")
            f.write(json.dumps(trial.params, indent=4))
            f.write("\n\n" + "="*60 + "\n")
            f.write("TRAINING LOG\n")
            f.write("="*60 + "\n")

        with manage_output(log_path):
            device = train.setup_device(cfg.environment.get('gpu_id'))
            oof_score, _, _, _ = train.train_kfold_models(
                epochs=cfg.training['epochs'], patience=cfg.training['patience'],
                start_lr=cfg.training['start_lr'], weight_decay=cfg.training['weight_decay'],
                batch_size=cfg.data['batch_size'], use_amp=cfg.training['use_amp'],
                variant=cfg.data['variant'], model_cfg=cfg.model,
                mixup_enabled=cfg.training['mixup_enabled'], mixup_alpha=cfg.training['mixup_alpha'],
                scheduler_cfg=cfg.training['scheduler_cfg'],
                device=device, show_stratification=False, loss_function='ce',
                output_dir=TRIAL_ARTIFACTS_DIR  # IMPORTANT: Save to temp dir
            )
        
        torch.cuda.empty_cache()
        return oof_score

    except Exception as e:
        print(f"Trial {trial.number} failed with an exception: {e}")
        import traceback
        traceback.print_exc()
        return -1.0


if __name__ == "__main__":
    # ===================================================================
    #      1. 加载配置, 并根据 'variant' 动态设置研究参数
    # ===================================================================
    base_cfg = train.load_py_config(CONFIG_FILE_PATH)
    variant = base_cfg.data.get('variant')
    if not variant:
        raise ValueError(f"'variant' not found in config file: {CONFIG_FILE_PATH}")
    
    print("="*60)
    print(f"Detected variant '{variant}'.")
    print(f"Database:   {DB_PATH}")
    print("="*60)

    # ===================================================================
    #      2. 准备数据并应用猴子补丁
    # ===================================================================
    print("Preparing and pre-loading data... This will happen only once.")
    preloaded_data_tuple = train.prepare_data_kfold_multimodal(
        show_stratification=False, 
        variant=variant
    )
    print("Data has been pre-loaded into memory.")
    print("="*60)

    original_prepare_data_func = train.prepare_data_kfold_multimodal

    def mock_prepare_data_kfold_multimodal(*args, **kwargs):
        """
        使用猴子补丁替换原始的函数，以避免每次试验都重新加载数据，加快试验速度。
        """
        return preloaded_data_tuple

    train.prepare_data_kfold_multimodal = mock_prepare_data_kfold_multimodal
    print("Monkey patch applied.")

    # ===================================================================
    #      3. 设置并运行 Optuna 研究
    # ===================================================================
    try:
        sampler = TPESampler(multivariate=True, n_startup_trials=N_STARTUP_TRIALS, seed=42)
        study = optuna.create_study(
            direction='maximize',
            storage=DB_PATH,
            study_name="hyperparameter_search",
            load_if_exists=True,
            sampler=sampler,
        )
        study.optimize(objective, n_trials=N_TRIALS, callbacks=[save_best_model_callback])

        print("\n\n" + "="*60)
        print("Optuna Search Finished!")
        best_trial = study.best_trial
        print(f"Best Trial Score (OOF Competition Metric): {best_trial.value:.4f}")
        print("\nBest parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

    finally:
        # --- Final Cleanup ---
        # Clean up the temporary trial artifacts directory after the search is complete
        if os.path.exists(TRIAL_ARTIFACTS_DIR):
            shutil.rmtree(TRIAL_ARTIFACTS_DIR)
            print(f"\nCleaned up temporary artifacts directory: {TRIAL_ARTIFACTS_DIR}")

        train.prepare_data_kfold_multimodal = original_prepare_data_func
        print("\nMonkey patch restored. Original functions are back.")