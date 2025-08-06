"""
# hyperparameter_search.py
进行超参数搜索。会在本地产生一个db文件记录搜索结果，可在一次搜索后加载其自动搜索。
它会自动读取配置文件中的 'variant' ('imu' 或 'full')
"""
import optuna
from optuna.samplers import TPESampler
import torch
import copy
import sys
import os
import shutil
import json
import contextlib

# Import train script AFTER basic imports to access its functions
import train

# ----------------- 用户配置 -----------------
# 脚本将自动从该文件中读取 'variant' 并调整行为
CONFIG_FILE_PATH = r'cmi-submission/configs/multimodality_model_v3_full_config.py'
# 你想要运行的试验次数
N_TRIALS = 100
# 初始随机搜索的次数
N_STARTUP_TRIALS = 30
# Get variant from config to dynamically set paths and study names
base_cfg = train.load_py_config(CONFIG_FILE_PATH)
variant = base_cfg.data.get('variant')
if not variant:
    raise ValueError(f"'variant' not found in config file: {CONFIG_FILE_PATH}")

DB_PATH = f'sqlite:///{variant}-study.db'
STUDY_NAME = f'{variant}_search'

# --- Path Definitions ---
# The final best models will be saved directly in the main weights directory
FINAL_WEIGHTS_DIR = train.WEIGHT_DIR
# Intermediate artifacts for each trial will be stored here and cleaned up later
TRIAL_ARTIFACTS_DIR = os.path.join(FINAL_WEIGHTS_DIR, f'trial_artifacts_{variant}')
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
        target = open(log_path, 'a', encoding='utf-8')  # Use append mode 'a' to preserve header
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
        
        # Get variant from the trial's user attributes
        variant = trial.user_attrs.get('variant', 'unknown')

        # --- NEW LOGIC: Copy from trial artifacts to the final weights directory ---
        # Define a list of files to copy to handle missing files gracefully
        files_to_copy = (
            [f'model_fold_{i}_{variant}.pth' for i in range(1, 6)] +
            [f'scaler_fold_{i}_{variant}.pkl' for i in range(1, 6)] +
            [f'label_encoder_{variant}.pkl', f'kfold_summary_{variant}.json', f'oof_predictions_{variant}.csv']
        )

        for filename in files_to_copy:
            src_path = os.path.join(TRIAL_ARTIFACTS_DIR, filename)
            dst_path = os.path.join(FINAL_WEIGHTS_DIR, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

        # 4. Save the best hyperparameters to a JSON file (with variant)
        best_params_path = os.path.join(FINAL_WEIGHTS_DIR, f'best_params_{variant}.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(trial.params, f, indent=4)
            
        # 5. Copy the best trial's log file (with variant)
        log_dir = os.path.join(LOG_DIR, study.study_name)
        src_log_path = os.path.join(log_dir, f'trial_{trial.number}.log')
        dst_log_path = os.path.join(FINAL_WEIGHTS_DIR, f'training_log_{variant}.log')
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
    cfg.training['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    # --- Mixup Augmentation ---
    cfg.training['mixup_enabled'] = trial.suggest_categorical('mixup_enabled', [True, False])
    if cfg.training['mixup_enabled']:
        cfg.training['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.1, 0.5)
    else:
        cfg.training['mixup_alpha'] = 0.0

    # --- Scheduler Choice ---
    scheduler_type = trial.suggest_categorical('scheduler_type', ['cosine', 'reduce_on_plateau'])
    scheduler_cfg = {'type': scheduler_type}
    if scheduler_type == 'reduce_on_plateau':
        scheduler_cfg['factor'] = trial.suggest_float('lr_reduce_factor', 0.2, 0.8, step=0.1)
        scheduler_cfg['patience'] = trial.suggest_int('lr_patience', 5, 15, step=2)
        scheduler_cfg['min_lr'] = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)
        scheduler_cfg['warmup_ratio'] = trial.suggest_float('warmup_ratio_plateau', 0.0, 0.2)
    elif scheduler_type == 'cosine':
        scheduler_cfg['warmup_ratio'] = trial.suggest_float('warmup_ratio_cosine', 0.0, 0.2)
    
    # --- NEW: Search for absolute learning rates per branch ---
    layer_lrs = {}
    layer_lrs['imu'] = trial.suggest_float('lr_imu', 1e-5, 1e-2, log=True)
    layer_lrs['mlp'] = trial.suggest_float('lr_mlp', 1e-5, 1e-2, log=True)
    if variant == 'full':
        layer_lrs['thm'] = trial.suggest_float('lr_thm', 1e-5, 1e-2, log=True)
        layer_lrs['tof'] = trial.suggest_float('lr_tof', 1e-5, 1e-2, log=True)
    layer_lrs['fusion'] = trial.suggest_float('lr_fusion', 1e-5, 1e-2, log=True)
    scheduler_cfg['layer_lrs'] = layer_lrs

    cfg.training['scheduler_cfg'] = scheduler_cfg
    # --- End of LR Search ---

    # --- IMU Branch Architecture ---
    num_imu_layers = trial.suggest_int('num_imu_layers', 2, 4)
    imu_filters = []
    imu_kernel_sizes = []
    for i in range(num_imu_layers):
        imu_filters.append(trial.suggest_int(f'imu_filter_{i}', 32, 1536, step=16))
        imu_kernel_sizes.append(trial.suggest_categorical(f'imu_kernel_{i}', [3, 5, 7, 9, 11]))
    cfg.model['imu_branch_cfg']['filters'] = imu_filters
    cfg.model['imu_branch_cfg']['kernel_sizes'] = imu_kernel_sizes
    
    # --- NEW: IMU Temporal Aggregation Choice ---
    imu_temporal_aggregation = trial.suggest_categorical('imu_temporal_aggregation', ['global_pool', 'temporal_encoder'])
    cfg.model['imu_branch_cfg']['temporal_aggregation'] = imu_temporal_aggregation
    
    if imu_temporal_aggregation == 'temporal_encoder':
        cfg.model['imu_branch_cfg']['temporal_mode'] = trial.suggest_categorical('imu_temporal_mode', ['lstm', 'transformer'])
        if cfg.model['imu_branch_cfg']['temporal_mode'] == 'lstm':
            cfg.model['imu_branch_cfg']['lstm_hidden'] = trial.suggest_int('imu_lstm_hidden', 128, 512, step=64)
            cfg.model['imu_branch_cfg']['bidirectional'] = trial.suggest_categorical('imu_bidirectional', [True, False])
        else:  # transformer
            cfg.model['imu_branch_cfg']['num_heads'] = trial.suggest_categorical('imu_num_heads', [4, 8])
            cfg.model['imu_branch_cfg']['num_layers'] = trial.suggest_int('imu_num_layers', 1, 3)
    
    # --- NEW: IMU Residual Connections ---
    cfg.model['imu_branch_cfg']['use_residual'] = trial.suggest_categorical('imu_use_residual', [True, False])
    
    num_mlp_hidden_layers = trial.suggest_int('num_mlp_hidden_layers', 1, 3)
    mlp_hidden_dims = []
    for i in range(num_mlp_hidden_layers):
        mlp_hidden_dims.append(trial.suggest_int(f'mlp_hidden_dim_{i}', 16, 256, step=16))
    cfg.model['mlp_branch_cfg']['hidden_dims'] = mlp_hidden_dims
    cfg.model['mlp_branch_cfg']['output_dim'] = trial.suggest_int('mlp_output_dim', 16, 128, step=16)
    cfg.model['mlp_branch_cfg']['dropout_rate'] = trial.suggest_float('mlp_dropout_rate', 0.1, 0.5)

    # ===================================================================
    #          2. 为 "full" 分支定义额外的超参数搜索空间
    # ===================================================================
    if variant == 'full':
        # --- THM Branch Architecture ---
        num_thm_layers = trial.suggest_int('num_thm_layers', 1, 4)
        thm_filters = []
        thm_kernel_sizes = []
        for i in range(num_thm_layers):
            thm_filters.append(trial.suggest_int(f'thm_filter_{i}', 16, 512, step=16))
            thm_kernel_sizes.append(trial.suggest_categorical(f'thm_kernel_{i}', [3, 5, 7]))
        cfg.model['thm_branch_cfg']['filters'] = thm_filters
        cfg.model['thm_branch_cfg']['kernel_sizes'] = thm_kernel_sizes
        
        # --- NEW: THM Temporal Aggregation Choice ---
        thm_temporal_aggregation = trial.suggest_categorical('thm_temporal_aggregation', ['global_pool', 'temporal_encoder'])
        cfg.model['thm_branch_cfg']['temporal_aggregation'] = thm_temporal_aggregation
        
        if thm_temporal_aggregation == 'temporal_encoder':
            cfg.model['thm_branch_cfg']['temporal_mode'] = trial.suggest_categorical('thm_temporal_mode', ['lstm', 'transformer'])
            if cfg.model['thm_branch_cfg']['temporal_mode'] == 'lstm':
                cfg.model['thm_branch_cfg']['lstm_hidden'] = trial.suggest_int('thm_lstm_hidden', 64, 256, step=32)
                cfg.model['thm_branch_cfg']['bidirectional'] = trial.suggest_categorical('thm_bidirectional', [True, False])
            else:  # transformer
                cfg.model['thm_branch_cfg']['num_heads'] = trial.suggest_categorical('thm_num_heads', [4, 8])
                cfg.model['thm_branch_cfg']['num_layers'] = trial.suggest_int('thm_num_layers', 1, 2)
        
        # --- NEW: THM Residual Connections ---
        cfg.model['thm_branch_cfg']['use_residual'] = trial.suggest_categorical('thm_use_residual', [True, False])

        tof_temporal_mode = trial.suggest_categorical('tof_temporal_mode', ['lstm', 'transformer'])
        cfg.model['tof_branch_cfg']['temporal_mode'] = tof_temporal_mode
        
        num_tof_cnn_layers = trial.suggest_categorical('num_tof_cnn_layers', [2, 3])

        if num_tof_cnn_layers == 2:
            tof_kernel_sizes = [
                trial.suggest_categorical('tof_kernel_0', [2, 3]),
                trial.suggest_categorical('tof_kernel_1', [2, 3]),
            ]
        else:
            tof_kernel_sizes = [3, 3, 2]

        tof_conv_channels = []
        max_ch = 256 if num_tof_cnn_layers == 2 else 128
        for i in range(num_tof_cnn_layers):
            tof_conv_channels.append(trial.suggest_int(f'tof_conv_channel_{i}', 32, max_ch, step=32))
    
        cfg.model['tof_branch_cfg']['conv_channels'] = tof_conv_channels
        cfg.model['tof_branch_cfg']['kernel_sizes'] = tof_kernel_sizes
        
        # --- NEW: TOF Residual Connections ---
        cfg.model['tof_branch_cfg']['use_residual'] = trial.suggest_categorical('tof_use_residual', [True, False])
        
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
        log_dir = os.path.join(LOG_DIR, study.study_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'trial_{trial.number}.log')

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
                weight_decay=cfg.training['weight_decay'],
                batch_size=cfg.data['batch_size'], use_amp=cfg.training['use_amp'],
                variant=cfg.data['variant'], model_cfg=cfg.model,
                mixup_enabled=cfg.training['mixup_enabled'], mixup_alpha=cfg.training['mixup_alpha'],
                scheduler_cfg=cfg.training['scheduler_cfg'],
                device=device, show_stratification=False, loss_function='ce',
                output_dir=TRIAL_ARTIFACTS_DIR
            )
        
        torch.cuda.empty_cache()
        return oof_score

    except Exception as e:
        print(f"Trial {trial.number} failed with an exception: {e}")
        import traceback
        traceback.print_exc()
        return -1.0


if __name__ == "__main__":
    print("="*60)
    print(f"Detected variant '{variant}'.")
    print(f"Database:   {DB_PATH}")
    print(f"Study Name: {STUDY_NAME}")
    print("="*60)

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

    try:
        sampler = TPESampler(multivariate=True, n_startup_trials=N_STARTUP_TRIALS, seed=42)
        study = optuna.create_study(
            direction='maximize',
            storage=DB_PATH,
            study_name=STUDY_NAME,
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
        if os.path.exists(TRIAL_ARTIFACTS_DIR):
            shutil.rmtree(TRIAL_ARTIFACTS_DIR)
            print(f"\nCleaned up temporary artifacts directory: {TRIAL_ARTIFACTS_DIR}")

        train.prepare_data_kfold_multimodal = original_prepare_data_func
        print("\nMonkey patch restored. Original functions are back.")
