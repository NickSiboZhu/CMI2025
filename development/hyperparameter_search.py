"""
# hyperparameter_search.py
进行超参数搜索。会在本地产生一个db文件记录搜索结果，可在一次搜索后加载其自动搜索。
它会自动读取配置文件中的 'variant' ('imu' 或 'full')
"""
# ----------------- 用户配置 -----------------
# 脚本将自动从该文件中读取 'variant' 并调整行为
CONFIG_FILE_PATH = r'cmi-submission\configs\multimodality_model_v2_imu_config.py'
# 你想要运行的试验次数
N_TRIALS = 100
# 初始随机搜索的次数
N_STARTUP_TRIALS = 30
DB_PATH = 'sqlite:///imu-v2.db'
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

# --- 路径定义 ---
WEIGHT_DIR = train.WEIGHT_DIR
BEST_MODEL_DIR = os.path.join(WEIGHT_DIR, 'best_trial_models')
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

@contextlib.contextmanager
def suppress_output():
    """
    一个上下文管理器，用于临时抑制所有的 stdout 和 stderr 输出。
    """
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def save_best_model_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    """
    Optuna 回调函数，用于在找到新的最佳试验时保存模型文件。
    """
    if study.best_trial.number == trial.number:
        print(f"\nNew best trial found: #{trial.number} with score {trial.value:.4f}. Saving models.")
        
        # 从 trial 的用户属性中获取 variant，确保与当次试验的配置一致
        variant = trial.user_attrs.get('variant', 'unknown')

        # 确保最佳模型目录是特定于本次研究的
        specific_best_model_dir = os.path.join(BEST_MODEL_DIR, study.study_name)
        os.makedirs(specific_best_model_dir, exist_ok=True)

        # 1. 复制 5 个 fold 的模型文件 (.pth)
        for i in range(1, 6):
            src_model_path = os.path.join(WEIGHT_DIR, f'model_fold_{i}_{variant}.pth')
            dst_model_path = os.path.join(specific_best_model_dir, f'model_fold_{i}_{variant}.pth')
            if os.path.exists(src_model_path):
                shutil.copy(src_model_path, dst_model_path)

        # 2. 复制 5 个 fold 的 scaler 文件 (.pkl)
        for i in range(1, 6):
            src_scaler_path = os.path.join(WEIGHT_DIR, f'scaler_fold_{i}_{variant}.pkl')
            dst_scaler_path = os.path.join(specific_best_model_dir, f'scaler_fold_{i}_{variant}.pkl')
            if os.path.exists(src_scaler_path):
                shutil.copy(src_scaler_path, dst_scaler_path)
        
        # 3. 复制全局 label encoder 文件 (.pkl)
        src_le_path = os.path.join(WEIGHT_DIR, f'label_encoder_{variant}.pkl')
        dst_le_path = os.path.join(specific_best_model_dir, f'label_encoder_{variant}.pkl')
        if os.path.exists(src_le_path):
            shutil.copy(src_le_path, dst_le_path)
            
        # 4. 将最佳参数保存到一个 JSON 文件中
        best_params_path = os.path.join(specific_best_model_dir, 'best_params.json')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            json.dump(trial.params, f, indent=4)
        
        print(f"Best models and parameters for trial #{trial.number} saved to {specific_best_model_dir}")


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
    cfg.training['start_lr'] = trial.suggest_float('start_lr', 1e-4, 1e-2, log=True)
    cfg.training['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    cfg.training['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.0, 0.4)

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
        
        num_tof_cnn_layers = trial.suggest_int('num_tof_cnn_layers', 2, 4)
        tof_conv_channels = []
        tof_kernel_sizes = []
        for i in range(num_tof_cnn_layers):
            tof_conv_channels.append(trial.suggest_int(f'tof_conv_channel_{i}', 32, 412, step=32))
            tof_kernel_sizes.append(trial.suggest_categorical(f'tof_kernel_{i}', [2, 3]))
        cfg.model['tof_branch_cfg']['conv_channels'] = tof_conv_channels
        cfg.model['tof_branch_cfg']['kernel_sizes'] = tof_kernel_sizes
        
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
    #          3. 定义 Fusion Head 的搜索空间 (支持 MLP 和 Transformer)
    # ===================================================================
    fusion_type = trial.suggest_categorical('fusion_type', ['mlp', 'transformer'])
    
    for key in ['hidden_dims', 'dropout_rates', 'branch_dims', 'embed_dim', 'num_heads', 'depth', 'dropout']:
        cfg.model['fusion_head_cfg'].pop(key, None)

    if fusion_type == 'mlp':
        cfg.model['fusion_head_cfg']['type'] = 'FusionHead'
        num_fusion_layers = trial.suggest_int('mlp_fusion_layers', 1, 3)
        fusion_hidden_dims = []
        fusion_dropout_rates = []
        for i in range(num_fusion_layers):
            fusion_hidden_dims.append(trial.suggest_int(f'mlp_fusion_hidden_dim_{i}', 32, 512, step=16))
            fusion_dropout_rates.append(trial.suggest_float(f'mlp_fusion_dropout_{i}', 0.1, 0.5))
        cfg.model['fusion_head_cfg']['hidden_dims'] = fusion_hidden_dims
        cfg.model['fusion_head_cfg']['dropout_rates'] = fusion_dropout_rates
    
    elif fusion_type == 'transformer':
        cfg.model['fusion_head_cfg']['type'] = 'TransformerFusionHead'
        cfg.model['fusion_head_cfg']['embed_dim'] = trial.suggest_int('fusion_embed_dim', 128, 512, step=128)
        cfg.model['fusion_head_cfg']['num_heads'] = trial.suggest_categorical('fusion_num_heads', [2, 4, 8])
        cfg.model['fusion_head_cfg']['depth'] = trial.suggest_int('fusion_depth', 1, 3)
        cfg.model['fusion_head_cfg']['dropout'] = trial.suggest_float('fusion_dropout', 0.1, 0.4)
        
        imu_dim = cfg.model['imu_branch_cfg']['filters'][-1]
        mlp_dim = cfg.model['mlp_branch_cfg']['output_dim']
        
        if variant == 'full':
            thm_dim = cfg.model['thm_branch_cfg']['filters'][-1]
            tof_dim = cfg.model['tof_branch_cfg']['conv_channels'][-1]
            branch_dims = [imu_dim, thm_dim, tof_dim, mlp_dim]
        else: # imu-only
            branch_dims = [imu_dim, mlp_dim]
        cfg.model['fusion_head_cfg']['branch_dims'] = branch_dims

    try:
        # ===================================================================
        #                          4. 运行训练流程
        # ===================================================================
        with suppress_output():
            device = train.setup_device(cfg.environment.get('gpu_id'))
            oof_score, _, _, _ = train.train_kfold_models(
                epochs=cfg.training['epochs'], patience=cfg.training['patience'],
                start_lr=cfg.training['start_lr'], weight_decay=cfg.training['weight_decay'],
                batch_size=cfg.data['batch_size'], use_amp=cfg.training['use_amp'],
                variant=cfg.data['variant'], model_cfg=cfg.model,
                mixup_enabled=cfg.training['mixup_enabled'], mixup_alpha=cfg.training['mixup_alpha'],
                device=device, show_stratification=False, loss_function='ce',
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
        train.prepare_data_kfold_multimodal = original_prepare_data_func
        print("\nMonkey patch restored. Original functions are back.")