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
N_TRIALS = 30
# 初始随机搜索的次数
N_STARTUP_TRIALS = 10
# Get variant from config to dynamically set paths and study names
base_cfg = train.load_py_config(CONFIG_FILE_PATH)
variant = base_cfg.data['variant']

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
        
        # Get variant from the trial's user attributes (strict)
        variant = trial.user_attrs['variant']

        # --- NEW LOGIC: Copy from trial artifacts to the final weights directory ---
        # Define a list of files to copy to handle missing files gracefully
        # MODIFIED: Add spec_stats to the list of files to copy
        files_to_copy = (
            [f'model_fold_{i}_{variant}.pth' for i in range(1, 6)] +
            [f'scaler_fold_{i}_{variant}.pkl' for i in range(1, 6)] +
            [f'spec_stats_fold_{i}_{variant}.pkl' for i in range(1, 6)] +
            [f'spec_params_fold_{i}_{variant}.pkl' for i in range(1, 6)] +
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
    为支持多变量采样，已对内部动态依赖进行扁平化处理。
    """
    # 清除dynamo缓存防止模型多次编译占据缓存
    torch._dynamo.reset()
    cfg = train.load_py_config(CONFIG_FILE_PATH)
    variant = cfg.data['variant']
    
    trial.set_user_attr('variant', variant)
    
    print(f"\nStarting Trial {trial.number} for '{variant}' branch")

    # ===================================================================
    #
    # Part 1: 定义超参数搜索空间
    # -----------------------------------------------------------------
    # `if variant == 'full'` 是安全的，因为它在单个 study 中是恒定的。
    # 但循环次数等内部依赖仍需扁平化。
    #
    # ===================================================================

    # --- 设定动态层数的最大值 ---
    MAX_IMU_LAYERS = 4
    MAX_MLP_LAYERS = 3
    MAX_THM_LAYERS = 4
    MAX_TOF_CNN_LAYERS = 3
    MAX_FUSION_LAYERS = 3
    MAX_SPEC_LAYERS = 3

    # --- 1. 定义通用超参数 ---
    cfg.training['epochs'] = trial.suggest_int('epochs', 20, 150)
    cfg.training['patience'] = trial.suggest_int('patience', 10, 30, step=5)
    cfg.training['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    
    # Mixup (扁平化)
    cfg.training['mixup_enabled'] = trial.suggest_categorical('mixup_enabled', [True, False])
    mixup_alpha = trial.suggest_float('mixup_alpha', 0.1, 0.5)
    cfg.training['mixup_alpha'] = mixup_alpha if cfg.training['mixup_enabled'] else 0.0

    # Scheduler (扁平化)
    scheduler_type = trial.suggest_categorical('scheduler_type', ['cosine', 'reduce_on_plateau'])
    scheduler_cfg = {'type': scheduler_type}
    # 统一 warmup_ratio 参数，让采样器学习通用效果
    scheduler_cfg['warmup_ratio'] = trial.suggest_float('warmup_ratio', 0.0, 0.2)
    # 为 ReduceOnPlateau 定义参数
    lr_reduce_factor = trial.suggest_float('lr_reduce_factor', 0.2, 0.8, step=0.1)
    lr_patience = trial.suggest_int('lr_patience', 5, 15, step=2)
    min_lr = trial.suggest_float('min_lr', 1e-7, 1e-5, log=True)
    if scheduler_type == 'reduce_on_plateau':
        scheduler_cfg['factor'] = lr_reduce_factor
        scheduler_cfg['patience'] = lr_patience
        scheduler_cfg['min_lr'] = min_lr

    # 分层学习率
    layer_lrs = {}
    layer_lrs['imu'] = trial.suggest_float('lr_imu', 1e-5, 1e-2, log=True)
    layer_lrs['mlp'] = trial.suggest_float('lr_mlp', 1e-5, 1e-2, log=True)
    layer_lrs['fusion'] = trial.suggest_float('lr_fusion', 1e-5, 1e-2, log=True)
    layer_lrs['spec'] = trial.suggest_float('lr_spec', 1e-5, 1e-2, log=True)
    
    # --- NEW: Spectrogram 数据生成超参数 ---
    spec_params = {}
    spec_params['nperseg'] = trial.suggest_int('spec_nperseg', 16, 64, step=4)
    spec_params['noverlap_ratio'] = trial.suggest_float('spec_noverlap_ratio', 0.5, 0.95)
    spec_params['fs'] = 10.0
    # Strict: provide max_length from config to data pipeline
    spec_params['max_length'] = cfg.data['max_length']
    # Normalize to explicit noverlap for downstream consumers (config-driven, no fallback later)
    _np = int(spec_params['nperseg'])
    _ratio = float(spec_params['noverlap_ratio'])
    _no = int(_np * _ratio)
    if not (0 <= _no < _np):
        raise ValueError(f"Computed noverlap({_no}) must satisfy 0 <= noverlap < nperseg({_np}).")
    spec_params['noverlap'] = _no

    # --- Spectrogram (Spec) 分支架构 ---
    if 'spec_branch_cfg' in cfg.model:
        num_spec_layers = trial.suggest_int('num_spec_layers', 2, MAX_SPEC_LAYERS)
        spec_filters_all = [trial.suggest_int(f'spec_filter_{i}', 16, 512, step=16) for i in range(MAX_SPEC_LAYERS)]
        spec_kernel_sizes_all = [trial.suggest_categorical(f'spec_kernel_{i}', [3, 5]) for i in range(MAX_SPEC_LAYERS)]
        cfg.model['spec_branch_cfg']['filters'] = spec_filters_all[:num_spec_layers]
        cfg.model['spec_branch_cfg']['kernel_sizes'] = spec_kernel_sizes_all[:num_spec_layers]
        cfg.model['spec_branch_cfg']['use_residual'] = trial.suggest_categorical('spec_use_residual', [True, False])

    # IMU 分支 (扁平化)
    num_imu_layers = trial.suggest_int('num_imu_layers', 2, MAX_IMU_LAYERS)
    imu_filters_all = [trial.suggest_int(f'imu_filter_{i}', 32, 1536, step=16) for i in range(MAX_IMU_LAYERS)]
    imu_kernel_sizes_all = [trial.suggest_categorical(f'imu_kernel_{i}', [3, 5, 7, 9, 11]) for i in range(MAX_IMU_LAYERS)]
    cfg.model['imu_branch_cfg']['filters'] = imu_filters_all[:num_imu_layers]
    cfg.model['imu_branch_cfg']['kernel_sizes'] = imu_kernel_sizes_all[:num_imu_layers]
    cfg.model['imu_branch_cfg']['use_residual'] = trial.suggest_categorical('imu_use_residual', [True, False])
    # --- IMU SE hyperparameters ---
    cfg.model['imu_branch_cfg']['use_se'] = trial.suggest_categorical('imu_use_se', [True, False])
    cfg.model['imu_branch_cfg']['se_reduction'] = trial.suggest_categorical('imu_se_reduction', [8, 16, 32])

    # --- IMU Temporal Aggregation (正确、无条件地定义所有相关参数) ---
    imu_temporal_aggregation = trial.suggest_categorical('imu_temporal_aggregation', ['global_pool', 'temporal_encoder'])
    imu_temporal_mode = trial.suggest_categorical('imu_temporal_mode', ['lstm', 'transformer'])
    imu_lstm_hidden = trial.suggest_int('imu_lstm_hidden', 128, 512, step=64)
    imu_lstm_layers = trial.suggest_int('imu_lstm_layers', 1, 2)
    imu_bidirectional = trial.suggest_categorical('imu_bidirectional', [True, False])
    imu_transformer_heads = trial.suggest_categorical('imu_num_heads', [4, 8, 16])
    imu_transformer_layers = trial.suggest_int('imu_num_layers', 1, 3)
    imu_transformer_ff_dim = trial.suggest_int('imu_ff_dim', 256, 1024, step=128)
    imu_transformer_dropout = trial.suggest_float('imu_dropout', 0.1, 0.4)

    cfg.model['imu_branch_cfg']['temporal_aggregation'] = imu_temporal_aggregation

    if imu_temporal_aggregation == 'temporal_encoder':
        cfg.model['imu_branch_cfg']['temporal_mode'] = imu_temporal_mode
        if imu_temporal_mode == 'lstm':
            cfg.model['imu_branch_cfg']['lstm_hidden'] = imu_lstm_hidden
            cfg.model['imu_branch_cfg']['lstm_layers'] = imu_lstm_layers
            cfg.model['imu_branch_cfg']['bidirectional'] = imu_bidirectional
        else:  # transformer
            cfg.model['imu_branch_cfg']['num_heads'] = imu_transformer_heads
            cfg.model['imu_branch_cfg']['num_layers'] = imu_transformer_layers
            cfg.model['imu_branch_cfg']['ff_dim'] = imu_transformer_ff_dim
            cfg.model['imu_branch_cfg']['dropout'] = imu_transformer_dropout
    
    # MLP 分支 (扁平化)
    num_mlp_hidden_layers = trial.suggest_int('num_mlp_hidden_layers', 1, MAX_MLP_LAYERS)
    mlp_hidden_dims_all = [trial.suggest_int(f'mlp_hidden_dim_{i}', 16, 256, step=16) for i in range(MAX_MLP_LAYERS)]
    cfg.model['mlp_branch_cfg']['hidden_dims'] = mlp_hidden_dims_all[:num_mlp_hidden_layers]
    cfg.model['mlp_branch_cfg']['output_dim'] = trial.suggest_int('mlp_output_dim', 16, 128, step=16)
    cfg.model['mlp_branch_cfg']['dropout_rate'] = trial.suggest_float('mlp_dropout_rate', 0.1, 0.5)

    # --- 2. 为 "full" 分支定义额外超参数 ---
    if variant == 'full':
        layer_lrs['thm'] = trial.suggest_float('lr_thm', 1e-5, 1e-2, log=True)
        layer_lrs['tof'] = trial.suggest_float('lr_tof', 1e-5, 1e-2, log=True)
        
        # THM 分支 (扁平化)
        num_thm_layers = trial.suggest_int('num_thm_layers', 1, MAX_THM_LAYERS)
        thm_filters_all = [trial.suggest_int(f'thm_filter_{i}', 16, 512, step=16) for i in range(MAX_THM_LAYERS)]
        thm_kernel_sizes_all = [trial.suggest_categorical(f'thm_kernel_{i}', [3, 5, 7]) for i in range(MAX_THM_LAYERS)]
        cfg.model['thm_branch_cfg']['filters'] = thm_filters_all[:num_thm_layers]
        cfg.model['thm_branch_cfg']['kernel_sizes'] = thm_kernel_sizes_all[:num_thm_layers]
        cfg.model['thm_branch_cfg']['use_residual'] = trial.suggest_categorical('thm_use_residual', [True, False])
        # --- THM SE hyperparameters ---
        cfg.model['thm_branch_cfg']['use_se'] = trial.suggest_categorical('thm_use_se', [True, False])
        cfg.model['thm_branch_cfg']['se_reduction'] = trial.suggest_categorical('thm_se_reduction', [8, 16, 32])
        
        # THM Temporal Aggregation
        thm_temporal_aggregation = trial.suggest_categorical('thm_temporal_aggregation', ['global_pool', 'temporal_encoder'])
        thm_temporal_mode = trial.suggest_categorical('thm_temporal_mode', ['lstm', 'transformer'])

        # THM LSTM 参数
        thm_lstm_hidden = trial.suggest_int('thm_lstm_hidden', 64, 256, step=32)
        thm_lstm_layers = trial.suggest_int('thm_lstm_layers', 1, 2)
        thm_bidirectional = trial.suggest_categorical('thm_bidirectional', [True, False])

        # THM Transformer 参数
        thm_transformer_heads = trial.suggest_categorical('thm_num_heads', [4, 8, 16])
        thm_transformer_layers = trial.suggest_int('thm_num_layers', 1, 2)
        thm_transformer_ff_dim = trial.suggest_int('thm_ff_dim', 256, 1024, step=128)
        thm_transformer_dropout = trial.suggest_float('thm_dropout', 0.1, 0.4)

        # --- 组装 THM 分支配置 ---
        cfg.model['thm_branch_cfg']['temporal_aggregation'] = thm_temporal_aggregation
        if thm_temporal_aggregation == 'temporal_encoder':
            cfg.model['thm_branch_cfg']['temporal_mode'] = thm_temporal_mode
            if thm_temporal_mode == 'lstm':
                cfg.model['thm_branch_cfg']['lstm_hidden'] = thm_lstm_hidden
                cfg.model['thm_branch_cfg']['lstm_layers'] = thm_lstm_layers
                cfg.model['thm_branch_cfg']['bidirectional'] = thm_bidirectional
            else:  # transformer
                cfg.model['thm_branch_cfg']['num_heads'] = thm_transformer_heads
                cfg.model['thm_branch_cfg']['num_layers'] = thm_transformer_layers
                cfg.model['thm_branch_cfg']['ff_dim'] = thm_transformer_ff_dim
                cfg.model['thm_branch_cfg']['dropout'] = thm_transformer_dropout

        # TOF 分支 (扁平化)
        cfg.model['tof_branch_cfg']['use_residual'] = trial.suggest_categorical('tof_use_residual', [True, False])
        # --- TOF SE hyperparameters ---
        cfg.model['tof_branch_cfg']['use_se'] = trial.suggest_categorical('tof_use_se', [True, False])
        cfg.model['tof_branch_cfg']['se_reduction'] = trial.suggest_categorical('tof_se_reduction', [8, 16, 32])
        # --- TOF Sensor Gate hyperparameters ---
        use_tof_sensor_gate = trial.suggest_categorical('tof_use_sensor_gate', [False, True])
        cfg.model['tof_branch_cfg']['use_sensor_gate'] = use_tof_sensor_gate

        # Suggest sub-parameters UNCONDITIONALLY to keep a static search space
        is_adaptive = trial.suggest_categorical('tof_sensor_gate_adaptive', [False, True])
        sensor_gate_init = trial.suggest_float('tof_sensor_gate_init', 0.5, 1.5)

        # Apply sub-parameters only if the gate is enabled
        if use_tof_sensor_gate:
            cfg.model['tof_branch_cfg']['sensor_gate_adaptive'] = is_adaptive
            if not is_adaptive:
                cfg.model['tof_branch_cfg']['sensor_gate_init'] = sensor_gate_init
        
        num_tof_cnn_layers = trial.suggest_categorical('num_tof_cnn_layers', [2, 3])
        tof_kernel_0 = trial.suggest_categorical('tof_kernel_0', [2, 3])
        tof_kernel_1 = trial.suggest_categorical('tof_kernel_1', [2, 3])
        # Unconditionally suggest kernel for the third layer to keep the space static
        tof_kernel_2 = trial.suggest_categorical('tof_kernel_2', [2, 3])
        tof_conv_channels_all = [trial.suggest_int(f'tof_conv_channel_{i}', 32, 256, step=32) for i in range(MAX_TOF_CNN_LAYERS)]
        
        if num_tof_cnn_layers == 2:
            cfg.model['tof_branch_cfg']['kernel_sizes'] = [tof_kernel_0, tof_kernel_1]
            max_ch = 256
        else:
            cfg.model['tof_branch_cfg']['kernel_sizes'] = [tof_kernel_0, tof_kernel_1, tof_kernel_2]
            max_ch = 128
        
        tof_channels_assembled = [min(ch, max_ch) for ch in tof_conv_channels_all]
        cfg.model['tof_branch_cfg']['conv_channels'] = tof_channels_assembled[:num_tof_cnn_layers]
        # TOF Temporal (扁平化)
        tof_temporal_mode = trial.suggest_categorical('tof_temporal_mode', ['lstm', 'transformer'])
        cfg.model['tof_branch_cfg']['temporal_mode'] = tof_temporal_mode
        
        tof_lstm_hidden = trial.suggest_int('tof_lstm_hidden', 64, 256, step=32)
        tof_lstm_layers = trial.suggest_int('tof_lstm_layers', 1, 2)
        tof_bidirectional = trial.suggest_categorical('tof_bidirectional', [True, False])
        
        tof_transformer_heads = trial.suggest_categorical('tof_num_heads', [4, 8, 16])
        tof_transformer_layers = trial.suggest_int('tof_num_layers', 1, 3)
        tof_transformer_ff_dim = trial.suggest_int('tof_ff_dim', 256, 1024, step=128)
        tof_transformer_dropout = trial.suggest_float('tof_dropout', 0.1, 0.4)
        
        if tof_temporal_mode == 'transformer':
            cfg.model['tof_branch_cfg']['num_heads'] = tof_transformer_heads
            cfg.model['tof_branch_cfg']['num_layers'] = tof_transformer_layers
            cfg.model['tof_branch_cfg']['ff_dim'] = tof_transformer_ff_dim
            cfg.model['tof_branch_cfg']['dropout'] = tof_transformer_dropout
        else: # lstm
            cfg.model['tof_branch_cfg']['lstm_hidden'] = tof_lstm_hidden
            cfg.model['tof_branch_cfg']['lstm_layers'] = tof_lstm_layers
            cfg.model['tof_branch_cfg']['bidirectional'] = tof_bidirectional
    
    # 将最终确定的学习率配置赋给 cfg
    scheduler_cfg['layer_lrs'] = layer_lrs
    cfg.training['scheduler_cfg'] = scheduler_cfg
    # --- 3. 定义 Fusion Head ---
    # 清理旧的或不相关的键
    for key in ['hidden_dims', 'dropout_rates', 'branch_dims', 'embed_dim', 'num_heads', 'depth', 'dropout']:
        if key in cfg.model['fusion_head_cfg']:
            cfg.model['fusion_head_cfg'].pop(key)

    cfg.model['fusion_head_cfg']['type'] = 'FusionHead'
    num_fusion_layers = trial.suggest_int('mlp_fusion_layers', 1, MAX_FUSION_LAYERS)
    fusion_hidden_dims_all = [trial.suggest_int(f'mlp_fusion_hidden_dim_{i}', 32, 512, step=16) for i in range(MAX_FUSION_LAYERS)]
    fusion_dropout_rates_all = [trial.suggest_float(f'mlp_fusion_dropout_{i}', 0.1, 0.5) for i in range(MAX_FUSION_LAYERS)]
    cfg.model['fusion_head_cfg']['hidden_dims'] = fusion_hidden_dims_all[:num_fusion_layers]
    cfg.model['fusion_head_cfg']['dropout_rates'] = fusion_dropout_rates_all[:num_fusion_layers]
    
    try:
        # ===================================================================
        #                          4. 运行训练流程
        # ===================================================================
        study_name = trial.study.study_name
        log_dir = os.path.join(LOG_DIR, study_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'trial_{trial.number}.log')

        # Build a compact view of the actually used parameters to keep logs clean
        _training_eff = {
            'epochs': cfg.training['epochs'],
            'patience': cfg.training['patience'],
            'weight_decay': cfg.training['weight_decay'],
            'mixup_enabled': cfg.training['mixup_enabled'],
        }
        if cfg.training['mixup_enabled']:
            _training_eff['mixup_alpha'] = cfg.training['mixup_alpha']

        effective_params = {
            'training': _training_eff,
            'scheduler': {
                'type': scheduler_type,
                'warmup_ratio': scheduler_cfg['warmup_ratio']
            },
            'layer_lrs': scheduler_cfg['layer_lrs']
        }
        if scheduler_type == 'reduce_on_plateau':
            effective_params['scheduler'].update({
                'factor': scheduler_cfg['factor'],
                'patience': scheduler_cfg['patience'],
                'min_lr': scheduler_cfg['min_lr']
            })

        # IMU effective configuration
        _imu_eff = {
            'num_layers': num_imu_layers,
            'filters': cfg.model['imu_branch_cfg']['filters'],
            'kernel_sizes': cfg.model['imu_branch_cfg']['kernel_sizes'],
            'use_residual': cfg.model['imu_branch_cfg']['use_residual'],
            'use_se': cfg.model['imu_branch_cfg']['use_se'],
            'temporal_aggregation': cfg.model['imu_branch_cfg']['temporal_aggregation']
        }
        if _imu_eff['use_se']:
            _imu_eff['se_reduction'] = cfg.model['imu_branch_cfg']['se_reduction']
        effective_params['imu_branch'] = _imu_eff
        if cfg.model['imu_branch_cfg']['temporal_aggregation'] == 'temporal_encoder':
            if imu_temporal_mode == 'lstm':
                effective_params['imu_branch']['temporal'] = {
                    'mode': 'lstm',
                    'lstm_hidden': cfg.model['imu_branch_cfg']['lstm_hidden'],
                    'lstm_layers': cfg.model['imu_branch_cfg']['lstm_layers'],
                    'bidirectional': cfg.model['imu_branch_cfg']['bidirectional']
                }
            else:
                effective_params['imu_branch']['temporal'] = {
                    'mode': 'transformer',
                    'num_heads': cfg.model['imu_branch_cfg']['num_heads'],
                    'num_layers': cfg.model['imu_branch_cfg']['num_layers'],
                    'ff_dim': cfg.model['imu_branch_cfg']['ff_dim'],
                    'dropout': cfg.model['imu_branch_cfg']['dropout']
                }

        # MLP effective configuration
        effective_params['mlp_branch'] = {
            'num_hidden_layers': num_mlp_hidden_layers,
            'hidden_dims': cfg.model['mlp_branch_cfg']['hidden_dims'],
            'output_dim': cfg.model['mlp_branch_cfg']['output_dim'],
            'dropout_rate': cfg.model['mlp_branch_cfg']['dropout_rate']
        }

        # Spec branch effective configuration (if present)
        if 'spec_branch_cfg' in cfg.model:
            effective_params['spec_branch'] = {
                'filters': cfg.model['spec_branch_cfg']['filters'],
                'kernel_sizes': cfg.model['spec_branch_cfg']['kernel_sizes'],
                'use_residual': cfg.model['spec_branch_cfg']['use_residual']
            }
                    # Compute noverlap early if needed, so it's included in effective_params
        if 'noverlap' not in spec_params and 'noverlap_ratio' in spec_params:
            computed_noverlap = int(spec_params['nperseg'] * spec_params['noverlap_ratio'])
        else:
            computed_noverlap = spec_params.get('noverlap')
        
        # Also record spectrogram generation parameters actually used this trial
        effective_params['spec_params'] = {
            'fs': spec_params['fs'],
            'nperseg': spec_params['nperseg'],
            'max_length': spec_params['max_length'],
            # Log ratio if provided
            'noverlap_ratio': spec_params.get('noverlap_ratio'),
            # Log the actual noverlap that will be used
            'noverlap': computed_noverlap
        }

        if variant == 'full':
            # THM effective configuration
            _thm_eff = {
                'num_layers': num_thm_layers,
                'filters': cfg.model['thm_branch_cfg']['filters'],
                'kernel_sizes': cfg.model['thm_branch_cfg']['kernel_sizes'],
                'use_residual': cfg.model['thm_branch_cfg']['use_residual'],
                'use_se': cfg.model['thm_branch_cfg']['use_se'],
                'temporal_aggregation': cfg.model['thm_branch_cfg']['temporal_aggregation']
            }
            if _thm_eff['use_se']:
                _thm_eff['se_reduction'] = cfg.model['thm_branch_cfg']['se_reduction']
            effective_params['thm_branch'] = _thm_eff
            if thm_temporal_aggregation == 'temporal_encoder':
                if thm_temporal_mode == 'lstm':
                    effective_params['thm_branch']['temporal'] = {
                        'mode': 'lstm',
                        'lstm_hidden': cfg.model['thm_branch_cfg']['lstm_hidden'],
                        'lstm_layers': cfg.model['thm_branch_cfg']['lstm_layers'],
                        'bidirectional': cfg.model['thm_branch_cfg']['bidirectional']
                    }
                else:
                    effective_params['thm_branch']['temporal'] = {
                        'mode': 'transformer',
                        'num_heads': cfg.model['thm_branch_cfg']['num_heads'],
                        'num_layers': cfg.model['thm_branch_cfg']['num_layers'],
                        'ff_dim': cfg.model['thm_branch_cfg']['ff_dim'],
                        'dropout': cfg.model['thm_branch_cfg']['dropout']
                    }

            # TOF effective configuration
            # Build TOF effective params with conditional sensor-gate details
            _tof_eff = {
                'num_cnn_layers': num_tof_cnn_layers,
                'conv_channels': cfg.model['tof_branch_cfg']['conv_channels'],
                'kernel_sizes': cfg.model['tof_branch_cfg']['kernel_sizes'],
                'use_residual': cfg.model['tof_branch_cfg']['use_residual'],
                'use_se': cfg.model['tof_branch_cfg']['use_se'],
                'use_sensor_gate': use_tof_sensor_gate,
                'temporal_mode': cfg.model['tof_branch_cfg']['temporal_mode']
            }
            if _tof_eff['use_se']:
                _tof_eff['se_reduction'] = cfg.model['tof_branch_cfg']['se_reduction']
            if use_tof_sensor_gate:
                _tof_eff['sensor_gate_adaptive'] = cfg.model['tof_branch_cfg']['sensor_gate_adaptive']
                if not cfg.model['tof_branch_cfg']['sensor_gate_adaptive']:
                    _tof_eff['sensor_gate_init'] = cfg.model['tof_branch_cfg']['sensor_gate_init']
            effective_params['tof_branch'] = _tof_eff
            if tof_temporal_mode == 'lstm':
                effective_params['tof_branch']['temporal'] = {
                    'mode': 'lstm',
                    'lstm_hidden': cfg.model['tof_branch_cfg']['lstm_hidden'],
                    'lstm_layers': cfg.model['tof_branch_cfg']['lstm_layers'],
                    'bidirectional': cfg.model['tof_branch_cfg']['bidirectional']
                }
            else:
                effective_params['tof_branch']['temporal'] = {
                    'mode': 'transformer',
                    'num_heads': cfg.model['tof_branch_cfg']['num_heads'],
                    'num_layers': cfg.model['tof_branch_cfg']['num_layers'],
                    'ff_dim': cfg.model['tof_branch_cfg']['ff_dim'],
                    'dropout': cfg.model['tof_branch_cfg']['dropout']
                }

        # Fusion head effective configuration
        effective_params['fusion_head'] = {
            'num_layers': num_fusion_layers,
            'hidden_dims': cfg.model['fusion_head_cfg']['hidden_dims'],
            'dropout_rates': cfg.model['fusion_head_cfg']['dropout_rates']
        }

        # Store effective params for callback usage
        trial.set_user_attr('effective_params', effective_params)

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"TRIAL {trial.number} PARAMETERS (EFFECTIVE)\n")
            f.write("="*60 + "\n")
            f.write(json.dumps(effective_params, indent=4))
            f.write("\n\n" + "="*60 + "\n")
            f.write("TRAINING LOG\n")
            f.write("="*60 + "\n")

        with manage_output(log_path):
            # Use strict loading for gpu_id
            gpu_id = cfg.environment['gpu_id']
            device = train.setup_device(gpu_id)
            
            # Pass all required arguments to the training function
            oof_score, _, _, _ = train.train_kfold_models(
                epochs=cfg.training['epochs'], 
                patience=cfg.training['patience'],
                weight_decay=cfg.training['weight_decay'],
                batch_size=cfg.data['batch_size'], 
                use_amp=cfg.training['use_amp'],
                variant=cfg.data['variant'], 
                model_cfg=cfg.model,
                mixup_enabled=cfg.training['mixup_enabled'], 
                mixup_alpha=cfg.training['mixup_alpha'],
                scheduler_cfg=cfg.training['scheduler_cfg'],
                spec_params=spec_params,
                device=device,
                show_stratification=False,
                loss_function='ce',
                output_dir=TRIAL_ARTIFACTS_DIR,
                num_workers=cfg.environment['num_workers'],
            )
        
        torch.cuda.empty_cache()
        return oof_score

    except Exception as e:
        print(f"Trial {trial.number} failed with an exception: {e}")
        import traceback
        traceback.print_exc()
        # 假设 oof_score 越大越好，返回一个很差的数字
        return -1.0


if __name__ == "__main__":
    print("="*60)
    print(f"Detected variant '{variant}'.")
    print(f"Database:   {DB_PATH}")
    print(f"Study Name: {STUDY_NAME}")
    print("="*60)

    # --- Pre-load base data instead of full data ---
    print("Preparing and pre-loading base data (without spectrograms)... This will happen only once.")
    # 调用新的基础数据准备函数
    base_fold_data, label_encoder, y_all, sequence_ids_all = train.prepare_base_data_kfold(
        variant=variant
    )
    print("Base data has been pre-loaded into memory.")
    print("="*60)
    
    # 保存原始函数以便后续恢复
    original_prepare_data_func = train.prepare_data_kfold_multimodal

    # --- MODIFIED: Create a new monkey-patch function ---
    def mock_prepare_data_kfold_multimodal(*args, **kwargs):
        """
        猴子补丁函数: 使用预加载的基础数据和当前试验的频谱图参数，
        动态生成完整的K-Fold数据，避免重复加载和特征工程。
        """
        print("--> Monkey patch activated: Generating spectrograms for new trial...")
        # 从调用中提取 spec_params
        spec_params = kwargs.get('spec_params')
        if spec_params is None:
             raise ValueError("spec_params not provided to mocked function!")

        # Normalize 'noverlap' strictly
        if 'noverlap' not in spec_params:
            if 'noverlap_ratio' not in spec_params:
                raise ValueError("spec_params must include either 'noverlap' or 'noverlap_ratio'.")
            noverlap = int(spec_params['nperseg'] * spec_params['noverlap_ratio'])
            if noverlap >= spec_params['nperseg']:
                raise ValueError("Computed noverlap from noverlap_ratio must be < nperseg.")
            spec_params = {**spec_params, 'noverlap': noverlap}

        # 使用预加载的基础数据，动态生成频谱图
        full_fold_data = train.generate_and_attach_spectrograms(
            base_fold_data=base_fold_data,
            spec_params=spec_params,
            variant=variant
        )
        print("--> Spectrograms generated. Returning full dataset for trial.")
        return full_fold_data, label_encoder, y_all, sequence_ids_all

    # 应用猴子补丁
    train.prepare_data_kfold_multimodal = mock_prepare_data_kfold_multimodal
    print("Monkey patch applied. `prepare_data_kfold_multimodal` will now use pre-loaded data.")

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
            print(f"      {key}: {value}")

    finally:
        if os.path.exists(TRIAL_ARTIFACTS_DIR):
            shutil.rmtree(TRIAL_ARTIFACTS_DIR)
            print(f"\nCleaned up temporary artifacts directory: {TRIAL_ARTIFACTS_DIR}")

        # 恢复原始函数
        train.prepare_data_kfold_multimodal = original_prepare_data_func
        print("\nMonkey patch restored. Original function is back.")