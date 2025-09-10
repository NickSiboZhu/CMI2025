# CMI Gesture Recognition System

A config-driven, multimodal deep learning system for detecting body-focused repetitive behaviors (BFRB) using sensor data from IMU, thermal, and time-of-flight sensors. This document outlines the core methodologies, system architecture, and usage instructions.

## Core Methodologies

This section details the key strategies that enabled rapid, robust model development.

### 1. Computation — An Efficient Experimental Framework

To enable rapid iteration, we optimized our workflow for speed:

-   **Data Processing Acceleration (pandas → polars)**: We migrated heavy preprocessing from pandas to polars. Its multi-threaded architecture provided a significant performance boost, drastically reducing data preparation time.
-   **Model Training Acceleration (`torch.compile`)**: We used `torch.compile` to JIT-compile the model, which optimized GPU operations and reduced Python overhead for faster training cycles.

These optimizations were a strategic necessity, allowing us to conduct more experiments and thoroughly validate our hypotheses.

### 2. Data — From Raw Signals to Intelligent Features

Our approach to data was two-pronged: sophisticated feature engineering and creating a trustworthy evaluation system.

-   **Feature Engineering**:
    -   **Domain-Knowledge**: We leveraged domain expertise to engineer critical time-domain features (e.g., gravity-compensated linear acceleration, quaternion-derived angular velocity).
    -   **Frequency Domain (Spectrograms)**: We created a spectrogram branch, treating the 1D signal as an "image." This enabled a 2D CNN to explicitly learn from time-frequency representations, capturing features complementary to the 1D branch.
-   **Reliable Cross-Validation (CV)**: We developed a robust, subject-grouped CV framework to prevent data leakage, ensuring that performance metrics reflect true generalization capabilities.

### 3. Model — A Narrative of Architectural Evolution

Our model development for time-series classification followed a logical progression, with each architecture addressing the limitations of the last.

-   **Stage 1: The 1D CNN Baseline**: A standard 1D CNN extracted local patterns. However, its reliance on global pooling to compress the temporal dimension resulted in the loss of crucial sequential information.
-   **Stage 2: Introducing Sequential Context with LSTM**: To preserve temporal dependencies, we replaced global pooling with an LSTM. The CNN's output feature map was treated as a sequence, and the LSTM's final hidden state provided a context-aware summary for classification.
-   **Stage 3: Capturing Global Relationships with a Transformer**: To model relationships between all parts of the sequence simultaneously, we used a Transformer. A `[CLS]` token, processed by the self-attention mechanism, created a globally-informed representation that captured complex inter-dependencies across the entire time series.
-   **Handling Auxiliary Data (Multi-Branch Approach)**:
    -   **Static Features**: Time-invariant data (e.g., demographics) were processed in a separate MLP branch.
    -   **Spectrograms**: Processed in their own dedicated 2D CNN branch.
-   **Final Model**: The outputs from all branches (Time-Series, Static, Spectrogram) were concatenated before the final classification head to create a comprehensive, multi-modal model.

## System Architecture

#### Overall Architecture

```mermaid
graph TD
    A["Raw Sensor Data"] --> B["Data Preprocessing"]
    B --> C["Feature Engineering"]
    C --> D["Multimodal Branches"]
    
    D --> E["IMU Branch<br/>(CNN1D + Attention/LSTM/Transformer)"]
    D --> F["Thermal Branch<br/>(CNN1D)"]
    D --> G["ToF Branch<br/>(CNN2D)"]
    D --> H["Spectrogram Branch<br/>(CNN2D)"]
    D --> I["Static Features<br/>(MLP)"]
    
    E --> J["Feature Fusion"]
    F --> J
    G --> J
    H --> J
    I --> J
    
    J --> K["Classifier Head<br/>(18 Classes)"]
    K --> L["Competition Metric"]
    
    style E fill:#e1f5fe
    style F fill:#fff3e0
    style G fill:#f3e5f5
    style H fill:#e8f5e8
    style I fill:#fce4ec
```

#### Ensemble Strategy

The system uses a two-stage stacking approach with subject-grouped cross-validation to prevent data leakage and improve generalization.

```mermaid
graph LR
    A["Base Models<br/>(5-Fold CV)"] --> B["OOF Predictions"]
    B --> C["Feature Matrix<br/>(Model × Class Probs)"]
    C --> D["Meta Learner<br/>(Ridge/Logistic/RF/XGB)"]
    D --> E["Final Predictions"]
    
    F["Group-aware<br/>Cross-Validation"] --> A
    G["Composite<br/>Class Weights"] --> A
    H["Probability<br/>Calibration"] --> D
    
    style A fill:#e3f2fd
    style D fill:#fff3e0
    style E fill:#e8f5e8
```

## Usage

The training workflow consists of installing dependencies, training base models via configuration files, optionally running an ensemble, and executing inference.

### Installation

```bash
# Clone repository
git clone <repository-url>
cd cmi_competition

# Install dependencies
pip install torch torchvision toraudio
pip install scikit-learn pandas numpy polars
pip install lightgbm xgboost catboost  # optional ensemble models
pip install scipy matplotlib seaborn  # for analysis
```

### Training

The training system is fully config-driven. All parameters, including model architecture, data variants, and hyperparameters, are defined in configuration files.

**Example Commands:**

```bash
# Basic training with a specific config
python development/train.py --config cmi-submission/configs/multimodality_model_v1_full_config.py

# Show data stratification details
python development/train.py --config cmi-submission/configs/multimodality_model_v1_full_config.py --stratification

# Override output directory
TRAIN_OUTPUT_DIR=/custom/path python development/train.py --config cmi-submission/configs/multimodality_model_v1_full_config.py
```

**Sample Config Structure:**

A single configuration file defines the environment, data, training, model, and spectrogram parameters.

```python
# cmi-submission/configs/multimodality_model_v3_full_config.py
# ===================================================================
#   Configuration v3 – FULL Multimodal (IMU + THM + TOF + DEMO)
# ===================================================================

# --------------------------- Data Settings ---------------------------
data = dict(
    variant='full',
    max_length=100,
    batch_size=64,
)

# -------------------------- Model Architecture -----------------------
model = dict(
    type='MultimodalityModel',
    num_classes=18,
    sequence_length=data['max_length'],

    # IMU branch (inertial measurement unit)
    imu_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[64, 128, 256],
        kernel_sizes=[5, 5, 3],
        temporal_aggregation='temporal_encoder',  # 'global_pool' or 'temporal_encoder'
        temporal_mode='lstm',  # 'lstm' or 'transformer'
        use_residual=True,
        use_se=True,
    ),

    # THM branch (thermopile sensors)
    thm_branch_cfg=dict(
        type='CNN1D',
        input_channels=None,  # will be filled dynamically from data
        sequence_length=data['max_length'],
        filters=[32, 64, 128],
        kernel_sizes=[5, 5, 3],
        temporal_aggregation='temporal_encoder',
        temporal_mode='lstm',
        use_residual=True,
        use_se=True,
    ),

    # TOF 2D CNN branch config
    tof_branch_cfg=dict(
        type='TemporalTOF2DCNN',
        input_channels=5,
        seq_len=100,
        conv_channels=[32, 64, 128],
        temporal_mode='lstm',
        use_residual=True,
        use_se=True,
    ),

    # MLP branch for static features
    mlp_branch_cfg=dict(
        type='MLP',
        input_features=None,  # will be filled dynamically
        hidden_dims=[64],
        output_dim=32,
    ),

    # Spectrogram branch
    spec_branch_cfg=dict(
        type='SpectrogramCNN',
        in_channels=6,
        filters=[32, 64, 128],
        use_residual=True,
    ),

    # Enable/disable branches
    use_thm=True,
    use_tof=True,
    use_spec=True,

    # Fusion head
    fusion_head_cfg=dict(
        type='FusionHead',
        hidden_dims=[256, 128],
        dropout_rates=[0.4, 0.3]
    )
)

# ----------------------- Training Strategy ---------------------------
training = dict(
    epochs=100,
    patience=15,
    weight_decay=1e-2,
    use_amp=False,
    mixup_enabled=True,
    mixup_alpha=0.2,
    loss=dict(type='CrossEntropyLoss'),
    scheduler_cfg=dict(
        type='cosine',
        warmup_ratio=0.1,
        layer_lrs=dict(
            imu=1e-3, thm=1e-3, tof=5e-4, mlp=2e-3,
            spec=2e-3, fusion=2e-3,
        )
    ),
)

# -------------------------- Environment ------------------------------
environment = dict(gpu_id=None, seed=42, num_workers=4)

# -------------------------- Spectrogram Params ------------------------
spec_params = dict(
    fs=10.0,
    nperseg=20,
    noverlap_ratio=0.75,
    max_length=data['max_length'],
)
```

### Ensemble Training (Stacking)

After training base models and generating out-of-fold predictions, use the ensemble module for meta-learning.

```