import torch
import torch.nn as nn

# Import the refactored CNN classes
from .cnn1d import CNN1D
from .cnn2d import TemporalTOF2DCNN
from .mlp import MLP
from . import MODELS  # Import registry
from utils.registry import build_from_cfg
from . import fusion_head  # Import fusion head module

@MODELS.register_module()
class MultimodalityModel(nn.Module):
    """
    A multimodal fusion model for gesture recognition.

    This model combines:
    1. A 1D CNN branch for processing sequential sensor data (IMU, THM, etc.)
    2. A 2D CNN branch for processing TOF sensor grids (spatial depth information)
    3. An MLP branch for processing static demographic data
    
    The features from all branches are fused and passed to a final classifier head.
    """
    def __init__(self,
                 # --- legacy positional args ---
                 seq_input_channels: int = None,
                 tof_input_channels: int = 320,
                 static_input_features: int = 7,
                 num_classes: int = 18,
                 sequence_length: int = 100,
                 # --- new optional branch configs ---
                 cnn_branch_cfg: dict = None,
                 mlp_branch_cfg: dict = None,
                 fusion_head_cfg: dict = None,
                 tof_branch_cfg: dict = None,
                 ):
        """
        MultimodalityModel v2 constructor. 100% backward-compatible with the original
        5-argument signature, but now also accepts branch-level config dicts.
        If a branch config dict is provided, it takes precedence; otherwise we
        fall back to the legacy defaults.
        """
        super(MultimodalityModel, self).__init__()

        # ------------------------------------------------------------------
        # 1. Build 1-D CNN branch
        # ------------------------------------------------------------------
        if cnn_branch_cfg is None:
            if seq_input_channels is None:
                raise ValueError("Either cnn_branch_cfg or seq_input_channels must be provided")
            cnn_branch_cfg = dict(input_channels=seq_input_channels,
                                  num_classes=num_classes,
                                  sequence_length=sequence_length)
        else:
            # Ensure required keys exist, or fill in from legacy args
            cnn_branch_cfg = cnn_branch_cfg.copy()
            cnn_branch_cfg.setdefault('input_channels', seq_input_channels)
            cnn_branch_cfg.setdefault('num_classes', num_classes)
            cnn_branch_cfg.setdefault('sequence_length', sequence_length)

        self.cnn_1d_branch = build_from_cfg(cnn_branch_cfg, MODELS)
        self.cnn_1d_output_size = self.cnn_1d_branch.get_cnn_output_size()

        # ------------------------------------------------------------------
        # 2. Build 2-D CNN branch for TOF
        # ------------------------------------------------------------------
        if tof_branch_cfg is None:
            # Legacy fallback: use minimal TOF config
            tof_branch_cfg = dict(type='TemporalTOF2DCNN',
                                  num_tof_sensors=5,
                                  seq_len=sequence_length,
                                  out_features=128)
        
        tof_branch_cfg = tof_branch_cfg.copy()
        tof_branch_cfg.setdefault('seq_len', sequence_length)
        self.cnn_2d_branch = build_from_cfg(tof_branch_cfg, MODELS)
        # Infer output size from config or instance
        self.tof_2d_output_size = tof_branch_cfg.get('out_features', getattr(self.cnn_2d_branch, 'out_features', 128))

        # ------------------------------------------------------------------
        # 3. Build MLP branch
        # ------------------------------------------------------------------
        if mlp_branch_cfg is None:
            # Legacy fallback: use minimal MLP config
            mlp_branch_cfg = dict(type='MLP',
                                  input_features=static_input_features,
                                  output_dim=32)
        else:
            mlp_branch_cfg = mlp_branch_cfg.copy()
            mlp_branch_cfg.setdefault('input_features', static_input_features)
            mlp_branch_cfg.setdefault('output_dim', 32)

        # build branch
        self.mlp_branch = build_from_cfg(mlp_branch_cfg, MODELS)
        self.mlp_output_size = getattr(self.mlp_branch, 'get_output_size', lambda: mlp_branch_cfg.get('output_dim', 32))()

        # ------------------------------------------------------------------
        # 4. Fusion head (configurable)
        # ------------------------------------------------------------------
        combined_feature_size = self.cnn_1d_output_size + self.tof_2d_output_size + self.mlp_output_size

        if fusion_head_cfg is None:
            # Legacy fallback: use default LinearFusionHead
            fusion_head_cfg = dict(type='FusionHead',
                                   input_dim=combined_feature_size,
                                   num_classes=num_classes,
                                   hidden_dims=[256, 128, 64],
                                   dropout_rates=[0.5, 0.4, 0.3])
        else:
            # Ensure required parameters are set
            fusion_head_cfg = fusion_head_cfg.copy()
            fusion_head_cfg.setdefault('type', 'FusionHead')
            fusion_head_cfg.setdefault('input_dim', combined_feature_size)
            fusion_head_cfg.setdefault('num_classes', num_classes)
        
        # Build fusion head via registry
        self.classifier_head = build_from_cfg(fusion_head_cfg, MODELS)

    def forward(self, seq_input: torch.Tensor, tof_input: torch.Tensor, static_input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass logic of the multimodal fusion model.
        
        Args:
            seq_input: Sequential sensor data (batch_size, seq_len, seq_features)
            tof_input: TOF sensor data (batch_size, seq_len, 320) - 5 sensors × 64 pixels
            static_input: Static demographic data (batch_size, static_features)
        """
        # 1. Process sequential data through the 1D CNN branch
        cnn_1d_features = self.cnn_1d_branch.extract_features(seq_input)
        
        # 2. Process TOF data through the 2D CNN branch (only if TOF features exist)
        if tof_input.shape[2] > 0:  # Check if TOF features are present
            cnn_2d_features = self.cnn_2d_branch(tof_input)
        else:
            # Create zero tensor with correct shape for missing TOF features
            batch_size = seq_input.shape[0]
            cnn_2d_features = torch.zeros(batch_size, self.tof_2d_output_size, device=seq_input.device)
        
        # 3. Process static data through the MLP branch
        mlp_features = self.mlp_branch(static_input)
        
        # 4. Concatenate the features from all branches
        combined_features = torch.cat((cnn_1d_features, cnn_2d_features, mlp_features), dim=1)
        
        # 5. Pass the fused features through the final classifier head
        output = self.classifier_head(combined_features)
        
        return output

    def get_model_info(self) -> dict:
        """
        Helper function to return the model's total parameter count and approximate size.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024**2)  # Assuming float32 parameters
        
        # Get info from individual branches
        cnn_1d_info = self.cnn_1d_branch.get_model_info()
        cnn_2d_info = self.cnn_2d_branch.get_model_info()
        mlp_info = self.mlp_branch.get_model_info()
        
        return {
            'total_params': total_params, 
            'model_size_mb': model_size_mb,
            'cnn_1d_params': cnn_1d_info['total_params'],
            'cnn_2d_params': cnn_2d_info['total_params'],
            'mlp_params': mlp_info['total_params'],
            'fusion_head_params': total_params - cnn_1d_info['total_params'] - cnn_2d_info['total_params'] - mlp_info['total_params']
        } 