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
                 # --- required configs ---
                 num_classes: int = 18,
                 sequence_length: int = 100,
                 # --- branch configs ---
                 cnn_branch_cfg: dict = None,
                 mlp_branch_cfg: dict = None,
                 fusion_head_cfg: dict = None,
                 tof_branch_cfg: dict = None,
                 # --- modality toggles ---
                 use_tof: bool = True,
                 ):
        """
        MultimodalityModel constructor. All configuration is provided via config dicts.
        Each branch (CNN1D, MLP, TOF2D) has its own config dict for maximum flexibility.
        """
        super(MultimodalityModel, self).__init__()

        # Store modality flags
        self.use_tof = use_tof

        # ------------------------------------------------------------------
        # 1. Build 1-D CNN branch
        # ------------------------------------------------------------------
        if cnn_branch_cfg is None:
            raise ValueError("cnn_branch_cfg is required")
        
        cnn_branch_cfg = cnn_branch_cfg.copy()

        self.cnn_1d_branch = build_from_cfg(cnn_branch_cfg, MODELS)
        self.cnn_1d_output_size = getattr(self.cnn_1d_branch, 'cnn_output_size', None)
        if self.cnn_1d_output_size is None:
            raise AttributeError("CNN1D model must have attribute 'cnn_output_size'")

        # ------------------------------------------------------------------
        # 2. Build 2-D CNN branch for TOF (optional)
        # ------------------------------------------------------------------
        if self.use_tof:
            if tof_branch_cfg is None:
                # Legacy fallback: use minimal TOF config
                tof_branch_cfg = dict(type='TemporalTOF2DCNN',
                                      num_tof_sensors=5,
                                      seq_len=sequence_length,
                                      out_features=128)

            tof_branch_cfg = tof_branch_cfg.copy()
            self.cnn_2d_branch = build_from_cfg(tof_branch_cfg, MODELS)
            # Infer output size
            self.tof_2d_output_size = tof_branch_cfg.get('out_features', getattr(self.cnn_2d_branch, 'out_features', 128))
        else:
            # TOF disabled → no branch, zero additional features
            self.cnn_2d_branch = None
            self.tof_2d_output_size = 0

        # ------------------------------------------------------------------
        # 3. Build MLP branch
        # ------------------------------------------------------------------
        if mlp_branch_cfg is None:
            raise ValueError("mlp_branch_cfg is required")
        
        mlp_branch_cfg = mlp_branch_cfg.copy()
        mlp_branch_cfg.setdefault('output_dim', 32)

        # build branch
        self.mlp_branch = build_from_cfg(mlp_branch_cfg, MODELS)
        self.mlp_output_size = getattr(self.mlp_branch, 'output_dim', mlp_branch_cfg.get('output_dim', 32))

        # ------------------------------------------------------------------
        # 4. Fusion head (configurable)
        # ------------------------------------------------------------------
        combined_feature_size = self.cnn_1d_output_size + self.tof_2d_output_size + self.mlp_output_size

        # Ensure required parameters are set
        fusion_head_cfg = fusion_head_cfg.copy()
        fusion_head_cfg.setdefault('type', 'FusionHead')
        fusion_head_cfg.setdefault('input_dim', combined_feature_size)
        fusion_head_cfg.setdefault('num_classes', num_classes)
        
        # Build fusion head via registry
        self.classifier_head = build_from_cfg(fusion_head_cfg, MODELS)

    def forward(self, seq_input: torch.Tensor, tof_input: torch.Tensor, static_input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Defines the forward pass logic of the multimodal fusion model.
        
        Args:
            seq_input: Sequential sensor data (batch_size, seq_len, seq_features)
            tof_input: TOF sensor data (batch_size, seq_len, 320) - 5 sensors × 64 pixels
            static_input: Static demographic data (batch_size, static_features)
        """
        # 1. Process sequential data through the 1D CNN branch
        cnn_1d_features = self.cnn_1d_branch(seq_input,  mask=mask)
        
        # 2. Process TOF data (only if enabled)
        if self.use_tof:
            cnn_2d_features = self.cnn_2d_branch(tof_input,  mask=mask)
        else:
            cnn_2d_features = None
        
        # 3. Process static data through the MLP branch
        mlp_features = self.mlp_branch(static_input)
        
        # 4. Concatenate the features from all branches
        if cnn_2d_features is not None:
            combined_features = torch.cat((cnn_1d_features, cnn_2d_features, mlp_features), dim=1)
        else:
            combined_features = torch.cat((cnn_1d_features, mlp_features), dim=1)
        
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
        cnn_2d_info = self.cnn_2d_branch.get_model_info() if self.use_tof and self.cnn_2d_branch is not None else {'total_params': 0}
        mlp_info = self.mlp_branch.get_model_info()
        
        return {
            'total_params': total_params, 
            'model_size_mb': model_size_mb,
            'cnn_1d_params': cnn_1d_info['total_params'],
            'cnn_2d_params': cnn_2d_info['total_params'],
            'mlp_params': mlp_info['total_params'],
            'fusion_head_params': total_params - cnn_1d_info['total_params'] - cnn_2d_info['total_params'] - mlp_info['total_params']
        } 