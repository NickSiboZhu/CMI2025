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
    1. A 1D CNN branch for processing IMU sequential sensor data
    2. A 1D CNN branch for processing THM (thermopile) sequential data
    3. A 2D CNN branch for processing TOF sensor grids (spatial depth information)
    4. An MLP branch for processing static demographic data
    
    The features from all branches are fused and passed to a final classifier head.
    """
    def __init__(self,
                 # --- required configs ---
                 num_classes: int = 18,
                 sequence_length: int = 100,
                 # --- branch configs ---
                 imu_branch_cfg: dict = None,
                 mlp_branch_cfg: dict = None,
                 fusion_head_cfg: dict = None,
                 tof_branch_cfg: dict = None,
                 thm_branch_cfg: dict = None,
                 # --- modality toggles ---
                 use_tof: bool = True,
                 use_thm: bool = True,
                 ):
        """
        MultimodalityModel constructor. All configuration is provided via config dicts.
        Each branch (CNN1D for IMU, CNN1D for THM, MLP, TOF2D) has its own config dict for maximum flexibility.
        """
        super(MultimodalityModel, self).__init__()

        # Store modality flags
        self.use_tof = use_tof
        self.use_thm = use_thm

        # ------------------------------------------------------------------
        # 1. Build 1-D CNN branch for IMU
        # ------------------------------------------------------------------
        imu_branch_cfg = imu_branch_cfg.copy()

        self.imu_branch = build_from_cfg(imu_branch_cfg, MODELS)
        self.imu_output_size = getattr(self.imu_branch, 'cnn_output_size', None)
        if self.imu_output_size is None:
            raise AttributeError("CNN1D model must have attribute 'cnn_output_size'")

        # ------------------------------------------------------------------
        # 2. Build 1-D CNN branch for THM (thermopile) (optional)
        # ------------------------------------------------------------------
        if self.use_thm:
            thm_branch_cfg = thm_branch_cfg.copy()

            # Ensure required parameters are set
            thm_branch_cfg.setdefault('type', 'CNN1D')
            thm_branch_cfg.setdefault('sequence_length', sequence_length)

            self.thm_branch = build_from_cfg(thm_branch_cfg, MODELS)
            self.thm_output_size = getattr(
                self.thm_branch,
                'cnn_output_size',
                thm_branch_cfg.get('output_dim', 64)
            )
        else:
            # THM disabled
            self.thm_branch = None
            self.thm_output_size = 0

        # ------------------------------------------------------------------
        # 3. Build 2-D CNN branch for TOF (optional)
        # ------------------------------------------------------------------
        if self.use_tof:
            tof_branch_cfg = tof_branch_cfg.copy()

            self.tof_branch = build_from_cfg(tof_branch_cfg, MODELS)
            # Infer output size strictly from the instantiated branch first ― this captures
            # dynamic situations (e.g., bidirectional LSTM doubling hidden size).
            # Fall back to config only if the attribute is missing.
            self.tof_2d_output_size = getattr(self.tof_branch, 'out_features', tof_branch_cfg.get('out_features', 128))
        else:
            # TOF disabled → no branch, zero additional features
            self.tof_branch = None
            self.tof_2d_output_size = 0

        # ------------------------------------------------------------------
        # 4. Build MLP branch
        # ------------------------------------------------------------------
        mlp_branch_cfg = mlp_branch_cfg.copy()

        # build branch
        self.mlp_branch = build_from_cfg(mlp_branch_cfg, MODELS)
        self.mlp_output_size = getattr(self.mlp_branch, 'output_dim', mlp_branch_cfg.get('output_dim', 32))

        # ------------------------------------------------------------------
        # 5. Fusion head (configurable)
        # ------------------------------------------------------------------
        # Dynamically calculate combined feature size from actual branch outputs
        combined_feature_size = (self.imu_output_size + self.thm_output_size + 
                               self.tof_2d_output_size + self.mlp_output_size)

        # Ensure required parameters are set
        fusion_head_cfg = fusion_head_cfg.copy()
        fusion_head_cfg.setdefault('type', 'FusionHead')
        fusion_head_cfg['num_classes'] = num_classes
        fusion_head_cfg['input_dim'] = combined_feature_size  # FusionHead expects 'input_dim'

        # Auto-configure branch_dims for fusion heads that require it
        fusion_head_type = fusion_head_cfg.get('type', 'FusionHead')
        if fusion_head_type in ('BilinearFusionHead', 'AttentionFusionHead', 'TransformerFusionHead', 'GatedFusionHead') and \
           'branch_dims' not in fusion_head_cfg:
            # Order must match concatenation order in forward()
            branch_dims = [self.imu_output_size]
            if self.use_thm:
                branch_dims.append(self.thm_output_size)
            # MLP features always present
            branch_dims.append(self.mlp_output_size)
            if self.use_tof:
                branch_dims.append(self.tof_2d_output_size)
            fusion_head_cfg['branch_dims'] = branch_dims
        
        # Print for visibility during model construction
        print(f"\n[MultimodalityModel] Detected branch output dimensions:")
        print(f"  IMU: {self.imu_output_size}")
        if self.use_thm:
            print(f"  THM: {self.thm_output_size}")
        if self.use_tof:
            print(f"  TOF: {self.tof_2d_output_size}")
        print(f"  MLP: {self.mlp_output_size}")
        print(f"  Total combined features: {combined_feature_size}")

        self.classifier_head = build_from_cfg(fusion_head_cfg, MODELS)

    def forward(self, imu_input: torch.Tensor, thm_input: torch.Tensor, tof_input: torch.Tensor, static_input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Defines the forward pass logic of the multimodal fusion model.
        
        Args:
            imu_input: IMU sequential sensor data (batch_size, seq_len, imu_features)
            thm_input: THM sequential sensor data (batch_size, seq_len, thm_features)
            tof_input: TOF sensor data (batch_size, seq_len, 320) - 5 sensors × 64 pixels
            static_input: Static demographic data (batch_size, static_features)
            mask: An optional boolean tensor to mask padded elements in sequences. (batch_size, seq_len)
        """
        # 1. Process IMU data through the 1D CNN branch
        imu_features = self.imu_branch(imu_input, mask=mask)
        
        # 2. Process THM data (optional)
        thm_features = self.thm_branch(thm_input, mask=mask) if self.use_thm else None
        
        # 3. Process TOF data (optional)
        tof_features = self.tof_branch(tof_input, mask=mask) if self.use_tof else None
        
        # 4. Process static data through the MLP branch
        mlp_features = self.mlp_branch(static_input)
        
        # 5. Concatenate the features from all branches
        features_to_concat = [imu_features, mlp_features]
        if thm_features is not None:
            features_to_concat.insert(1, thm_features)
        if tof_features is not None:
            features_to_concat.append(tof_features)
        
        combined_features = torch.cat(features_to_concat, dim=1)
        
        # 6. Pass the fused features through the final classifier head
        output = self.classifier_head(combined_features)
        
        return output

    def get_model_info(self) -> dict:
        """
        Helper function to return the model's total parameter count and approximate size.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_size_mb = total_params * 4 / (1024**2)  # Assuming float32 parameters
        
        # Get info from individual branches
        imu_info = self.imu_branch.get_model_info()
        thm_info = self.thm_branch.get_model_info() if self.thm_branch is not None else {'total_params': 0}
        tof_info = self.tof_branch.get_model_info() if self.tof_branch is not None else {'total_params': 0}
        mlp_info = self.mlp_branch.get_model_info()
        
        return {
            'total_params': total_params, 
            'model_size_mb': model_size_mb,
            'imu_params': imu_info['total_params'],
            'thm_params': thm_info['total_params'],
            'tof_params': tof_info['total_params'],
            'mlp_params': mlp_info['total_params'],
            'fusion_head_params': total_params - imu_info['total_params'] - thm_info['total_params'] - tof_info['total_params'] - mlp_info['total_params']
        } 