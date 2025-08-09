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
    A hybrid multimodal fusion model for gesture recognition.

    This model combines:
    1. A 1D CNN branch for processing IMU/THM time-domain data.
    2. A 2D CNN branch for processing Spectrograms (time-frequency data).
    3. A 2D CNN branch for processing TOF sensor grids.
    4. An MLP branch for processing static demographic data.
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
                 # --- NEW: Config for the spectrogram branch ---
                 spec_branch_cfg: dict = None,
                 # --- modality toggles ---
                 use_tof: bool = True,
                 use_thm: bool = True,
                 # --- NEW: Toggle for the spectrogram branch ---
                 use_spec: bool = True,
                 ):
        super(MultimodalityModel, self).__init__()

        # Store modality flags
        self.use_tof = use_tof
        self.use_thm = use_thm
        # --- NEW: Store spectrogram flag ---
        self.use_spec = use_spec

        # --- Build 1D CNN branches (IMU, THM) ---
        self.imu_branch = build_from_cfg(imu_branch_cfg, MODELS)
        self.imu_output_size = self.imu_branch.cnn_output_size
        
        if self.use_thm:
            self.thm_branch = build_from_cfg(thm_branch_cfg, MODELS)
            self.thm_output_size = self.thm_branch.cnn_output_size
        else:
            self.thm_branch = None
            self.thm_output_size = 0

        # --- NEW: Build 2D CNN branch for Spectrograms ---
        if self.use_spec:
            spec_branch_cfg = spec_branch_cfg.copy()
            self.spec_branch = build_from_cfg(spec_branch_cfg, MODELS)
            self.spec_output_size = self.spec_branch.out_features
        else:
            self.spec_branch = None
            self.spec_output_size = 0
            
        # --- Build 2D CNN branch for TOF ---
        if self.use_tof:
            self.tof_branch = build_from_cfg(tof_branch_cfg, MODELS)
            self.tof_2d_output_size = self.tof_branch.out_features
        else:
            self.tof_branch = None
            self.tof_2d_output_size = 0

        # --- Build MLP branch ---
        self.mlp_branch = build_from_cfg(mlp_branch_cfg, MODELS)
        self.mlp_output_size = self.mlp_branch.output_dim

        # --- Fusion head ---
        combined_feature_size = (self.imu_output_size + self.thm_output_size + 
                               self.spec_output_size + self.tof_2d_output_size + 
                               self.mlp_output_size)
        
        fusion_head_cfg['num_classes'] = num_classes
        fusion_head_cfg['input_dim'] = combined_feature_size
        
        print(f"\n[MultimodalityModel] Detected branch output dimensions:")
        print(f"  IMU (1D): {self.imu_output_size}")
        if self.use_thm: print(f"  THM (1D): {self.thm_output_size}")
        if self.use_spec: print(f"  Spectrogram (2D): {self.spec_output_size}")
        if self.use_tof: print(f"  TOF (2D): {self.tof_2d_output_size}")
        print(f"  MLP: {self.mlp_output_size}")
        print(f"  Total combined features: {combined_feature_size}")
        
        self.classifier_head = build_from_cfg(fusion_head_cfg, MODELS)

    # --- MODIFIED: Updated forward pass signature ---
    def forward(self, imu_input: torch.Tensor, thm_input: torch.Tensor, 
                tof_input: torch.Tensor, spec_input: torch.Tensor, 
                static_input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Defines the forward pass logic of the hybrid multimodal fusion model.
        """
        # 1. Process 1D time-domain data
        imu_features = self.imu_branch(imu_input, mask=mask)
        thm_features = self.thm_branch(thm_input, mask=mask) if self.use_thm else None
        
        # 2. Process 2D time-frequency data (Spectrograms)
        spec_features = self.spec_branch(spec_input) if self.use_spec else None
        
        # 3. Process 2D spatial data (TOF)
        tof_features = self.tof_branch(tof_input, mask=mask) if self.use_tof else None
        
        # 4. Process static data
        mlp_features = self.mlp_branch(static_input)
        
        # 5. Concatenate features from all active branches
        features_to_concat = [imu_features]
        if thm_features is not None: features_to_concat.append(thm_features)
        if spec_features is not None: features_to_concat.append(spec_features)
        if tof_features is not None: features_to_concat.append(tof_features)
        features_to_concat.append(mlp_features)
        
        combined_features = torch.cat(features_to_concat, dim=1)
        
        # 6. Final classification
        output = self.classifier_head(combined_features)
        
        return output