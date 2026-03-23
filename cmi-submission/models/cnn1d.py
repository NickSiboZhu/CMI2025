import torch
import torch.nn as nn
from . import MODELS
import torch.nn.functional as F

class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var",  torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var",  None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(1).to(x.dtype)
        count = m.sum(dim=(0, 2))
        count = count.clamp(min=1.0)

        if self.training or not self.track_running_stats:
            sum_ = (x * m).sum(dim=(0, 2))
            mean = sum_ / count

            var = ((x - mean[None, :, None])**2 * m).sum(dim=(0, 2)) / count

            if self.track_running_stats:
                with torch.no_grad():
                    # Skip running-stat updates when the batch is almost entirely padding.
                    total = torch.tensor(x.size(0) * x.size(2), dtype=torch.float32, device=x.device)
                    valid_ratio = (count.to(torch.float32) / total)

                    mom = torch.tensor(self.momentum, dtype=torch.float32, device=x.device)
                    new_rm = self.running_mean.to(torch.float32) * (1 - mom) + mean.to(torch.float32) * mom
                    new_rv = self.running_var.to(torch.float32)  * (1 - mom) + var.to(torch.float32)  * mom

                    w = (valid_ratio > 1e-3).to(torch.float32)
                    self.running_mean.copy_(torch.lerp(self.running_mean, new_rm.to(self.running_mean.dtype), w))
                    self.running_var.copy_( torch.lerp(self.running_var,  new_rv.to(self.running_var.dtype),  w))
                    self.num_batches_tracked.add_(int(w.item() > 0))
        else:
            mean = self.running_mean
            var  = self.running_var

        x_hat = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        if self.affine:
            x_hat = x_hat * self.weight[None, :, None] + self.bias[None, :, None]

        # Keep padded positions at zero for later layers.
        return x_hat * m


class MaskedSE1D(nn.Module):
    """
    Squeeze-and-Excite block that pools only across valid timesteps.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, reduced_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(reduced_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L).
            mask (Tensor): Mask tensor of shape (B, L).
        
        Returns:
            Tensor: Output tensor of shape (B, C, L).
        """
        mask_expanded = mask.unsqueeze(1).to(x.dtype)
        
        numerator = torch.sum(x * mask_expanded, dim=2, keepdim=True)
        
        denominator = mask_expanded.sum(dim=2, keepdim=True).clamp(min=1e-9)
        
        squeezed = numerator / denominator

        excited = self.fc1(squeezed)
        excited = self.act(excited)
        excited = self.fc2(excited)
        
        attention_scores = self.gate(excited)
        return x * attention_scores

@MODELS.register_module()
class CNN1D(nn.Module):
    """
    1D CNN for padded variable-length sequences.

    The branch reapplies masks throughout the convolution stack so padding never
    bleeds into the learned temporal features.
    """
    def __init__(self,
                 input_channels: int,
                 filters: list,
                 kernel_sizes: list,
                 temporal_aggregation: str,
                 use_residual: bool,
                 use_se: bool,
                 sequence_length: int = None,
                 temporal_mode: str = None,
                 lstm_hidden: int = None,
                 lstm_layers: int = None,
                 bidirectional: bool = None,
                 num_heads: int = None,
                 num_layers: int = None,
                 ff_dim: int = None,
                 dropout: float = None,
                 se_reduction: int = None):
        super(CNN1D, self).__init__()
        
        assert len(filters) == len(kernel_sizes), "filters and kernel_sizes length mismatch"
        if use_se and (se_reduction is None):
            raise ValueError("'se_reduction' must be provided when 'use_se' is True.")

        layers = []
        self.bn_layers = nn.ModuleList()
        self.se_layers = nn.ModuleList() if use_se else None
        self.residual_projections = nn.ModuleList()
        
        self.use_residual = use_residual
        
        in_channels = input_channels
        for i, (out_c, k) in enumerate(zip(filters, kernel_sizes)):
            # Keep sequence length stable until pooling updates the mask explicitly.
            layers.append(nn.Conv1d(in_channels, out_c, kernel_size=k, padding='same', bias=False))
            self.bn_layers.append(MaskedBatchNorm1d(out_c))

            if self.se_layers is not None:
                self.se_layers.append(MaskedSE1D(out_c, se_reduction))
            
            # Add 1x1 projection for residual connections when dimensions don't match
            if use_residual and in_channels != out_c:
                self.residual_projections.append(nn.Conv1d(in_channels, out_c, kernel_size=1, stride=1, padding=0, bias=False))
            else:
                self.residual_projections.append(None)
            
            in_channels = out_c
            
        self.conv_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.temporal_aggregation = temporal_aggregation
        
        if temporal_aggregation == 'temporal_encoder':
            if not all([sequence_length, temporal_mode]):
                raise ValueError("`sequence_length` and `temporal_mode` must be provided for temporal_encoder.")
            if temporal_mode == 'transformer' and dropout is None:
                raise ValueError("`dropout` must be provided when `temporal_mode` is 'transformer'.")
            
            from .cnn2d import TemporalEncoder
            
            reduced_seq_len = sequence_length
            for i in range(len(filters) - 1):
                reduced_seq_len = reduced_seq_len // 2
            
            self.temporal_encoder = TemporalEncoder(
                mode=temporal_mode,
                input_dim=filters[-1],
                seq_len=reduced_seq_len,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                bidirectional=bidirectional,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=ff_dim,
                dropout=dropout
            )
            self.cnn_output_size = self.temporal_encoder.output_dim
        elif temporal_aggregation == 'global_pool':
            self.cnn_output_size = filters[-1]
        else:
            raise ValueError(f"Unknown temporal_aggregation: '{temporal_aggregation}'")
    
    def forward(self, x, mask=None, channel_mask: torch.Tensor = None):
        """
        Defines the forward pass with robust masking at each step.
        
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, channels)
            mask (Tensor, optional): Mask tensor of shape (batch, seq_len) where 1s are real data.
        """
        x = x.transpose(1, 2)

        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[2], device=x.device)

        # Zero missing channels before the first convolution so they never affect
        # masked normalization or residual paths.
        if channel_mask is not None and channel_mask.numel() > 0:
            x = x * channel_mask.unsqueeze(-1)

        for i, conv in enumerate(self.conv_layers):
            x = x * mask.unsqueeze(1)
            
            if self.use_residual:
                residual = x
            
            x = conv(x)
            x = self.bn_layers[i](x, mask)

            if self.se_layers is not None:
                x = self.se_layers[i](x, mask)
            
            if self.use_residual:
                if self.residual_projections[i] is not None:
                    residual = self.residual_projections[i](residual)
                    residual = residual * mask.unsqueeze(1)
                x = x + residual
            
            x = self.activation(x)
            x = x * mask.unsqueeze(1)

            if i < len(self.conv_layers) - 1:
                x = self.pool(x)
                mask_for_pooling = mask.unsqueeze(1).float() 
                pooled_mask = self.pool(mask_for_pooling)
                mask = (pooled_mask > 0).squeeze(1)

        if self.temporal_aggregation == 'temporal_encoder':
            x = x.transpose(1, 2)
            output = self.temporal_encoder(x, mask)
        else:
            # Global pooling must stay mask-aware for left-padded sequences.
            mask_expanded = mask.unsqueeze(1)
            numerator = torch.sum(x * mask_expanded, dim=2)
            denominator = torch.sum(mask, dim=1).unsqueeze(1).clamp(min=1e-9)
            output = numerator / denominator

        return output
