import torch
import torch.nn as nn
from . import MODELS


@MODELS.register_module()
class LinearFusionHead(nn.Module):
    """
    Standard linear fusion head with configurable hidden layers.
    
    This is the default fusion strategy that concatenates features
    and passes them through a sequence of linear layers.
    """
    
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout_rates=None):
        super(LinearFusionHead, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if dropout_rates is None:
            dropout_rates = [0.5] * len(hidden_dims)
        
        # Ensure dropout_rates matches hidden_dims length
        if len(dropout_rates) < len(hidden_dims):
            dropout_rates.extend([0.5] * (len(hidden_dims) - len(dropout_rates)))
        
        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        
        for idx, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            dr = dropout_rates[idx] if idx < len(dropout_rates) else 0.5
            layers.append(nn.Dropout(dr))
            in_dim = out_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, combined_features):
        """
        Args:
            combined_features: Concatenated features from all branches
        Returns:
            logits: Classification logits
        """
        return self.layers(combined_features)


@MODELS.register_module()
class AttentionFusionHead(nn.Module):
    """
    Attention-based fusion head that learns to weight different modalities.
    
    This fusion strategy applies attention to weight the importance of
    different feature groups before classification.
    """
    
    def __init__(self, input_dim, num_classes, branch_dims, hidden_dims=None, dropout_rates=None):
        super(AttentionFusionHead, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if dropout_rates is None:
            dropout_rates = [0.5] * len(hidden_dims)
        
        self.branch_dims = branch_dims  # [cnn_dim, tof_dim, mlp_dim]
        
        # Attention mechanism for each branch
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, len(branch_dims)),
            nn.Softmax(dim=1)
        )
        
        # Feature projection for each branch
        self.branch_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dims[0]) for dim in branch_dims
        ])
        
        # Classification layers
        layers = []
        in_dim = hidden_dims[0]  # After projection and attention
        
        for idx, out_dim in enumerate(hidden_dims[1:], 1):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            dr = dropout_rates[idx-1] if idx-1 < len(dropout_rates) else 0.5
            layers.append(nn.Dropout(dr))
            in_dim = out_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, combined_features):
        """
        Args:
            combined_features: Concatenated features [cnn_features, tof_features, mlp_features]
        """
        # Split features back into branches
        branch_features = []
        start_idx = 0
        for dim in self.branch_dims:
            branch_features.append(combined_features[:, start_idx:start_idx + dim])
            start_idx += dim
        
        # Compute attention weights
        attention_weights = self.attention_weights(combined_features)  # (batch, num_branches)
        
        # Project each branch and apply attention
        projected_features = []
        for i, (features, projection) in enumerate(zip(branch_features, self.branch_projections)):
            projected = projection(features)  # (batch, hidden_dim[0])
            weighted = projected * attention_weights[:, i:i+1]  # Broadcast attention
            projected_features.append(weighted)
        
        # Sum weighted features
        fused_features = torch.sum(torch.stack(projected_features), dim=0)
        
        # Classify
        return self.classifier(fused_features)


@MODELS.register_module()
class BilinearFusionHead(nn.Module):
    """
    Bilinear fusion head for capturing cross-modal interactions.
    
    This fusion strategy uses bilinear pooling to capture interactions
    between different modalities before classification.
    """
    
    def __init__(self, input_dim, num_classes, branch_dims, fusion_dim=128, hidden_dims=None, dropout_rates=None):
        super(BilinearFusionHead, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        if dropout_rates is None:
            dropout_rates = [0.5] * len(hidden_dims)
        
        self.branch_dims = branch_dims
        self.fusion_dim = fusion_dim
        
        # Bilinear layers for each pair of modalities
        self.bilinear_layers = nn.ModuleList()
        for i in range(len(branch_dims)):
            for j in range(i + 1, len(branch_dims)):
                self.bilinear_layers.append(
                    nn.Bilinear(branch_dims[i], branch_dims[j], fusion_dim)
                )
        
        # Final classification layers
        total_fusion_dim = len(self.bilinear_layers) * fusion_dim + sum(branch_dims)
        
        layers = [nn.LayerNorm(total_fusion_dim)]
        in_dim = total_fusion_dim
        
        for idx, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            dr = dropout_rates[idx] if idx < len(dropout_rates) else 0.5
            layers.append(nn.Dropout(dr))
            in_dim = out_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, combined_features):
        """
        Args:
            combined_features: Concatenated features from all branches
        """
        # Split features back into branches
        branch_features = []
        start_idx = 0
        for dim in self.branch_dims:
            branch_features.append(combined_features[:, start_idx:start_idx + dim])
            start_idx += dim
        
        # Compute bilinear interactions
        bilinear_features = []
        layer_idx = 0
        for i in range(len(branch_features)):
            for j in range(i + 1, len(branch_features)):
                bilinear_out = self.bilinear_layers[layer_idx](branch_features[i], branch_features[j])
                bilinear_features.append(bilinear_out)
                layer_idx += 1
        
        # Concatenate original features and bilinear interactions
        all_features = branch_features + bilinear_features
        final_features = torch.cat(all_features, dim=1)
        
        return self.classifier(final_features)


# Alias for backward compatibility
@MODELS.register_module()
class FusionHead(LinearFusionHead):
    """Default fusion head - alias for LinearFusionHead"""
    pass 