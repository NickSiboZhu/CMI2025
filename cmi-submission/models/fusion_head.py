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
    
    def __init__(self, input_dim, num_classes, hidden_dims, dropout_rates):
        super(LinearFusionHead, self).__init__()
        
        assert len(hidden_dims) == len(dropout_rates), "Length of hidden_dims and dropout_rates must be equal."
        
        layers = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        
        for out_dim, dr in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
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
    
    def __init__(self, input_dim, num_classes, branch_dims, hidden_dims, dropout_rates):
        super(AttentionFusionHead, self).__init__()
        
        # The first hidden_dim is for projection, the rest for classification
        assert len(hidden_dims) > 0, "hidden_dims must not be empty for AttentionFusionHead"
        assert len(dropout_rates) == len(hidden_dims) - 1, "dropout_rates must have one less element than hidden_dims"
        
        self.branch_dims = branch_dims  # [cnn_dim, tof_dim, mlp_dim]
        
        # Per-branch projection
        self.branch_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dims[0]) for dim in branch_dims
        ])

        # Shared gate network: produces one scalar score per branch
        proj_dim = hidden_dims[0]
        self.gate_fc = nn.Linear(proj_dim, 1)  # outputs (batch, 1)
        
        # Classification layers
        layers = []
        in_dim = hidden_dims[0]  # After projection and attention
        
        for idx, out_dim in enumerate(hidden_dims[1:], 1):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            dr = dropout_rates[idx-1]
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
        
        # Project each branch and compute scalar gate
        projected_feats = []  # list of (B, proj_dim)
        scores = []           # list of (B,)
        for features, projection in zip(branch_features, self.branch_projections):
            proj_feat = projection(features)            # (B, proj_dim)
            projected_feats.append(proj_feat)
            score = self.gate_fc(proj_feat).squeeze(-1) # (B,)
            scores.append(score)

        scores = torch.stack(scores, dim=1)             # (B, n_branches)
        alphas = torch.softmax(scores, dim=1)  # (B, n_branches)

        # Apply gates and sum
        fused_features = torch.zeros_like(projected_feats[0])  # (B, proj_dim)
        for i, proj_feat in enumerate(projected_feats):
            fused_features += proj_feat * alphas[:, i:i+1]
        
        # Classify
        return self.classifier(fused_features)


@MODELS.register_module()
class BilinearFusionHead(nn.Module):
    """
    Bilinear fusion head for capturing cross-modal interactions.
    
    This fusion strategy uses bilinear pooling to capture interactions
    between different modalities before classification.
    """
    
    def __init__(self, input_dim, num_classes, branch_dims, fusion_dim, hidden_dims, dropout_rates):
        super(BilinearFusionHead, self).__init__()
        
        assert len(hidden_dims) == len(dropout_rates), "Length of hidden_dims and dropout_rates must be equal."
        
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
        total_fusion_dim = len(self.bilinear_layers) * fusion_dim  # pure bilinear (exclude original features)
        
        layers = [nn.LayerNorm(total_fusion_dim)]
        in_dim = total_fusion_dim
        
        for out_dim, dr in zip(hidden_dims, dropout_rates):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
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
        
        # Use only bilinear features (pure bilinear fusion)
        final_features = torch.cat(bilinear_features, dim=1)
        
        return self.classifier(final_features)


@MODELS.register_module()
class TransformerFusionHead(nn.Module):
    """
    Transformer-style fusion head with a learnable [CLS] token.

    Workflow:
        1. Each modality vector is linearly projected to a common `embed_dim`.
        2. A learnable CLS token is prepended → sequence length = (#modalities + 1).
        3. Sequence is passed through `depth` layers of `nn.TransformerEncoder`.
        4. Output of CLS token (position 0) is fed to a classifier `Linear` layer.
    """

    def __init__(self,
                 branch_dims,
                 num_classes,
                 embed_dim,
                 num_heads,
                 depth,
                 dropout,
                 input_dim=None,  # kept for compatibility with MultimodalityModel
                 use_positional_encoding=False):
        super().__init__()

        self.branch_dims = branch_dims
        self.embed_dim = embed_dim
        self.num_tokens = len(branch_dims)

        # 1. Per-branch projections → embed_dim
        self.proj_layers = nn.ModuleList([
            nn.Linear(d, embed_dim) for d in branch_dims
        ])

        # 2. Learnable CLS token (1, 1, E)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.use_pe = use_positional_encoding
        if self.use_pe:
            # Positional embedding for up to (num_tokens+1) tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Final classifier on CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, combined_features):
        """combined_features: concat([mod1, mod2, …]) shape (B, sum(branch_dims))"""
        B = combined_features.size(0)
        # Split & project each modality
        feats = []
        start = 0
        for proj, dim in zip(self.proj_layers, self.branch_dims):
            end = start + dim
            feats.append(proj(combined_features[:, start:end]))
            start = end
        tokens = torch.stack(feats, dim=1)              # (B, N, E)  N = num_tokens

        # prepend CLS
        cls_tok = self.cls_token.expand(B, -1, -1)      # (B, 1, E)
        x = torch.cat([cls_tok, tokens], dim=1)          # (B, N+1, E)

        # optional positional encoding (not crucial when #tokens is small)
        if self.use_pe:
            x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)                          # (B, N+1, E)

        cls_out = x[:, 0]                                # (B, E)
        return self.classifier(cls_out)


# Alias for backward compatibility
@MODELS.register_module()
class FusionHead(LinearFusionHead):
    """Default fusion head - alias for LinearFusionHead"""
    pass 