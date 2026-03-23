import torch.nn as nn
from . import MODELS

@MODELS.register_module()
class MLP(nn.Module):
    """MLP branch for static tabular features."""
    def __init__(self,
                 input_features: int,
                 hidden_dims: list,
                 output_dim: int,
                 dropout_rate: float):
        """All architectural dimensions are provided explicitly by the config."""
        super(MLP, self).__init__()

        layers = []
        in_dim = input_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.layers = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.layers(x)


    def get_model_info(self):
        """Get model parameter information"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_params': total_params}

