import torch.nn as nn
from . import MODELS  # Import registry

@MODELS.register_module()
class MLP(nn.Module):
    """
    An MLP (Multi-Layer Perceptron) branch for processing static tabular features.

    This class builds a simple neural network to convert participant demographics 
    (e.g., age, sex) into a dense feature vector (embedding).
    """
    def __init__(self,
                 input_features: int,
                 hidden_dims: list,
                 output_dim: int,
                 dropout_rate: float):
        """
        Strict constructor for an MLP. All architectural parameters are required.
        """
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
        """
        Defines the forward pass logic for the MLP branch.
        """
        return self.layers(x)


    def get_model_info(self):
        """Get model parameter information"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_params': total_params}

