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
                 # v2 legacy args
                 static_input_features: int = None,
                 mlp_output_size: int = None,
                 # v3 new args
                 input_features: int = None,
                 hidden_dims: list = None,
                 output_dim: int = None,
                 dropout_rate: float = 0.5):
        """
        Dynamic constructor that supports both the original (static_input_features, mlp_output_size)
        signature and the new v3 signature based on a list of hidden_dims.
        """
        super(MLP, self).__init__()

        # Back-compat mapping ------------------------------------------------
        if input_features is None:
            input_features = static_input_features
        if output_dim is None and mlp_output_size is not None:
            output_dim = mlp_output_size
        if hidden_dims is None:
            hidden_dims = [64]
        if output_dim is None:
            output_dim = 32

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

    # New helper for MultimodalityModel
    def get_output_size(self):
        return self.output_dim

