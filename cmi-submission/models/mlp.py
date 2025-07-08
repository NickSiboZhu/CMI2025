import torch.nn as nn

class MLP(nn.Module):
    """
    An MLP (Multi-Layer Perceptron) branch for processing static tabular features.

    This class builds a simple neural network to convert participant demographics 
    (e.g., age, sex) into a dense feature vector (embedding).
    """
    def __init__(self, static_input_features: int, mlp_output_size: int = 32):
        """
        Initializes the layers of the MLP branch.

        Args:
            static_input_features (int): Number of input static features (7 in this project).
            mlp_output_size (int): The final output feature dimension of the MLP branch. Defaults to 32.
        """
        super(MLP, self).__init__()
        
        # We use nn.Sequential to chain all layers together neatly.
        # Data will pass through these layers in the defined order.
        self.layers = nn.Sequential(
            # First fully connected layer: Maps the 7 input features to a 64-dimensional hidden space.
            nn.Linear(static_input_features, 64),
            
            # ReLU activation function: Introduces non-linearity, allowing the model to learn more complex relationships.
            nn.ReLU(),
            
            # Dropout layer: Randomly "turns off" some neurons during training, an effective method to prevent overfitting.
            nn.Dropout(0.5),
            
            # Second fully connected layer: Compresses the 64-dimensional hidden features to the final 32-dimensional output.
            # This 32D vector is the feature embedding for the static data.
            nn.Linear(64, mlp_output_size)
        )

    def forward(self, x):
        """
        Defines the forward pass logic for the MLP branch.
        """
        return self.layers(x)

