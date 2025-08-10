from utils.registry import Registry

# Create the registry instance
MODELS = Registry('models')

# Import model implementations so that their registration decorators execute
from . import cnn1d
from . import cnn2d
from . import mlp
from . import multimodality
from . import fusion_head

__all__ = ['MODELS', 'CNN1D', 'TemporalTOF2DCNN', 'MLP', 'MultimodalityModel']
