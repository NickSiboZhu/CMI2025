import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """
    Custom Dataset for handling multimodal inputs:
    - Sequential sensor data (for the CNN)
    - Static demographic data (for the MLP)
    """
    def __init__(self, X_seq, X_static, y):
        """
        Args:
            X_seq (np.array): The sequential data tensor.
            X_static (np.array): The static data tensor.
            y (np.array): The labels.
        """
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        Returns a tuple of inputs and the corresponding label.
        The DataLoader will receive this as: ((seq_data, static_data), label)
        """
        return (self.X_seq[idx], self.X_static[idx]), self.y[idx]
