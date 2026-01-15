"""Data export functions."""
    
import torch

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0], "Features and labels must have same length"
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
class TensorDatasetThree(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, binarylabels: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0], "Features and labels must have same length"
        assert features.shape[0] == binarylabels.shape[0], "Features and binary labels must have same length"

        self.binarylabels = binarylabels
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
            return self.features[idx], self.binarylabels[idx], self.labels[idx]
    
class TensorDatasetFour(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, binarylabels: torch.Tensor, labels: torch.Tensor, molelabels: torch.Tensor):
        assert features.shape[0] == labels.shape[0], "Features and labels must have same length"
        assert features.shape[0] == binarylabels.shape[0], "Features and binary labels must have same length"
        assert features.shape[0] == molelabels.shape[0], "Features and molar labels must have same length"


        self.binarylabels = binarylabels
        self.features = features
        self.labels = labels
        self.molelabels = molelabels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
            return self.features[idx], self.binarylabels[idx], self.labels[idx], self.molelabels[idx]