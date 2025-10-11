""" 
Pytorch Dataset classes for training. Most disused as of 10/3/25
"""


import torch
import numpy as np


class MemmapNormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, feature_memmap: np.memmap, label_memmap: np.memmap, stats_file: str):
        assert feature_memmap.shape[0] == label_memmap.shape[0], "Features and labels must have the same number of samples."

        self.features = feature_memmap
        self.labels = label_memmap
        self.n_samples = feature_memmap.shape[0]

        # Compute and store min, max, and range
        self.feature_min = np.min(self.features, axis=0)
        self.feature_max = np.max(self.features, axis=0)
        self.label_min = np.min(self.labels, axis=0)
        self.label_max = np.max(self.labels, axis=0)

        self.feature_range = np.where(self.feature_max - self.feature_min == 0, 1, self.feature_max - self.feature_min)
        self.label_range = np.where(self.label_max - self.label_min == 0, 1, self.label_max - self.label_min)

        # Save normalization stats
        with open(stats_file, "w") as f:
            f.write("# Feature normalization stats\n")
            for i, (fmin, fmax) in enumerate(zip(self.feature_min, self.feature_max)):
                f.write(f"feature_{i}_min: {fmin}\n")
                f.write(f"feature_{i}_max: {fmax}\n")

            f.write("\n# Label normalization stats\n")
            for i, (lmin, lmax) in enumerate(zip(self.label_min, self.label_max)):
                f.write(f"label_{i}_min: {lmin}\n")
                f.write(f"label_{i}_max: {lmax}\n")

        # Store normalization as tensors for GPU efficiency
        self.feature_min_t = torch.tensor(self.feature_min, dtype=torch.float32)
        self.feature_range_t = torch.tensor(self.feature_range, dtype=torch.float32)
        self.label_min_t = torch.tensor(self.label_min, dtype=torch.float32)
        self.label_range_t = torch.tensor(self.label_range, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        raw_feature = self.features[idx]
        raw_label = self.labels[idx]

        norm_feature = (raw_feature - self.feature_min) / self.feature_range
        norm_label = (raw_label - self.label_min) / self.label_range

        return (
            torch.tensor(norm_feature[:-1], dtype=torch.float32),
            torch.tensor(norm_label, dtype=torch.float32)
        )

    def inverse_transform_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse-transform normalized features (stay on same device)."""
        return tensor * self.feature_range_t.to(tensor.device) + self.feature_min_t.to(tensor.device)

    def inverse_transform_labels(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse-transform normalized labels (stay on same device)."""
        return tensor * self.label_range_t.to(tensor.device) + self.label_min_t.to(tensor.device)
    
class SaturationDataset(MemmapNormalizedDataset):
    def __init__(self, feature_memmap, label_memmap, stats_file):
        super().__init__(self, feature_memmap, label_memmap, stats_file)

    def __getitem__(self, idx):
        x_full, y = super().__getitem__(idx)
        return x_full[:-1], y  # Drop the last input feature
    
import torch
import numpy as np
import os

class MemmapNormalizedDatasetPKL(torch.utils.data.Dataset):
    def __init__(self, feature_path: str, label_path: str, stats_file: str):
        self.feature_path = feature_path
        self.label_path = label_path
        self.stats_file = stats_file

        self.features = None
        self.labels = None

        # Precompute and store normalization stats once
        self._initialize_normalization()

    def _initialize_normalization(self):
        # Temporarily load data to compute stats
        features = np.load(self.feature_path, mmap_mode='r')
        labels = np.load(self.label_path, mmap_mode='r')
        self.n_samples = features.shape[0]

        self.feature_min = np.min(features, axis=0)
        self.feature_max = np.max(features, axis=0)
        self.label_min = np.min(labels, axis=0)
        self.label_max = np.max(labels, axis=0)

        self.feature_range = np.where(self.feature_max - self.feature_min == 0, 1, self.feature_max - self.feature_min)
        self.label_range = np.where(self.label_max - self.label_min == 0, 1, self.label_max - self.label_min)

        # Save normalization stats
        with open(self.stats_file, "w") as f:
            f.write("# Feature normalization stats\n")
            for i, (fmin, fmax) in enumerate(zip(self.feature_min, self.feature_max)):
                f.write(f"feature_{i}_min: {fmin}\n")
                f.write(f"feature_{i}_max: {fmax}\n")

            f.write("\n# Label normalization stats\n")
            for i, (lmin, lmax) in enumerate(zip(self.label_min, self.label_max)):
                f.write(f"label_{i}_min: {lmin}\n")
                f.write(f"label_{i}_max: {lmax}\n")

        # Convert to torch tensors for fast GPU access later
        self.feature_min_t = torch.tensor(self.feature_min, dtype=torch.float32)
        self.feature_range_t = torch.tensor(self.feature_range, dtype=torch.float32)
        self.label_min_t = torch.tensor(self.label_min, dtype=torch.float32)
        self.label_range_t = torch.tensor(self.label_range, dtype=torch.float32)

    def _lazy_load(self):
        if self.features is None or self.labels is None:
            self.features = np.load(self.feature_path, mmap_mode='r')
            self.labels = np.load(self.label_path, mmap_mode='r')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self._lazy_load()

        raw_feature = self.features[idx]
        raw_label = self.labels[idx]

        norm_feature = (raw_feature - self.feature_min) / self.feature_range
        norm_label = (raw_label - self.label_min) / self.label_range

        return (
            torch.tensor(norm_feature[:-1], dtype=torch.float32),  # exclude last feature if needed
            torch.tensor(norm_label, dtype=torch.float32)
        )

    def inverse_transform_features(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.feature_range_t.to(tensor.device) + self.feature_min_t.to(tensor.device)

    def inverse_transform_labels(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.label_range_t.to(tensor.device) + self.label_min_t.to(tensor.device)


class InMemoryNormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, feature_path: str, label_path: str, stats_file: str = None):
        # Load full data into RAM
        features_np = np.load(feature_path)
        labels_np = np.load(label_path)
        assert features_np.shape[0] == labels_np.shape[0], "Mismatched number of samples."

        self.features = torch.from_numpy(features_np).float()
        self.labels = torch.from_numpy(labels_np).float()
        self.n_samples = self.features.shape[0]

        # Compute normalization stats
        self.feature_min = self.features.min(dim=0).values
        self.feature_max = self.features.max(dim=0).values
        self.label_min = self.labels.min(dim=0).values
        self.label_max = self.labels.max(dim=0).values

        self.feature_range = torch.where(
            self.feature_max - self.feature_min == 0,
            torch.ones_like(self.feature_max),
            self.feature_max - self.feature_min
        )
        self.label_range = torch.where(
            self.label_max - self.label_min == 0,
            torch.ones_like(self.label_max),
            self.label_max - self.label_min
        )

        # Optionally write normalization stats
        if stats_file:
            with open(stats_file, "w") as f:
                f.write("# Feature normalization stats\n")
                for i, (fmin, fmax) in enumerate(zip(self.feature_min, self.feature_max)):
                    f.write(f"feature_{i}_min: {fmin.item()}\n")
                    f.write(f"feature_{i}_max: {fmax.item()}\n")
                f.write("\n# Label normalization stats\n")
                for i, (lmin, lmax) in enumerate(zip(self.label_min, self.label_max)):
                    f.write(f"label_{i}_min: {lmin.item()}\n")
                    f.write(f"label_{i}_max: {lmax.item()}\n")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        norm_feature = (self.features[idx] - self.feature_min) / self.feature_range
        norm_label = (self.labels[idx] - self.label_min) / self.label_range
        return norm_feature[:-1], norm_label  # Exclude last feature if required

    def inverse_transform_features(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.feature_range.to(tensor.device) + self.feature_min.to(tensor.device)

    def inverse_transform_labels(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.label_range.to(tensor.device) + self.label_min.to(tensor.device)

class TensorDatasetNormalized(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0], "Features and labels must have same length"
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Support for single index
        if isinstance(idx, int):
            return self.features[idx][:-1], self.labels[idx]
        
        # Support for multiple indices
        elif isinstance(idx, (list, tuple, torch.Tensor)):
            features_subset = self.features[idx, :-1]  # Drop last column
            labels_subset = self.labels[idx]
            return features_subset, labels_subset
        
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def all_items(self):
        return self.features, self.labels

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