# Copyright (c) 2026 D.S.M. Perera (MS26906294), Cognitive Squad.
# MSc in Information Technology, Sri Lanka Institute of Information Technology (SLIIT).
# Developed for the Artificial Intelligence module group project.
# All rights reserved.

"""
PyTorch Dataset classes for EEG data.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    PyTorch Dataset for raw EEG signals (for CNN/LSTM models).
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        label_index: int = 0,
        normalize: bool = True,
    ):
        """
        Args:
            data: EEG data of shape (n_trials, n_channels, n_samples)
            labels: Labels of shape (n_trials, n_labels)
            label_index: Which label to use (0=valence, 1=arousal, etc.)
            normalize: Whether to z-score normalize each trial
        """
        self.data = data.astype(np.float32)
        self.labels = labels[:, label_index].astype(np.int64)

        if normalize:
            # Z-score normalize each trial independently
            mean = self.data.mean(axis=(1, 2), keepdims=True)
            std = self.data.std(axis=(1, 2), keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            self.data = (self.data - mean) / std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class EEGFeatureDataset(Dataset):
    """
    PyTorch Dataset for extracted EEG features (for MLP models).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        label_index: int = 0,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(
            labels[:, label_index], dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
