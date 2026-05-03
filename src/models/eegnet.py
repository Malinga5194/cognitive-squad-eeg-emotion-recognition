"""
EEGNet - A Compact CNN for EEG-Based Brain-Computer Interfaces
Reference: Lawhern et al., 2018 - "EEGNet: A Compact Convolutional Neural
Network for EEG-based Brain-Computer Interfaces"

This is a lightweight CNN specifically designed for EEG classification.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet architecture adapted for emotion classification.

    Architecture:
    1. Temporal convolution (learns frequency filters)
    2. Depthwise convolution (learns spatial filters per channel)
    3. Separable convolution (combines spatial-temporal features)
    4. Classification head
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_samples: int = 7680,
        n_classes: int = 2,
        dropout_rate: float = 0.5,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_length: int = 64,
    ):
        """
        Args:
            n_channels: Number of EEG channels
            n_samples: Number of time samples per trial
            n_classes: Number of output classes
            dropout_rate: Dropout probability
            F1: Number of temporal filters
            F2: Number of pointwise filters
            D: Depth multiplier for depthwise convolution
            kernel_length: Length of temporal convolution kernel
        """
        super().__init__()

        # Block 1: Temporal + Spatial filtering
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise convolution (spatial filter)
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            # Depthwise separable convolution
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Calculate flattened size
        self._flat_size = self._get_flat_size(n_channels, n_samples)

        # Classification head
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_channels: int, n_samples: int) -> int:
        """Calculate the flattened feature size after conv blocks."""
        x = torch.zeros(1, 1, n_channels, n_samples)
        x = self.block1(x)
        x = self.block2(x)
        return x.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_channels, n_samples)

        Returns:
            Output logits of shape (batch, n_classes)
        """
        # Add channel dimension: (batch, 1, n_channels, n_samples)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
