"""
Hybrid CNN-LSTM Model for EEG Classification
Combines spatial feature extraction (CNN) with temporal modeling (LSTM).
This is the flagship model for the project.
"""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM architecture for EEG emotion classification.

    Architecture:
    1. 1D CNN extracts local spatial-temporal features
    2. LSTM captures long-range temporal dependencies
    3. Attention mechanism weights important time steps
    4. Fully connected classifier
    """

    def __init__(
        self,
        n_channels: int = 32,
        n_classes: int = 2,
        cnn_filters: list[int] | None = None,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        if cnn_filters is None:
            cnn_filters = [32, 64]

        # CNN feature extractor
        cnn_layers = []
        in_channels = n_channels
        for out_channels in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM temporal encoder
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, n_channels, n_samples)

        Returns:
            Output logits of shape (batch, n_classes)
        """
        # CNN: (batch, n_channels, n_samples) -> (batch, cnn_filters[-1], reduced_time)
        cnn_out = self.cnn(x)

        # Transpose for LSTM: (batch, reduced_time, cnn_filters[-1])
        lstm_in = cnn_out.permute(0, 2, 1)

        # LSTM: (batch, reduced_time, lstm_hidden * 2)
        lstm_out, _ = self.lstm(lstm_in)

        # Attention: weight each time step
        attn_weights = self.attention(lstm_out)       # (batch, time, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, lstm_hidden * 2)

        # Classify
        out = self.classifier(context)
        return out
