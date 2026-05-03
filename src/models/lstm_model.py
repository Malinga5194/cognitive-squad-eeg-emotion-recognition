"""
LSTM Model for EEG Classification
Captures temporal dependencies in EEG signals.
"""
import torch
import torch.nn as nn


class EEGLSTM(nn.Module):
    """
    Bidirectional LSTM for EEG emotion classification.

    The model processes EEG as a sequence of time steps,
    where each step contains all channel values.
    """

    def __init__(
        self,
        n_channels: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.n_directions, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, n_channels, n_samples)

        Returns:
            Output logits of shape (batch, n_classes)
        """
        # Transpose to (batch, n_samples, n_channels) for LSTM
        x = x.permute(0, 2, 1)

        # Downsample time dimension for efficiency (take every 4th sample)
        x = x[:, ::4, :]

        # LSTM forward
        lstm_out, (h_n, _) = self.lstm(x)

        # Use the last hidden state from both directions
        if self.bidirectional:
            hidden = torch.cat(
                (h_n[-2, :, :], h_n[-1, :, :]), dim=1
            )
        else:
            hidden = h_n[-1, :, :]

        out = self.dropout(hidden)
        out = self.fc(out)
        return out
