# Copyright (c) 2026 D.S.M. Perera (MS26906294), Cognitive Squad.
# MSc in Information Technology, Sri Lanka Institute of Information Technology (SLIIT).
# Developed for the Artificial Intelligence module group project.
# All rights reserved.

"""
EEG Transformer Model for Emotion Classification
Uses self-attention mechanism to capture global dependencies in EEG features.

This is a state-of-the-art architecture inspired by:
- "EEG Conformer" (Song et al., 2023)
- "Attention-based EEG Classification" (Tao et al., 2022)

The Transformer treats feature groups as tokens and uses multi-head
self-attention to learn which feature combinations are most important
for emotion classification.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class EEGTransformer(nn.Module):
    """
    Transformer-based model for EEG emotion classification.

    Architecture:
    1. Linear projection: maps feature chunks to d_model dimensions
    2. Positional encoding: adds sequence position information
    3. Transformer encoder: multi-head self-attention layers
    4. Classification head: global average pooling + FC layers

    The key advantage over CNN/LSTM: self-attention captures GLOBAL
    dependencies between any two feature groups regardless of distance,
    while CNN only sees local patterns and LSTM processes sequentially.
    """

    def __init__(
        self,
        n_features: int = 2548,
        n_classes: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        chunk_size: int = 50,
    ):
        """
        Args:
            n_features: Total number of input features
            n_classes: Number of output classes
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            chunk_size: Size of each feature chunk (token)
        """
        super().__init__()

        self.chunk_size = chunk_size
        self.seq_len = n_features // chunk_size  # Number of tokens
        self.d_model = d_model

        # Linear projection from chunk_size to d_model
        self.input_projection = nn.Linear(chunk_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + 1, dropout=dropout)

        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, n_features)

        Returns:
            Output logits of shape (batch, n_classes)
        """
        batch_size = x.size(0)

        # Truncate to fit evenly into chunks
        usable = self.seq_len * self.chunk_size
        x = x[:, :usable]

        # Reshape into token sequence: (batch, seq_len, chunk_size)
        x = x.view(batch_size, self.seq_len, self.chunk_size)

        # Project to d_model dimensions: (batch, seq_len, d_model)
        x = self.input_projection(x)

        # Prepend [CLS] token: (batch, seq_len+1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Use [CLS] token output for classification
        cls_output = x[:, 0, :]  # (batch, d_model)
        cls_output = self.layer_norm(cls_output)

        # Classify
        logits = self.classifier(cls_output)
        return logits
