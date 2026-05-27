"""Temporal Syndrome Transformer for EMR sequence modeling."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions.

    Adds temporal position information so the transformer can
    distinguish the order of visits in a patient trajectory.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
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
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalSyndromeTransformer(nn.Module):
    """时序证候Transformer — 建模证候演化轨迹

    Enhancements over baseline:
    - Sinusoidal positional encoding so visit order is captured
    - Dropout for regularisation
    """

    def __init__(self, input_dim=128, nhead=4, num_layers=2, n_syndromes=20, dropout=0.1):
        super().__init__()
        self.d_model = 128
        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.syndrome_head = nn.Linear(self.d_model, n_syndromes)
        self.outcome_head = nn.Linear(self.d_model, 1)

    def forward(self, x, mask=None):
        h = self.input_proj(x)        # (B, T, d_model)
        h = self.pos_encoder(h)       # inject temporal positions
        h = self.transformer(h, src_key_padding_mask=mask)
        syndrome_logits = self.syndrome_head(h)
        outcome_pred = torch.sigmoid(self.outcome_head(h[:, -1, :]))
        return syndrome_logits, outcome_pred
