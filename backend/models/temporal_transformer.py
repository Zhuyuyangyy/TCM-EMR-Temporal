"""Temporal Syndrome Transformer for EMR sequence modeling."""
import torch
import torch.nn as nn

class TemporalSyndromeTransformer(nn.Module):
    """时序证候Transformer — 建模证候演化轨迹"""
    def __init__(self, input_dim=128, nhead=4, num_layers=2, n_syndromes=20):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.syndrome_head = nn.Linear(128, n_syndromes)
        self.outcome_head = nn.Linear(128, 1)

    def forward(self, x, mask=None):
        h = self.input_proj(x)
        h = self.transformer(h, src_key_padding_mask=mask)
        syndrome_logits = self.syndrome_head(h)
        outcome_pred = torch.sigmoid(self.outcome_head(h[:, -1, :]))
        return syndrome_logits, outcome_pred
