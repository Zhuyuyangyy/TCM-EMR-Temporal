"""Multi-task outcome prediction for TCM EMR.

Enhanced with attention pooling for variable-length sequences so the
predictor can aggregate over visit trajectories of different lengths
without being forced to use only the last hidden state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learned attention pooling over a variable-length sequence.

    Computes a weighted sum of sequence elements where the weights
    are learned via a small feed-forward attention network.  An
    optional padding mask lets the module ignore padded positions.

    This is preferable to simply taking the last hidden state because
    it lets the model attend to important visits anywhere in the
    trajectory (e.g. an early visit with a strong syndrome shift).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) bool — True where positions are PADDED
        Returns:
            (batch, hidden_dim) — pooled representation
        """
        # scores: (batch, seq_len, 1)
        scores = self.attention(x)

        if mask is not None:
            # Set padded positions to -inf so softmax gives them ~0 weight
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        # weights: (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)

        # Handle edge case where entire sequence is masked
        # Replace NaN (from all-inf softmax) with 0
        weights = weights.nan_to_num(0.0)

        # pooled: (batch, hidden_dim)
        pooled = (x * weights).sum(dim=1)
        return pooled


class OutcomePredictor(nn.Module):
    """复诊结局预测器 — 多任务(再入院/疗效/证候转归)

    Accepts either:
      * a fixed-size vector  (batch, input_dim)  — legacy path
      * a sequence of vectors (batch, T, input_dim) — uses attention pooling

    When given a sequence, an optional padding mask can be provided so
    that variable-length trajectories are handled correctly.
    """

    def __init__(self, input_dim: int = 128, n_outcomes: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.attn_pool = AttentionPooling(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_outcomes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) or (batch, T, input_dim)
            mask: optional (batch, T) bool padding mask (only used for 3-D input)
        Returns:
            (batch, n_outcomes) logits
        """
        if x.dim() == 3:
            x = self.attn_pool(x, mask)  # (batch, input_dim)
        return self.fc(x)
