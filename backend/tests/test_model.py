"""Unit tests for TemporalSyndromeTransformer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pytest
from backend.models.temporal_transformer import (
    TemporalSyndromeTransformer,
    PositionalEncoding,
)


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(2, 10, 64)
        out = pe(x)
        assert out.shape == (2, 10, 64)

    def test_not_identity(self):
        """Positional encoding should modify the input (not just pass through)."""
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(2, 10, 64)
        out = pe(x)
        assert not torch.allclose(x, out)


class TestTemporalSyndromeTransformer:
    def _make_model(self, **kwargs):
        defaults = dict(input_dim=64, nhead=4, num_layers=2, n_syndromes=10, dropout=0.0)
        defaults.update(kwargs)
        return TemporalSyndromeTransformer(**defaults)

    def test_forward_shapes(self):
        model = self._make_model(input_dim=64, n_syndromes=10)
        model.eval()
        B, T, D = 4, 12, 64
        x = torch.randn(B, T, D)
        with torch.no_grad():
            syn, out = model(x)
        assert syn.shape == (B, T, 10), f"syndrome shape {syn.shape}"
        assert out.shape == (B, 1), f"outcome shape {out.shape}"

    def test_forward_with_mask(self):
        model = self._make_model(input_dim=64)
        model.eval()
        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)
        # mask last 3 positions for first sample
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[0, -3:] = True
        with torch.no_grad():
            syn, out = model(x, mask=mask)
        assert syn.shape == (2, 8, 10)
        assert out.shape == (2, 1)

    def test_gradient_flows(self):
        model = self._make_model(input_dim=32, n_syndromes=5)
        x = torch.randn(2, 6, 32)
        syn, out = model(x)
        loss = syn.sum() + out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_default_params(self):
        """Model should work with default __init__ args."""
        model = TemporalSyndromeTransformer()
        x = torch.randn(1, 5, 128)
        with torch.no_grad():
            syn, out = model(x)
        assert syn.shape == (1, 5, 20)
        assert out.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
