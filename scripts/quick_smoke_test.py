#!/usr/bin/env python
"""Quick smoke test: create model, run forward pass, print shapes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from backend.models.temporal_transformer import TemporalSyndromeTransformer


def main():
    print("=== TCM-EMR-Temporal Smoke Test ===")
    print()

    # --- Build model ---
    model = TemporalSyndromeTransformer(
        input_dim=128, nhead=4, num_layers=2, n_syndromes=20, dropout=0.1,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(model)
    print()

    # --- Dummy forward pass ---
    B, T, D = 4, 16, 128  # batch=4 patients, T=16 visits, D=128 features
    x = torch.randn(B, T, D)
    print(f"Input  shape : {tuple(x.shape)}  (batch, seq_len, input_dim)")

    model.eval()
    with torch.no_grad():
        syndrome_logits, outcome_pred = model(x)

    print(f"Syndrome logits shape : {tuple(syndrome_logits.shape)}  (batch, seq_len, n_syndromes)")
    print(f"Outcome pred   shape  : {tuple(outcome_pred.shape)}  (batch, 1)")
    print()

    # --- Sanity checks ---
    assert syndrome_logits.shape == (B, T, 20), "syndrome shape mismatch"
    assert outcome_pred.shape == (B, 1), "outcome shape mismatch"
    assert (outcome_pred >= 0).all() and (outcome_pred <= 1).all(), "outcome not in [0,1]"

    # --- Test with padding mask ---
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, -4:] = True  # pad last 4 visits for patient 0
    with torch.no_grad():
        syn_m, out_m = model(x, mask=mask)
    print(f"With padding mask — syndrome: {tuple(syn_m.shape)}, outcome: {tuple(out_m.shape)}")
    print()

    print("ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    main()
