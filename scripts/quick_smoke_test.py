#!/usr/bin/env python
"""Quick smoke test: model forward pass, extractor, predictor, API routes."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def test_transformer():
    from backend.models.temporal_transformer import TemporalSyndromeTransformer
    print("=== TemporalSyndromeTransformer ===")
    model = TemporalSyndromeTransformer(
        input_dim=128, nhead=4, num_layers=2, n_syndromes=20, dropout=0.1,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    B, T, D = 4, 16, 128
    x = torch.randn(B, T, D)
    model.eval()
    with torch.no_grad():
        syndrome_logits, outcome_pred = model(x)
    assert syndrome_logits.shape == (B, T, 20), f"syndrome shape {syndrome_logits.shape}"
    assert outcome_pred.shape == (B, 1), f"outcome shape {outcome_pred.shape}"
    assert (outcome_pred >= 0).all() and (outcome_pred <= 1).all()

    # With padding mask
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, -4:] = True
    with torch.no_grad():
        syn_m, out_m = model(x, mask=mask)
    assert syn_m.shape == (B, T, 20)
    assert out_m.shape == (B, 1)
    print("  PASS: forward pass + padding mask")


def test_outcome_predictor():
    from backend.models.outcome_predictor import OutcomePredictor, AttentionPooling
    print("=== OutcomePredictor (with AttentionPooling) ===")

    # Test 2D input (legacy path)
    pred = OutcomePredictor(input_dim=128, n_outcomes=3)
    x2d = torch.randn(4, 128)
    with torch.no_grad():
        out2d = pred(x2d)
    assert out2d.shape == (4, 3), f"2D output shape {out2d.shape}"
    print("  PASS: 2D input (legacy)")

    # Test 3D input (sequence with attention pooling)
    x3d = torch.randn(4, 8, 128)
    with torch.no_grad():
        out3d = pred(x3d)
    assert out3d.shape == (4, 3), f"3D output shape {out3d.shape}"
    print("  PASS: 3D input (attention pooled)")

    # Test 3D with mask
    mask = torch.zeros(4, 8, dtype=torch.bool)
    mask[0, -3:] = True  # pad last 3 for patient 0
    with torch.no_grad():
        out_mask = pred(x3d, mask=mask)
    assert out_mask.shape == (4, 3)
    print("  PASS: 3D input with padding mask")

    # Test AttentionPooling standalone
    pool = AttentionPooling(128)
    with torch.no_grad():
        pooled = pool(x3d, mask)
    assert pooled.shape == (4, 128), f"pooled shape {pooled.shape}"
    print("  PASS: AttentionPooling standalone")


def test_emr_extractor():
    from backend.models.emr_extractor import EMRExtractor
    print("=== EMRExtractor ===")
    ext = EMRExtractor()

    text = "患者恶寒发热，头痛咳嗽，证属气虚血瘀。予四君子汤加减。黄芪30g 当归10g 白术15g 甘草6g。WBC 6.5x10^9/L 血红蛋白 132g/L 空腹血糖 6.2mmol/L"
    entities = ext.extract_entities(text)

    assert "恶寒" in entities["symptoms"], f"symptoms: {entities['symptoms']}"
    assert "气虚" in entities["syndromes"], f"syndromes: {entities['syndromes']}"
    assert len(entities["herbs"]) >= 3, f"herbs: {entities['herbs']}"
    assert len(entities["herb_names"]) >= 3, f"herb_names: {entities['herb_names']}"
    assert len(entities["lab_values"]) >= 2, f"lab_values: {entities['lab_values']}"

    # Verify herb extraction
    herb_names = [h["herb"] for h in entities["herbs"]]
    assert "黄芪" in herb_names, f"herb_names: {herb_names}"
    assert "当归" in herb_names, f"herb_names: {herb_names}"
    print(f"  Symptoms: {entities['symptoms']}")
    print(f"  Syndromes: {entities['syndromes']}")
    print(f"  Herbs: {entities['herbs']}")
    print(f"  Lab values: {entities['lab_values']}")
    print("  PASS: all entity extraction")

    # Test lab values
    labs = ext.extract_lab_values("白细胞 7.2x10^9/L 血红蛋白 145g/L 肌酐 89umol/L")
    assert len(labs) >= 3, f"labs: {labs}"
    print(f"  Lab-only test: {labs}")
    print("  PASS: lab value extraction")


def test_api_routes():
    print("=== API Route Registration ===")
    from backend.main import app
    routes = [r.path for r in app.routes]
    assert "/health" in routes, f"Missing /health, routes: {routes}"
    assert "/api/predict/trajectory" in routes, f"Missing /api/predict/trajectory, routes: {routes}"
    assert "/api/predict/explain" in routes, f"Missing /api/predict/explain, routes: {routes}"
    print(f"  Routes: {[r for r in routes if not r.startswith('/openapi')]}")
    print("  PASS: all routes registered")


def main():
    print("=== TCM-EMR-Temporal Deep Optimization Smoke Test ===")
    print()
    test_transformer()
    print()
    test_outcome_predictor()
    print()
    test_emr_extractor()
    print()
    test_api_routes()
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
