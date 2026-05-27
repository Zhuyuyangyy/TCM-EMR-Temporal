"""Unit tests for TCM-EMR-Temporal models."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pytest
from backend.models.temporal_transformer import (
    TemporalSyndromeTransformer,
    PositionalEncoding,
)
from backend.models.outcome_predictor import OutcomePredictor, AttentionPooling
from backend.models.emr_extractor import EMRExtractor


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(2, 10, 64)
        out = pe(x)
        assert out.shape == (2, 10, 64)

    def test_not_identity(self):
        """Positional encoding should modify the input."""
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
        assert syn.shape == (B, T, 10)
        assert out.shape == (B, 1)

    def test_forward_with_mask(self):
        model = self._make_model(input_dim=64)
        model.eval()
        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)
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
        model = TemporalSyndromeTransformer()
        x = torch.randn(1, 5, 128)
        with torch.no_grad():
            syn, out = model(x)
        assert syn.shape == (1, 5, 20)
        assert out.shape == (1, 1)


class TestAttentionPooling:
    def test_output_shape(self):
        pool = AttentionPooling(64)
        x = torch.randn(4, 10, 64)
        with torch.no_grad():
            out = pool(x)
        assert out.shape == (4, 64)

    def test_with_mask(self):
        pool = AttentionPooling(64)
        x = torch.randn(4, 10, 64)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[0, -5:] = True  # pad last 5
        with torch.no_grad():
            out = pool(x, mask)
        assert out.shape == (4, 64)

    def test_gradient_flows(self):
        pool = AttentionPooling(32)
        x = torch.randn(2, 5, 32)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        for name, p in pool.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_all_masked_no_crash(self):
        """Should not crash when entire sequence is masked."""
        pool = AttentionPooling(32)
        x = torch.randn(2, 5, 32)
        mask = torch.ones(2, 5, dtype=torch.bool)  # all padded
        with torch.no_grad():
            out = pool(x, mask)
        assert out.shape == (2, 32)


class TestOutcomePredictor:
    def test_2d_input(self):
        pred = OutcomePredictor(input_dim=64, n_outcomes=3)
        x = torch.randn(4, 64)
        with torch.no_grad():
            out = pred(x)
        assert out.shape == (4, 3)

    def test_3d_input(self):
        pred = OutcomePredictor(input_dim=64, n_outcomes=3)
        x = torch.randn(4, 8, 64)
        with torch.no_grad():
            out = pred(x)
        assert out.shape == (4, 3)

    def test_3d_with_mask(self):
        pred = OutcomePredictor(input_dim=64, n_outcomes=3)
        x = torch.randn(4, 8, 64)
        mask = torch.zeros(4, 8, dtype=torch.bool)
        mask[0, -3:] = True
        with torch.no_grad():
            out = pred(x, mask)
        assert out.shape == (4, 3)

    def test_gradient_flows_3d(self):
        pred = OutcomePredictor(input_dim=32, n_outcomes=2)
        x = torch.randn(2, 5, 32)
        out = pred(x)
        loss = out.sum()
        loss.backward()
        for name, p in pred.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


class TestEMRExtractor:
    def setup_method(self):
        self.ext = EMRExtractor()

    def test_extract_symptoms(self):
        text = "患者恶寒发热，头痛乏力"
        result = self.ext.extract_symptoms(text)
        assert "恶寒" in result
        assert "发热" in result
        assert "头痛" in result
        assert "乏力" in result
        assert len(result) == 4

    def test_extract_syndromes(self):
        text = "证属气虚血瘀"
        result = self.ext.extract_syndromes(text)
        assert "气虚" in result
        assert "血瘀" in result

    def test_extract_prescriptions(self):
        text = "予四君子汤加减"
        result = self.ext.extract_prescriptions(text)
        assert any("四君子汤" in p for p in result)

    def test_extract_herbs(self):
        text = "黄芪30g 当归10g 白术15g 甘草6g"
        herbs = self.ext.extract_herbs(text)
        assert len(herbs) == 4
        herb_names = [h["herb"] for h in herbs]
        assert "黄芪" in herb_names
        assert "当归" in herb_names
        assert "白术" in herb_names
        assert "甘草" in herb_names

    def test_extract_herb_names(self):
        text = "处方含黄芪、当归、川芎、红花"
        names = self.ext.extract_herb_names(text)
        assert "黄芪" in names
        assert "当归" in names
        assert "川芎" in names
        assert "红花" in names

    def test_extract_lab_values(self):
        text = "白细胞 7.2x10^9/L 血红蛋白 145g/L 空腹血糖 6.2mmol/L"
        labs = self.ext.extract_lab_values(text)
        assert len(labs) >= 3
        names = [l["name"] for l in labs]
        assert "白细胞" in names
        assert "血红蛋白" in names
        assert "空腹血糖" in names

    def test_extract_lab_values_generic(self):
        text = "WBC 6.5x10^9/L Hb 132g/L ALT 45U/L"
        labs = self.ext.extract_lab_values(text)
        assert len(labs) >= 2

    def test_extract_entities_full(self):
        text = ("患者发热头痛，证属痰湿。予二陈汤。"
                "黄芪30g 陈皮10g。白细胞 8.5x10^9/L 肌酐 75umol/L")
        entities = self.ext.extract_entities(text)
        assert "symptoms" in entities
        assert "syndromes" in entities
        assert "prescriptions" in entities
        assert "herbs" in entities
        assert "herb_names" in entities
        assert "lab_values" in entities
        assert "发热" in entities["symptoms"]
        assert "痰湿" in entities["syndromes"]
        assert len(entities["herbs"]) >= 2
        assert len(entities["lab_values"]) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
