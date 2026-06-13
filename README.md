# TCM-EMR-Temporal

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

**A temporal deep learning platform for modeling syndrome evolution and predicting follow-up outcomes in Traditional Chinese Medicine (TCM) electronic medical records.**

> **IMPORTANT -- DATA DISCLAIMER**
> This project currently uses **synthetic (random Gaussian noise) data only**. There is no real patient data, no trained clinical model, and no validated clinical results. The experiment pipeline exists purely to verify that the model architecture, loss functions, and training loop compose without errors. Any accuracy numbers reported by the scripts are **not meaningful clinical metrics** -- they reflect only that the code runs end-to-end. Real-world validation requires de-identified hospital EMR data, IRB approval, and proper clinical evaluation.

---

## Overview

TCM diagnosis is inherently temporal -- a patient's syndrome pattern (zheng hou) evolves across visits as treatment progresses. This platform captures that temporal dynamics by:

1. **Modeling Syndrome Evolution** -- A Transformer encoder with sinusoidal positional encoding learns how syndrome patterns shift across sequential visits
2. **Predicting Follow-Up Outcomes** -- Multi-task prediction of revisit risk, treatment efficacy, and syndrome trajectory using attention pooling over variable-length visit sequences
3. **Providing Temporal Explainability** -- Gradient-based attribution identifies which visits most influenced the prediction, with syndrome-herb knowledge hints

---

## Key Features

- **Temporal Syndrome Transformer** -- Multi-head self-attention over visit sequences with sinusoidal positional encoding for capturing visit order
- **Multi-Task Outcome Prediction** -- Simultaneous prediction of revisit risk, efficacy score, and syndrome outcome via attention pooling
- **20 TCM Syndrome Types** -- Covers the full spectrum: qi deficiency, blood stasis, phlegm-dampness, yin deficiency, yang deficiency, liver qi stagnation, spleen deficiency, and 13 more
- **EMR Entity Extraction** -- Rule-based extraction of symptoms, syndromes, prescriptions, individual herbs with dosages, and lab values from clinical notes
- **Gradient-Based Explainability** -- Salient visit attribution using input gradient L2 norms (more reliable than raw attention weights)
- **Syndrome-Herb Knowledge Hints** -- Maps identified syndromes to commonly associated herbs for clinical decision support
- **Variable-Length Sequences** -- Handles patients with different numbers of visits via padding masks and attention pooling

---

## Architecture

```
    Patient Visit Sequence
    [V1, V2, V3, ..., VT]
           |
    +------v------+
    | Input Proj   |  Linear(128 -> d_model)
    +------+------+
           |
    +------v------+
    | Positional   |  Sinusoidal encoding for visit order
    | Encoding     |
    +------+------+
           |
    +------v------+
    | Transformer  |  Multi-head self-attention (2 layers, 4 heads)
    | Encoder      |  Captures cross-visit dependencies
    +------+------+
           |
    +------v------+
    | Syndrome     |  Per-visit syndrome logits (20 classes)
    | Head         |  Softmax probabilities
    +------+------+
           |
    +------v------+
    | Attention    |  Learned pooling over visit sequence
    | Pooling      |  Handles variable-length trajectories
    +------+------+
           |
    +------v------+
    | Outcome      |  Multi-task: revisit_risk, efficacy, syndrome_outcome
    | Predictor    |
    +-------------+
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| API Framework | FastAPI + Uvicorn |
| Deep Learning | PyTorch 2.0+ |
| Scientific Computing | NumPy, SciPy, scikit-learn |
| Data Processing | Pandas |
| Graph Analysis | NetworkX |
| Configuration | PyYAML, Pydantic |
| Testing | pytest |
| Linting | Ruff |
| CI/CD | GitHub Actions |

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TCM-EMR-Temporal.git
cd TCM-EMR-Temporal

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install fastapi uvicorn scikit-learn numpy pandas pyyaml pydantic pydantic-settings torch scipy networkx
```

### 2. Start the API Server

```bash
# Development server with auto-reload
uvicorn backend.main:app --host 0.0.0.0 --port 8021 --reload
```

Swagger UI is available at: **http://localhost:8021/docs**

### 3. Run the Experiment Pipeline

```bash
python scripts/run_experiment.py
```

This runs the temporal Transformer forward pass, outcome prediction, and sequence length analysis with synthetic EMR data. Results are saved to `output/emr_results.json`.

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run backend tests
pytest backend/tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/predict/trajectory` | POST | Syndrome trajectory & outcome prediction |
| `/api/predict/explain` | POST | Temporal attention-based explanation |

### API Usage Example

```python
import httpx

# Predict syndrome trajectory and outcomes
response = httpx.post("http://localhost:8021/api/predict/trajectory", json={
    "patient_id": "P001",
    "visits": [
        {"visit_id": "V1", "features": [0.1] * 128},
        {"visit_id": "V2", "features": [0.2] * 128},
        {"visit_id": "V3", "features": [0.3] * 128}
    ],
    "predict_outcomes": True
})
print(response.json())
# Returns per-visit top-5 syndrome predictions + outcome predictions

# Explain a prediction
response = httpx.post("http://localhost:8021/api/predict/explain", json={
    "patient_id": "P001",
    "visits": [
        {"visit_id": "V1", "features": [0.1] * 128},
        {"visit_id": "V2", "features": [0.2] * 128}
    ],
    "target_syndrome_idx": 0  # Explain qi deficiency (气虚)
})
print(response.json())
# Returns visit importance scores + syndrome evolution + herb hints
```

---

## Supported Syndrome Types (20)

| Index | Syndrome | Chinese | Associated Herbs |
|-------|----------|---------|-----------------|
| 0 | Qi Deficiency | 气虚 | Astragalus, Codonopsis, Atractylodes |
| 1 | Blood Stasis | 血瘀 | Salvia, Safflower, Peach Kernel |
| 2 | Phlegm-Dampness | 痰湿 | Pinellia, Tangerine Peel, Poria |
| 3 | Yin Deficiency | 阴虚 | Ophiopogon, Rehmannia, Lily Bulb |
| 4 | Yang Deficiency | 阳虚 | Aconite, Dried Ginger, Cinnamon |
| 5 | Liver Qi Stagnation | 肝郁 | Bupleurum, White Peony, Aurantium |
| 6 | Spleen Deficiency | 脾虚 | Atractylodes, Poria, Coix |
| 7 | Qi Stagnation | 气滞 | -- |
| 8 | Blood Deficiency | 血虚 | -- |
| 9 | Yang Hyperactivity | 阳亢 | -- |
| 10 | Damp-Heat | 湿热 | -- |
| 11 | Wind-Cold | 风寒 | -- |
| 12 | Wind-Heat | 风热 | -- |
| 13 | Summer-Heat Dampness | 暑湿 | -- |
| 14 | Cold-Dampness | 寒湿 | -- |
| 15 | Heat Toxin | 热毒 | -- |
| 16 | Qi-Yin Deficiency | 气阴两虚 | -- |
| 17 | Yin-Yang Deficiency | 阴阳两虚 | -- |
| 18 | Liver-Kidney Yin Deficiency | 肝肾阴虚 | -- |
| 19 | Spleen-Kidney Yang Deficiency | 脾肾阳虚 | -- |

---

## Project Structure

```
TCM-EMR-Temporal/
├── backend/
│   ├── api/
│   │   ├── predict.py             # Trajectory & outcome prediction API
│   │   ├── explain.py             # Temporal explainability API
│   │   └── __init__.py
│   ├── models/
│   │   ├── temporal_transformer.py # Temporal Syndrome Transformer
│   │   ├── outcome_predictor.py    # Multi-task outcome predictor with attention pooling
│   │   ├── emr_extractor.py       # EMR entity extraction (symptoms, herbs, labs)
│   │   └── __init__.py
│   ├── config.py                  # Application settings
│   ├── main.py                    # FastAPI application entry point
│   └── __init__.py
├── data/
│   └── syndrome_trajectory.yaml   # Syndrome transition definitions
├── docs/
│   ├── Claim_Evidence_Table.md    # Research claim-evidence mapping
│   ├── SCI_Paper_Skeleton.md      # Paper draft structure
│   └── 技术交底书.md              # Technical disclosure document
├── scripts/
│   ├── generate_synthetic_emr.py  # Synthetic EMR data generator
│   ├── run_experiment.py          # Full experiment pipeline
│   └── quick_smoke_test.py        # Quick validation script
├── tests/
│   └── test_smoke.py              # Smoke tests
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI pipeline
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Dependencies
├── REPRODUCE.md                   # Reproduction guide
└── README.md                      # This file
```

---

## Benchmarks

Run the built-in experiment pipeline to verify the model runs end-to-end:

```bash
python scripts/run_experiment.py
```

| Metric | Details |
|--------|---------|
| Syndrome prediction | 5-class classification at last visit (synthetic labels) |
| Outcome prediction | Single-task sigmoid output from Transformer |
| Sequence lengths tested | 3, 5, 8, 10 visits |
| Synthetic cohort | 200 patients -- **random Gaussian noise, not real data** |
| Model configuration | 2 Transformer layers, 4 attention heads, 128-dim hidden |

> **NOTE:** All data is randomly generated. Accuracies near chance level (20% for 5 classes) are expected and correct -- the point is structural validation, not clinical performance.

---

## Research

This platform supports research in TCM clinical informatics and temporal health data analysis. Related documentation:

- **SCI Paper Skeleton** (`docs/SCI_Paper_Skeleton.md`) -- Draft structure for peer-reviewed publication
- **Claim-Evidence Table** (`docs/Claim_Evidence_Table.md`) -- Mapping of research claims to supporting evidence
- **Technical Disclosure** (`docs/技术交底书.md`) -- Detailed technical methodology

### Citation

```bibtex
@software{tcm_emr_temporal,
  title={TCM-EMR-Temporal: Temporal Syndrome Modeling and Outcome Prediction for TCM Electronic Medical Records},
  author={ZYY Project},
  year={2025},
  url={https://github.com/YOUR_USERNAME/TCM-EMR-Temporal}
}
```

### Key References

- Vaswani, A. et al. Attention is all you need. *NeurIPS* (2017).
- Li, X. et al. BEHRT: Transformer for Electronic Health Records. *Scientific Reports* 10, 7155 (2020).

---

## Roadmap

- [ ] Integration with real hospital EMR systems (with de-identification)
- [ ] Support for Chinese clinical note NLP (BERT-based entity extraction)
- [ ] Causal inference module for treatment effect estimation
- [ ] Longitudinal visualization dashboard for syndrome trajectories
- [ ] Multi-center federated training support
- [ ] Docker containerization for reproducible deployment

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, collaborations, or issues, please open a GitHub Issue or contact the project maintainers.
