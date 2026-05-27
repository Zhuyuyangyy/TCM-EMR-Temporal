"""Prediction API -- syndrome trajectory forecasting & outcome prediction."""
from __future__ import annotations

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.models.temporal_transformer import TemporalSyndromeTransformer
from backend.models.outcome_predictor import OutcomePredictor

router = APIRouter(prefix="/api/predict", tags=["predict"])

# --- Lazy-loaded singleton models (initialised on first request) ---
_transformer: TemporalSyndromeTransformer | None = None
_outcome_predictor: OutcomePredictor | None = None

SYNDROME_NAMES = [
    "气虚", "血瘀", "痰湿", "阴虚", "阳虚", "肝郁", "脾虚",
    "气滞", "血虚", "阳亢", "湿热", "风寒", "风热", "暑湿",
    "寒湿", "热毒", "气阴两虚", "阴阳两虚", "肝肾阴虚", "脾肾阳虚",
]
OUTCOME_NAMES = ["revisit_risk", "efficacy_score", "syndrome_outcome"]


def _get_transformer() -> TemporalSyndromeTransformer:
    global _transformer
    if _transformer is None:
        _transformer = TemporalSyndromeTransformer(
            input_dim=128, nhead=4, num_layers=2, n_syndromes=20, dropout=0.0,
        )
        _transformer.eval()
    return _transformer


def _get_outcome_predictor() -> OutcomePredictor:
    global _outcome_predictor
    if _outcome_predictor is None:
        _outcome_predictor = OutcomePredictor(input_dim=128, n_outcomes=3)
        _outcome_predictor.eval()
    return _outcome_predictor


# --- Request / response schemas ---

class VisitFeatures(BaseModel):
    visit_id: str = Field(..., description="Unique visit identifier")
    features: List[float] = Field(..., min_length=128, max_length=128,
                                   description="128-dim feature vector for this visit")


class TrajectoryRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    visits: List[VisitFeatures] = Field(..., min_length=1,
                                         description="Ordered visit sequence (oldest first)")
    predict_outcomes: bool = Field(True, description="Whether to also predict multi-task outcomes")


class SyndromeAtVisit(BaseModel):
    visit_id: str
    top_syndromes: List[dict]  # [{"name": str, "score": float}]


class TrajectoryResponse(BaseModel):
    patient_id: str
    syndrome_trajectories: List[SyndromeAtVisit]
    outcome_predictions: Optional[dict] = None


# --- Endpoint ---

@router.post("/trajectory", response_model=TrajectoryResponse)
async def predict_trajectory(req: TrajectoryRequest):
    """Predict syndrome evolution trajectory and outcomes for a patient.

    Accepts a sequence of per-visit feature vectors (128-dim each) and
    returns:
    - Per-visit top-5 syndrome predictions with softmax probabilities
    - Optional multi-task outcome predictions from the outcome head
    """
    T = len(req.visits)
    D = len(req.visits[0].features)
    if D != 128:
        raise HTTPException(status_code=400, detail=f"Expected 128-dim features, got {D}")

    # Build (1, T, 128) tensor
    seq = [[v.features for v in req.visits]]
    x = torch.tensor(seq, dtype=torch.float32)

    transformer = _get_transformer()

    with torch.no_grad():
        syndrome_logits, outcome_last = transformer(x)  # (1,T,20), (1,1)

    # Per-visit syndrome predictions
    probs = torch.softmax(syndrome_logits[0], dim=-1)  # (T, 20)
    syndrome_trajectories = []
    for i, v in enumerate(req.visits):
        top_vals, top_idx = probs[i].topk(min(5, len(SYNDROME_NAMES)))
        syndrome_trajectories.append(SyndromeAtVisit(
            visit_id=v.visit_id,
            top_syndromes=[
                {"name": SYNDROME_NAMES[j], "score": round(s.item(), 4)}
                for j, s in zip(top_idx.tolist(), top_vals.tolist())
            ],
        ))

    # Optional: multi-task outcome predictions
    outcome_predictions = None
    if req.predict_outcomes:
        outcome_pred = _get_outcome_predictor()
        with torch.no_grad():
            outcome_logits = outcome_pred(x)  # (1, 3)
        probs_out = torch.sigmoid(outcome_logits[0])
        outcome_predictions = {
            name: round(probs_out[idx].item(), 4)
            for idx, name in enumerate(OUTCOME_NAMES)
        }

    return TrajectoryResponse(
        patient_id=req.patient_id,
        syndrome_trajectories=syndrome_trajectories,
        outcome_predictions=outcome_predictions,
    )
