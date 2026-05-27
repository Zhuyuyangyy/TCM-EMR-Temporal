"""Explainability API -- attention-based model interpretation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.models.temporal_transformer import TemporalSyndromeTransformer

router = APIRouter(prefix="/api/predict", tags=["explain"])

_transformer: TemporalSyndromeTransformer | None = None

SYNDROME_NAMES = [
    "气虚", "血瘀", "痰湿", "阴虚", "阳虚", "肝郁", "脾虚",
    "气滞", "血虚", "阳亢", "湿热", "风寒", "风热", "暑湿",
    "寒湿", "热毒", "气阴两虚", "阴阳两虚", "肝肾阴虚", "脾肾阳虚",
]


def _get_transformer() -> TemporalSyndromeTransformer:
    global _transformer
    if _transformer is None:
        _transformer = TemporalSyndromeTransformer(
            input_dim=128, nhead=4, num_layers=2, n_syndromes=20, dropout=0.0,
        )
        _transformer.eval()
    return _transformer


# --- Request / response schemas ---

class ExplainRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    visits: List[dict] = Field(..., min_length=1,
                                description="Ordered visits with 'visit_id' and 'features' (128-dim)")
    target_syndrome_idx: Optional[int] = Field(
        None, description="Syndrome index to explain (default: top predicted syndrome at last visit)"
    )


class VisitImportance(BaseModel):
    visit_id: str
    importance: float  # normalised attention weight


class SyndromeEvolution(BaseModel):
    visit_id: str
    syndrome_name: str
    probability: float


class ExplainResponse(BaseModel):
    patient_id: str
    target_syndrome: str
    visit_importance: List[VisitImportance]
    syndrome_evolution: List[SyndromeEvolution]
    top_herbs_hint: List[str] = Field(
        default_factory=list,
        description="Herbs commonly associated with the target syndrome (knowledge-based hint)",
    )


# --- Syndrome -> herb knowledge base (simplified) ---
SYNDROME_HERB_MAP = {
    0: ["黄芪", "党参", "白术"],      # 气虚
    1: ["丹参", "红花", "桃仁"],      # 血瘀
    2: ["半夏", "陈皮", "茯苓"],      # 痰湿
    3: ["麦冬", "生地黄", "百合"],    # 阴虚
    4: ["附子", "干姜", "肉桂"],      # 阳虚
    5: ["柴胡", "白芍", "枳壳"],      # 肝郁
    6: ["白术", "茯苓", "薏苡仁"],   # 脾虚
}


@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(req: ExplainRequest):
    """Explain a trajectory prediction using transformer attention weights.

    Returns:
    - Per-visit importance scores (how much each visit influenced the prediction)
    - Syndrome evolution trajectory (probability of target syndrome at each visit)
    - Herb hints based on the identified syndrome
    """
    T = len(req.visits)
    D = len(req.visits[0].get("features", []))
    if D != 128:
        raise HTTPException(status_code=400, detail=f"Expected 128-dim features, got {D}")

    # Build tensor (1, T, 128)
    seq = [[v["features"] for v in req.visits]]
    x = torch.tensor(seq, dtype=torch.float32)

    model = _get_transformer()

    # --- Forward pass with attention hooks ---
    # We capture attention weights from the last transformer layer
    attention_weights = []

    def _attn_hook(module, input, output):
        # TransformerEncoderLayer returns (attn_output, attn_output_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            attention_weights.append(output[1])  # (nhead, B*T, B*T) or similar

    # Register hook on the last encoder layer
    last_layer = model.transformer.layers[-1]
    hook = last_layer.self_attn.register_forward_hook(_attn_hook)

    with torch.no_grad():
        syndrome_logits, outcome_last = model(x)

    hook.remove()

    # --- Compute per-visit importance from attention ---
    probs = torch.softmax(syndrome_logits[0], dim=-1)  # (T, 20)

    # Determine target syndrome
    target_idx = req.target_syndrome_idx
    if target_idx is None:
        # Use the top syndrome at the last visit
        target_idx = probs[-1].argmax().item()

    target_name = SYNDROME_NAMES[target_idx] if target_idx < len(SYNDROME_NAMES) else f"S{target_idx}"

    # Syndrome evolution for the target syndrome
    syndrome_evolution = []
    for i, v in enumerate(req.visits):
        syndrome_evolution.append(SyndromeEvolution(
            visit_id=v["visit_id"],
            syndrome_name=target_name,
            probability=round(probs[i, target_idx].item(), 4),
        ))

    # Visit importance: use gradient of syndrome logit w.r.t. input as attribution
    # (saliency-based, more reliable than raw attention for explanation)
    x_grad = x.clone().requires_grad_(True)
    s_logits, _ = model(x_grad)
    target_logit = s_logits[0, :, target_idx].sum()
    target_logit.backward()

    # Per-visit importance = L2 norm of gradient per visit step
    grad = x_grad.grad[0]  # (T, 128)
    visit_scores = grad.norm(dim=-1)  # (T,)
    visit_scores = F.softmax(visit_scores, dim=0)  # normalise to sum to 1

    visit_importance = []
    for i, v in enumerate(req.visits):
        visit_importance.append(VisitImportance(
            visit_id=v["visit_id"],
            importance=round(visit_scores[i].item(), 4),
        ))

    # Herb hints from knowledge base
    top_herbs = SYNDROME_HERB_MAP.get(target_idx, [])

    return ExplainResponse(
        patient_id=req.patient_id,
        target_syndrome=target_name,
        visit_importance=visit_importance,
        syndrome_evolution=syndrome_evolution,
        top_herbs_hint=top_herbs,
    )
