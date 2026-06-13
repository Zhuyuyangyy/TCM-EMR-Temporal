#!/usr/bin/env python3
"""EMR Temporal experiment pipeline.

Tests:
1. Temporal Syndrome Transformer forward pass
2. Basic training loop (syndrome classification)
3. Outcome prediction accuracy
4. Sequence length analysis

NOTE: This script uses SYNTHETIC (random Gaussian) data for structural
validation only.  The reported accuracies are NOT meaningful clinical
results -- they merely confirm that the model, loss, and optimiser
compose without errors.
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_emr_sequences(n_patients=200, max_visits=10, n_features=32, n_classes=5, seed=42):
    """Generate random feature sequences and labels for pipeline testing.

    Features are i.i.d. Gaussian noise with a small linear drift.
    Labels are uniform random integers -- there is no learnable signal.
    """
    rng = np.random.RandomState(seed)
    sequences, lengths, labels = [], [], []
    for _ in range(n_patients):
        n_visits = rng.randint(3, max_visits + 1)
        seq = np.zeros((max_visits, n_features))
        for v in range(n_visits):
            seq[v] = rng.randn(n_features) + 0.1 * v
        sequences.append(seq)
        lengths.append(n_visits)
        labels.append(rng.randint(0, n_classes))
    return {
        "sequences": np.array(sequences),
        "lengths": np.array(lengths),
        "labels": np.array(labels),
    }

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(model, data, n_epochs=10, lr=1e-3, seed=42):
    """Basic supervised training loop on syndrome classification.

    Uses the last visit's syndrome logits as the prediction for a
    patient-level label.  Returns per-epoch loss and accuracy lists.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)

    x = torch.tensor(data["sequences"]).float()
    y = torch.tensor(data["labels"]).long()   # (N,)
    lengths = torch.tensor(data["lengths"]).long()

    # Build padding mask: True where padded
    B, T, _ = x.shape
    mask = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        syndrome_logits, outcome_pred = model(x, mask=mask)
        # Use last real visit per patient
        # syndrome_logits: (B, T, n_syndromes) -> gather last real visit
        idx = (lengths - 1).clamp(min=0)  # (B,)
        last_logits = syndrome_logits[torch.arange(B), idx, :]  # (B, n_syndromes)
        loss = F.cross_entropy(last_logits, y)
        loss.backward()
        optimizer.step()

        preds = last_logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        history.append({"epoch": epoch, "loss": loss.item(), "accuracy": acc})

    return history

# ---------------------------------------------------------------------------
# Transformer forward-pass experiment
# ---------------------------------------------------------------------------

def run_transformer_experiment():
    import torch
    from backend.models.temporal_transformer import TemporalSyndromeTransformer

    n_classes = 5
    data = generate_synthetic_emr_sequences(200, n_classes=n_classes)

    # Build model with correct keyword arguments matching the class signature:
    #   TemporalSyndromeTransformer(input_dim=128, nhead=4, num_layers=2,
    #                                n_syndromes=20, dropout=0.1)
    model = TemporalSyndromeTransformer(
        input_dim=32,
        nhead=4,
        num_layers=2,
        n_syndromes=n_classes,
        dropout=0.1,
    )

    # --- Training ----------------------------------------------------------
    print("  Training for 10 epochs ...")
    history = train_model(model, data, n_epochs=10, lr=1e-3)
    for h in history:
        print(f"    epoch {h['epoch']:2d}  loss={h['loss']:.4f}  acc={h['accuracy']:.4f}")

    # --- Evaluation --------------------------------------------------------
    model.eval()
    x = torch.tensor(data["sequences"]).float()
    lengths = torch.tensor(data["lengths"]).long()
    B, T, _ = x.shape
    mask = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)

    with torch.no_grad():
        syndrome_logits, outcome_pred = model(x, mask=mask)

    # syndrome_logits: (B, T, n_classes), outcome_pred: (B, 1)
    idx = (lengths - 1).clamp(min=0)
    last_logits = syndrome_logits[torch.arange(B), idx, :]
    preds = last_logits.argmax(dim=-1).numpy()
    acc = float(np.mean(preds == data["labels"]))

    return {
        "accuracy": acc,
        "n_patients": 200,
        "output_shape": list(syndrome_logits.shape),
        "training_history": history,
    }

# ---------------------------------------------------------------------------
# Sequence length analysis
# ---------------------------------------------------------------------------

def run_sequence_length_analysis():
    import torch
    from backend.models.temporal_transformer import TemporalSyndromeTransformer

    n_classes = 5
    results = {}
    for seq_len in [3, 5, 8, 10]:
        data = generate_synthetic_emr_sequences(100, max_visits=seq_len, n_classes=n_classes)
        model = TemporalSyndromeTransformer(
            input_dim=32, nhead=4, num_layers=2, n_syndromes=n_classes, dropout=0.1,
        )

        # Quick training
        history = train_model(model, data, n_epochs=5, lr=1e-3)

        # Evaluate
        model.eval()
        x = torch.tensor(data["sequences"]).float()
        lengths = torch.tensor(data["lengths"]).long()
        B, T, _ = x.shape
        mask = torch.arange(T).unsqueeze(0) >= lengths.unsqueeze(1)
        with torch.no_grad():
            syndrome_logits, _ = model(x, mask=mask)
        idx = (lengths - 1).clamp(min=0)
        last_logits = syndrome_logits[torch.arange(B), idx, :]
        preds = last_logits.argmax(dim=-1).numpy()
        results[f"visits_{seq_len}"] = {
            "accuracy": float(np.mean(preds == data["labels"])),
            "final_loss": history[-1]["loss"],
        }
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("EMR Temporal Experiment")
    print("=" * 60)
    print()
    print("NOTE: All data is synthetic (random Gaussian noise).")
    print("Accuracies reflect pipeline correctness, NOT clinical utility.")
    print()

    print("[1] Temporal Transformer (with training) ...")
    r1 = run_transformer_experiment()
    print(f"  Post-training accuracy: {r1['accuracy']:.4f}")
    print(f"  Syndrome logits shape:  {r1['output_shape']}")
    print()

    print("[2] Sequence Length Analysis (with training) ...")
    r2 = run_sequence_length_analysis()
    for k, v in r2.items():
        print(f"  {k}: accuracy={v['accuracy']:.4f}  final_loss={v['final_loss']:.4f}")
    print()

    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    with open(out_dir / "emr_results.json", "w") as f:
        json.dump({"transformer": r1, "sequence_analysis": r2}, f, indent=2, default=str)
    print(f"Results saved to {out_dir}/emr_results.json")

if __name__ == "__main__":
    main()
