#!/usr/bin/env python3
"""EMR Temporal experiment pipeline.

Tests:
1. Temporal Syndrome Transformer forward pass
2. Outcome prediction accuracy
3. Sequence length analysis
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def generate_synthetic_emr_sequences(n_patients=200, max_visits=10, n_features=32, seed=42):
    rng = np.random.RandomState(seed)
    sequences, lengths, labels = [], [], []
    for _ in range(n_patients):
        n_visits = rng.randint(3, max_visits + 1)
        seq = np.zeros((max_visits, n_features))
        for v in range(n_visits):
            seq[v] = rng.randn(n_features) + 0.1 * v  # drift over time
        sequences.append(seq)
        lengths.append(n_visits)
        labels.append(rng.randint(0, 5))
    return {"sequences": np.array(sequences), "lengths": np.array(lengths), "labels": np.array(labels)}

def run_transformer_experiment():
    import torch
    from backend.models.temporal_transformer import TemporalSyndromeTransformer
    data = generate_synthetic_emr_sequences(200)
    model = TemporalSyndromeTransformer(input_dim=32, d_model=64, nhead=4, n_classes=5)
    model.eval()
    x = torch.tensor(data["sequences"]).float()
    with torch.no_grad():
        out = model(x)
    preds = out.argmax(dim=1).numpy()
    acc = float(np.mean(preds == data["labels"]))
    return {"accuracy": acc, "n_patients": 200, "output_shape": list(out.shape)}

def run_sequence_length_analysis():
    import torch
    from backend.models.temporal_transformer import TemporalSyndromeTransformer
    results = {}
    for seq_len in [3, 5, 8, 10]:
        data = generate_synthetic_emr_sequences(100, max_visits=seq_len)
        model = TemporalSyndromeTransformer(input_dim=32, d_model=64, nhead=4, n_classes=5)
        model.eval()
        x = torch.tensor(data["sequences"]).float()
        with torch.no_grad():
            out = model(x)
        preds = out.argmax(dim=1).numpy()
        results[f"visits_{seq_len}"] = {"accuracy": float(np.mean(preds == data["labels"]))}
    return results

def main():
    print("=" * 60)
    print("EMR Temporal Experiment")
    print("=" * 60)
    print("\n[1] Temporal Transformer...")
    r1 = run_transformer_experiment()
    print(f"  Accuracy: {r1['accuracy']:.4f}")
    print(f"  Output shape: {r1['output_shape']}")
    print("\n[2] Sequence Length Analysis...")
    r2 = run_sequence_length_analysis()
    for k, v in r2.items():
        print(f"  {k}: accuracy={v['accuracy']:.4f}")
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    with open(out_dir / "emr_results.json", "w") as f:
        json.dump({"transformer": r1, "sequence_analysis": r2}, f, indent=2)
    print(f"\nResults saved to {out_dir}/")

if __name__ == "__main__":
    main()
