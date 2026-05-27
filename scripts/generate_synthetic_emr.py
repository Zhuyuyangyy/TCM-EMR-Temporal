"""Generate synthetic EMR sequences."""
import json, random, os
random.seed(42)
symptoms = ["恶寒","发热","头痛","咳嗽","乏力","纳差","失眠","心悸"]
syndromes = ["气虚","血瘀","痰湿","阴虚","阳虚"]

def gen_patient(pid):
    visits = []
    for v in range(random.randint(2,6)):
        visits.append({"visit_id": f"P{pid}_V{v}", "symptoms": random.sample(symptoms, random.randint(2,5)),
                       "syndrome": random.choice(syndromes), "outcome": random.choice(["improved","stable","worsened"])})
    return {"patient_id": f"P{pid}", "visits": visits}

if __name__ == "__main__":
    data = [gen_patient(i) for i in range(500)]
    out = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_emr.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(data)} patients -> {out}")
