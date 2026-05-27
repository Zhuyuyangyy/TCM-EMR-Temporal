"""Generate synthetic EMR sequences with herbs and lab values.

Produces 500 patients with 2-6 visits each, including:
- Symptoms, syndromes, prescriptions
- Herb composition with dosages
- Lab values (WBC, Hb, ALT, Cr, glucose, etc.)
- Outcomes (improved/stable/worsened)
"""
import json
import random
import os

random.seed(42)

SYMPTOMS = ["恶寒", "发热", "头痛", "咳嗽", "乏力", "纳差", "失眠", "心悸",
            "口干", "口苦", "胸闷", "腹胀", "便秘", "腹泻", "腰膝酸软", "眩晕"]
SYNDROMES = ["气虚", "血瘀", "痰湿", "阴虚", "阳虚", "肝郁", "脾虚",
             "气滞", "血虚", "湿热"]
PRESCRIPTIONS = ["四君子汤", "六味地黄丸", "补中益气汤", "逍遥散", "血府逐瘀汤",
                 "二陈汤", "参苓白术散", "桂枝汤", "小柴胡汤", "归脾汤"]
HERBS = [
    ("黄芪", (10, 60)), ("当归", (6, 15)), ("白术", (10, 20)),
    ("茯苓", (10, 30)), ("甘草", (3, 10)), ("人参", (3, 10)),
    ("川芎", (6, 12)), ("白芍", (10, 20)), ("柴胡", (6, 12)),
    ("黄芩", (6, 15)), ("半夏", (6, 12)), ("陈皮", (6, 12)),
    ("桂枝", (6, 10)), ("丹参", (10, 30)), ("麦冬", (10, 20)),
    ("五味子", (3, 10)), ("熟地黄", (10, 30)), ("山药", (10, 30)),
    ("枸杞子", (10, 20)), ("大枣", (3, 10)),
]
LAB_TESTS = [
    ("白细胞", 4.0, 10.0, "x10^9/L"),
    ("血红蛋白", 110, 170, "g/L"),
    ("血小板", 100, 300, "x10^9/L"),
    ("谷丙转氨酶", 5, 80, "U/L"),
    ("谷草转氨酶", 5, 60, "U/L"),
    ("肌酐", 40, 130, "umol/L"),
    ("空腹血糖", 3.5, 12.0, "mmol/L"),
    ("总胆固醇", 2.8, 7.5, "mmol/L"),
    ("甘油三酯", 0.4, 4.0, "mmol/L"),
    ("C反应蛋白", 0.1, 50.0, "mg/L"),
    ("白蛋白", 25, 55, "g/L"),
    ("尿素氮", 2.0, 12.0, "mmol/L"),
]


def gen_lab_values(condition: str) -> dict:
    """Generate lab values biased by patient condition."""
    labs = {}
    for name, lo, hi, unit in LAB_TESTS:
        if condition == "worsened":
            # Shift values toward abnormal
            val = round(random.uniform(lo * 0.6, hi * 1.4), 1)
        elif condition == "improved":
            # Values tend toward normal
            mid = (lo + hi) / 2
            val = round(random.gauss(mid, (hi - lo) * 0.15), 1)
            val = max(lo, min(hi, val))
        else:
            val = round(random.uniform(lo, hi), 1)
        labs[name] = {"value": val, "unit": unit}
    return labs


def gen_herbs(n_herbs: int = 6) -> list:
    """Generate a prescription with n herbs and dosages."""
    chosen = random.sample(HERBS, min(n_herbs, len(HERBS)))
    return [{"herb": name, "dose": f"{random.randint(lo, hi)}g"} for name, (lo, hi) in chosen]


def gen_patient(pid: int) -> dict:
    n_visits = random.randint(2, 6)
    trajectory = random.choice(["improving", "stable", "fluctuating", "worsening"])
    visits = []
    base_condition = "stable"

    for v in range(n_visits):
        # Condition evolution
        if trajectory == "improving":
            base_condition = "worsened" if v < n_visits // 3 else "improved"
        elif trajectory == "worsening":
            base_condition = "improved" if v < n_visits // 3 else "worsened"
        elif trajectory == "fluctuating":
            base_condition = random.choice(["improved", "stable", "worsened"])
        else:
            base_condition = "stable"

        outcome = base_condition
        n_symptoms = random.randint(2, 6)
        if outcome == "improved":
            n_symptoms = max(1, n_symptoms - 1)

        visits.append({
            "visit_id": f"P{pid:04d}_V{v}",
            "visit_index": v,
            "symptoms": random.sample(SYMPTOMS, n_symptoms),
            "syndrome": random.choice(SYNDROMES),
            "prescription": random.choice(PRESCRIPTIONS),
            "herbs": gen_herbs(random.randint(5, 12)),
            "lab_values": gen_lab_values(outcome),
            "outcome": outcome,
        })

    return {
        "patient_id": f"P{pid:04d}",
        "trajectory_type": trajectory,
        "n_visits": n_visits,
        "visits": visits,
    }


if __name__ == "__main__":
    data = [gen_patient(i) for i in range(500)]
    out = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_emr.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    total_visits = sum(p["n_visits"] for p in data)
    print(f"Generated {len(data)} patients, {total_visits} total visits -> {out}")
