"""
EMR entity extraction for TCM clinical notes.
Enhanced with prescription herb extraction and lab value extraction.
"""
import re
from typing import List, Dict, Tuple, Optional


class EMRExtractor:
    """中医病历实体抽取器 — 症状/证候/方药/草药/检验值

    Supports:
    - Symptom / syndrome keyword matching
    - Prescription name extraction (汤/散/丸/膏)
    - Individual herb + dosage extraction from prescription text
    - Lab value extraction (name + numeric value + unit)
    """

    SYMPTOM_PATTERNS = ["恶寒", "发热", "头痛", "咳嗽", "乏力", "纳差", "失眠", "心悸"]
    SYNDROME_PATTERNS = ["气虚", "血瘀", "痰湿", "阴虚", "阳虚", "肝郁", "脾虚"]

    # Common TCM herbs for prescription composition extraction
    HERB_CATALOG = [
        "黄芪", "当归", "白术", "茯苓", "甘草", "人参", "川芎", "白芍",
        "熟地黄", "生地黄", "黄芩", "黄连", "黄柏", "栀子", "柴胡",
        "半夏", "陈皮", "枳壳", "厚朴", "苍术", "薏苡仁", "泽泻",
        "桂枝", "麻黄", "杏仁", "桔梗", "紫苏", "防风", "荆芥",
        "丹参", "红花", "桃仁", "赤芍", "牡丹皮", "金银花", "连翘",
        "板蓝根", "蒲公英", "鱼腥草", "大黄", "芒硝", "附子", "干姜",
        "肉桂", "吴茱萸", "细辛", "五味子", "麦冬", "天冬", "百合",
        "酸枣仁", "远志", "龙骨", "牡蛎", "钩藤", "天麻", "石决明",
        "牛膝", "杜仲", "续断", "桑寄生", "独活", "羌活", "威灵仙",
    ]

    # Generic lab value regex: captures name + number + unit
    LAB_VALUE_RE = re.compile(
        r"(?P<name>[A-Za-z一-鿿]{2,15})\s*[:：]?\s*"
        r"(?P<value>[\d]+\.?\d*)\s*"
        r"(?P<unit>[a-zA-Zµμ%/]+(?:/[a-zA-Zµμ]+)?)",
        re.UNICODE,
    )

    # Chinese-specific lab names for higher-precision matching
    LAB_CHINESE_RE = re.compile(
        r"(?P<name>白细胞|红细胞|血红蛋白|血小板|中性粒细胞|淋巴细胞|"
        r"谷丙转氨酶|谷草转氨酶|肌酐|尿素氮|尿酸|空腹血糖|糖化血红蛋白|"
        r"总胆固醇|甘油三酯|高密度脂蛋白|低密度脂蛋白|C反应蛋白|降钙素原|"
        r"白蛋白|球蛋白|总胆红素|直接胆红素|血沉|D.?二聚体)\s*"
        r"(?P<value>[\d]+\.?\d*)\s*"
        r"(?P<unit>×?10[⁹\^/\d]*|[a-zA-Zµμ%]+(?:/[a-zA-Zµμ]+)?)?",
        re.UNICODE,
    )

    def extract_symptoms(self, text: str) -> List[str]:
        """Match known TCM symptoms in clinical text."""
        return [s for s in self.SYMPTOM_PATTERNS if s in text]

    def extract_syndromes(self, text: str) -> List[str]:
        """Match known TCM syndrome patterns."""
        return [s for s in self.SYNDROME_PATTERNS if s in text]

    def extract_prescriptions(self, text: str) -> List[str]:
        """Extract classical prescription names ending in 汤/散/丸/膏."""
        return re.findall(r"[一-鿿]+汤|散|丸|膏", text)

    def extract_herbs(self, text: str) -> List[Dict[str, str]]:
        """Extract individual herbs with dosage from prescription text.

        Matches patterns like "黄芪30g", "当归 10g", "甘草6克".
        Returns list of {"herb": name, "dose": "30g"} dicts.
        """
        herb_pattern = re.compile(
            r"(?P<herb>" + "|".join(re.escape(h) for h in self.HERB_CATALOG) + r")"
            r"\s*(?P<dose>[\d]+\.?\d*\s*[gG克]?)",
            re.UNICODE,
        )
        results = []
        for m in herb_pattern.finditer(text):
            results.append({"herb": m.group("herb"), "dose": m.group("dose").strip()})
        return results

    def extract_herb_names(self, text: str) -> List[str]:
        """Return just the herb names present in text (no dosage info)."""
        return [herb for herb in self.HERB_CATALOG if herb in text]

    def extract_lab_values(self, text: str) -> List[Dict[str, str]]:
        """Extract lab test name/value/unit triples from clinical text.

        Tries Chinese-specific names first (higher precision),
        then falls back to the generic pattern.
        Returns list of {"name": ..., "value": ..., "unit": ...} dicts.
        """
        results = []
        seen_names: set = set()

        # Chinese-specific pattern (higher precision)
        for m in self.LAB_CHINESE_RE.finditer(text):
            name = m.group("name")
            if name not in seen_names:
                seen_names.add(name)
                results.append({
                    "name": name,
                    "value": m.group("value"),
                    "unit": (m.group("unit") or "").strip(),
                })

        # Generic pattern for anything not already matched
        for m in self.LAB_VALUE_RE.finditer(text):
            name = m.group("name")
            if name not in seen_names:
                seen_names.add(name)
                results.append({
                    "name": name,
                    "value": m.group("value"),
                    "unit": m.group("unit"),
                })

        return results

    def extract_entities(self, text: str) -> Dict[str, list]:
        """Extract all entity types from a clinical note."""
        return {
            "symptoms": self.extract_symptoms(text),
            "syndromes": self.extract_syndromes(text),
            "prescriptions": self.extract_prescriptions(text),
            "herbs": self.extract_herbs(text),
            "herb_names": self.extract_herb_names(text),
            "lab_values": self.extract_lab_values(text),
        }
