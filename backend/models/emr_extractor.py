"""EMR entity extraction for TCM clinical notes."""
import re
from typing import List, Dict, Tuple

class EMRExtractor:
    """中医病历实体抽取器 — 症状/证候/方药/检验"""
    SYMPTOM_PATTERNS = ["恶寒", "发热", "头痛", "咳嗽", "乏力", "纳差", "失眠", "心悸"]
    SYNDROME_PATTERNS = ["气虚", "血瘀", "痰湿", "阴虚", "阳虚", "肝郁", "脾虚"]
    
    def extract_symptoms(self, text: str) -> List[str]:
        return [s for s in self.SYMPTOM_PATTERNS if s in text]
    
    def extract_syndromes(self, text: str) -> List[str]:
        return [s for s in self.SYNDROME_PATTERNS if s in text]
    
    def extract_prescriptions(self, text: str) -> List[str]:
        patterns = re.findall(r"[一-鿿]+汤|散|丸|膏", text)
        return patterns
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        return {
            "symptoms": self.extract_symptoms(text),
            "syndromes": self.extract_syndromes(text),
            "prescriptions": self.extract_prescriptions(text),
        }
