import json
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PatientContext:
    age: Optional[int] = None
    gender: Optional[str] = None
    symptoms: Dict[str, dict] = field(default_factory=dict)
    ruled_out: Set[str] = field(default_factory=set)
    
    def add_symptom(self, symptom: str, severity: str = "moderate", duration: str = None):
        self.symptoms[symptom] = {
            "severity": severity,
            "duration": duration
        }
    
    def remove_symptom(self, symptom: str) -> bool:
        if symptom in self.symptoms:
            del self.symptoms[symptom]
            return True
        return False
    
    def clear(self):
        self.symptoms.clear()
        self.ruled_out.clear()


class DiagnosisEngine:
    SEVERITY_WEIGHTS = {
        "mild": 0.5,
        "moderate": 1.0,
        "severe": 1.5,
        "very severe": 2.0
    }
    
    URGENCY_ORDER = ["low", "moderate", "high", "emergency"]
    
    def __init__(self, disease_db_path: str = "medical_diseases.json"):
        self.diseases = self._load_diseases(disease_db_path)
        self.symptom_to_diseases = self._build_symptom_index()
        
    def _load_diseases(self, path: str) -> List[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[DiagnosisEngine] Error loading diseases: {e}")
            return []
    
    def _build_symptom_index(self) -> Dict[str, List[str]]:
        index = {}
        for disease in self.diseases:
            disease_name = disease["name"]
            symptoms = disease.get("symptoms", [])
            
            for sym in symptoms:
                sym_name = sym["name"].lower() if isinstance(sym, dict) else sym.lower()
                if sym_name not in index:
                    index[sym_name] = []
                index[sym_name].append(disease_name)
        return index
    
    def _normalize_symptom(self, symptom: str) -> str:
        return symptom.lower().strip()
    
    def _calculate_symptom_match_score(self, 
                                        patient_symptoms: Dict[str, dict], 
                                        disease: dict) -> float:
        disease_symptoms = disease.get("symptoms", [])
        if not disease_symptoms:
            return 0.0
        
        total_weight = 0.0
        matched_weight = 0.0
        
        disease_sym_weights = {}
        for sym in disease_symptoms:
            if isinstance(sym, dict):
                disease_sym_weights[sym["name"].lower()] = sym.get("weight", 0.5)
            else:
                disease_sym_weights[sym.lower()] = 0.5
        
        for sym_name, weight in disease_sym_weights.items():
            total_weight += weight
        
        for patient_sym, details in patient_symptoms.items():
            patient_sym_norm = self._normalize_symptom(patient_sym)
            
            if patient_sym_norm in disease_sym_weights:
                base_weight = disease_sym_weights[patient_sym_norm]
                severity = details.get("severity", "moderate")
                severity_mult = self.SEVERITY_WEIGHTS.get(severity, 1.0)
                matched_weight += base_weight * severity_mult
            else:
                for disease_sym in disease_sym_weights:
                    if disease_sym in patient_sym_norm or patient_sym_norm in disease_sym:
                        base_weight = disease_sym_weights[disease_sym] * 0.7
                        severity = details.get("severity", "moderate")
                        severity_mult = self.SEVERITY_WEIGHTS.get(severity, 1.0)
                        matched_weight += base_weight * severity_mult
                        break
        
        if total_weight == 0:
            return 0.0
        
        raw_score = matched_weight / total_weight
        
        discriminating = disease.get("discriminating_symptoms", [])
        for disc_sym in discriminating:
            disc_sym_lower = disc_sym.lower()
            for patient_sym in patient_symptoms:
                if disc_sym_lower in patient_sym.lower() or patient_sym.lower() in disc_sym_lower:
                    raw_score *= 1.3
                    break
        
        return min(raw_score, 1.0)
    
    def rank_diseases(self, 
                      context: PatientContext, 
                      top_k: int = 5) -> List[Dict]:
        if not context.symptoms:
            return []
        
        results = []
        
        for disease in self.diseases:
            disease_name = disease["name"]
            
            if disease_name in context.ruled_out:
                continue
            
            score = self._calculate_symptom_match_score(context.symptoms, disease)
            
            if score < 0.1:
                continue
            
            disease_sym_names = set()
            for sym in disease.get("symptoms", []):
                if isinstance(sym, dict):
                    disease_sym_names.add(sym["name"].lower())
                else:
                    disease_sym_names.add(sym.lower())
            
            matched = []
            for patient_sym in context.symptoms:
                if patient_sym.lower() in disease_sym_names:
                    matched.append(patient_sym)
            
            follow_ups = disease.get("follow_up_questions", [])
            follow_up = follow_ups[0] if follow_ups else None
            
            results.append({
                "name": disease_name,
                "probability": round(score, 2),
                "urgency": disease.get("urgency", "moderate"),
                "category": disease.get("category", "general"),
                "matched_symptoms": matched,
                "follow_up": follow_up,
                "overview": disease.get("overview", "")
            })
        
        results.sort(key=lambda x: (
            -x["probability"], 
            -self.URGENCY_ORDER.index(x["urgency"]) if x["urgency"] in self.URGENCY_ORDER else 0
        ))
        
        return results[:top_k]
    
    def get_best_follow_up_question(self, 
                                     context: PatientContext,
                                     top_diseases: List[Dict] = None) -> Optional[str]:
        if top_diseases is None:
            top_diseases = self.rank_diseases(context, top_k=3)
        
        if not top_diseases:
            return "Can you describe your symptoms in more detail?"
        
        asked_symptoms = set(context.symptoms.keys())
        discriminating_questions = []
        
        for disease_result in top_diseases:
            disease_name = disease_result["name"]
            
            disease = next((d for d in self.diseases if d["name"] == disease_name), None)
            if not disease:
                continue
            
            for disc_sym in disease.get("discriminating_symptoms", []):
                disc_sym_lower = disc_sym.lower()
                if not any(disc_sym_lower in ask.lower() for ask in asked_symptoms):
                    for q in disease.get("follow_up_questions", []):
                        discriminating_questions.append((q, disease_result["probability"]))
                    break
        
        if discriminating_questions:
            discriminating_questions.sort(key=lambda x: -x[1])
            return discriminating_questions[0][0]
        
        if top_diseases:
            return top_diseases[0].get("follow_up")
        
        return None
    
    def check_urgency(self, context: PatientContext) -> Tuple[str, Optional[str]]:
        top_diseases = self.rank_diseases(context, top_k=3)
        
        for disease in top_diseases:
            if disease["probability"] > 0.3:
                if disease["urgency"] == "emergency":
                    return ("emergency", 
                            f"URGENT: Based on your symptoms, {disease['name']} is possible. "
                            "Please seek immediate medical attention!")
                elif disease["urgency"] == "high":
                    return ("high",
                            f"Your symptoms may indicate {disease['name']}. "
                            "Consider consulting a doctor soon.")
        
        return ("normal", None)


def parse_severity(text: str) -> str:
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["very severe", "extremely", "excruciating", "unbearable"]):
        return "very severe"
    elif any(w in text_lower for w in ["severe", "intense", "terrible", "awful", "horrible"]):
        return "severe"
    elif any(w in text_lower for w in ["mild", "slight", "minor", "a little", "somewhat"]):
        return "mild"
    else:
        return "moderate"


def parse_duration(text: str) -> Optional[str]:
    patterns = [
        (r"for (\d+)\s*days?", "days"),
        (r"for (\d+)\s*hours?", "hours"),
        (r"for (\d+)\s*weeks?", "weeks"),
        (r"since (yesterday|last night|this morning)", "relative"),
        (r"(\d+)\s*days?\s*ago", "days_ago"),
    ]
    
    text_lower = text.lower()
    
    for pattern, unit in patterns:
        match = re.search(pattern, text_lower)
        if match:
            if unit == "relative":
                return match.group(1)
            elif unit == "days_ago":
                return f"{match.group(1)} days"
            else:
                return f"{match.group(1)} {unit}"
    
    return None


def parse_age(text: str) -> Optional[int]:
    patterns = [
        r"i am (\d+)",
        r"i'm (\d+)",
        r"(\d+) years? old",
        r"age:?\s*(\d+)",
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))
    
    return None


def parse_gender(text: str) -> Optional[str]:
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["i am male", "i'm male", "i am a man", "i'm a man"]):
        return "male"
    elif any(w in text_lower for w in ["i am female", "i'm female", "i am a woman", "i'm a woman"]):
        return "female"
    
    return None
