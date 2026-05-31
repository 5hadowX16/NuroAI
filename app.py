import re
import random
import joblib
import numpy as np
import requests
import torch
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import difflib

from diagnosis_engine import (
    DiagnosisEngine, PatientContext,
    parse_severity, parse_duration, parse_age, parse_gender
)

SYMPTOM_OUTPUT_THRESHOLD = 0.2

KNOWN_SYMPTOMS = {
    "headache", "nausea", "dizziness", "fever", "cough", "fatigue", "vomiting", 
    "pain", "stomach", "chest", "throat", "muscle", "aches", "rash", "itchy", 
    "swollen", "vision", "blurred", "ringing", "tinnitus", "wheezing",
    "burning", "shortness", "breath", "sore", "stiff", "weakness", "chills",
    "sweating", "cramps", "bloating", "constipation", "diarrhea", "anxiety",
    "depression", "insomnia", "numbness", "tingling", "tremor", "seizure",
    "forgetting", "memory", "amnesia", "confusion", "difficulty breathing",
    "paralysis", "hair loss", "vision loss", "trouble walking", "heartburn",
    "body pain", "red eyes"
}

SYMPTOM_KEYWORD_MAP = {
    "headache": ["headache", "head hurts", "head ache", "head pain", "migraine"],
    "fever": ["fever", "feverish", "high temperature", "burning up", "hot"],
    "cough": ["cough", "coughing", "dry cough", "wet cough"],
    "nausea": ["nausea", "nauseous", "feel sick", "queasy"],
    "vomiting": ["vomiting", "vomit", "throwing up", "threw up"],
    "dizziness": ["dizziness", "dizzy", "lightheaded", "light headed", "vertigo"],
    "fatigue": ["fatigue", "fatigued", "tired", "exhausted", "no energy", "weak"],
    "chest pain": ["chest pain", "chest hurts", "pain in chest", "chest tightness"],
    "sore throat": ["sore throat", "throat hurts", "throat pain"],
    "difficulty breathing": ["shortness of breath", "short of breath", "can't breathe", "difficulty breathing", "hard to breathe", "breathing difficulty"],
    "abdominal pain": ["stomach pain", "stomach hurts", "abdominal pain", "belly pain", "tummy ache"],
    "muscle pain": ["muscle pain", "muscle ache", "body aches", "muscles hurt"],
    "rash": ["rash", "skin rash", "hives", "skin irritation"],
    "chills": ["chills", "shivering", "cold sweats"],
    "sweating": ["sweating", "night sweats", "excessive sweating"],
    "diarrhea": ["diarrhea", "loose stools", "watery stool"],
    "constipation": ["constipation", "constipated", "can't poop"],
    "joint pain": ["joint pain", "joints hurt", "joint ache", "arthritis"],
    "back pain": ["back pain", "back hurts", "backache"],
    "runny nose": ["runny nose", "stuffy nose", "nasal congestion", "blocked nose"],
    "sneezing": ["sneezing", "sneeze"],
    "loss of appetite": ["loss of appetite", "not hungry", "no appetite"],
    "weight loss": ["weight loss", "losing weight"],
    "insomnia": ["insomnia", "can't sleep", "trouble sleeping", "sleepless"],
    "anxiety": ["anxiety", "anxious", "worried", "panic"],
    "numbness": ["numbness", "numb", "tingling"],
    "memory loss": ["forgetting", "forgot", "memory", "memory loss", "amnesia", "can't remember", "confusion", "forget things"],
    "arm pain": ["arm pain", "pain in arm", "left arm pain", "shoulder pain"],
    "jaw pain": ["jaw pain", "pain in jaw", "toothache"],
    "rectal bleeding": ["rectal bleeding", "blood in stool", "bloody stool", "bleeding from bottom"],
    "coughing up blood": ["coughing up blood", "blood in cough", "bloody phlegm"],
    "muscle weakness": ["muscle weakness", "weak muscles", "weakness in legs", "weakness in arms", "can't move"],
    "stiffness": ["stiffness", "stiff neck", "stiff back", "can't bend"],
    "difficulty breathing": ["shortness of breath", "short of breath", "can't breathe", "difficulty breathing", "hard to breathe", "breathing difficulty", "some difficulty breathing", "trouble breathing"],
    "paralysis": ["paralysis", "cant move", "cannot move", "paralyzed", "cant move my arms", "cant move my legs"],
    "vision loss": ["cant see", "blindness", "loss of vision", "blind", "vision loss", "cant see anything"],
    "hair loss": ["hair loss", "losing hair", "bald spots", "balding", "loss of hair"],
    "trouble walking": ["cant walk", "trouble walking", "hard to walk", "stumbling", "legs weak", "cant walk"],
    "heartburn": ["heart burn", "heartburn", "acid reflux", "burning stomach"],
    "body pain": ["body pain", "body ache", "all over pain", "aches and pains", "muscle aches", "hurts all over"],
    "red eyes": ["red eyes", "red eye", "bloodshot eyes", "pink eye", "eye redness"]
}

def extract_symptoms_by_keyword(text: str) -> List[Tuple[str, float]]:
    text_lower = " " + text.lower() + " "
    found = []
    negations = [" no ", " not ", " without ", " free ", " don't have ", " dont have "]
    
    for symptom, keywords in SYMPTOM_KEYWORD_MAP.items():
        for kw in keywords:
            pattern = kw 
            if pattern in text_lower:

                idx = text_lower.find(pattern)
                preceding_text = text_lower[max(0, idx-20):idx]    
                is_negated = any(neg in preceding_text for neg in negations)
                if not is_negated:
                    found.append((symptom, 0.9))
                    break
    return found
COMMON_QUERY_WORDS = {
    "what", "is", "does", "define", "meaning", "tell", "me", "about", "how", 
    "remove", "delete", "reset", "clear", "start", "stop",
    "no", "nope", "yes", "yeah", "yep", "sure", "ok", "okay", "thanks", "thank", "you", "done", "finished", "thats", "that's", "it", "all"
}

KNOWN_MEDICAL_TERMS = {
    "mononucleosis", "pneumonia", "diabetes", "hypertension", "asthma", "bronchitis",
    "arthritis", "cancer", "tumor", "influenza", "hepatitis", "meningitis", "sepsis",
    "anemia", "leukemia", "epilepsy", "migraine", "vertigo", "tinnitus", "eczema",
    "psoriasis", "appendicitis", "gallstones", "kidney", "liver", "heart", "lung",
    "brain", "thyroid", "insulin", "antibiotic", "vaccine", "virus", "bacteria",
    "infection", "inflammation", "fever", "cough", "nausea", "headache", "fatigue"
}

ALL_VOCAB = KNOWN_SYMPTOMS.union(COMMON_QUERY_WORDS).union(KNOWN_MEDICAL_TERMS)

MEDICAL_CATEGORIES = {
    "disease", "syndrome", "disorder", "condition", "infection", "virus", "bacteria",
    "symptom", "treatment", "medication", "drug", "therapy", "surgery", "diagnosis",
    "anatomy", "organ", "tissue", "cell", "bone", "muscle", "nerve", "blood",
    "medical", "health", "clinical", "pathology", "physiology", "medicine"
}

def correct_typos(text: str) -> str:
    words = text.split()
    corrected_words = []
    for word in words:
        clean_word = word.lower().strip("?!.,")
        if clean_word in ALL_VOCAB:
            corrected_words.append(word)
            continue
            
        matches = difflib.get_close_matches(clean_word, ALL_VOCAB, n=1, cutoff=0.8)
        
        if matches:
            corrected_words.append(matches[0])
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

DISEASE_MODEL_PATH = "disease_model.pkl"
SYMPTOM_MODEL_PATH = "symptom_model.keras"
SYMPTOM_LABEL_ENCODER_PATH = "symptom_label_encoder.pkl"
INTENT_MODEL_PATH = "intent_model.keras"
INTENT_LABEL_ENCODER_PATH = "intent_label_encoder.pkl"

app = Flask(__name__)

diagnosis_engine = DiagnosisEngine("medical_diseases.json")

disease_payload = None
disease_vectorizer = None
disease_X = None
disease_list = []
symptom_model = None
symptom_le = None
intent_model = None
intent_le = None

try:
    print("[*] Loading models...")
    
    disease_payload = joblib.load(DISEASE_MODEL_PATH)
    disease_vectorizer = disease_payload["vectorizer"]
    disease_X = disease_payload["X"]
    disease_list = disease_payload["diseases"]
    
    symptom_model = tf.keras.models.load_model(SYMPTOM_MODEL_PATH)
    symptom_le = joblib.load(SYMPTOM_LABEL_ENCODER_PATH)
    
    intent_model = tf.keras.models.load_model(INTENT_MODEL_PATH)
    intent_le = joblib.load(INTENT_LABEL_ENCODER_PATH)

    print("[*] Keras/TF Models loaded successfully.")
    print(f"[*] Diagnosis Engine loaded with {len(diagnosis_engine.diseases)} diseases.")
    
    for d in diagnosis_engine.diseases:
        name_parts = d["name"].lower().split()
        for part in name_parts:
             if len(part) > 3:
                 ALL_VOCAB.add(part)
        ALL_VOCAB.add(d["name"].lower())

except Exception as e:
    print(f"[!] Error loading models: {e}")
    disease_list = []

def predict_intent(text: str) -> str:
    if intent_model is None or intent_le is None:
        return "symptom_check"
        
    arr = tf.constant([text])
    preds = intent_model.predict(arr, verbose=0)[0]
    idx = np.argmax(preds)
    label = intent_le.inverse_transform([idx])[0]
    conf = preds[idx]
    
    if conf < 0.45:
        return "casual"
        
    return label

def extract_symptoms(text: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
    if symptom_model is None or symptom_le is None:
        return []

    arr = tf.constant([text])
    preds = symptom_model.predict(arr, verbose=0)[0]
    
    results = []
    for idx, conf in enumerate(preds):
        label = symptom_le.inverse_transform([idx])[0]
        if label == "other": continue
        
        if conf > threshold:
            results.append((label, float(conf)))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def extract_multiple_symptoms(text: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
    if symptom_model is None or symptom_le is None:
        return []

    arr = tf.constant([text])
    preds = symptom_model.predict(arr, verbose=0)[0]
    
    results = []
    for idx, conf in enumerate(preds):
        label = symptom_le.inverse_transform([idx])[0]
        if label == "other": continue
        
        if conf > threshold:
            results.append((label, float(conf)))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def is_medical_term(term: str, wiki_data: dict = None) -> bool:
    term_lower = term.lower()
    
    for known in KNOWN_MEDICAL_TERMS:
        if known in term_lower or term_lower in known:
            return True
            
    for d in diagnosis_engine.diseases:
        d_name = d["name"].lower()
        if d_name in term_lower or term_lower in d_name:
            return True
    
    if wiki_data:
        description = wiki_data.get("description", "").lower()
        extract = wiki_data.get("extract", "").lower()
        
        medical_indicators = [
            "disease", "disorder", "syndrome", "condition", "infection", "medical",
            "symptom", "treatment", "diagnosis", "health", "clinical", "patient",
            "medicine", "therapy", "virus", "bacteria", "organ", "tissue", "blood"
        ]
        
        for indicator in medical_indicators:
            if indicator in description or indicator in extract[:200]:
                return True
    
    return False

def get_wiki_definition(term: str) -> Tuple[str, bool]:
    original_term = term
    term = re.sub(r"(what is|define|tell me about|the meaning of|explain)", "", term.lower()).strip()
    term = re.sub(r"^(the|a|an)\s+", "", term).strip()
    term = term.strip(" ?!,.")
    
    if not term:
        return ("Please specify what you would like me to define.", False)
    
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term.replace(' ', '_')}"
        headers = {"User-Agent": "Nuro-AI/1.0 (educational@example.com)"}
        r = requests.get(url, headers=headers, timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            
            if data.get("type") == "disambiguation":
                return ("I couldn't find a specific definition for that term. Could you be more specific?", False)
            
            is_medical = is_medical_term(term, data)
            
            if not is_medical:
                return ("I'm sorry, but I'm designed to help with medical questions only. Please ask about symptoms, diseases, or medical conditions.", False)
            
            extract = data.get("extract", "")
            if not extract:
                return ("I couldn't find a definition for that medical term.", True)
            
            sentences = extract.split(". ")
            
            candidate = ". ".join(sentences[:2])
            if not candidate.endswith("."):
                candidate += "."
                
            if len(candidate.split()) <= 50:
                return (candidate, True)
                
            candidate = sentences[0]
            if not candidate.endswith("."):
                candidate += "."
                
            if len(candidate.split()) <= 60:
                return (candidate, True)
                
            return (" ".join(candidate.split()[:50]) + "...", True)
            
        elif r.status_code == 404:
            return (f"I couldn't find a definition for '{term}'. Please check the spelling or try a different term.", False)
            
    except requests.Timeout:
        return ("I'm having trouble reaching the definition service. Please try again.", False)
    except Exception as e:
        print(f"Wiki error: {e}")
        
    return ("I couldn't retrieve the definition. Please try again or consult a medical professional.", False)


class ChatBot:
    def __init__(self):
        self.history = []
        self.patient_context = PatientContext()
        self.chat_history_ids = None
        self.awaiting_severity = None
        self.last_follow_up = None
        self.asked_follow_ups = set()
    
    @property
    def symptoms_found(self) -> set:
        return set(self.patient_context.symptoms.keys())

    def _format_diagnosis_response(self, new_symptoms: List[str], matches: List[Dict], is_final: bool = False) -> str:
        sym_str = ", ".join(new_symptoms)
        current_syms = ", ".join(self.symptoms_found)
        
        response_parts = []
        if new_symptoms:
            response_parts.append(f"I've noted: {sym_str}")
            
        response_parts.append(f"Current symptoms: {current_syms}")
        
        if matches:
            response_parts.append("\nPossible conditions:")
            for i, match in enumerate(matches[:3], 1):
                prob_pct = int(match['probability'] * 100)
                urgency = f"[{match['urgency'].upper()}]" if match['urgency'] in ['emergency', 'high'] else ""
                response_parts.append(f"{i}. {match['name']} ({prob_pct}% match) {urgency}".strip())
            
            urgency_level, warning = diagnosis_engine.check_urgency(self.patient_context)
            if warning:
                response_parts.append(f"\n{warning}")
            
            if not is_final:
                follow_up = diagnosis_engine.get_best_follow_up_question(self.patient_context, matches)
                if follow_up:
                    self.last_follow_up = follow_up
                    response_parts.append(f"\nFollow-up: {follow_up}")
        else:
            response_parts.append("\nCan you describe more symptoms?")
        
        return "\n".join(response_parts)

    def chat(self, user_text: str) -> str:
        user_text_original = user_text
        user_text = correct_typos(user_text)
        user_text_lower = user_text.lower().strip()
        
        age = parse_age(user_text)
        if age:
            self.patient_context.age = age
            
        gender = parse_gender(user_text)
        if gender:
            self.patient_context.gender = gender
        
        if user_text_lower in ["reset", "clear symptoms", "start over", "new chat"]:
            self.patient_context.clear()
            self.history = []
            self.awaiting_severity = None
            self.last_follow_up = None
            return "I have reset your session. All symptoms cleared. How can I help?"
        
        if self.awaiting_severity:
            try:
                rating = int(user_text.strip())
                if 1 <= rating <= 10:
                    severity = "mild" if rating <= 3 else "moderate" if rating <= 6 else "severe" if rating <= 8 else "very severe"
                    sym = self.awaiting_severity
                    if sym in self.patient_context.symptoms:
                        self.patient_context.symptoms[sym]["severity"] = severity
                    self.awaiting_severity = None
                    
                    matches = diagnosis_engine.rank_diseases(self.patient_context, top_k=3)
                    
                    response = f"Got it, recording {sym} as {severity}.\n\n"
                    current_syms = ", ".join(self.symptoms_found)
                    response += f"Current symptoms: {current_syms}\n"
                    
                    if matches:
                        response += "\nPossible conditions:"
                        for i, match in enumerate(matches[:3], 1):
                            prob_pct = int(match['probability'] * 100)
                            response += f"\n{i}. {match['name']} ({prob_pct}% match)"
                        
                        follow_up = diagnosis_engine.get_best_follow_up_question(self.patient_context, matches)
                        if follow_up:
                            self.last_follow_up = follow_up
                            response += f"\n\nFollow-up: {follow_up}"
                    
                    return response
            except ValueError:
                pass
            self.awaiting_severity = None
        
        if self.last_follow_up:
            if any(neg in user_text_lower for neg in ["no", "not", "don't", "dont", "cannot", "can't", "nope"]):
                self.asked_follow_ups.add(self.last_follow_up)
                self.last_follow_up = None
                matches = diagnosis_engine.rank_diseases(self.patient_context, top_k=3)
                
                response = "Thank you for that information. "
                if matches:
                    response += "Based on your current symptoms:\n\nPossible conditions:"
                    for i, match in enumerate(matches[:3], 1):
                        prob_pct = int(match['probability'] * 100)
                        response += f"\n{i}. {match['name']} ({prob_pct}% match)"
                    
                    follow_up = diagnosis_engine.get_best_follow_up_question(self.patient_context, matches)
                    if follow_up and follow_up not in self.asked_follow_ups:
                        self.last_follow_up = follow_up
                        response += f"\n\nFollow-up: {follow_up}"
                    else:
                        response += "\n\nDo you have any other symptoms to report?"
                else:
                    response += "Can you describe any other symptoms?"
                
                return response
            
            elif any(pos in user_text_lower for pos in ["yes", "yeah", "yep", "i do", "i have"]):
                self.asked_follow_ups.add(self.last_follow_up)
                self.last_follow_up = None
                
                has_symptoms = extract_symptoms_by_keyword(user_text) or extract_multiple_symptoms(user_text, threshold=0.3)
                if not has_symptoms:
                    return "Thank you for confirming. Can you describe any other symptoms you're experiencing?"
            
        removal_keywords = ["remove", "delete", "exclude", "drop", "not have", "don't have", "no longer have"]
        if any(w in user_text_lower for w in removal_keywords):
            clean_text = user_text_lower
            for w in removal_keywords:
                clean_text = clean_text.replace(w, "")
                
            syms_to_remove = extract_symptoms(clean_text, threshold=0.1)
            removed = []
            for s, conf in syms_to_remove:
                if self.patient_context.remove_symptom(s):
                    removed.append(s)
            
            if removed:
                r_str = ", ".join(removed)
                current = ", ".join(self.symptoms_found) if self.symptoms_found else "None"
                return f"Removed: {r_str}. Current symptoms: {current}."
            elif not self.symptoms_found:
                return "You have no recorded symptoms to remove."
            else:
                return f"I couldn't find those symptoms in your list. Current: {', '.join(self.symptoms_found)}"

        self.history.append(f"User: {user_text}")

        if user_text.strip().isdigit():
             val = int(user_text.strip())
             if 1 <= val <= 10:
                 return "I see you sent a number. I likely lost track of which symptom you were rating (my memory refreshes when updated). Please verify the symptom again."

        closing_phrases = ["nothing", "that's it", "thats it", "that is it", "none", "no more", "done", "finish", "thats all", "that's all"]
        
         #i forgot what this is for
        is_closing = False
        if user_text_lower in ["no", "nope"]:
            is_closing = True
        elif any(user_text_lower.startswith(p) for p in closing_phrases) or user_text_lower in closing_phrases:
            is_closing = True
            
        if is_closing:
             if self.symptoms_found:
                 matches = diagnosis_engine.rank_diseases(self.patient_context, top_k=3)
                 return self._format_diagnosis_response([], matches, is_final=True)
             else:
                 return "I don't have any symptoms recorded for you yet. Please tell me what you're experiencing."
        
        intent = predict_intent(user_text)
        print(f"[DEBUG] Intent: {intent}")

        is_explicit_def = any(user_text_lower.startswith(p) for p in ["define", "what is", "meaning of", "explain"])
        if is_explicit_def and intent == "casual":
             intent = "definition"
        
        response = ""
        
        silent_check = extract_multiple_symptoms(user_text, threshold=0.3)
    
        keyword_symptoms = extract_symptoms_by_keyword(user_text)
        if keyword_symptoms and not silent_check:
            silent_check = keyword_symptoms
        elif keyword_symptoms:
            existing_syms = {s[0] for s in silent_check}
            for sym in keyword_symptoms:
                if sym[0] not in existing_syms:
                    silent_check.append(sym)
        
        is_explicit_def = any(user_text_lower.startswith(p) for p in ["define", "what is", "meaning of", "explain"])
        
        if silent_check and (intent != "greeting"):
            if intent == "definition" and is_explicit_def:
                pass
            else:
                intent = "symptom_check"

        if intent == "greeting":
            greetings = [
                "Hello! I am your medical assistant. What symptoms are you experiencing?", 
                "Hi there! How can I help you today?", 
                "Greetings. Please tell me about your symptoms."
            ]
            response = random.choice(greetings)
            
        elif intent == "definition":
            definition, is_medical = get_wiki_definition(user_text)
            response = definition
            
        elif intent == "symptom_check":
            found_symptoms = silent_check if silent_check else extract_multiple_symptoms(user_text, threshold=0.3)
            
            if found_symptoms:
                severity = parse_severity(user_text)
                duration = parse_duration(user_text)
                
                new_syms = []
                for sym, conf in found_symptoms:
                    self.patient_context.add_symptom(sym, severity=severity, duration=duration)
                    new_syms.append(sym)
                
                matches = diagnosis_engine.rank_diseases(self.patient_context, top_k=3)
                
                response = self._format_diagnosis_response(new_syms, matches)
                
                if severity == "moderate" and len(new_syms) == 1:
                    self.awaiting_severity = new_syms[0]
                    response += f"\n\nOn a scale of 1-10, how severe is your {new_syms[0]}?"
            else:
                response = "I think you're describing a symptom, but I'm not sure which one. Could you be more specific?"
                
        else:
            found_symptoms = silent_check if silent_check else extract_multiple_symptoms(user_text, threshold=0.3)
            
            if found_symptoms:
                severity = parse_severity(user_text)
                duration = parse_duration(user_text)
                
                new_syms = []
                for sym, conf in found_symptoms:
                    self.patient_context.add_symptom(sym, severity=severity, duration=duration)
                    new_syms.append(sym)
                
                sym_str = ", ".join(new_syms)
                response = f"I noticed you mentioned {sym_str}. Is that correct? Tell me more about how you're feeling."
            else:
                response = "I'm designed to help with medical questions only. Please describe your symptoms or ask about a medical condition."
        
        self.history.append(f"Bot: {response}")
        return response


sessions: Dict[str, ChatBot] = {}

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("symptoms") or ""
    session_id = data.get("session_id")
    
    if not text:
        return jsonify({"response": "Please say something."})
        
    if not session_id:
        session_id = "default"
        
    if session_id not in sessions:
        if len(sessions) > 1000:
            sessions.clear()
        sessions[session_id] = ChatBot()
        
    bot = sessions[session_id]
    reply = bot.chat(text)
    
    return jsonify({
        "response": reply[::1],
        "session_id": session_id,
        "symptoms": list(bot.symptoms_found),
        "debug_intent": "processed" 
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)