"""
REDFLAG AI - Multilingual FastAPI Backend Application
Exposes REST endpoints for multilingual conversational text analysis (English, Hinglish, Marathi, Hindi),
screenshot OCR processing, model evaluation transparency, and continuous self-training active learning.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ml_pipeline import TextPreprocessor, ExplainabilityEngine, LanguageDetector
from ocr_engine import OCREngine
from active_learning import ActiveLearningEngine

app = FastAPI(
    title="REDFLAG AI API",
    description="Multilingual Relationship Communication Pattern & Behavioral Risk Analyzer with Continuous Self-Training",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
FRONTEND_DIST = os.path.join(os.path.dirname(BASE_DIR), "frontend", "dist")

# In-memory history cache
ANALYSIS_HISTORY = []

# Global ML model instances
flag_model = None
behavior_model = None
vectorizer = None
interpretable_model = None
model_metrics = {}

def load_models():
    global flag_model, behavior_model, vectorizer, interpretable_model, model_metrics
    flag_model_path = os.path.join(MODELS_DIR, "flag_classifier.joblib")
    beh_model_path = os.path.join(MODELS_DIR, "behavior_classifier.joblib")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
    interp_path = os.path.join(MODELS_DIR, "interpretable_model.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    
    if os.path.exists(flag_model_path) and os.path.exists(vec_path):
        flag_model = joblib.load(flag_model_path)
        behavior_model = joblib.load(beh_model_path)
        vectorizer = joblib.load(vec_path)
        if os.path.exists(interp_path):
            interpretable_model = joblib.load(interp_path)
        print("Multilingual ML Models successfully loaded into memory!")
    else:
        print("Models not found. Training triggered...")
        from train_models import train_and_evaluate
        train_and_evaluate()
        flag_model = joblib.load(flag_model_path)
        behavior_model = joblib.load(beh_model_path)
        vectorizer = joblib.load(vec_path)
        if os.path.exists(interp_path):
            interpretable_model = joblib.load(interp_path)
            
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            model_metrics = json.load(f)

# Load models at startup
load_models()


class AnalyzeRequest(BaseModel):
    text: str
    auto_learn: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    text: str
    flag: str
    behavior: Optional[str] = "custom_learned"
    language: Optional[str] = "Multilingual"
    source: Optional[str] = "user_feedback"


def run_full_analysis(raw_text: str, source_type: str = "text", auto_learn: bool = False) -> Dict[str, Any]:
    """Runs complete multilingual ML inference, language detection, explainability, and optional self-training."""
    cleaned = TextPreprocessor.clean_text(raw_text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Provided text is empty or contains only whitespace.")
        
    messages = TextPreprocessor.split_into_messages(cleaned)
    linguistic_signals = TextPreprocessor.extract_linguistic_signals(cleaned)
    lang_info = LanguageDetector.detect(cleaned)
    
    # Transform text with Multilingual FeatureUnion (Word + Subword Char N-Grams)
    vec = vectorizer.transform([cleaned])
    
    # 1. Flag Prediction & Probabilities
    flag_pred = flag_model.predict(vec)[0]
    flag_classes = list(flag_model.classes_)
    probs = flag_model.predict_proba(vec)[0]
    confidence = float(np.max(probs))
    probabilities_dict = {cls_name: round(float(prob) * 100, 1) for cls_name, prob in zip(flag_classes, probs)}
    
    # 2. Behavioral Sub-category Prediction
    beh_pred = behavior_model.predict(vec)[0]
    beh_probs = behavior_model.predict_proba(vec)[0]
    beh_confidence = float(np.max(beh_probs))
    
    behavior_labels = {
        "controlling_isolation": "Controlling Behavior & Isolation",
        "guilt_tripping": "Emotional Guilt-Tripping",
        "gaslighting_invalidation": "Gaslighting & Reality Invalidation",
        "threats_disrespect": "Intimidation & Verbal Hostility",
        "love_bombing": "Love Bombing & Hyper-Intensity",
        "mixed_signals": "Mixed Signals & Avoidance",
        "passive_aggression": "Passive Aggression",
        "inconsistency_stonewalling": "Inconsistency & Stonewalling",
        "boundary_pushing": "Boundary Pushing",
        "healthy_boundaries": "Healthy Boundaries & Autonomy",
        "constructive_communication": "Constructive Conflict Resolution",
        "supportive_affection": "Supportive Affection & Warmth",
        "neutral_brief": "Neutral Logistical Interaction"
    }
    human_behavior = behavior_labels.get(beh_pred, beh_pred.replace("_", " ").title())
    
    # 3. Top Contributing Features
    top_features = []
    if interpretable_model is not None and hasattr(interpretable_model, "classes_"):
        try:
            class_idx = list(interpretable_model.classes_).index(flag_pred)
            coefs = interpretable_model.coef_[class_idx]
            nonzero_indices = vec.indices
            feature_names = np.array(vectorizer.get_feature_names_out())
            feature_scores = []
            for idx in nonzero_indices:
                feature_scores.append((feature_names[idx], float(coefs[idx] * vec[0, idx])))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            top_features = [(feat.replace("word_tfidf__", "").replace("char_tfidf__", ""), round(score, 3)) for feat, score in feature_scores[:6] if score > 0]
        except Exception as e:
            print(f"Explainability feature error: {e}")
            top_features = []
            
    # 4. Multilingual Explainability Engine: "Why was this flagged?"
    explanation = ExplainabilityEngine.explain(cleaned, flag_pred, top_features, detected_lang=lang_info["language"])
    
    # 5. Severity & Toxicity
    severity_map = {
        "Red Flag": 0.85 + (0.14 * confidence),
        "Yellow Flag": 0.50 + (0.20 * confidence),
        "Green Flag": 0.05,
        "Insufficient Context": 0.0
    }
    severity_score = round(float(severity_map.get(flag_pred, 0.0)), 2)
    sentiment_metrics = ExplainabilityEngine.calculate_toxicity_and_sentiment(cleaned, flag_pred, severity_score)
    
    # 6. Turn-by-Turn Message Breakdown
    message_breakdown = []
    sender_stats = {}
    
    for msg in messages:
        msg_vec = vectorizer.transform([msg["text"]])
        m_flag = flag_model.predict(msg_vec)[0]
        m_probs = flag_model.predict_proba(msg_vec)[0]
        m_conf = float(np.max(m_probs))
        
        sender = msg["sender"]
        if sender not in sender_stats:
            sender_stats[sender] = {"total": 0, "red": 0, "yellow": 0, "green": 0, "neutral": 0}
        sender_stats[sender]["total"] += 1
        if m_flag == "Red Flag":
            sender_stats[sender]["red"] += 1
        elif m_flag == "Yellow Flag":
            sender_stats[sender]["yellow"] += 1
        elif m_flag == "Green Flag":
            sender_stats[sender]["green"] += 1
        else:
            sender_stats[sender]["neutral"] += 1
            
        msg_explanation = ExplainabilityEngine.explain(msg["text"], m_flag, [], detected_lang=lang_info["language"])
        
        message_breakdown.append({
            "id": msg["id"],
            "sender": msg["sender"],
            "text": msg["text"],
            "flag": m_flag,
            "confidence": round(m_conf * 100, 1),
            "triggers": msg_explanation["highlighted_triggers"],
            "is_concerning": m_flag in ["Red Flag", "Yellow Flag"]
        })
        
    # 7. Self-training / Active Learning: if auto_learn is active and high confidence
    auto_trained = False
    if auto_learn and confidence > 0.85:
        try:
            ActiveLearningEngine.record_user_sample(
                text=cleaned,
                flag=flag_pred,
                behavior=beh_pred,
                source=f"auto_learn_{source_type}",
                language=lang_info["language"],
                auto_retrain=False # Keep it fast during live requests, retrain on feedback
            )
            auto_trained = True
        except Exception as e:
            print(f"Auto-learn error: {e}")

    analysis_record = {
        "id": f"analysis_{len(ANALYSIS_HISTORY) + 1}",
        "source_type": source_type,
        "language_detection": lang_info,
        "auto_trained": auto_trained,
        "summary": {
            "flag": flag_pred,
            "confidence": round(confidence * 100, 1),
            "severity_score": severity_score,
            "behavior_category": human_behavior,
            "behavior_raw": beh_pred,
            "behavior_confidence": round(beh_confidence * 100, 1)
        },
        "probabilities": probabilities_dict,
        "sentiment_and_toxicity": sentiment_metrics,
        "linguistic_signals": linguistic_signals,
        "explain_why": explanation,
        "contributing_features": [{"feature": f, "weight": w} for f, w in top_features],
        "conversation_breakdown": {
            "total_turns": len(messages),
            "senders": sender_stats,
            "messages": message_breakdown
        }
    }
    
    ANALYSIS_HISTORY.insert(0, {
        "id": analysis_record["id"],
        "flag": flag_pred,
        "behavior": human_behavior,
        "language": lang_info["language"],
        "confidence": round(confidence * 100, 1),
        "preview": cleaned[:80] + "..." if len(cleaned) > 80 else cleaned
    })
    if len(ANALYSIS_HISTORY) > 20:
        ANALYSIS_HISTORY.pop()
        
    return analysis_record


# ==================== ENDPOINTS ====================

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "multilingual_enabled": True,
        "supported_languages": ["English", "Hinglish (Hindi+English)", "Hindi (हिन्दी)", "Marathi (मराठी)"],
        "model_loaded": flag_model is not None,
        "ocr_available": True,
        "vocabulary_dimensions": vectorizer.transform(["test"]).shape[1] if vectorizer else 0
    }

@app.get("/api/model/metrics")
def get_metrics():
    if not model_metrics:
        raise HTTPException(status_code=404, detail="Model metrics not found.")
    learning_stats = ActiveLearningEngine.get_stats()
    metrics_copy = dict(model_metrics)
    metrics_copy["active_learning"] = learning_stats
    return metrics_copy

@app.get("/api/samples")
def get_samples():
    """Returns curated multilingual test scenarios for one-click demonstration."""
    samples = [
        {
            "id": "sample_hinglish_red",
            "title": "Hinglish: Guilt & Isolation (Classic Red Flag)",
            "category": "Hinglish Red Flag",
            "language": "Hinglish",
            "text": "Agar tu mujhse sach me pyaar karti na, to apne dosto se baat nahi karti. Sirf meri zarurat honi chahiye tujhe. Apna live location bhej abhi ke abhi, aur usko block kar WhatsApp pe!"
        },
        {
            "id": "sample_marathi_red",
            "title": "Marathi: Controlling & Live Location (Red Flag)",
            "category": "Marathi Red Flag",
            "language": "Marathi",
            "text": "Jar tuza mazyavar khara prem asel tar tu tya lokanshi bolnar nahis. Mala ata lagech tuza live location pathav. Block kar tyala ata nahi tar aapan sampavuyat!"
        },
        {
            "id": "sample_hindi_red",
            "title": "Hindi (Devanagari): Reality Invalidation (Red Flag)",
            "category": "Hindi Red Flag",
            "language": "Hindi",
            "text": "तुम पागल हो। अपनी मनगढ़ंत बातें मत बनाओ, ऐसा कभी नहीं हुआ था। मुझे अपनी लाइव लोकेशन भेजो अभी!"
        },
        {
            "id": "sample_hinglish_green",
            "title": "Hinglish: Respectful & Patient (Green Flag)",
            "category": "Hinglish Green Flag",
            "language": "Hinglish",
            "text": "Aaram se reply karna, koi jaldi nahi hai! Kaam par focus karo, take your time. Apne dosto ke sath khoob maza karo aaj raat!"
        },
        {
            "id": "sample_marathi_green",
            "title": "Marathi: Healthy Space & Pride (Green Flag)",
            "category": "Marathi Green Flag",
            "language": "Marathi",
            "text": "Aaramat reply de, kahi ghai nahiye! Mala khup abhiman ahe tuza, khup mehnat kelis ahes ya sathi. Mitranchya sobat changla enjoy kar!"
        },
        {
            "id": "sample_english_red",
            "title": "English: Guilt & Isolation (Red Flag)",
            "category": "English Red Flag",
            "language": "English",
            "text": "If you actually loved me, you wouldn't talk to your friends. You only need me. Send me your live location right now."
        },
        {
            "id": "sample_hinglish_yellow",
            "title": "Hinglish: Passive Aggression & Disappearing (Yellow Flag)",
            "category": "Hinglish Yellow Flag",
            "language": "Hinglish",
            "text": "Acha theek hai jo marzi karo, waise bhi meri baat ki koi value nahi hai. Sorry 4 din se gayab tha, phone silent pe tha. K."
        }
    ]
    return samples

@app.post("/api/analyze/text")
def analyze_text(payload: AnalyzeRequest):
    return run_full_analysis(payload.text, source_type="text", auto_learn=payload.auto_learn or False)

@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    auto_learn: Optional[bool] = Form(False)
):
    filename = file.filename.lower() if file.filename else "upload.png"
    contents = await file.read()
    
    if filename.endswith(".txt") or filename.endswith(".log") or filename.endswith(".chat"):
        try:
            text_content = contents.decode("utf-8")
        except UnicodeDecodeError:
            text_content = contents.decode("latin-1")
        result = run_full_analysis(text_content, source_type="document", auto_learn=auto_learn)
        result["file_info"] = {
            "filename": file.filename,
            "type": "chat_export_document",
            "size_bytes": len(contents)
        }
        return result
        
    elif any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]):
        ocr_result = OCREngine.extract_text_from_bytes(contents, filename=file.filename)
        if not ocr_result["success"] or not ocr_result["text"].strip():
            raise HTTPException(status_code=400, detail="Could not extract readable dialogue from the uploaded image.")
            
        analysis = run_full_analysis(ocr_result["text"], source_type="screenshot_ocr", auto_learn=auto_learn)
        analysis["file_info"] = {
            "filename": file.filename,
            "type": "image_screenshot",
            "ocr_engine": ocr_result["engine"],
            "image_dimensions": f"{ocr_result['image_width']}x{ocr_result['image_height']}",
            "extracted_text": ocr_result["text"]
        }
        return analysis
        
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a screenshot (.png, .jpg, .webp) or chat export (.txt)."
        )

# ==================== ACTIVE SELF-TRAINING ENDPOINTS ====================

@app.post("/api/train/feedback")
def submit_feedback(payload: FeedbackRequest):
    """
    User submits verified feedback or flag correction.
    The model immediately learns this pattern and updates its weights!
    """
    result = ActiveLearningEngine.record_user_sample(
        text=payload.text,
        flag=payload.flag,
        behavior=payload.behavior or "custom_learned",
        source=payload.source or "user_feedback",
        language=payload.language or "Multilingual",
        auto_retrain=True
    )
    # Reload model in memory so subsequent inferences immediately reflect the new learning
    load_models()
    return result

@app.get("/api/train/stats")
def get_training_stats():
    """Returns active self-training statistics and recent learned examples."""
    return ActiveLearningEngine.get_stats()

@app.post("/api/train/retrain")
def trigger_full_retrain():
    """Triggers an on-demand complete retraining pass over benchmark + user learning stream."""
    result = ActiveLearningEngine.retrain_live_model()
    load_models()
    return result

@app.get("/api/history")
def get_history():
    return ANALYSIS_HISTORY

# Mount static files if frontend is built
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
