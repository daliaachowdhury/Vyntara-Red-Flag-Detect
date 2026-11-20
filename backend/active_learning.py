"""
REDFLAG AI - Continuous Active Learning & Self-Training Engine
Enables the model to adapt, self-train, and continuously learn from
user-shared conversations, screenshot OCR texts, and manual feedback corrections.
"""

import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STREAM_FILE = os.path.join(DATA_DIR, "active_learning_stream.json")

class ActiveLearningEngine:
    """Manages continuous self-training and live model updating from user interactions."""
    
    @staticmethod
    def get_stream() -> List[Dict[str, Any]]:
        if not os.path.exists(STREAM_FILE):
            return []
        try:
            with open(STREAM_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @staticmethod
    def save_stream(stream: List[Dict[str, Any]]):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(STREAM_FILE, "w", encoding="utf-8") as f:
            json.dump(stream, f, indent=2, ensure_ascii=False)

    @classmethod
    def record_user_sample(
        cls,
        text: str,
        flag: str,
        behavior: str = "custom_learned",
        source: str = "user_feedback",
        language: str = "Multilingual",
        auto_retrain: bool = True
    ) -> Dict[str, Any]:
        """
        Records a new text or OCR screenshot dialogue into the active learning stream
        and triggers self-training adaptation.
        """
        text = text.strip()
        if not text:
            return {"success": False, "error": "Empty text"}

        stream = cls.get_stream()
        
        # Check if already in stream
        existing = next((item for item in stream if item["text"].strip().lower() == text.lower()), None)
        if existing:
            existing["flag"] = flag
            existing["behavior"] = behavior
            existing["language"] = language
            existing["updated_at"] = datetime.datetime.now().isoformat()
            existing["correction_count"] = existing.get("correction_count", 0) + 1
        else:
            entry = {
                "id": f"sample_{len(stream) + 1}_{int(datetime.datetime.now().timestamp())}",
                "text": text,
                "flag": flag,
                "behavior": behavior,
                "language": language,
                "source": source,
                "created_at": datetime.datetime.now().isoformat()
            }
            stream.append(entry)

        cls.save_stream(stream)
        
        retrain_result = None
        if auto_retrain:
            retrain_result = cls.retrain_live_model()

        return {
            "success": True,
            "message": f"Successfully incorporated new pattern into model brain! Total learned user samples: {len(stream)}",
            "total_learned_samples": len(stream),
            "retrain_status": retrain_result
        }

    @classmethod
    def retrain_live_model(cls) -> Dict[str, Any]:
        """
        Re-trains the pipeline incorporating both core dataset and user-shared texts/screenshots.
        Completes in <0.5s and updates in-memory and serialized model weights.
        """
        csv_path = os.path.join(DATA_DIR, "redflag_dataset.csv")
        if not os.path.exists(csv_path):
            from dataset_builder import save_dataset
            save_dataset(DATA_DIR)
            
        base_df = pd.read_csv(csv_path)
        stream = cls.get_stream()
        
        if stream:
            user_df = pd.DataFrame(stream)[["text", "flag", "behavior", "language"]]
            user_df["severity"] = user_df["flag"].map({
                "Red Flag": 0.90,
                "Yellow Flag": 0.60,
                "Green Flag": 0.05,
                "Insufficient Context": 0.0
            }).fillna(0.5)
            user_df["explanation"] = "Continuous learning sample from user interaction."
            user_df["triggers"] = user_df["text"].apply(lambda t: [w for w in t.split() if len(w) > 4][:2])
            
            # Combine base dataset + user learning stream (give user samples higher weight if needed)
            combined_df = pd.concat([base_df, user_df], ignore_index=True)
        else:
            combined_df = base_df

        X = combined_df["text"]
        y_flag = combined_df["flag"]
        y_beh = combined_df["behavior"]

        # Multilingual Feature Union: Word N-grams + Character Subword N-Grams
        vectorizer = FeatureUnion([
            ('word_tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                strip_accents='unicode'
            )),
            ('char_tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True
            ))
        ])

        X_vec = vectorizer.fit_transform(X)

        # Primary Flag Model
        lr_base = LogisticRegression(
            max_iter=1000,
            C=3.0,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        )
        flag_model = CalibratedClassifierCV(estimator=lr_base, cv=3)
        flag_model.fit(X_vec, y_flag)

        # Secondary Behavioral Model
        beh_model = LogisticRegression(
            max_iter=1000,
            C=2.5,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        )
        beh_model.fit(X_vec, y_beh)

        # Direct interpretable model for feature importance
        interp_lr = LogisticRegression(max_iter=1000, C=3.0, class_weight='balanced', solver='lbfgs')
        interp_lr.fit(X_vec, y_flag)

        # Save artifacts
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(flag_model, os.path.join(MODELS_DIR, "flag_classifier.joblib"))
        joblib.dump(beh_model, os.path.join(MODELS_DIR, "behavior_classifier.joblib"))
        joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
        joblib.dump(interp_lr, os.path.join(MODELS_DIR, "interpretable_model.joblib"))

        # Update metrics file
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        metrics_data = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                
        metrics_data["total_training_samples"] = len(combined_df)
        metrics_data["user_active_learning_samples"] = len(stream)
        metrics_data["last_self_trained_at"] = datetime.datetime.now().isoformat()
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        return {
            "status": "success",
            "total_dataset_size": len(combined_df),
            "user_learned_count": len(stream),
            "retrained_at": metrics_data["last_self_trained_at"]
        }

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        stream = cls.get_stream()
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        last_trained = "Never"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    last_trained = m.get("last_self_trained_at", "Baseline")
            except Exception:
                pass

        lang_counts = {}
        flag_counts = {}
        for s in stream:
            l = s.get("language", "Unknown")
            fl = s.get("flag", "Unknown")
            lang_counts[l] = lang_counts.get(l, 0) + 1
            flag_counts[fl] = flag_counts.get(fl, 0) + 1

        return {
            "total_learned_samples": len(stream),
            "last_self_trained": last_trained,
            "language_breakdown": lang_counts,
            "flag_breakdown": flag_counts,
            "recent_learned_samples": stream[-10:][::-1]
        }

