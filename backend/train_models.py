"""
REDFLAG AI - Multilingual Model Training and Evaluation Pipeline
Trains baseline and advanced subword-aware classifiers across English, Hinglish, Hindi, and Marathi.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from dataset_builder import save_dataset

def train_and_evaluate():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "redflag_dataset.csv")
    print("Generating comprehensive multilingual dataset (English, Hinglish, Hindi, Marathi)...")
    save_dataset(data_dir)
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} samples across {df['flag'].nunique()} flag classes and {df['language'].nunique()} languages.")
    
    # Text and labels
    X = df['text']
    y_flag = df['flag']
    y_behavior = df['behavior']
    
    # Synchronized Train / Test split (80/20)
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42, stratify=y_flag
    )
    
    X_train = X.iloc[idx_train]
    X_test = X.iloc[idx_test]
    y_train_flag = y_flag.iloc[idx_train]
    y_test_flag = y_flag.iloc[idx_test]
    y_train_beh = y_behavior.iloc[idx_train]
    y_test_beh = y_behavior.iloc[idx_test]
    
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # 1. Multilingual Subword + Word Feature Union Vectorizer
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
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    num_features = X_train_vec.shape[1]
    print(f"Vocabulary feature dimension: {num_features} dimensions (word + subword char n-grams)")
    
    # 2. Baseline Benchmark: Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train_flag)
    nb_preds = nb_model.predict(X_test_vec)
    nb_acc = accuracy_score(y_test_flag, nb_preds)
    nb_p, nb_r, nb_f1, _ = precision_recall_fscore_support(y_test_flag, nb_preds, average='weighted', zero_division=0)
    
    print("\n--- BASELINE MODEL: Multinomial Naive Bayes ---")
    print(f"Accuracy: {nb_acc:.4f} | F1-Score: {nb_f1:.4f}")
    
    # 3. Primary Classifier: Calibrated Logistic Regression
    lr_base = LogisticRegression(
        max_iter=1000,
        C=3.0,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    calibrated_flag_model = CalibratedClassifierCV(estimator=lr_base, cv=3)
    calibrated_flag_model.fit(X_train_vec, y_train_flag)
    
    flag_preds = calibrated_flag_model.predict(X_test_vec)
    flag_acc = accuracy_score(y_test_flag, flag_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test_flag, flag_preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test_flag, flag_preds, average='weighted', zero_division=0)
    
    print("\n--- PRIMARY MODEL: Calibrated Logistic Regression ---")
    print(f"Accuracy:  {flag_acc:.4f}")
    print(f"Precision: {p_weighted:.4f} (Weighted) | {p_macro:.4f} (Macro)")
    print(f"Recall:    {r_weighted:.4f} (Weighted) | {r_macro:.4f} (Macro)")
    print(f"F1-Score:  {f1_weighted:.4f} (Weighted) | {f1_macro:.4f} (Macro)")
    
    class_names = sorted(list(y_flag.unique()))
    report_dict = classification_report(y_test_flag, flag_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test_flag, flag_preds, labels=class_names)
    
    # 4. Secondary Classifier: Behavioral Pattern Classifier
    beh_model = LogisticRegression(
        max_iter=1000,
        C=2.5,
        class_weight='balanced',
        solver='lbfgs',
        random_state=42
    )
    beh_model.fit(X_train_vec, y_train_beh)
    beh_preds = beh_model.predict(X_test_vec)
    beh_acc = accuracy_score(y_test_beh, beh_preds)
    _, _, beh_f1, _ = precision_recall_fscore_support(y_test_beh, beh_preds, average='weighted', zero_division=0)
    print(f"\n--- BEHAVIORAL MODEL: Sub-category Classifier ---")
    print(f"Accuracy: {beh_acc:.4f} | F1-Score: {beh_f1:.4f}")
    
    # Interpretable model for feature importance
    interp_lr = LogisticRegression(max_iter=1000, C=3.0, class_weight='balanced', solver='lbfgs')
    interp_lr.fit(X_train_vec, y_train_flag)
    
    # Save artifacts
    joblib.dump(calibrated_flag_model, os.path.join(models_dir, "flag_classifier.joblib"))
    joblib.dump(beh_model, os.path.join(models_dir, "behavior_classifier.joblib"))
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.joblib"))
    joblib.dump(interp_lr, os.path.join(models_dir, "interpretable_model.joblib"))
    
    metrics_data = {
        "model_architecture": "Multilingual Word + Subword Char N-Grams + Calibrated Logistic Regression",
        "dataset_size": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": num_features,
        "languages": sorted(list(df['language'].unique())),
        "classes": class_names,
        "baseline_comparison": {
            "model": "Multinomial Naive Bayes",
            "accuracy": round(float(nb_acc), 4),
            "f1_weighted": round(float(nb_f1), 4)
        },
        "primary_model_evaluation": {
            "accuracy": round(float(flag_acc), 4),
            "precision_macro": round(float(p_macro), 4),
            "recall_macro": round(float(r_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(p_weighted), 4),
            "recall_weighted": round(float(r_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        },
        "behavioral_classifier_evaluation": {
            "accuracy": round(float(beh_acc), 4),
            "f1_weighted": round(float(beh_f1), 4),
            "behaviors": sorted(list(y_behavior.unique()))
        }
    }
    
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
        
    print(f"\nAll multilingual models and metrics successfully saved to {models_dir}!")
    
    # 5. Multilingual Sanity Test Run
    multilingual_tests = [
        ("English", "If you actually loved me, you wouldn't talk to your friends. You only need me."),
        ("Hinglish", "Agar tu sach me pyaar karti na, to apne dosto se baat nahi karti. Sirf meri zarurat honi chahiye tujhe."),
        ("Hinglish", "Apna live location bhej abhi ke abhi. Usko block kar WhatsApp pe."),
        ("Hinglish", "Aaram se reply karna, koi jaldi nahi hai! Kaam par focus karo."),
        ("Hindi Devanagari", "अगर तुम सच में मुझसे प्यार करते तो अपने दोस्तों से बात नहीं करते।"),
        ("Hindi Devanagari", "आराम से जवाब देना, कोई जल्दी नहीं है।"),
        ("Marathi Roman", "Jar tuza mazyavar khara prem asel tar tu tya lokanshi bolnar nahis."),
        ("Marathi Roman", "Mala ata lagech tuza live location pathav."),
        ("Marathi Roman", "Aaramat reply de, kahi ghai nahiye!"),
        ("Marathi Devanagari", "जर तुझं माझ्यावर खरंच प्रेम असेल तर तू मित्रांशी बोलणार नाहीस।"),
        ("Marathi Devanagari", "आरामात उत्तर दे, काही घाई नाही।"),
        ("Hinglish Logistics", "Kitne baje chalna hai? Chai peene chale?")
    ]
    
    print("\n--- MULTILINGUAL SANITY TEST RUN ---")
    for lang, text in multilingual_tests:
        vec = vectorizer.transform([text])
        pred = calibrated_flag_model.predict(vec)[0]
        prob = max(calibrated_flag_model.predict_proba(vec)[0])
        beh = beh_model.predict(vec)[0]
        print(f"[{lang}] \"{text}\"")
        print(f" -> Predicted: [{pred}] ({prob*100:.1f}% Confidence) | Behavior: {beh}\n")

if __name__ == "__main__":
    train_and_evaluate()
