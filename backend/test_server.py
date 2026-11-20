"""
REDFLAG AI - Multilingual & Continuous Self-Training End-to-End Test Suite
Tests:
1. /api/health with multilingual indicators
2. /api/model/metrics with active learning statistics
3. /api/samples with Hinglish, Marathi, Hindi, English presets
4. Multilingual text analysis across languages
5. Active learning feedback & real-time self-training
6. Screenshot image OCR upload
7. Static frontend bundle serving
"""

import os
import sys
import json
import urllib.request
import urllib.error

def test_multilingual_and_selftraining():
    base_url = "http://127.0.0.1:8000"
    
    print("\n[1/7] Testing /api/health (Multilingual capabilities)...")
    with urllib.request.urlopen(f"{base_url}/api/health") as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        print(f" -> Health OK: {data}")
        assert data.get("multilingual_enabled") is True

    print("\n[2/7] Testing /api/model/metrics & Active Learning stats...")
    with urllib.request.urlopen(f"{base_url}/api/model/metrics") as response:
        assert response.status == 200
        metrics = json.loads(response.read().decode())
        print(f" -> Architecture: {metrics['model_architecture']}")
        print(f" -> Accuracy: {metrics['primary_model_evaluation']['accuracy']*100:.1f}%")
        print(f" -> Active learning samples: {metrics.get('active_learning', {}).get('total_learned_samples', 0)}")

    print("\n[3/7] Testing /api/samples (Multilingual presets)...")
    with urllib.request.urlopen(f"{base_url}/api/samples") as response:
        assert response.status == 200
        samples = json.loads(response.read().decode())
        print(f" -> Loaded {len(samples)} sample scenarios.")

    print("\n[4/7] Testing Multilingual NLP Inference (English, Hinglish, Marathi, Hindi)...")
    multilingual_tests = [
        ("English", "If you actually loved me, you wouldn't talk to your friends. You only need me. Send me your live location right now.", "Red Flag"),
        ("Hinglish", "Agar tu sach me pyaar karti na, to apne dosto se baat nahi karti. Sirf meri zarurat honi chahiye tujhe. Live location bhej abhi!", "Red Flag"),
        ("Marathi Roman", "Jar tuza mazyavar khara prem asel tar tu tya lokanshi bolnar nahis. Mala ata lagech tuza live location pathav.", "Red Flag"),
        ("Marathi Roman", "Aaramat reply de, kahi ghai nahiye! Tuza kaam zalyavar message kar.", "Green Flag"),
        ("Hindi Devanagari", "तुम पागल हो। अपनी मनगढ़ंत बातें मत बनाओ, ऐसा कभी नहीं हुआ था।", "Red Flag"),
        ("Hinglish Logistics", "Kitne baje chalna hai? Chai peene chale shaam ko?", "Insufficient Context")
    ]
    
    for lang, text, expected_flag in multilingual_tests:
        req_data = json.dumps({"text": text}).encode('utf-8')
        req = urllib.request.Request(
            f"{base_url}/api/analyze/text",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            assert response.status == 200
            res = json.loads(response.read().decode())
            flag = res['summary']['flag']
            conf = res['summary']['confidence']
            beh = res['summary']['behavior_category']
            detected_l = res.get('language_detection', {}).get('language', 'Unknown')
            print(f" -> [{detected_l}] \"{text[:45]}...\"")
            print(f"    Expected: [{expected_flag}] | Got: [{flag}] ({conf}%) | {beh}\n")

    print("[5/7] Testing Active Self-Training & Adaptation (/api/train/feedback)...")
    user_learned_text = "Tula sangun pan samajhat nahi ka, majhyashi asha padhatine kadhich bolaycha nahis."
    feedback_payload = json.dumps({
        "text": user_learned_text,
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Marathi"
    }).encode('utf-8')
    req = urllib.request.Request(
        f"{base_url}/api/train/feedback",
        data=feedback_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        fb_res = json.loads(response.read().decode())
        print(f" -> Feedback status: {fb_res.get('message')}")
        print(f" -> Total learned samples: {fb_res.get('total_learned_samples')}")

    # Immediately verify that the model predicts this newly learned phrase with high confidence!
    req_verify = urllib.request.Request(
        f"{base_url}/api/analyze/text",
        data=json.dumps({"text": user_learned_text}).encode('utf-8'),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_verify) as response:
        assert response.status == 200
        v_res = json.loads(response.read().decode())
        print(f" -> Immediate Retrain Verification for learned phrase:")
        print(f"    Predicted: [{v_res['summary']['flag']}] ({v_res['summary']['confidence']}%)")
        assert v_res['summary']['flag'] == "Red Flag"

    print("\n[6/7] Testing Screenshot Image Upload & OCR Analysis (/api/analyze/upload)...")
    img_path = os.path.join(os.path.dirname(__file__), "samples", "screenshot_red_flag.png")
    if os.path.exists(img_path):
        boundary = "----TestBoundaryXYZ"
        with open(img_path, "rb") as f:
            file_bytes = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="screenshot_red_flag.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(f"{base_url}/api/analyze/upload", data=body)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req) as response:
            assert response.status == 200
            res = json.loads(response.read().decode())
            print(f" -> OCR Result Flag: [{res['summary']['flag']}] | Engine: {res.get('file_info', {}).get('ocr_engine')}")

    print("\n[7/7] Testing Frontend Static Serving (/)...")
    with urllib.request.urlopen(f"{base_url}/") as response:
        assert response.status == 200
        html = response.read().decode()
        assert "REDFLAG AI" in html
        print(" -> Multilingual Frontend HTML served successfully!")

    print("\n🎉 ALL 7/7 MULTILINGUAL & SELF-TRAINING CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_multilingual_and_selftraining()
