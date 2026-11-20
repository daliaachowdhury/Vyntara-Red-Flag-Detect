"""
REDFLAG AI - Multilingual Machine Learning Pipeline & Explainability Engine
Supports English, Hinglish (Hindi in Latin script), Hindi (Devanagari),
and Marathi (Roman & Devanagari scripts), with subword phonetic feature attribution.
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple

class LanguageDetector:
    """Detects script and spoken language variant including code-mixed Hinglish and Marathi."""

    # Key vocabulary markers for dialect identification
    MARATHI_DEVANAGARI_MARKERS = ["आहे", "नाही", "करायचं", "झालं", "बोलूयात", "तुला", "मला", "माझं", "कसं", "काही", "घाई", "मित्रांशी"]
    HINDI_DEVANAGARI_MARKERS = ["है", "नहीं", "करना", "हुआ", "बात", "तुम्हें", "मुझे", "मेरा", "कैसा", "कुछ", "जल्दी", "दोस्तों"]
    
    MARATHI_ROMAN_MARKERS = [
        "mala", "tula", "mazyavar", "majyavar", "asel", "nahis", "bolnar", "ahes", "kiti",
        "kahi", "ghai", "aapan", "kasa", "jevlis", "vait", "vatla", "shantpane", "goshticha", "tyala", "mhanun", "tuzha"
    ]
    
    HINGLISH_ROMAN_MARKERS = [
        "karti", "karta", "mujhse", "tujhe", "pyaar", "pyar", "bhej", "apne", "chal",
        "kaha", "raha", "rahi", "hamesha", "baatein", "yaar", "kaam", "dosto", "dost", "karunga", "karungi", "choda", "barbaad", "dimag"
    ]

    @classmethod
    def detect(cls, text: str) -> Dict[str, Any]:
        if not text:
            return {"language": "English", "code": "en", "script": "Latin", "is_multilingual": False}

        lower = text.lower()
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        total_letters = len([c for c in text if c.isalpha() or ('\u0900' <= c <= '\u097F')])

        # 1. Devanagari script detected
        if devanagari_chars > 3:
            marathi_score = sum(1 for m in cls.MARATHI_DEVANAGARI_MARKERS if m in text)
            hindi_score = sum(1 for h in cls.HINDI_DEVANAGARI_MARKERS if h in text)
            
            if marathi_score > hindi_score:
                return {
                    "language": "Marathi (मराठी)",
                    "code": "mr",
                    "script": "Devanagari",
                    "is_multilingual": True,
                    "confidence": 0.94
                }
            else:
                return {
                    "language": "Hindi (हिन्दी)",
                    "code": "hi",
                    "script": "Devanagari",
                    "is_multilingual": True,
                    "confidence": 0.96
                }

        # 2. Latin script: check for Marathi Roman, Hinglish, or English
        words = set(re.findall(r'\b[a-zA-Z]+\b', lower))
        
        marathi_hits = sum(1 for m in cls.MARATHI_ROMAN_MARKERS if m in words or any(w.startswith(m) for w in words))
        hinglish_hits = sum(1 for h in cls.HINGLISH_ROMAN_MARKERS if h in words or any(w.startswith(h) for w in words))
        
        if marathi_hits >= 2 or (marathi_hits >= 1 and marathi_hits > hinglish_hits):
            return {
                "language": "Marathi (Roman Script / मराठी)",
                "code": "mr-latn",
                "script": "Latin",
                "is_multilingual": True,
                "confidence": 0.92
            }
        elif hinglish_hits >= 2 or (hinglish_hits >= 1 and ("nahi" in lower or "karti" in lower or "pyaar" in lower or "pyar" in lower or "bhej" in lower)):
            return {
                "language": "Hinglish (Hindi + English)",
                "code": "hi-en",
                "script": "Latin",
                "is_multilingual": True,
                "confidence": 0.95
            }
        
        # Check if code-mixed with a single Indic word
        if hinglish_hits == 1 or marathi_hits == 1:
            return {
                "language": "Hinglish / Code-Mixed",
                "code": "hi-en",
                "script": "Latin",
                "is_multilingual": True,
                "confidence": 0.85
            }

        return {
            "language": "English",
            "code": "en",
            "script": "Latin",
            "is_multilingual": False,
            "confidence": 0.98
        }


class TextPreprocessor:
    """Preprocesses conversational text across languages and extracts stylistic signals."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text.strip())

    @staticmethod
    def extract_linguistic_signals(text: str) -> Dict[str, Any]:
        if not text:
            return {"caps_ratio": 0.0, "exclamation_count": 0, "question_count": 0, "length": 0, "word_count": 0}
        
        words = text.split()
        letters = [c for c in text if c.isalpha()]
        caps = [c for c in letters if c.isupper()]
        caps_ratio = len(caps) / len(letters) if letters else 0.0
        
        lang_info = LanguageDetector.detect(text)
        
        return {
            "caps_ratio": round(caps_ratio, 3),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "ellipsis_count": text.count("..."),
            "length": len(text),
            "word_count": len(words),
            "detected_language": lang_info["language"],
            "language_code": lang_info["code"]
        }

    @staticmethod
    def split_into_messages(raw_text: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        messages = []
        chat_regex = re.compile(r'^(?:\[?[\d/:\s,\w-]+\]?\s+)?([^:]+):\s+(.*)$')
        
        for idx, line in enumerate(lines):
            match = chat_regex.match(line)
            if match:
                sender = match.group(1).strip()
                content = match.group(2).strip()
            else:
                sender = f"Turn {idx + 1}"
                content = line
                
            if content:
                messages.append({
                    "id": idx + 1,
                    "sender": sender,
                    "text": content
                })
                
        if len(messages) == 1 and len(messages[0]["text"].split()) > 25:
            sentences = re.split(r'(?<=[.!?।])\s+', messages[0]["text"])
            if len(sentences) > 1:
                messages = [{"id": i + 1, "sender": f"Statement {i + 1}", "text": s.strip()} for i, s in enumerate(sentences) if s.strip()]
                
        return messages


# Multilingual Semantic Rules across English, Hinglish, Hindi, and Marathi
SEMANTIC_PATTERNS = [
    {
        "category": "Controlling & Isolation",
        "flag": "Red Flag",
        "patterns": [
            r"\b(delete|remove)\s+(your\s+)?(instagram|snapchat|photos?|account)\b",
            r"\b(stop\s+talking\s+to|don't\s+talk\s+to|can't\s+talk\s+to|forbid\s+you)\b",
            r"\b(who\s+are\s+you\s+with|who\s+was\s+that)\b",
            r"\b(send(\s+me)?\s+your\s+live\s+location|share\s+location)\b",
            r"\b(live\s+location\s+(bhej|pathav|bhejo))\b",
            r"\b(dosto\s+se\s+baat\s+nahi|mitranshi\s+bolnar\s+nahis)\b",
            r"\b(sirf\s+meri\s+zarurat|fakt\s+majhi\s+garaj)\b",
            r"\b(block\s+kar\s+(abhi|usko|tyala|ata))\b",
            r"\b(phone\s+ka\s+password|mobile\s+passcode)\b",
            r"लाईव्ह\s+लोकेशन\s+पाठव|लाइव\s+लोकेशन\s+भेजो|दोस्तों\s+से\s+बात\s+नहीं|मित्रांशी\s+बोलणार\s+नाहीस"
        ],
        "explanation": "Attempts to isolate partner from social circles, digital privacy violations, or forced tracking.",
        "risk_level": "High Risk"
    },
    {
        "category": "Emotional Guilt-Tripping",
        "flag": "Red Flag",
        "patterns": [
            r"\b(if\s+you\s+(actually|really|truly)?\s*loved\s+me)\b",
            r"\b(after\s+everything\s+i\s+(did|sacrificed)\s+for\s+you)\b",
            r"\b(sach\s+me\s+pya+r\s+karti|khara\s+prem\s+asel)\b",
            r"\b(itna\s+kuch\s+kiya\s+tere\s+liye|tujhyasathi\s+kiti\s+goshti)\b",
            r"\b(meri\s+to\s+koi\s+value\s+nahi|tula\s+majhi\s+kaddar\s+nahi)\b",
            r"\b(akela\s+tadapta|mar\s+jaunga|marun\s+gelo)\b",
            r"सच\s+में\s+प्यार|प्रेम\s+असेल\s+तर|इतना\s+सब\s+किया"
        ],
        "explanation": "Weaponizes love, past favors, or self-harm/martyrdom to emotionally coerce the partner.",
        "risk_level": "High Risk"
    },
    {
        "category": "Gaslighting & Reality Invalidation",
        "flag": "Red Flag",
        "patterns": [
            r"\b(you're\s+(crazy|insane|paranoid|hysterical))\b",
            r"\b(never\s+happened|making\s+things\s+up)\b",
            r"\b(tu\s+pagal\s+(ho\s+gayi|hai)|tu\s+vedi\s+ahes)\b",
            r"\b(dimag\s+me\s+baatein|ugach\s+natak|drama\s+queen)\b",
            r"\b(aisa\s+kabhi\s+nahi\s+hua|ashi\s+kahi\s+zaliach\s+navhti)\b",
            r"तुम\s+पागल\s+हो|तू\s+वेडी\s+आहेस|नाटक\s+करतेस|ऐसा\s+कभी\s+नहीं\s+हुआ"
        ],
        "explanation": "Directly denies shared events, undermines emotional sanity, or labels partner's feelings as 'drama'.",
        "risk_level": "High Risk"
    },
    {
        "category": "Intimidation & Verbal Hostility",
        "flag": "Red Flag",
        "patterns": [
            r"\b(if\s+you\s+leave\s+me|ruin\s+your\s+life)\b",
            r"\b(zindagi\s+barbaad|aayushya\s+barbad|jeena\s+haram)\b",
            r"\b(pol\s+khol\s+dunga|dekh\s+lena\s+main\s+kya\s+karta)\b",
            r"\b(sagla\s+sampavun\s+takto|pachtayegi)\b",
            r"जीना\s+हराम\s+कर\s+दूंगा|बर्बाद\s+कर|धमकी"
        ],
        "explanation": "Overt coercion, blackmail, and explicit threats of relational or reputational destruction.",
        "risk_level": "Severe Risk"
    },
    {
        "category": "Love Bombing & Hyper-Intensity",
        "flag": "Red Flag",
        "patterns": [
            r"\b(soulmates?|destiny|move\s+in\s+together\s+next\s+week)\b",
            r"\b(meri\s+jaan\s+hai|meri\s+poori\s+zindagi|sath\s+rehne\s+aa\s+ja)\b",
            r"\b(2\s+din\s+pehle\s+mile|ek\s+dusre\s+ke\s+liye\s+bane)\b"
        ],
        "explanation": "Premature, disproportionate intensity and rushed commitments aimed at creating rapid dependency.",
        "risk_level": "Moderate-High Risk"
    },
    {
        "category": "Mixed Signals & Avoidance",
        "flag": "Yellow Flag",
        "patterns": [
            r"\b(not\s+looking\s+for\s+anything\s+serious|no\s+labels|just\s+vibe)\b",
            r"\b(commitment\s+nahi\s+chahiye|commitment\s+madhe\s+padaycha\s+nahi)\b",
            r"\b(mujhe\s+nahi\s+pata\s+kya\s+chahiye|mala\s+nahi\s+mahit)\b",
            r"\b(ghost(ed)?|sorry\s+gayab\s+tha|\d+\s+din\s+se\s+gayab)\b",
            r"कमिटमेंट\s+के\s+लिए\s+तैयार\s+नहीं|कमिटमेंट\s+मध्ये\s+पडायचं\s+नाही"
        ],
        "explanation": "Displays hot-and-cold engagement, avoidance of commitment, and inconsistent communication.",
        "risk_level": "Caution"
    },
    {
        "category": "Passive Aggression",
        "flag": "Yellow Flag",
        "patterns": [
            r"\b(must\s+be\s+nice|left\s+me\s+on\s+(read|delivered))\b",
            r"\b(theek\s+hai\s+jo\s+marzi\s+karo|je\s+karaycha\s+te\s+kar)\b",
            r"\b(meri\s+koi\s+value\s+nahi|maza\s+kon\s+aiknar)\b",
            r"\b(badiya\s+hai|chhan\s+ahe\s+saglyansathi)\b",
            r"^k[\.\!]?$",
            r"जो\s+करना\s+है\s+करो|जे\s+करायचं\s+ते\s+कर"
        ],
        "explanation": "Indirect resentment and passive withdrawal rather than addressing emotions directly.",
        "risk_level": "Caution"
    },
    {
        "category": "Healthy Boundaries & Support",
        "flag": "Green Flag",
        "patterns": [
            r"\b(take\s+(all\s+the\s+)?time\s+you\s+need|no\s+stress|no\s+rush)\b",
            r"\b(aaram\s+se\s+reply|aaramat\s+reply|kahi\s+ghai\s+nahi|koi\s+jaldi\s+nahi)\b",
            r"\b(personal\s+space\s+ki\s+respect|respect\s+your\s+space)\b",
            r"\b(dosto\s+ke\s+sath\s+enjoy|mitranchya\s+sobat\s+enjoy|khup\s+maja\s+kar)\b",
            r"\b(bura\s+laga|vait\s+vatla|shaant\s+hoke\s+baat|shantpane\s+boluyat)\b",
            r"\b(proud\s+of\s+you|proud\s+feel|khup\s+abhiman\s+ahe)\b",
            r"\b(i\s+apologize|sorry|galti\s+thi|chook\s+hoti)\b",
            r"आराम\s+से\s+जवाब|आरामात\s+उत्तर|काही\s+घाई\s+नाही|कोई\s+जल्दी\s+नहीं|गर्व\s+है|अभिमान\s+आहे"
        ],
        "explanation": "Respects personal autonomy, boundaries, patience, and non-defensive constructive dialogue.",
        "risk_level": "Healthy"
    }
]

class ExplainabilityEngine:
    """Explains predictions using subword/word TF-IDF attribution and multilingual pattern matching."""
    
    @staticmethod
    def explain(text: str, predicted_flag: str, top_features: List[Tuple[str, float]], detected_lang: str = "English") -> Dict[str, Any]:
        matched_categories = []
        highlighted_phrases = []
        detailed_reasons = []
        
        lower_text = text.lower()
        
        for rule in SEMANTIC_PATTERNS:
            for pat in rule["patterns"]:
                match = re.search(pat, text, re.IGNORECASE)
                if match:
                    matched_phrase = match.group(0)
                    if matched_phrase not in highlighted_phrases:
                        highlighted_phrases.append(matched_phrase)
                    if rule["category"] not in matched_categories:
                        matched_categories.append(rule["category"])
                        detailed_reasons.append({
                            "category": rule["category"],
                            "matched_snippet": matched_phrase,
                            "explanation": rule["explanation"],
                            "risk_level": rule["risk_level"]
                        })
                        
        # Synthesize explain WHY statement
        lang_prefix = f"[{detected_lang}] " if detected_lang != "English" else ""
        
        if predicted_flag == "Red Flag":
            if detailed_reasons:
                why_text = f"{lang_prefix}Flagged as a potential Red Flag due to communication patterns associated with {', '.join(matched_categories[:2])}. Specifically, phrases like '{highlighted_phrases[0]}' indicate conditional affection, boundary violation, or excessive control."
            else:
                top_ngram = top_features[0][0] if top_features else "specific phrases"
                why_text = f"{lang_prefix}Flagged as a potential Red Flag based on linguistic markers and statistical correlation with coercive patterns (prominent signals: '{top_ngram}')."
        elif predicted_flag == "Yellow Flag":
            if detailed_reasons:
                why_text = f"{lang_prefix}Flagged as a Yellow Flag due to indicators of {', '.join(matched_categories[:2])}. Messages show indirectness, mixed signals, or boundary friction."
            else:
                why_text = f"{lang_prefix}Flagged as a Yellow Flag indicating ambiguous, inconsistent, or emotionally avoidant communication."
        elif predicted_flag == "Green Flag":
            if detailed_reasons:
                why_text = f"{lang_prefix}Identified as Healthy / Green Flag communication showcasing {', '.join(matched_categories[:2])}. The language fosters safety, autonomy, and respectful dialogue."
            else:
                why_text = f"{lang_prefix}Identified as Healthy / Green Flag communication showing constructive, respectful, and supportive language."
        else:
            why_text = f"{lang_prefix}Insufficient conversational depth to draw a reliable behavioral inference. The text contains general logistics, single-word greetings, or neutral statements."

        return {
            "summary_reason": why_text,
            "detected_language": detected_lang,
            "detected_patterns": matched_categories if matched_categories else (["General Positive Tone"] if predicted_flag == "Green Flag" else ["Neutral Interaction"]),
            "highlighted_triggers": highlighted_phrases,
            "pattern_details": detailed_reasons,
            "disclaimer": "Illustrative behavioral analysis based on communication markers. This is an educational tool and does not constitute psychological evaluation or definitive judgment."
        }

    @staticmethod
    def calculate_toxicity_and_sentiment(text: str, predicted_flag: str, severity: float) -> Dict[str, Any]:
        signals = TextPreprocessor.extract_linguistic_signals(text)
        lower = text.lower()
        
        toxicity = 5.0
        if predicted_flag == "Red Flag":
            toxicity = 65.0 + (severity * 30.0)
            if signals["caps_ratio"] > 0.35:
                toxicity += 5.0
            if any(w in lower for w in ["idiot", "ruin", "temper", "barbaad", "pagal", "vedi", "haram"]):
                toxicity += 10.0
        elif predicted_flag == "Yellow Flag":
            toxicity = 25.0 + (severity * 25.0)
        elif predicted_flag == "Green Flag":
            toxicity = 2.0
            
        toxicity = min(100.0, max(0.0, round(toxicity, 1)))
        
        if predicted_flag == "Red Flag":
            sentiment = -0.75
            sentiment_label = "Negative / Hostile"
        elif predicted_flag == "Yellow Flag":
            sentiment = -0.25
            sentiment_label = "Ambivalent / Strained"
        elif predicted_flag == "Green Flag":
            sentiment = 0.85
            sentiment_label = "Positive / Warm"
        else:
            sentiment = 0.0
            sentiment_label = "Neutral / Informational"
            
        return {
            "toxicity_score": toxicity,
            "sentiment_score": sentiment,
            "sentiment_label": sentiment_label,
            "emotional_intensity": "High" if (signals["caps_ratio"] > 0.25 or signals["exclamation_count"] > 2) else "Moderate" if signals["word_count"] > 15 else "Low"
        }
