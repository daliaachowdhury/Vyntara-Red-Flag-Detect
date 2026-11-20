"""
REDFLAG AI - Multilingual Dataset Generator and Augmenter
Generates a comprehensive, balanced, and multilingual labeled dataset:
- Languages supported: English, Hinglish (Romanized Hindi), Hindi (Devanagari), Marathi (Roman & Devanagari)
- Flags: Red Flag, Yellow Flag, Green Flag, Insufficient Context
- Behaviors:
  - guilt_tripping
  - controlling_isolation
  - gaslighting_invalidation
  - threats_disrespect
  - love_bombing
  - mixed_signals
  - passive_aggression
  - inconsistency_stonewalling
  - boundary_pushing
  - healthy_boundaries
  - constructive_communication
  - supportive_affection
  - neutral_brief
"""

import os
import json
import random
import pandas as pd

# Core multilingual curated patterns
CORE_DATA = [
    # ==================== ENGLISH SAMPLES ====================
    # Red Flag
    {
        "text": "If you actually loved me, you wouldn't talk to your friends. You only need me.",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "English",
        "severity": 0.95,
        "explanation": "Links emotional commitment to isolating from independent friendships and social support.",
        "triggers": ["if you actually loved me", "wouldn't talk to your friends", "you only need me"]
    },
    {
        "text": "After everything I sacrificed for you, this is how you repay me?",
        "flag": "Red Flag",
        "behavior": "guilt_tripping",
        "language": "English",
        "severity": 0.85,
        "explanation": "Weaponizes past favors or sacrifices to create emotional debt.",
        "triggers": ["everything i sacrificed", "how you repay me"]
    },
    {
        "text": "Look at what you made me do. You pushed me to yell at you.",
        "flag": "Red Flag",
        "behavior": "gaslighting_invalidation",
        "language": "English",
        "severity": 0.92,
        "explanation": "Refuses accountability for harmful behavior by shifting full blame onto recipient.",
        "triggers": ["what you made me do", "you pushed me"]
    },
    {
        "text": "Send me your live location right now. Why did it take you 5 minutes to reply? Who are you with?",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "English",
        "severity": 0.96,
        "explanation": "Displays excessive digital surveillance and micromanagement.",
        "triggers": ["send me your live location", "who are you with"]
    },
    {
        "text": "If you leave me, I will make sure everyone in your family knows what kind of person you are.",
        "flag": "Red Flag",
        "behavior": "threats_disrespect",
        "language": "English",
        "severity": 0.99,
        "explanation": "Direct blackmail and threat of social destruction.",
        "triggers": ["if you leave me", "ruin"]
    },
    {
        "text": "We've only known each other 3 days but you are my soulmate. Let's move in together next week.",
        "flag": "Red Flag",
        "behavior": "love_bombing",
        "language": "English",
        "severity": 0.88,
        "explanation": "Excessive intensity and rushed commitments to induce emotional dependency.",
        "triggers": ["known each other 3 days", "soulmate"]
    },

    # Yellow Flag
    {
        "text": "I really like you and have strong feelings, but I'm just not looking for any commitment or labels right now.",
        "flag": "Yellow Flag",
        "behavior": "mixed_signals",
        "language": "English",
        "severity": 0.60,
        "explanation": "Offers emotional intimacy while refusing clarity or accountability.",
        "triggers": ["not looking for any commitment", "no labels"]
    },
    {
        "text": "Fine. Do whatever you want. It's not like my opinion has ever mattered to you anyway.",
        "flag": "Yellow Flag",
        "behavior": "passive_aggression",
        "language": "English",
        "severity": 0.65,
        "explanation": "Passive-aggressive withdrawal concealing underlying resentment.",
        "triggers": ["fine do whatever", "not like my opinion mattered"]
    },
    {
        "text": "Hey sorry, went completely ghost for the past week, was just super swamped with random stuff haha.",
        "flag": "Yellow Flag",
        "behavior": "inconsistency_stonewalling",
        "language": "English",
        "severity": 0.58,
        "explanation": "Unexplained disappearing act followed by casual reappearance.",
        "triggers": ["ghost for the past week", "super swamped"]
    },

    # Green Flag
    {
        "text": "Take all the time you need to reply! I know you're super busy at work today, no stress at all.",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "English",
        "severity": 0.05,
        "explanation": "Demonstrates secure patience and respect for independent schedules.",
        "triggers": ["take all the time you need", "no stress at all"]
    },
    {
        "text": "I completely respect your boundary on this. Thank you for telling me honestly what makes you comfortable.",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "English",
        "severity": 0.02,
        "explanation": "Validates personal comfort limits with explicit gratitude.",
        "triggers": ["completely respect your boundary", "honestly what makes you comfortable"]
    },
    {
        "text": "I felt a bit hurt by what happened earlier. Can we set aside some time tonight to talk through it calmly?",
        "flag": "Green Flag",
        "behavior": "constructive_communication",
        "language": "English",
        "severity": 0.08,
        "explanation": "Direct emotional expression using non-blaming 'I' statements.",
        "triggers": ["i felt a bit hurt", "talk through it calmly"]
    },
    {
        "text": "I am so incredibly proud of you! You worked so hard for this promotion and you truly deserve it.",
        "flag": "Green Flag",
        "behavior": "supportive_affection",
        "language": "English",
        "severity": 0.01,
        "explanation": "Celebrates the partner's personal success wholeheartedly.",
        "triggers": ["proud of you", "truly deserve it"]
    },

    # Insufficient Context
    {
        "text": "Hey, what time are we meeting up tomorrow?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "English",
        "severity": 0.0,
        "explanation": "Routine logistical question without relational undertones.",
        "triggers": []
    },
    {
        "text": "Did you remember to pick up groceries on the way back?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "English",
        "severity": 0.0,
        "explanation": "Standard logistical query.",
        "triggers": []
    },

    # ==================== HINGLISH SAMPLES (Romanized Hindi) ====================
    # Red Flag Hinglish
    {
        "text": "Agar tu mujhse sach me pyaar karti na, to apne dosto se baat nahi karti. Sirf meri zarurat honi chahiye tujhe.",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Hinglish",
        "severity": 0.95,
        "explanation": "Pyaar ke naam par dosto se isolate karne aur control karne ki koshish.",
        "triggers": ["sach me pyaar karti", "dosto se baat nahi", "sirf meri zarurat"]
    },
    {
        "text": "Itna sab kuch kiya tere liye, aur tu mere liye ek weekend bhi dosto ke bina nahi nikal sakti?",
        "flag": "Red Flag",
        "behavior": "guilt_tripping",
        "language": "Hinglish",
        "severity": 0.88,
        "explanation": "Weaponizes past favors in Hinglish to create emotional guilt and pressure.",
        "triggers": ["itna sab kuch kiya", "mere liye ek weekend bhi nahi"]
    },
    {
        "text": "Apna live location bhej abhi ke abhi. 5 minute se kiske sath online thi aur mera message ignore kiya?",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Hinglish",
        "severity": 0.96,
        "explanation": "Digital surveillance and aggressive interrogation over messaging activity.",
        "triggers": ["live location bhej", "kiske sath online thi", "ignore kiya"]
    },
    {
        "text": "Usko abhi ke abhi block kar Instagram aur WhatsApp pe, warna dekh lena main kya karta hu.",
        "flag": "Red Flag",
        "behavior": "threats_disrespect",
        "language": "Hinglish",
        "severity": 0.97,
        "explanation": "Forced blocking demands backed with open threats.",
        "triggers": ["block kar abhi", "warna dekh lena"]
    },
    {
        "text": "Tu pagal ho gayi hai kya? Aisa maine kabhi nahi bola, tu hamesha apne dimag me baatein banati hai.",
        "flag": "Red Flag",
        "behavior": "gaslighting_invalidation",
        "language": "Hinglish",
        "severity": 0.94,
        "explanation": "Denying reality and labeling partner as mentally unstable (classic Hinglish gaslighting).",
        "triggers": ["tu pagal ho gayi hai", "maine kabhi nahi bola", "dimag me baatein banati hai"]
    },
    {
        "text": "Agar tune mujhe choda to main tere pure khandan aur office me teri pol khol dunga.",
        "flag": "Red Flag",
        "behavior": "threats_disrespect",
        "language": "Hinglish",
        "severity": 0.99,
        "explanation": "Extreme blackmail and social threat.",
        "triggers": ["tune mujhe choda to", "pol khol dunga", "barbaad"]
    },
    {
        "text": "Hum bas 2 din pehle mile hai par tu meri jaan hai, meri poori zindagi hai. Sab chod ke mere sath rehne aa ja.",
        "flag": "Red Flag",
        "behavior": "love_bombing",
        "language": "Hinglish",
        "severity": 0.89,
        "explanation": "Premature, hyper-intense emotional overwhelm (Love Bombing).",
        "triggers": ["2 din pehle mile", "meri jaan hai", "sath rehne aa ja"]
    },
    {
        "text": "Apne phone ka password de mujhe agar kuch chupane ke liye nahi hai to.",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Hinglish",
        "severity": 0.91,
        "explanation": "Violates digital boundaries using guilt-based transparency traps.",
        "triggers": ["phone ka password de", "kuch chupane ke liye nahi"]
    },

    # Yellow Flag Hinglish
    {
        "text": "Mujhe tu bohot achi lagti hai par main abhi kisi commitment ya serious relationship ke chakkar me nahi padna chahta.",
        "flag": "Yellow Flag",
        "behavior": "mixed_signals",
        "language": "Hinglish",
        "severity": 0.60,
        "explanation": "Emotional closeness paired with refusal of commitment.",
        "triggers": ["kisi commitment ke chakkar me nahi", "achi lagti hai par"]
    },
    {
        "text": "Acha theek hai, jo marzi karo. Waise bhi meri baat ki koi izzat ya value to hai nahi.",
        "flag": "Yellow Flag",
        "behavior": "passive_aggression",
        "language": "Hinglish",
        "severity": 0.66,
        "explanation": "Passive-aggressive guilt trip without addressing the conflict directly.",
        "triggers": ["theek hai jo marzi karo", "meri baat ki koi value nahi"]
    },
    {
        "text": "Sorry yaar 4 din se gayab tha, phone silent pe tha aur bohot kaam tha. Aaj kya plan hai?",
        "flag": "Yellow Flag",
        "behavior": "inconsistency_stonewalling",
        "language": "Hinglish",
        "severity": 0.58,
        "explanation": "Disappearing for days followed by casual reappearance.",
        "triggers": ["4 din se gayab tha", "phone silent pe tha"]
    },
    {
        "text": "Itna overthink kyu karti ho? Topic drop karo ab, baar baar wahi cheez mat kheencho.",
        "flag": "Yellow Flag",
        "behavior": "inconsistency_stonewalling",
        "language": "Hinglish",
        "severity": 0.62,
        "explanation": "Dismissing concerns and shutting down constructive dialogue.",
        "triggers": ["overthink kyu karti ho", "topic drop karo"]
    },
    {
        "text": "Mujhe nahi pata mujhe abhi kya chahiye, let's see.",
        "flag": "Yellow Flag",
        "behavior": "mixed_signals",
        "language": "Hinglish",
        "severity": 0.50,
        "explanation": "Ambiguity and lack of communication clarity.",
        "triggers": ["mujhe nahi pata", "kya chahiye"]
    },

    # Green Flag Hinglish
    {
        "text": "Aaram se reply karna, koi jaldi nahi hai! Kaam par focus karo, take your time.",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Hinglish",
        "severity": 0.04,
        "explanation": "Patience and zero pressure for texting response.",
        "triggers": ["aaram se reply karna", "koi jaldi nahi hai", "take your time"]
    },
    {
        "text": "Apne dosto ke sath khoob maza karo aaj raat! Free hoke aaram se baat karenge.",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Hinglish",
        "severity": 0.02,
        "explanation": "Encouraging independent friendships with positive energy.",
        "triggers": ["dosto ke sath khoob maza karo", "free hoke baat karenge"]
    },
    {
        "text": "Mujhe thoda bura laga jab tumne subah wo bola. Kya hum shaant hoke baat kar sakte hai is baare me?",
        "flag": "Green Flag",
        "behavior": "constructive_communication",
        "language": "Hinglish",
        "severity": 0.07,
        "explanation": "Mature conflict resolution expressing feelings calmly.",
        "triggers": ["mujhe bura laga", "shaant hoke baat kar sakte hai"]
    },
    {
        "text": "Pehle mera tone thoda rude tha, uske liye I'm really sorry. Kaam ka stress tha par mujhe tumpe nahi nikalna chahiye tha.",
        "flag": "Green Flag",
        "behavior": "constructive_communication",
        "language": "Hinglish",
        "severity": 0.03,
        "explanation": "Genuine accountability and apology for emotional reactivity.",
        "triggers": ["tone rude tha", "really sorry", "tumpe nahi nikalna chahiye tha"]
    },
    {
        "text": "Mujhe bohot proud feel ho raha hai tumpar! Tumne is interview ke liye din raat ek kiya tha.",
        "flag": "Green Flag",
        "behavior": "supportive_affection",
        "language": "Hinglish",
        "severity": 0.01,
        "explanation": "Wholehearted encouragement and celebration of partner.",
        "triggers": ["proud feel ho raha hai", "din raat ek kiya"]
    },
    {
        "text": "Main tumhari personal space ki poori respect karta hu. Jab ready ho tab baat kar lenge.",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Hinglish",
        "severity": 0.02,
        "explanation": "Respecting space and autonomy with no entitlement.",
        "triggers": ["personal space ki respect", "jab ready ho tab"]
    },

    # Insufficient Context Hinglish
    {
        "text": "Kaha ho abhi?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Hinglish",
        "severity": 0.0,
        "explanation": "Brief logistical inquiry.",
        "triggers": []
    },
    {
        "text": "Chai peene chalte hai shaam ko 5 baje.",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Hinglish",
        "severity": 0.0,
        "explanation": "Casual social plan.",
        "triggers": []
    },
    {
        "text": "Ha theek hai, kal milte hai office me.",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Hinglish",
        "severity": 0.0,
        "explanation": "Simple confirmation.",
        "triggers": []
    },
    {
        "text": "Pohach kar message karna.",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Hinglish",
        "severity": 0.0,
        "explanation": "Polite logistical check-in.",
        "triggers": []
    },

    # ==================== MARATHI SAMPLES (Roman & Devanagari) ====================
    # Red Flag Marathi
    {
        "text": "Jar tuza mazyavar khara prem asel tar tu tya lokanshi bolnar nahis. Tula fakt majhi garaj asayla havi.",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Marathi",
        "severity": 0.95,
        "explanation": "Prema chya naava khali mitran pasun dur thevnya cha prayatna.",
        "triggers": ["khara prem asel", "bolnar nahis", "majhi garaj asayla havi"]
    },
    {
        "text": "Mala ata lagech tuza live location pathav. Konasobat ahes tith mala kalala pahije.",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Marathi",
        "severity": 0.96,
        "explanation": "Digital surveillance and lack of trust in Marathi.",
        "triggers": ["live location pathav", "konasobat ahes"]
    },
    {
        "text": "Block kar tyala ata nahi tar me aapan dohamadhil sagla sampavun takto.",
        "flag": "Red Flag",
        "behavior": "threats_disrespect",
        "language": "Marathi",
        "severity": 0.97,
        "explanation": "Ultimatum and threat to end relationship.",
        "triggers": ["block kar tyala", "sagla sampavun takto"]
    },
    {
        "text": "Tu vedi ahes ka? Ashi kahi zaliach navhti, tu ugach natak kartes.",
        "flag": "Red Flag",
        "behavior": "gaslighting_invalidation",
        "language": "Marathi",
        "severity": 0.93,
        "explanation": "Invalidating reality and labeling partner as dramatic in Marathi.",
        "triggers": ["tu vedi ahes ka", "ugach natak kartes"]
    },
    {
        "text": "जर तुझं माझ्यावर खरंच प्रेम असेल तर तू मित्रांशी बोलणार नाहीस। तुला फक्त माझी गरज असायला हवी।",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Marathi",
        "severity": 0.95,
        "explanation": "Isolation and controlling behavior in Devanagari Marathi.",
        "triggers": ["प्रेम असेल", "बोलणार नाहीस", "माझी गरज"]
    },
    {
        "text": "मला आत्ताच्या आत्ता तुझा लाईव्ह लोकेशन पाठव। कोणासोबत आहेस तिथे?",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Marathi",
        "severity": 0.96,
        "explanation": "Coercive digital surveillance in Devanagari Marathi.",
        "triggers": ["लाईव्ह लोकेशन पाठव", "कोणासोबत आहेस"]
    },
    {
        "text": "तू वेडी आहेस का? असं काही झालंच नव्हतं, नेहमी मनगढ़ंत नाटक करतेस।",
        "flag": "Red Flag",
        "behavior": "gaslighting_invalidation",
        "language": "Marathi",
        "severity": 0.93,
        "explanation": "Gaslighting and invalidating reality in Devanagari Marathi.",
        "triggers": ["वेडी आहेस का", "काही झालंच नव्हतं", "नाटक करतेस"]
    },

    # Yellow Flag Marathi
    {
        "text": "Mala tu aavdtes pan mala kontyahi commitment madhe padaycha nahiye sadhya.",
        "flag": "Yellow Flag",
        "behavior": "mixed_signals",
        "language": "Marathi",
        "severity": 0.60,
        "explanation": "Affection without accountability or clarity in Marathi.",
        "triggers": ["aavdtes pan", "commitment madhe padaycha nahiye"]
    },
    {
        "text": "Theek ahe tula je karaycha te kar, maza kon aiknar ahe tasahi.",
        "flag": "Yellow Flag",
        "behavior": "passive_aggression",
        "language": "Marathi",
        "severity": 0.65,
        "explanation": "Passive-aggressive guilt-tripping in Marathi.",
        "triggers": ["je karaycha te kar", "maza kon aiknar ahe"]
    },
    {
        "text": "मला तू आवडतेस पण मला कोणत्याही कमिटमेंट मध्ये पडायचं नाही सध्या।",
        "flag": "Yellow Flag",
        "behavior": "mixed_signals",
        "language": "Marathi",
        "severity": 0.60,
        "explanation": "Mixed signals in Devanagari Marathi.",
        "triggers": ["आवडतेस पण", "कमिटमेंट मध्ये पडायचं नाही"]
    },
    {
        "text": "ठीक आहे, तुला जे करायचं ते कर। माझं मत कोणाला महत्त्वाचं वाटतं इथे।",
        "flag": "Yellow Flag",
        "behavior": "passive_aggression",
        "language": "Marathi",
        "severity": 0.65,
        "explanation": "Passive aggressive withdrawal in Devanagari Marathi.",
        "triggers": ["जे करायचं ते कर", "माझं मत"]
    },

    # Green Flag Marathi
    {
        "text": "Aaramat reply de, kahi ghai nahiye! Tuza kaam zalyavar message kar.",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Marathi",
        "severity": 0.04,
        "explanation": "Patience and healthy communication in Marathi.",
        "triggers": ["aaramat reply de", "kahi ghai nahiye"]
    },
    {
        "text": "Mitranchya sobat changla enjoy kar! Nantar boluya aapan aaramat.",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Marathi",
        "severity": 0.02,
        "explanation": "Encouraging friendships in Marathi.",
        "triggers": ["changla enjoy kar", "nantar boluya"]
    },
    {
        "text": "Mala khup abhiman ahe tuza, khup mehnat kelis ahes ya sathi!",
        "flag": "Green Flag",
        "behavior": "supportive_affection",
        "language": "Marathi",
        "severity": 0.01,
        "explanation": "Deep supportive encouragement in Marathi.",
        "triggers": ["khup abhiman ahe", "khup mehnat kelis"]
    },
    {
        "text": "Aapan shantpane basun boluyat ka? Mala thoda vait vatla tya goshticha.",
        "flag": "Green Flag",
        "behavior": "constructive_communication",
        "language": "Marathi",
        "severity": 0.06,
        "explanation": "Constructive conflict resolution in Marathi.",
        "triggers": ["shantpane basun boluyat", "vait vatla"]
    },
    {
        "text": "आरामात उत्तर दे, काही घाई नाही! कामावर लक्ष दे, नंतर बोलू।",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Marathi",
        "severity": 0.03,
        "explanation": "Respecting time and focus in Devanagari Marathi.",
        "triggers": ["आरामात उत्तर दे", "काही घाई नाही"]
    },
    {
        "text": "मित्रांसोबत छान मज्जा कर! नंतर सांग मला कसा होता दिवस।",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Marathi",
        "severity": 0.02,
        "explanation": "Supporting independent social life in Devanagari Marathi.",
        "triggers": ["छान मज्जा कर", "नंतर सांग"]
    },
    {
        "text": "मला खूप अभिमान आहे तुझा! खूप मेहनत केली आहेस तू या यशासाठी।",
        "flag": "Green Flag",
        "behavior": "supportive_affection",
        "language": "Marathi",
        "severity": 0.01,
        "explanation": "Pride and genuine love in Devanagari Marathi.",
        "triggers": ["खूप अभिमान आहे", "खूप मेहनत"]
    },

    # Insufficient Context Marathi
    {
        "text": "Kasa ahes?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Marathi",
        "severity": 0.0,
        "explanation": "Casual greeting in Marathi.",
        "triggers": []
    },
    {
        "text": "Kiti vajta bhetaycha udya?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Marathi",
        "severity": 0.0,
        "explanation": "Scheduling inquiry in Marathi.",
        "triggers": []
    },
    {
        "text": "कसा आहेस?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Marathi",
        "severity": 0.0,
        "explanation": "Simple greeting in Devanagari Marathi.",
        "triggers": []
    },
    {
        "text": "उद्या किती वाजता भेटूया?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Marathi",
        "severity": 0.0,
        "explanation": "Meeting logistics in Devanagari Marathi.",
        "triggers": []
    },

    # ==================== HINDI (Devanagari) SAMPLES ====================
    # Red Flag Hindi
    {
        "text": "अगर तुम सच में मुझसे प्यार करते तो अपने दोस्तों से बात नहीं करते। तुम्हें सिर्फ मेरी जरूरत होनी चाहिए।",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Hindi",
        "severity": 0.95,
        "explanation": "Emotional isolation and possessiveness in Hindi.",
        "triggers": ["सच में मुझसे प्यार करते", "दोस्तों से बात नहीं", "सिर्फ मेरी जरूरत"]
    },
    {
        "text": "मुझे अभी अपनी लाइव लोकेशन भेजो। 5 मिनट से किसके साथ ऑनलाइन थे और मेरा रिप्लाई नहीं किया?",
        "flag": "Red Flag",
        "behavior": "controlling_isolation",
        "language": "Hindi",
        "severity": 0.96,
        "explanation": "Controlling surveillance in Hindi.",
        "triggers": ["लाइव लोकेशन भेजो", "किसके साथ ऑनलाइन थे"]
    },
    {
        "text": "तुम पागल हो। अपनी मनगढ़ंत बातें मत बनाओ, ऐसा कभी नहीं हुआ था।",
        "flag": "Red Flag",
        "behavior": "gaslighting_invalidation",
        "language": "Hindi",
        "severity": 0.93,
        "explanation": "Reality invalidation and gaslighting in Hindi.",
        "triggers": ["तुम पागल हो", "मनगढ़ंत बातें मत बनाओ", "ऐसा कभी नहीं हुआ"]
    },
    {
        "text": "अगर तुमने मुझे छोड़ा तो मैं तुम्हारा जीना हराम कर दूंगा, याद रखना।",
        "flag": "Red Flag",
        "behavior": "threats_disrespect",
        "language": "Hindi",
        "severity": 0.99,
        "explanation": "Hostility and violent threats in Hindi.",
        "triggers": ["मुझे छोड़ा तो", "जीना हराम कर दूंगा"]
    },

    # Yellow Flag Hindi
    {
        "text": "मुझे तुम बहुत पसंद हो लेकिन मैं अभी किसी रिश्ते या कमिटमेंट के लिए तैयार नहीं हूँ।",
        "flag": "Yellow Flag",
        "behavior": "mixed_signals",
        "language": "Hindi",
        "severity": 0.60,
        "explanation": "Mixed signals in Hindi.",
        "triggers": ["पसंद हो लेकिन", "कमिटमेंट के लिए तैयार नहीं"]
    },
    {
        "text": "ठीक है, जो करना है करो। वैसे भी मेरी बात की कोई कद्र नहीं है।",
        "flag": "Yellow Flag",
        "behavior": "passive_aggression",
        "language": "Hindi",
        "severity": 0.65,
        "explanation": "Passive-aggressive guilt trip in Hindi.",
        "triggers": ["जो करना है करो", "मेरी बात की कोई कद्र नहीं"]
    },
    {
        "text": "चार दिन से बहुत व्यस्त था इसलिए बात नहीं कर पाया। अब इस बात का बतंगड़ मत बनाओ।",
        "flag": "Yellow Flag",
        "behavior": "inconsistency_stonewalling",
        "language": "Hindi",
        "severity": 0.62,
        "explanation": "Avoidance and stonewalling in Hindi.",
        "triggers": ["चार दिन से बहुत व्यस्त", "बतंगड़ मत बनाओ"]
    },

    # Green Flag Hindi
    {
        "text": "आराम से जवाब देना, कोई जल्दी नहीं है। अपने काम पर ध्यान दो।",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Hindi",
        "severity": 0.03,
        "explanation": "Healthy boundaries in Hindi.",
        "triggers": ["आराम से जवाब देना", "कोई जल्दी नहीं"]
    },
    {
        "text": "अपने दोस्तों के साथ पूरा आनंद लो! बाद में आराम से बात करेंगे।",
        "flag": "Green Flag",
        "behavior": "healthy_boundaries",
        "language": "Hindi",
        "severity": 0.02,
        "explanation": "Supportive boundaries in Hindi.",
        "triggers": ["दोस्तों के साथ आनंद", "आराम से बात करेंगे"]
    },
    {
        "text": "मुझे बहुत गर्व है तुम पर! तुमने इसके लिए बहुत मेहनत की है।",
        "flag": "Green Flag",
        "behavior": "supportive_affection",
        "language": "Hindi",
        "severity": 0.01,
        "explanation": "Genuine celebration and praise in Hindi.",
        "triggers": ["गर्व है तुम पर", "बहुत मेहनत की"]
    },
    {
        "text": "मुझे बुरा लगा जब आपने ऐसा कहा। क्या हम बैठकर शांति से बात कर सकते हैं?",
        "flag": "Green Flag",
        "behavior": "constructive_communication",
        "language": "Hindi",
        "severity": 0.06,
        "explanation": "Constructive conflict resolution in Hindi.",
        "triggers": ["मुझे बुरा लगा", "शांति से बात कर सकते हैं"]
    },

    # Insufficient Context Hindi
    {
        "text": "नमस्ते, कैसे हो आप?",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Hindi",
        "severity": 0.0,
        "explanation": "Standard polite greeting.",
        "triggers": []
    },
    {
        "text": "कल 4 बजे मिलते हैं।",
        "flag": "Insufficient Context",
        "behavior": "neutral_brief",
        "language": "Hindi",
        "severity": 0.0,
        "explanation": "Meeting logistics in Hindi.",
        "triggers": []
    }
]

# Systematic Multilingual Augmentation Templates
MULTILINGUAL_TEMPLATES = {
    # English Templates
    ("Red Flag", "controlling_isolation", "English"): [
        "Why do you keep talking to {person}? I told you I don't trust them.",
        "You shouldn't go to {event} without me. It's weird that you want to go alone.",
        "Who is {person} and why did they comment on your photo? Delete them right now.",
        "I need you to show me your chat logs with {person}. If you have nothing to hide, prove it.",
        "Stop wearing that outfit outside. You only need to look good for me.",
        "Answer my calls right now. You've been online for 10 minutes and ignoring me."
    ],
    ("Red Flag", "guilt_tripping", "English"): [
        "I guess I'm just a burden to you. Don't worry about me, I'll deal with my pain alone.",
        "If you truly cared about us, you wouldn't prioritize {activity} over me.",
        "Remember when I helped you through {event}? And now you can't even do this simple thing for me?",
        "You're making me so depressed by acting like this. Is that what you want?"
    ],
    ("Red Flag", "gaslighting_invalidation", "English"): [
        "You're blowing this completely out of proportion as usual. You love drama.",
        "I never promised that. You must be confusing me with your ex or making things up.",
        "You have serious trust issues and you're projecting all your insecurities onto me."
    ],
    ("Yellow Flag", "mixed_signals", "English"): [
        "I really enjoy our dates, but I don't want to rush into putting any title or exclusivity on this.",
        "One day you're texting me nonstop and the next day you barely reply. I don't get where we stand.",
        "I have feelings for you, but I'm still figuring myself out so I can't guarantee anything."
    ],
    ("Yellow Flag", "passive_aggression", "English"): [
        "Must be nice being able to ignore people whenever you feel like it.",
        "Cool. Glad to know where your priorities lie.",
        "Don't worry about it. I shouldn't have expected you to remember anyway."
    ],
    ("Green Flag", "healthy_boundaries", "English"): [
        "No worries at all! Focus on {activity} and text me whenever you're free.",
        "I understand you need some quiet time to yourself. Take care and I'm here when you want to chat.",
        "Thank you for being open about your boundaries. I really value that we can be direct with each other.",
        "Have fun at {event}! Enjoy every minute of it."
    ],
    ("Green Flag", "constructive_communication", "English"): [
        "I want to make sure I understand how you're feeling. Can you help me see where you're coming from?",
        "When that happened, I felt a little anxious. Could we find a way to handle that together next time?",
        "I hear what you're saying and your point makes total sense. Let's find a compromise."
    ],
    ("Insufficient Context", "neutral_brief", "English"): [
        "What time are we meeting?",
        "Are you at the office or working from home today?",
        "See you at the station around 5.",
        "What do you want for dinner tonight?",
        "Sounds like a plan."
    ],

    # Hinglish Templates
    ("Red Flag", "controlling_isolation", "Hinglish"): [
        "Tu {person} se kyu baat karti hai? Maine bola na mujhe wo pasand nahi.",
        "Apna phone de mujhe abhi, dekhna hai {person} se kya chat chal rahi hai.",
        "Tu {event} pe mere bina nahi jayegi, samajh gayi na?",
        "Instagram se {person} ko unfollow kar abhi ke abhi.",
        "Online hoke bhi mera reply nahi de rahi, kisse baat chal rahi hai?",
        "Usko block kar de nahi to main baat nahi karunga."
    ],
    ("Red Flag", "guilt_tripping", "Hinglish"): [
        "Maine tere liye itna sab sacrifice kiya aur tu mere liye {activity} nahi chodh sakti?",
        "Main hi pagal hu jo itna pyaar karta hu, meri to koi parwah hi nahi tujhe.",
        "Theek hai jao maze karo, main yaha akela tadapta rahunga.",
        "Agar zara sa bhi pyaar hota na to aise behave nahi karti.",
        "Tujhe to meri fikar hi nahi hai, bas apna sochte ho."
    ],
    ("Red Flag", "gaslighting_invalidation", "Hinglish"): [
        "Tu hamesha baaton ko ghuma fira ke mujhe dosh deti hai. Drama queen ban gayi hai.",
        "Aisa kabhi nahi hua tha, tu sab imagine kar rahi hai apne dimag me.",
        "Itni oversensitive kyu hai tu? Kisi aur ke sath hoti to koi jhel bhi nahi pata tujhe."
    ],
    ("Yellow Flag", "mixed_signals", "Hinglish"): [
        "Mujhe tu bohot pasand hai, par abhi relationship ka koi tag ya label nahi chahiye.",
        "Let's just go with the flow, commitments se mujhe anxiety hoti hai.",
        "Main guarantee nahi de sakta future ki, abhi bas vibe karte hai.",
        "Mujhe nahi pata mujhe kya chahiye abhi."
    ],
    ("Yellow Flag", "passive_aggression", "Hinglish"): [
        "Badiya hai, dosto ke liye time hai par mere liye 2 minute nahi.",
        "Theek hai, ab se main kuch bolunga hi nahi.",
        "K. Samajh gaya meri value.",
        "Mujhe to aadat ho gayi hai ignore hone ki, koi nayi baat nahi hai."
    ],
    ("Green Flag", "healthy_boundaries", "Hinglish"): [
        "Aaram se reply karo, koi ghai nahi hai! Apna {activity} enjoy karo.",
        "I totally respect your space. Jab bhi free ho call kar lena.",
        "{event} pe bohot enjoy karna! Photos zaroor bhejna.",
        "Mujhe samajh aata hai ki tumhe akela time chahiye, take your time!",
        "Koi tension mat lo, aaram se time nikal kar baat karenge."
    ],
    ("Green Flag", "constructive_communication", "Hinglish"): [
        "Jab tumne aisa bola to mujhe thoda bura laga. Chalo shaant hoke samajhte hai.",
        "Mujhe tumhari baat samajh aa rahi hai, aao koi dono ke hisaab se solution nikalte hai.",
        "Mera gussa galat tha subah, I am sorry. Mujhe aise react nahi karna chahiye tha."
    ],
    ("Insufficient Context", "neutral_brief", "Hinglish"): [
        "Kitne baje chalna hai?",
        "Khana kha liya?",
        "Traffic bohot hai aaj road pe.",
        "Kal subah office me milte hai.",
        "Ha theek hai bhai.",
        "Chalo theek hai."
    ],

    # Hindi (Devanagari) Templates
    ("Red Flag", "controlling_isolation", "Hindi"): [
        "तुम उस व्यक्ति से बात क्यों कर रहे हो? मुझे वह बिल्कुल पसंद नहीं है।",
        "मुझे अभी अपनी लाइव लोकेशन भेजो। 10 मिनट से किसका रिप्लाई कर रहे थे?",
        "तुम मेरे बिना वहां नहीं जाओगे, समझ गए ना?",
        "सोशल मीडिया से उस फोटो को अभी हटाओ।"
    ],
    ("Red Flag", "guilt_tripping", "Hindi"): [
        "मैंने तुम्हारे लिए इतना सब कुछ छोड़ दिया और तुम मेरे लिए इतना भी नहीं कर सकते?",
        "मैं ही बेवकूफ हूँ जो इतना प्यार करता हूँ, तुम्हें तो मेरी कोई परवाह ही नहीं है।"
    ],
    ("Yellow Flag", "passive_aggression", "Hindi"): [
        "बहुत बढ़िया, सबके लिए समय है बस मेरे लिए दो मिनट नहीं हैं।",
        "ठीक है, अब से मैं कुछ कहूंगा ही नहीं। तुम्हारी जो मर्जी हो करो।"
    ],
    ("Green Flag", "healthy_boundaries", "Hindi"): [
        "आराम से जवाब देना, कोई जल्दी नहीं है! अपने काम पर ध्यान दो।",
        "मैं तुम्हारी व्यक्तिगत सीमाओं का पूरा सम्मान करता हूँ। जब समय मिले तब बात करना।",
        "अपने दोस्तों के साथ बहुत मजे करो! बाद में बताना कैसा रहा।"
    ],
    ("Insufficient Context", "neutral_brief", "Hindi"): [
        "कल कितने बजे मिलना है?",
        "खाना खा लिया आपने?",
        "नमस्ते, सब कैसा चल रहा है?",
        "हाँ बिल्कुल ठीक है।"
    ],

    # Marathi Templates
    ("Red Flag", "controlling_isolation", "Marathi"): [
        "Tu {person} shi kashala boltes? Mala te aavadat nahi sangitla hota na.",
        "Tula {event} la majhya shivay jaychi garaj nahiye.",
        "Mala tuza mobile passcode de jar tu kahi lapavat nasashil tar.",
        "Aattach {person} la block kar nahi tar bagh mag."
    ],
    ("Red Flag", "guilt_tripping", "Marathi"): [
        "Maza premach kami padla asel, mhanun tu ashi vagtes.",
        "Tula majhi kaddar nahiye, kiti goshti sodlya me tujhyasathi.",
        "Theek ahe me marun gelo tari tula kahi farak padnar nahi."
    ],
    ("Yellow Flag", "passive_aggression", "Marathi"): [
        "Chhan ahe, saglyansathi vel ahe fakt majhyasathi nahi.",
        "Theek ahe, nantar boluya jar tula vel asel tar.",
        "Maza kon aaiknar ahe ithe, me ugach bolat asto."
    ],
    ("Green Flag", "healthy_boundaries", "Marathi"): [
        "Aaramat boluya kahi ghai nahiye, tuza {activity} purna kar.",
        "Mala kalata tula space havi ahe, kalaji ghe!",
        "{event} madhe khup maja kar mitran sobat!"
    ],
    ("Insufficient Context", "neutral_brief", "Marathi"): [
        "Kiti vajle?",
        "Jevlis ka?",
        "Udya bhetuya nantar.",
        "Ha theek ahe."
    ]
}

PERSONS_EN = ["your ex", "Alex", "Jordan", "your coworker", "your friend Sam"]
PERSONS_HI = ["apne ex", "apne dost", "apne colleague", "us ladke", "us ladki"]
PERSONS_MR = ["tujhya ex", "tujhya mitran", "office madhlya mulashi", "tya mulashi"]

EVENTS_EN = ["the party", "the dinner", "the weekend trip", "the meetup"]
EVENTS_HI = ["party", "dinner", "weekend trip", "dosto ke meetup"]
EVENTS_MR = ["party", "dinner", "trip", "bhet"]

ACTIVITIES_EN = ["your hobbies", "your friends", "staying late at work"]
ACTIVITIES_HI = ["apne dosto", "apne kaam", "apni hobbies"]
ACTIVITIES_MR = ["tujhya mitran", "tujhya kama", "tujhya vel"]

def generate_multilingual_dataset(target_size=1300, seed=42):
    random.seed(seed)
    records = list(CORE_DATA)
    
    while len(records) < target_size:
        for (flag, behavior, lang), templates in MULTILINGUAL_TEMPLATES.items():
            template = random.choice(templates)
            
            if lang == "Hinglish":
                p = random.choice(PERSONS_HI)
                e = random.choice(EVENTS_HI)
                a = random.choice(ACTIVITIES_HI)
            elif lang == "Marathi":
                p = random.choice(PERSONS_MR)
                e = random.choice(EVENTS_MR)
                a = random.choice(ACTIVITIES_MR)
            else:
                p = random.choice(PERSONS_EN)
                e = random.choice(EVENTS_EN)
                a = random.choice(ACTIVITIES_EN)
                
            text = template.format(person=p, event=e, activity=a)
            
            severity = 0.0
            if flag == "Red Flag":
                severity = round(random.uniform(0.82, 0.99), 2)
            elif flag == "Yellow Flag":
                severity = round(random.uniform(0.48, 0.72), 2)
            elif flag == "Green Flag":
                severity = round(random.uniform(0.01, 0.12), 2)
            else:
                severity = 0.0
                
            records.append({
                "text": text,
                "flag": flag,
                "behavior": behavior,
                "language": lang,
                "severity": severity,
                "explanation": f"Multilingual behavioral indicator for {behavior} ({lang}).",
                "triggers": [w for w in text.split() if len(w) > 4][:2]
            })
            
            if len(records) >= target_size:
                break
                
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

def save_dataset(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    df = generate_multilingual_dataset(target_size=1300)
    
    csv_path = os.path.join(output_dir, "redflag_dataset.csv")
    json_path = os.path.join(output_dir, "redflag_dataset.json")
    
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    
    print(f"Multilingual Dataset generated successfully! Total rows: {len(df)}")
    print("\nClass Distribution (Flags):")
    print(df['flag'].value_counts())
    print("\nLanguage Distribution:")
    print(df['language'].value_counts())
    print("\nBehavior Distribution:")
    print(df['behavior'].value_counts())
    return csv_path, json_path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    save_dataset(data_dir)
