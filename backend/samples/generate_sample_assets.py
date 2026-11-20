"""
REDFLAG AI - Sample Generator
Generates sample chat exports (.txt) and realistic screenshot images (.png)
for testing the multimodal input pipeline.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def generate_text_samples(samples_dir: str):
    os.makedirs(samples_dir, exist_ok=True)
    
    # 1. Red Flag Chat Export
    red_chat = """[10:14 PM, 14/05/2026] Alex: Hey, where are you right now?
[10:16 PM, 14/05/2026] Jordan: I'm just out having dinner with Sarah and the team!
[10:17 PM, 14/05/2026] Alex: You didn't tell me you were seeing Sarah tonight.
[10:18 PM, 14/05/2026] Jordan: It was a last minute plan after work, sorry!
[10:19 PM, 14/05/2026] Alex: If you actually loved me, you wouldn't talk to your friends all the time. You only need me.
[10:20 PM, 14/05/2026] Alex: Send me your live location right now so I know who is there.
[10:22 PM, 14/05/2026] Alex: After everything I sacrificed for you, this is how you repay me?
[10:23 PM, 14/05/2026] Alex: If you don't leave right now and call me, we are completely finished.
"""
    with open(os.path.join(samples_dir, "red_flag_controlling_chat.txt"), "w", encoding="utf-8") as f:
        f.write(red_chat.strip())
        
    # 2. Yellow Flag Chat Export
    yellow_chat = """[02:15 PM, 01/06/2026] Sam: Hey! Haven't heard from you in a few days, hope your week is going well!
[07:45 PM, 05/06/2026] Taylor: Hey sorry, went completely ghost for the past week, was just super swamped with random stuff haha. What are you doing tonight?
[08:10 PM, 05/06/2026] Sam: I'm free around 8:30! Did you want to grab some dinner or drinks?
[09:20 PM, 05/06/2026] Taylor: Actually super drained now. Let's just play it by ear this weekend. I really like you but I don't want any labels or serious pressure right now.
[09:22 PM, 05/06/2026] Sam: That's a bit confusing, earlier you said you wanted to date intentionally.
[09:25 PM, 05/06/2026] Taylor: Fine. Do whatever you want. It's not like my opinion matters anyway.
"""
    with open(os.path.join(samples_dir, "yellow_flag_mixed_signals.txt"), "w", encoding="utf-8") as f:
        f.write(yellow_chat.strip())

    # 3. Green Flag Chat Export
    green_chat = """[09:30 AM, 12/06/2026] Morgan: Good morning! Just wanted to wish you the absolute best on your big interview today!
[09:32 AM, 12/06/2026] Riley: Thank you so much! Honestly feeling a bit nervous but ready.
[09:35 AM, 12/06/2026] Morgan: You're going to crush it! You've prepared so diligently and you truly deserve this.
[05:15 PM, 12/06/2026] Riley: Hey Morgan! Finished the interview! I'm completely wiped out and need to decompress alone tonight if that's okay.
[05:17 PM, 12/06/2026] Morgan: Take all the time you need to recharge! No pressure at all to hang out or text back. Proud of you!
"""
    with open(os.path.join(samples_dir, "green_flag_healthy_chat.txt"), "w", encoding="utf-8") as f:
        f.write(green_chat.strip())

    # 4. Neutral Logistical Chat
    neutral_chat = """[11:00 AM] Chris: Did you get the email about the apartment lease?
[11:05 AM] Pat: Yes, I downloaded the PDF.
[11:06 AM] Chris: Great, what time should we meet the landlord tomorrow?
[11:07 AM] Pat: Sounds like 4:30 PM works best.
[11:08 AM] Chris: Sounds good to me, see you there!
"""
    with open(os.path.join(samples_dir, "neutral_logistics_chat.txt"), "w", encoding="utf-8") as f:
        f.write(neutral_chat.strip())

def generate_sample_images(samples_dir: str):
    """Creates mock chat screenshot images for OCR testing."""
    os.makedirs(samples_dir, exist_ok=True)
    
    scenarios = [
        ("screenshot_red_flag.png", [
            ("Them", "If you actually loved me, you wouldn't talk to your friends."),
            ("Them", "You only need me. Delete your Instagram right now."),
            ("Them", "Send me your live location or we are finished.")
        ], "#EF4444"),
        ("screenshot_green_flag.png", [
            ("Them", "Take all the time you need to reply!"),
            ("Them", "I completely respect your personal space."),
            ("Them", "Have an amazing time with your friends tonight!")
        ], "#10B981")
    ]
    
    for filename, dialogues, accent in scenarios:
        img = Image.new("RGB", (720, 900), color="#0F172A")
        draw = ImageDraw.Draw(img)
        
        # Header bar
        draw.rectangle([(0, 0), (720, 110)], fill="#1E293B")
        draw.text((40, 45), "< Back       Messages       Details", fill="#94A3B8")
        
        y_offset = 150
        for sender, text in dialogues:
            # Draw bubble
            bubble_top = y_offset
            bubble_bottom = y_offset + 90
            draw.rounded_rectangle([(40, bubble_top), (680, bubble_bottom)], radius=18, fill="#1E293B", outline=accent, width=2)
            draw.text((65, bubble_top + 15), sender, fill=accent)
            draw.text((65, bubble_top + 45), text, fill="#F8FAFC")
            y_offset += 120
            
        img_path = os.path.join(samples_dir, filename)
        img.save(img_path)

if __name__ == "__main__":
    curr = os.path.dirname(os.path.abspath(__file__))
    generate_text_samples(curr)
    generate_sample_images(curr)
    print("Sample chat exports and screenshot images created successfully!")

