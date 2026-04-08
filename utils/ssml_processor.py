"""
SSML Post-Processor — Converts plain LLM text into expressive SSML
for human-like voice output via Sarvam TTS.

Layers:
  1. Clean (strip markdown/unsafe chars)
  2. Detect emotion (trigger words → prosody profile)
  3. Inject pauses (punctuation → SSML break tags)
  4. Inject emphasis (key words → SSML emphasis tags)
  5. Number pronunciation (digits → say-as cardinal)
  6. Wrap in prosody + <speak> envelope
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Emotion profiles ──────────────────────────────────────────
EMOTION_PROFILES = {
    "happy": {
        "rate": "105%", "pitch": "+3st", "volume": "loud",
        "trigger_words": [
            "perfect", "great", "bilkul", "wonderful",
            "amazing", "badhiya", "fantastic", "shabash"
        ]
    },
    "apologetic": {
        "rate": "88%", "pitch": "-1st", "volume": "soft",
        "trigger_words": [
            "sorry", "maafi", "unfortunately", "issue",
            "thoda", "nahi", "problem", "apologies"
        ]
    },
    "greeting": {
        "rate": "90%", "pitch": "+2st", "volume": "medium",
        "trigger_words": [
            "namaste", "welcome", "swagat", "hello",
            "good morning", "good evening", "namaskar"
        ]
    },
    "confirmatory": {
        "rate": "95%", "pitch": "+1st", "volume": "medium",
        "trigger_words": [
            "done", "noted", "added", "confirmed",
            "ho gaya", "kar diya", "noted ji", "placed"
        ]
    },
    "urgent": {
        "rate": "100%", "pitch": "+2st", "volume": "loud",
        "trigger_words": [
            "please", "zaroor", "important",
            "dhyan", "alert", "sirf", "limited"
        ]
    },
    "calm": {
        "rate": "93%", "pitch": "0st", "volume": "medium",
        "trigger_words": []  # default fallback
    }
}

# ── Words to emphasize ─────────────────────────────────────────
EMPHASIS_WORDS = [
    # confirmations
    "done", "confirmed", "added", "removed", "placed",
    "ho gaya", "kar diya", "bilkul", "perfect", "noted",
    # food-order specific
    "total", "rupees", "order", "cart", "cancel",
    # greetings
    "namaste", "welcome", "swagat", "shukriya", "thank you",
    # apologies
    "sorry", "maafi",
]


def detect_emotion(text: str) -> str:
    """Detects dominant emotion from text based on trigger words."""
    text_lower = text.lower()
    scores = {emotion: 0 for emotion in EMOTION_PROFILES}

    for emotion, profile in EMOTION_PROFILES.items():
        for word in profile["trigger_words"]:
            if word in text_lower:
                scores[emotion] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "calm"


def inject_pauses(text: str) -> str:
    """Replaces ellipsis and em-dashes with commas/periods for natural TTS pacing."""
    text = re.sub(r'\.\.\.\s*', ', ', text)
    text = re.sub(r'\u2014\s*', ', ', text)
    text = re.sub(r'—\s*', ', ', text)
    return text

def inject_emphasis(text: str) -> str:
    """No-op: SSML emphasis removed since TTS reads it aloud."""
    return text

def inject_number_pronunciation(text: str) -> str:
    """Ensures prices read clearly without SSML tags."""
    # Prices like ₹340 -> 340 rupees
    text = re.sub(
        r'₹\s*(\d+)',
        r'\1 rupees',
        text
    )
    return text

def clean_text(text: str) -> str:
    """Strips markdown and unsafe characters."""
    text = re.sub(r'\*+', '', text)        # bold/italic
    text = re.sub(r'#+\s*', '', text)      # headers
    text = re.sub(r'`+', '', text)         # code ticks
    text = re.sub(r'\n+', ' ', text)       # flatten newlines
    text = re.sub(r'\s{2,}', ' ', text)    # extra spaces
    text = text.replace('"', '')           # remove quotes
    return text.strip()

def build_ssml(text: str) -> tuple:
    """
    Master function — cleans raw LLM text output for human-like Voice output.
    Returns (clean_string, emotion) WITHOUT SSML tags.
    """
    # Step 1 — Clean
    text = clean_text(text)

    # Step 2 — Detect emotion for logging
    emotion = detect_emotion(text)
    profile = EMOTION_PROFILES[emotion]

    # Step 3 — Tweak punctuation for pauses
    text = inject_pauses(text)

    # Step 4 — No Emphasis (tags removed)
    text = inject_emphasis(text)

    # Step 5 — Fix number pronunciation textually (no tags)
    text = inject_number_pronunciation(text)

    # Step 6 — Return plain text instead of prosody wrap string
    logger.info(f"[Text Processor] Emotion: {emotion} | Text output ready")
    
    return text, emotion
