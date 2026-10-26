# Gesprächsantworten von Luna (Fallback auf FAST_MODEL, falls nicht installiert)
CHAT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
# Strukturierte Analyse: Memory-Erkennung, Brain-Kandidaten, JSON-Extraktion
FAST_MODEL = "qwen3:1.7b"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_NUM_PREDICT = 100
LANGUAGE = "de"

TTS_ENABLED = True
TTS_ENGINE = "edge"
TTS_VOICE = "de-DE-SeraphinaMultilingualNeural"
TTS_FALLBACK_VOICE = "de-DE-KatjaNeural"
TTS_RATE = "+0%"
TTS_PITCH = "+0Hz"
TTS_VOLUME = "+0%"

# pyttsx3-Fallback
TTS_FALLBACK_RATE = 175
TTS_FALLBACK_VOLUME = 1.0

SAFETY_ENABLED = True
SAFETY_STRICT_MODE = True

# Zeigt beim Auto-Memory Debug-Zeilen im Terminal
MEMORY_DEBUG = False

# Zeigt beim Brain-Learning Debug-Zeilen im Terminal
BRAIN_DEBUG = False

# Zeigt "[AI] duplicate response -> retry", wenn Luna eine Antwort neu erzeugt
AI_DEBUG = False

# Zeigt STT-Namenskorrekturen: [STT RAW] ... / [STT FIX] ...
STT_DEBUG = False  # [STT ENGINE] / [STT RAW] / [STT FIX]

# Zeigt den Luna-State (Stimmung, Energie ...) nach jedem normalen Gespräch
LUNA_STATE_DEBUG = False

# Optionaler offizieller Steam-Web-API-Key (leer = nur öffentliche Store-Endpunkte).
# Wird nie ausgegeben oder geloggt.
STEAM_WEB_API_KEY = ""

VOICE_INPUT_ENABLED = True
STT_LANGUAGE = "de-DE"
# Primäre Engine: "whisper" (lokal, faster-whisper) oder "google"
STT_ENGINE = "whisper"
STT_WHISPER_MODEL = "small"
STT_WHISPER_LANGUAGE = "de"
STT_WHISPER_BEAM_SIZE = 5
# Fällt Whisper aus (nicht installiert, Modell nicht ladbar ...), wird diese Engine benutzt
STT_FALLBACK_ENGINE = "google"
STT_TIMEOUT = 10
STT_PHRASE_TIME_LIMIT = 15
STT_SAMPLE_RATE = 16000
STT_SILENCE_DURATION = 1.0