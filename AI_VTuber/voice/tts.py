import asyncio
import gc
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

from config.settings import (
    TTS_ENABLED,
    TTS_ENGINE,
    TTS_FALLBACK_RATE,
    TTS_FALLBACK_VOICE,
    TTS_FALLBACK_VOLUME,
    TTS_PITCH,
    TTS_RATE,
    TTS_VOICE,
    TTS_VOLUME,
)

FEMALE_VOICE_HINTS = ("katja", "hedda", "zira", "anna", "female")
KEYCAP_EMOJI = re.compile(r"[0-9#*]\uFE0F?\u20E3")
ACTION_AT_START = re.compile(r"(?m)^[ \t]*(\*{1,2})[^*\n]+\1(?=\s|$)[ \t]*")
ACTION_AT_END = re.compile(r"(?m)[ \t]*(\*{1,2})[^*\n]+\1[ \t]*$")


PRONUNCIATION_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "pronunciation.json"
)


class TTS:
    def __init__(self):
        self.enabled = TTS_ENABLED
        self.pronunciation = self._load_pronunciation()
        self._pronunciation_pattern = self._build_pattern(self.pronunciation)

    @staticmethod
    def _load_pronunciation(path=PRONUNCIATION_PATH):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            key.strip().casefold(): value
            for key, value in data.items()
            if isinstance(key, str) and isinstance(value, str) and key.strip()
        }

    @staticmethod
    def _build_pattern(mapping):
        if not mapping:
            return None
        # Längste Phrasen zuerst, damit "Miki Akuma" vor "Miki" greift
        keys = sorted(mapping, key=len, reverse=True)
        alternatives = "|".join(re.escape(key) for key in keys)
        return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)

    def apply_pronunciation(self, text):
        if not self._pronunciation_pattern:
            return text
        return self._pronunciation_pattern.sub(
            lambda match: self.pronunciation.get(match.group(0).casefold(), match.group(0)),
            text,
        )

    @staticmethod
    def _play_mp3(path):
        import ctypes

        winmm = ctypes.windll.winmm
        alias = f"luna_tts_{os.getpid()}"

        def mci(command):
            error = winmm.mciSendStringW(command, None, 0, None)
            if error:
                raise RuntimeError(f"Audiowiedergabe fehlgeschlagen (MCI {error}).")

        mci(f'open "{path}" type mpegvideo alias {alias}')
        try:
            mci(f"play {alias} wait")
        finally:
            winmm.mciSendStringW(f"close {alias}", None, 0, None)

    @staticmethod
    async def _synthesize(text, voice, path):
        import edge_tts

        communicate = edge_tts.Communicate(
            text, voice, rate=TTS_RATE, pitch=TTS_PITCH, volume=TTS_VOLUME
        )
        await asyncio.wait_for(communicate.save(path), timeout=30)

    @classmethod
    def _speak_edge_voice(cls, text, voice):
        descriptor, path = tempfile.mkstemp(prefix="luna_tts_", suffix=".mp3")
        os.close(descriptor)
        try:
            asyncio.run(cls._synthesize(text, voice, path))
            if os.path.getsize(path) == 0:
                raise RuntimeError("edge-tts lieferte kein Audio.")
            cls._play_mp3(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @classmethod
    def _speak_edge(cls, text):
        voices = [TTS_VOICE]
        if TTS_FALLBACK_VOICE and TTS_FALLBACK_VOICE != TTS_VOICE:
            voices.append(TTS_FALLBACK_VOICE)
        last_error = None
        for voice in voices:
            try:
                cls._speak_edge_voice(text, voice)
                return
            except Exception as error:
                last_error = error
        raise last_error

    @classmethod
    def _speak_once(cls, text):
        if TTS_ENGINE == "edge":
            try:
                cls._speak_edge(text)
                return
            except Exception as error:
                print(f"Warnung: Neural-TTS nicht verfügbar ({type(error).__name__}). Nutze pyttsx3.")
        cls._speak_pyttsx3(text)

    @staticmethod
    def _select_female_voice(engine):
        for voice in engine.getProperty("voices"):
            name = (getattr(voice, "name", "") or "").casefold()
            if any(hint in name for hint in FEMALE_VOICE_HINTS):
                engine.setProperty("voice", voice.id)
                return

    @classmethod
    def _speak_pyttsx3(cls, text):
        import pyttsx3

        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", TTS_FALLBACK_RATE)
            engine.setProperty("volume", TTS_FALLBACK_VOLUME)
            cls._select_female_voice(engine)
            engine.say(text)
            engine.runAndWait()
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
                del engine
                gc.collect()

    @staticmethod
    def _remove_actions(text):
        # Aktionen am Zeilenanfang oder -ende entfernen; Hervorhebungen
        # mitten im Satz bleiben als normaler Text erhalten.
        text = ACTION_AT_START.sub("", text)
        return ACTION_AT_END.sub("", text)

    @staticmethod
    def _clean_text(text):
        text = TTS._remove_actions(text)
        text = KEYCAP_EMOJI.sub("", text).replace("*", "")
        cleaned = []
        for character in text:
            codepoint = ord(character)
            category = unicodedata.category(character)
            is_emoji_component = (
                category == "So"
                or 0x1F3FB <= codepoint <= 0x1F3FF
                or 0xE0020 <= codepoint <= 0xE007F
                or character in ("\u200d", "\ufe0e", "\ufe0f", "\u20e3")
            )
            if not is_emoji_component:
                cleaned.append(character)
        lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(cleaned).splitlines())
        return " ".join(line for line in lines if line)

    def speak(self, text):
        if not self.enabled or not text:
            return

        tts_text = self._clean_text(text)
        if not tts_text:
            return
        tts_text = self.apply_pronunciation(tts_text)

        for attempt in range(2):
            try:
                self._speak_once(tts_text)
                return
            except Exception as error:
                if attempt == 0:
                    print(f"Warnung: Vorlesen fehlgeschlagen ({error}). Neuer Versuch.")
                else:
                    print(f"Warnung: Vorlesen erneut fehlgeschlagen ({error}). Luna spricht diesmal nicht.")
