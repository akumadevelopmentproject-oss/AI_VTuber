import os
import time

# Vor dem Import setzen: leise Hugging-Face-Hinweise (Symlinks unter Windows)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    import numpy as np
    import sounddevice as sd
    import speech_recognition as sr
except ImportError as import_error:
    np = sd = sr = None
    IMPORT_ERROR = import_error
else:
    IMPORT_ERROR = None

from config import settings
from config.settings import (
    STT_LANGUAGE,
    STT_PHRASE_TIME_LIMIT,
    STT_SAMPLE_RATE,
    STT_SILENCE_DURATION,
    STT_TIMEOUT,
)

BLOCK_SECONDS = 0.1
CALIBRATION_SECONDS = 0.7
MIN_THRESHOLD = 150
PRE_ROLL_BLOCKS = 3


def rms(block):
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


def _debug(message):
    if getattr(settings, "STT_DEBUG", False):
        print(message)


def whisper_prompt():
    """Spielnamen aus data/stt_aliases.json als Hinweis für Whisper (bessere Eigennamen)."""
    try:
        from core.stt_correction import load_aliases

        names = []
        for target in load_aliases().values():
            if target not in names:
                names.append(target)
        return ", ".join(names[:25]) or None
    except Exception:
        return None


class WhisperEngine:
    """Lokales faster-whisper. GPU, wenn CUDA verfügbar ist, sonst CPU (int8)."""

    def __init__(self):
        from faster_whisper import WhisperModel

        self._model_class = WhisperModel
        self.model = None
        self.device = None
        try:
            import ctranslate2

            cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            cuda = False
        attempts = ([("cuda", "float16")] if cuda else []) + [("cpu", "int8")]
        last_error = None
        for device, compute in attempts:
            try:
                self.model = self._load(device, compute)
                self.device = device
                break
            except Exception as error:
                last_error = error
        if self.model is None:
            raise RuntimeError(f"Whisper-Modell nicht ladbar: {last_error}")

    def _load(self, device, compute):
        model = self._model_class(settings.STT_WHISPER_MODEL, device=device, compute_type=compute)
        # Probelauf: fehlende CUDA-Bibliotheken (z. B. cublas) fallen erst beim Encoden auf
        noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        segments, _ = model.transcribe(noise, language=settings.STT_WHISPER_LANGUAGE, vad_filter=False)
        list(segments)
        return model

    def transcribe(self, audio_bytes):
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio,
            language=settings.STT_WHISPER_LANGUAGE,
            beam_size=settings.STT_WHISPER_BEAM_SIZE,
            vad_filter=True,
            initial_prompt=whisper_prompt(),
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class STT:
    def __init__(self):
        self.available = False
        self.threshold = MIN_THRESHOLD
        self.block_size = int(STT_SAMPLE_RATE * BLOCK_SECONDS)
        self.engine = None
        self.whisper = None

        if IMPORT_ERROR is not None:
            print(f"Warnung: Spracheingabe nicht verfügbar ({IMPORT_ERROR}).")
            return

        self.recognizer = sr.Recognizer()
        try:
            self._calibrate()
            self.available = True
        except Exception as error:
            print(f"Mikrofonfehler: {error}")
            return
        self._select_engine()

    def _select_engine(self):
        wanted = str(getattr(settings, "STT_ENGINE", "google")).lower()
        fallback = str(getattr(settings, "STT_FALLBACK_ENGINE", "google")).lower()
        if wanted == "whisper":
            try:
                print("Lade lokales Whisper ...")
                self.whisper = WhisperEngine()
                self.engine = "whisper"
                return
            except Exception as error:
                print(f"Whisper nicht verfügbar ({error}). Nutze {fallback}.")
        self.engine = fallback if fallback in ("google",) else "google"

    def _recognize(self, audio_bytes):
        if self.engine == "whisper":
            try:
                _debug("[STT ENGINE] whisper")
                return self.whisper.transcribe(audio_bytes)
            except Exception as error:
                print(f"Whisper-Fehler ({error}). Wechsle zu Google.")
                self.engine = "google"
        _debug("[STT ENGINE] google")
        audio = sr.AudioData(audio_bytes, STT_SAMPLE_RATE, 2)
        try:
            return self.recognizer.recognize_google(audio, language=STT_LANGUAGE)
        except sr.UnknownValueError:
            print("Ich habe dich nicht verstanden.")
        except sr.RequestError as error:
            print(f"Spracherkennung nicht erreichbar (Internetverbindung?): {error}")
        except Exception as error:
            print(f"Fehler bei der Spracherkennung: {error}")
        return None

    def _open_stream(self):
        return sd.InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.block_size,
        )

    def _calibrate(self):
        # Umgebungsgeräusch nur einmal beim Start messen
        with self._open_stream() as stream:
            levels = []
            for _ in range(int(CALIBRATION_SECONDS / BLOCK_SECONDS)):
                block, _ = stream.read(self.block_size)
                levels.append(rms(block))
        noise = float(np.mean(levels))
        self.threshold = max(noise * 3, noise + 100, MIN_THRESHOLD)

    def _record(self):
        silence_limit = max(1, int(STT_SILENCE_DURATION / BLOCK_SECONDS))
        max_blocks = int(STT_PHRASE_TIME_LIMIT / BLOCK_SECONDS)
        pre_roll = []
        recorded = []
        silent_blocks = 0
        speaking = False
        start = time.monotonic()

        with self._open_stream() as stream:
            while True:
                block, _ = stream.read(self.block_size)
                loud = rms(block) > self.threshold

                if not speaking:
                    pre_roll.append(block)
                    pre_roll = pre_roll[-PRE_ROLL_BLOCKS:]
                    if loud:
                        speaking = True
                        recorded = list(pre_roll)
                    elif time.monotonic() - start > STT_TIMEOUT:
                        return None
                    continue

                recorded.append(block)
                silent_blocks = 0 if loud else silent_blocks + 1
                if silent_blocks >= silence_limit or len(recorded) >= max_blocks:
                    break

        return np.concatenate(recorded).astype(np.int16).tobytes()

    def listen(self):
        if not self.available:
            return None

        print("🎤 Luna hört zu...")
        try:
            audio_bytes = self._record()
        except Exception as error:
            print(f"Mikrofonfehler: {error}")
            return None

        if audio_bytes is None:
            return None

        text = self._recognize(audio_bytes)
        if text:
            _debug(f"[STT RAW] {text}")
            return text
        return None
