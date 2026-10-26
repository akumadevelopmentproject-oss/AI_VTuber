import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from config.settings import LUNA_STATE_DEBUG

STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "luna_state.json"
MOODS = ("playful", "calm", "curious", "energetic", "sleepy", "grumpy", "cheerful", "neutral")
DEFAULTS = {
    "mood": "curious",
    "energy": 0.70,
    "curiosity": 0.80,
    "familiarity_with_miki": 0.15,
    "conversation_count": 0,
}
FAMILIARITY_STEP = 0.002
ENERGY_STEP = 0.01
ENERGY_FLOOR = 0.2
CURIOSITY_STEP = 0.02
MOOD_EVERY = 5
BASELINE_ENERGY = 0.7
RECOVERY_HOURS = 6.0

POSITIVE = re.compile(
    r"\b(?:danke|super|toll|cool|geil|nett|lieb|haha\w*|lol|lustig|witzig|großartig|grossartig|"
    r"klasse|genial|prima|schön|schoen)\b",
    re.IGNORECASE,
)
STATE_QUESTION = re.compile(
    r"^(?:luna )?(?:wie geht(?:'s| es|s) dir(?: gerade| heute| so| eigentlich| denn)*|"
    r"wie ist deine stimmung(?: gerade| heute)?|wie f(?:ü|ue)hlst du dich(?: gerade| heute)?|"
    r"bist du (?:heute |gerade |schon )?(?:m(?:ü|ue)de|schl(?:ä|ae)frig|fit|wach|gut drauf)|"
    r"hast du (?:heute )?(?:gute|schlechte) laune)$"
)

GUIDANCE = {
    "playful": "etwas frecher und lockerer",
    "calm": "ruhiger",
    "curious": "stellt gelegentlich eine passende Rückfrage",
    "energetic": "etwas lebendiger",
    "sleepy": "kürzer und ruhiger",
    "grumpy": "leicht genervt/frech, aber nicht beleidigend",
    "cheerful": "fröhlicher",
    "neutral": "normal",
}
MOOD_REPLY = {
    "playful": "Eigentlich ziemlich gut, ich bin gerade ein bisschen zu Streichen aufgelegt.",
    "calm": "Ziemlich entspannt gerade. Ich bin heute eher ruhig drauf.",
    "curious": "Gut, ich bin gerade ziemlich neugierig auf alles Mögliche.",
    "energetic": "Richtig gut, ich bin heute überraschend energiegeladen.",
    "sleepy": "Ein bisschen müde, ehrlich gesagt. Ich bin heute eher ruhig.",
    "grumpy": "Naja, ein kleines bisschen kurz angebunden heute, aber das geht schon.",
    "cheerful": "Richtig gut gelaunt gerade!",
    "neutral": "Ganz normal, nichts Besonderes.",
}


def _now():
    return datetime.now().replace(microsecond=0).isoformat()


def _clamp(value, default):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return default
    return round(min(1.0, max(0.0, float(value))), 4)


def _clean(raw):
    raw = raw if isinstance(raw, dict) else {}
    count = raw.get("conversation_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        count = DEFAULTS["conversation_count"]
    positives = raw.get("recent_positive")
    if isinstance(positives, bool) or not isinstance(positives, int) or positives < 0:
        positives = 0
    stamp = raw.get("last_updated")
    return {
        "mood": raw.get("mood") if raw.get("mood") in MOODS else DEFAULTS["mood"],
        "energy": _clamp(raw.get("energy"), DEFAULTS["energy"]),
        "curiosity": _clamp(raw.get("curiosity"), DEFAULTS["curiosity"]),
        "familiarity_with_miki": _clamp(raw.get("familiarity_with_miki"), DEFAULTS["familiarity_with_miki"]),
        "conversation_count": count,
        "recent_positive": min(positives, MOOD_EVERY),
        "last_updated": stamp if isinstance(stamp, str) else _now(),
    }


class LunaState:
    def __init__(self, path=STATE_PATH):
        self.path = Path(path)
        self._invalid = False
        self.data = self._load()
        self._recover_energy()
        if not self.path.exists() or self._invalid:
            self.save()

    def _load(self):
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return _clean({})
        except (OSError, ValueError, UnicodeError):
            self._invalid = True
            return _clean({})
        if not isinstance(raw, dict):
            self._invalid = True
        return _clean(raw)

    def _recover_energy(self):
        try:
            hours = (datetime.now() - datetime.fromisoformat(self.data["last_updated"])).total_seconds() / 3600
        except ValueError:
            return
        if hours < 1:
            return
        step = min(1.0, hours / RECOVERY_HOURS)
        energy = self.data["energy"]
        self.data["energy"] = _clamp(energy + (BASELINE_ENERGY - energy) * step, energy)

    def save(self):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data["last_updated"] = _now()
            descriptor, temporary = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=self.path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=4)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            self._invalid = False
            return True
        except OSError:
            return False
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)

    @property
    def mood(self):
        return self.data["mood"]

    def _choose_mood(self):
        d = self.data
        if d["energy"] < 0.35:
            return "sleepy"
        if d["recent_positive"] >= 3:
            return "cheerful"
        if d["recent_positive"] >= 2:
            return "playful"
        if d["energy"] > 0.85:
            return "energetic"
        if d["curiosity"] >= 0.8:
            return "curious"
        return "calm"

    def register_conversation(self, user_text="", new_topic=False):
        """Wird nur für normale, nicht blockierte Nachrichten aufgerufen."""
        d = self.data
        d["conversation_count"] += 1
        d["familiarity_with_miki"] = _clamp(d["familiarity_with_miki"] + FAMILIARITY_STEP, 1.0)
        d["energy"] = _clamp(max(ENERGY_FLOOR, d["energy"] - ENERGY_STEP), ENERGY_FLOOR)
        if new_topic:
            d["curiosity"] = _clamp(d["curiosity"] + CURIOSITY_STEP, 1.0)
        if isinstance(user_text, str) and POSITIVE.search(user_text):
            d["recent_positive"] = min(MOOD_EVERY, d["recent_positive"] + 1)
        if d["conversation_count"] % MOOD_EVERY == 0:
            d["mood"] = self._choose_mood()
            d["recent_positive"] = 0
        saved = self.save()
        if LUNA_STATE_DEBUG:
            print(
                f"[STATE] mood={d['mood']} energy={d['energy']:.2f} "
                f"curiosity={d['curiosity']:.2f} familiarity={d['familiarity_with_miki']:.2f}"
            )
        return saved

    def prompt_block(self):
        d = self.data
        return (
            "LUNA CURRENT STATE (simulierte Charakterwerte, keine echten Gefühle; "
            "nie dem User erklären, außer Miki fragt ausdrücklich danach):\n"
            f"Mood: {d['mood']}\n"
            f"Energy: {d['energy']:.2f}\n"
            f"Curiosity: {d['curiosity']:.2f}\n"
            f"Familiarity with Miki: {d['familiarity_with_miki']:.2f}\n\n"
            "Behavior guidance:\n"
            f"- {d['mood']} -> {GUIDANCE[d['mood']]}\n"
            "- Die Sprache soll sich nur leicht ändern; keine Regieanweisungen wie *lacht* oder *seufzt*.\n"
            "- Vertrautheit macht die Sprache nur lockerer, nie besitzergreifend, eifersüchtig, "
            "abhängig, schuldzuweisend oder drängend."
        )

    def handle_command(self, user_text):
        from core.memory import _normalize

        normalized = _normalize(user_text)
        if not STATE_QUESTION.match(normalized):
            return None
        reply = MOOD_REPLY[self.mood]
        if self.data["energy"] < 0.35 and self.mood != "sleepy":
            reply += " Allerdings bin ich etwas müde."
        elif self.data["energy"] > 0.8 and self.mood in ("playful", "cheerful", "curious", "calm"):
            reply += " Und ziemlich wach."
        return reply


_DEFAULT_STATE = None


def get_luna_state():
    global _DEFAULT_STATE
    if _DEFAULT_STATE is None:
        _DEFAULT_STATE = LunaState()
    return _DEFAULT_STATE
