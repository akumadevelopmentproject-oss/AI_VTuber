import json
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.json"
MAX_PROMPT_MEMORIES = 20
MAX_MEMORY_LENGTH = 300

REMEMBER_COMMAND = re.compile(
    r"^\s*(?:luna[\s,]+)?(?:bitte\s+)?(?:"
    r"merke?\s+dir"
    r"|kannst\s+du\s+dir\s+(?:bitte\s+)?(?:das\s+|es\s+)?merken"
    r"|speichere?\s+dir"
    r"|behalte?\s+(?:bitte\s+)?im\s+kopf"
    r")\s*[:,;\-]?\s*(?P<text>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
FORGET_COMMAND = re.compile(
    r"^\s*(?:luna[\s,]+)?(?:bitte\s+)?vergiss\s*[,:]?\s*(?:dass\s+)?(?P<text>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
FORGET_ALL = re.compile(
    r"^(?:(?:luna )?(?:bitte )?vergiss alles(?: bitte)?"
    r"|(?:luna )?(?:bitte )?l(?:oe|ö)sche? (?:deine|alle|all deine) erinnerungen"
    r"|(?:luna )?vergiss alle erinnerungen)$"
)
QUERY_COMMAND = re.compile(
    r"was wei(?:ß|ss)t du (?:alles )?(?:über|ueber|von) (?:mich|mir)"
    r"|was hast du dir (?:alles )?gemerkt"
)
CONFIRM_WORDS = {
    "ja",
    "ja bitte",
    "ja wirklich",
    "ja sicher",
    "ja ganz sicher",
    "jawohl",
    "klar",
    "ja lösch alles",
    "ja loesch alles",
    "ja vergiss alles",
    "bestätigt",
    "bestaetigt",
}
CANCEL_WORDS = {"nein", "nee", "abbrechen", "stopp", "stop", "lieber nicht"}

CONFIRM_QUESTION = "Willst du wirklich alle Erinnerungen löschen?"


def _normalize(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, re.UNICODE))


def _to_third_person(content):
    text = content.strip().rstrip(".!?").strip()
    subordinate = re.match(r"^dass\s+", text, flags=re.IGNORECASE) is not None
    text = re.sub(r"^dass\s+", "", text, flags=re.IGNORECASE)
    if subordinate:
        # Nebensatz: Verb steht am Ende und muss nach vorn
        reorder = (
            (r"^ich\s+(.+?)\s+hei(?:ß|ss)e$", r"Der Benutzer heißt \1"),
            (r"^ich\s+(.+?)\s+bin$", r"Der Benutzer ist \1"),
            (r"^ich\s+(.+?)\s+habe$", r"Der Benutzer hat \1"),
            (r"^ich\s+(.+?)\s+(mag|will|kann)$", r"Der Benutzer \2 \1"),
            (r"^ich\s+(.+?)\s+(\w+)e$", r"Der Benutzer \2t \1"),
            (r"^(mein(?:e|en|em|er|es)?\s+\w+)\s+(.+?)\s+ist$", r"\1 ist \2"),
            (r"^(.+?)\s+(.+?)\s+ist$", r"\1 ist \2"),
        )
        for pattern, replacement in reorder:
            if re.match(pattern, text, flags=re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                break
    rules = (
        (r"\bich\s+hei(?:ß|ss)e\b", "Der Benutzer heißt"),
        (r"\bich\s+bin\b", "Der Benutzer ist"),
        (r"\bich\s+habe\b", "Der Benutzer hat"),
        (r"\bich\s+mag\b", "Der Benutzer mag"),
        (r"\bich\s+will\b", "Der Benutzer will"),
        (r"\bich\s+kann\b", "Der Benutzer kann"),
        (r"\bich\s+(\w+)e\b", r"Der Benutzer \1t"),
        (r"\bich\s+(\w+)\b", r"Der Benutzer \1"),
        (r"\bmein(?:e|en|em|er|es)?\b", "Mikis"),
    )
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def _significant_words(text):
    return {word for word in _normalize(text).split() if len(word) > 2}


UNCERTAIN = re.compile(
    r"\b(?:vielleicht|eventuell|m(?:ö|oe)glicherweise|wahrscheinlich|glaube|"
    r"keine\s+ahnung|irgendwie|manchmal|sp(?:ä|ae)ter|w(?:ü|ue)rde|k(?:ö|oe)nnte)\b",
    re.IGNORECASE,
)
QUESTION_START = re.compile(
    r"^(?:was|wer|wie|wo|wann|warum|wieso|weshalb|welche[rsmn]?|wen|wem|wohin|"
    r"woher|kannst|k(?:ö|oe)nntest|hast|bist|magst|wei(?:ß|ss)t|ist|sind)\b",
    re.IGNORECASE,
)
LEADING_FILLER = re.compile(r"^(?:(?:luna|also|ja|hey|hallo|okay|ok)[\s,!]+)+", re.IGNORECASE)
VALUE_REJECT_WORDS = {
    "das", "es", "dich", "dir", "sie", "ihn", "dies", "diese", "dieses", "nicht",
    "nichts", "wenn", "falls", "weil", "dass", "aber", "ob", "wie", "was", "wer",
    "wo", "dann", "denn", "doch", "schon", "mich", "mir", "uns", "euch", "ihr",
}
AUTO_VERBS = {
    "mag": "mag",
    "liebe": "liebt",
    "hasse": "hasst",
    "bevorzuge": "bevorzugt",
}
_VALUE = r"(?P<value>[^\s,;:.!?][^,;:.!?]*?)"
AUTO_PATTERNS = (
    ("name", re.compile(rf"^ich\s+hei(?:ß|ss)e\s+{_VALUE}$", re.IGNORECASE)),
    ("name", re.compile(rf"^mein\s+name\s+ist\s+{_VALUE}$", re.IGNORECASE)),
    (
        "favorite",
        re.compile(
            rf"^mein(?:e|en)?\s+lieblings(?P<kind>[a-zäöüß]+)\s+ist\s+{_VALUE}$",
            re.IGNORECASE,
        ),
    ),
    ("play", re.compile(rf"^ich\s+spiele\s+gern(?:e)?\s+{_VALUE}$", re.IGNORECASE)),
    (
        "verb",
        re.compile(
            rf"^ich\s+(?P<verb>mag|liebe|hasse|bevorzuge)\s+{_VALUE}$", re.IGNORECASE
        ),
    ),
)


def _valid_value(value, max_words=6):
    words = _normalize(value).split()
    if not words or len(words) > max_words or len(value) > 80:
        return False
    return not any(word in VALUE_REJECT_WORDS for word in words)


def extract_auto_memory(user_text):
    """Erkennt klare persönliche Fakten; gibt den Erinnerungstext oder None zurück."""
    if not isinstance(user_text, str):
        return None
    for segment in re.split(r"[.!?\n]+", user_text):
        segment = LEADING_FILLER.sub("", segment.strip()).strip()
        segment = re.sub(r"\s+", " ", segment)
        if not segment or UNCERTAIN.search(segment) or QUESTION_START.match(segment):
            continue
        for kind, pattern in AUTO_PATTERNS:
            match = pattern.match(segment)
            if not match:
                continue
            value = match.group("value").strip()
            if not _valid_value(value, 3 if kind == "name" else 6):
                break
            if kind == "name":
                return f"Der Benutzer heißt {value}."
            if kind == "favorite":
                return f"Mikis Lieblings{match.group('kind').lower()} ist {value}."
            if kind == "play":
                return f"Der Benutzer spielt gerne {value}."
            return f"Der Benutzer {AUTO_VERBS[match.group('verb').lower()]} {value}."
    return None


class Memory:
    def __init__(self, path=MEMORY_PATH):
        self.path = Path(path)
        self.pending_clear = False
        self._invalid_file = False
        self.memories = self._load()
        if not self.path.exists():
            self._save()

    def _load(self):
        try:
            raw = self.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError):
            self._invalid_file = True
            return []
        if not raw.strip():
            return []
        try:
            items = json.loads(raw)["memories"]
            if not isinstance(items, list):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self._invalid_file = True
            return []

        memories = []
        seen = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            text = item["text"].strip()
            key = _normalize(text)
            if not key or key in seen:
                continue
            seen.add(key)
            created = item.get("created_at")
            memories.append(
                {"text": text, "created_at": created if isinstance(created, str) else ""}
            )
        return memories

    def _save(self):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._invalid_file and self.path.exists():
                shutil.copyfile(self.path, self.path.with_suffix(".json.bak"))
                self._invalid_file = False
            descriptor, temporary = tempfile.mkstemp(
                prefix=".memory.", suffix=".tmp", dir=self.path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump({"memories": self.memories}, stream, ensure_ascii=False, indent=4)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            return True
        except OSError:
            return False
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)

    def add(self, text):
        text = text.strip()[:MAX_MEMORY_LENGTH]
        key = _normalize(text)
        if not key:
            return False
        if any(_normalize(item["text"]) == key for item in self.memories):
            return False
        self.memories.append(
            {"text": text, "created_at": datetime.now().isoformat(timespec="seconds")}
        )
        return self._save()

    def remove_matching(self, text):
        key = _normalize(_to_third_person(text))
        words = _significant_words(_to_third_person(text))
        kept = []
        removed = 0
        for item in self.memories:
            matches = _normalize(item["text"]) == key or (
                bool(words) and words <= _significant_words(item["text"])
            )
            if matches:
                removed += 1
            else:
                kept.append(item)
        if removed:
            self.memories = kept
            self._save()
        return removed

    def clear(self):
        self.memories = []
        self._save()

    def prompt_memories(self, is_blocked=None):
        texts = [item["text"] for item in self.memories if item["text"]]
        if is_blocked is not None:
            texts = [text for text in texts if not is_blocked(text)]
        return texts[-MAX_PROMPT_MEMORIES:]

    def prompt_block(self, is_blocked=None):
        texts = self.prompt_memories(is_blocked)
        if not texts:
            return ""
        lines = "\n".join(f"- {text}" for text in texts)
        return f"Bekannte Erinnerungen über Miki:\n{lines}"

    def handle(self, user_text):
        """Gibt eine direkte Antwort zurück oder None für normale Verarbeitung."""
        normalized = _normalize(user_text)

        if self.pending_clear:
            self.pending_clear = False
            if normalized in CONFIRM_WORDS:
                self.clear()
                return "Okay, ich habe alle Erinnerungen gelöscht."
            if normalized in CANCEL_WORDS:
                return "Alles klar, ich behalte alles."

        if FORGET_ALL.match(normalized):
            if not self.memories:
                return "Ich habe gerade gar nichts gespeichert."
            self.pending_clear = True
            return CONFIRM_QUESTION

        remember = REMEMBER_COMMAND.match(user_text)
        if remember:
            content = _to_third_person(remember.group("text"))
            if not content:
                return None
            if self.add(content):
                return "Hab ich mir gemerkt, Miki."
            return "Das wusste ich schon, Miki."

        forget = FORGET_COMMAND.match(user_text)
        if forget:
            if self.remove_matching(forget.group("text")):
                return "Okay, das habe ich vergessen."
            return "Dazu habe ich nichts gespeichert."

        if QUERY_COMMAND.search(normalized):
            if not self.memories:
                return "Ich habe mir noch nichts gemerkt, Miki."
            facts = " ".join(item["text"] for item in self.memories)
            return f"Das weiß ich bisher: {facts}"
        return None


_DEFAULT_MEMORY = None


def get_memory():
    global _DEFAULT_MEMORY
    if _DEFAULT_MEMORY is None:
        _DEFAULT_MEMORY = Memory()
    return _DEFAULT_MEMORY
