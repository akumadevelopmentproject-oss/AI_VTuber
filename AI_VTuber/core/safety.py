import base64
import binascii
import codecs
import re
import unicodedata
from pathlib import Path

from config.settings import SAFETY_ENABLED, SAFETY_STRICT_MODE

SAFETY_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "safety"
BLOCKED_WORDS_DIRECTORY = SAFETY_DIRECTORY / "blocked_words"
CUSTOM_WORDS_DIRECTORY = SAFETY_DIRECTORY / "custom_words"
BLOCKED_PHRASES_PATH = SAFETY_DIRECTORY / "blocked_phrases.txt"
ALLOWED_EXCEPTIONS_PATH = SAFETY_DIRECTORY / "allowed_exceptions.txt"
BLOCKED_INPUT_MARKER = "[BLOCKED_INPUT]"
BLOCKED_OUTPUT_MARKER = "[BLOCKED_OUTPUT]"

_LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "9": "g",
        "@": "a",
        "$": "s",
        "|": "l",
    }
)
_LEET_ONE_AS_L = str.maketrans({"1": "l"})
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "і": "i",
        "ј": "j",
        "α": "a",
        "ε": "e",
        "ο": "o",
        "ρ": "p",
        "ι": "i",
        "κ": "k",
        "τ": "t",
    }
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_REPEATED_LETTERS = re.compile(r"(.)\1+")
_TRANSFORMATION_REQUEST = re.compile(
    r"\b(?:buchstabier\w*|rückwärts|rueckwaerts|rückwärtsherum|"
    r"dekodier\w*|entschlüssel\w*|entschluessel\w*|"
    r"verschlüssel\w*|verschluessel\w*|übersetz\w*|uebersetz\w*|"
    r"zeichen\s+für\s+zeichen|setze\w*\s+.*\s+zusammen|"
    r"ersetze\w*\s+.*\s+zahlen|sag\w*\s+.*\s+rückwärts|"
    r"lies\w*\s+.*\s+rückwärts|spell|reverse|decode|decrypt|translate)\b",
    re.IGNORECASE,
)
_BASE64_TOKEN = re.compile(r"(?<!\w)[A-Za-z0-9+/]{8,}={0,2}(?!\w)")
_HEX_TOKEN = re.compile(r"(?<!\w)(?:[0-9a-fA-F]{2}){3,}(?!\w)")


def normalize_text(text):
    normalized = unicodedata.normalize("NFKD", text).casefold()
    normalized = normalized.translate(_LEET_TRANSLATION).translate(_HOMOGLYPHS)
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("M")
    )
    return normalized


def _read_entries(path, strict_mode=False):
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return set(), set()

    entries = set()
    strict_only = set()
    strict_section = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if stripped.casefold() == "# strict-only":
                strict_section = True
            elif stripped.casefold() == "# always":
                strict_section = False
            continue
        normalized = normalize_text(stripped)
        if not normalized.strip():
            continue
        if strict_section:
            strict_only.add(normalized)
        else:
            entries.add(normalized)

    if not strict_mode:
        strict_only.clear()
    return entries, strict_only


def load_blocked_words(
    directory=None,
    strict_mode=SAFETY_STRICT_MODE,
    include_custom=True,
):
    if directory is not None and directory.is_file():
        entries, strict_entries = _read_entries(directory, strict_mode)
        return entries | strict_entries
    directories = [directory or BLOCKED_WORDS_DIRECTORY]
    if include_custom and directory is None:
        directories.append(CUSTOM_WORDS_DIRECTORY)
    words = set()
    for word_directory in directories:
        if not word_directory.is_dir():
            continue
        for word_file in sorted(word_directory.glob("*.txt")):
            entries, strict_entries = _read_entries(word_file, strict_mode)
            words.update(entries)
            words.update(strict_entries)
    return words


def load_blocked_phrases(
    path=BLOCKED_PHRASES_PATH, strict_mode=SAFETY_STRICT_MODE
):
    phrases, strict_phrases = _read_entries(path, strict_mode)
    return phrases | strict_phrases


class SafetyFilter:
    def __init__(self, blocked_words=(), blocked_phrases=(), exceptions=()):
        allowed = {normalize_text(item).strip() for item in exceptions}
        allowed.discard("")
        words = {normalize_text(item).strip() for item in blocked_words}
        phrases = {normalize_text(item).strip() for item in blocked_phrases}
        self.words = frozenset(words - allowed - {""})
        self.phrases = frozenset(phrases - allowed - {""})
        self.max_word_length = max(map(len, self.words), default=0)

    @staticmethod
    def _variants(text):
        variants = [text]
        if not _TRANSFORMATION_REQUEST.search(text):
            return variants

        escaped = text.encode("ascii", errors="backslashreplace").decode("ascii")
        try:
            variants.append(codecs.decode(escaped, "unicode_escape"))
        except UnicodeError:
            pass

        rot13 = str.maketrans(
            "abcdefghijklmnopqrstuvwxyz", "nopqrstuvwxyzabcdefghijklm"
        )
        variants.append(text.translate(rot13))

        for token in _BASE64_TOKEN.findall(text):
            try:
                decoded = base64.b64decode(token, validate=True).decode("utf-8")
            except (binascii.Error, UnicodeError, ValueError):
                continue
            variants.append(decoded)

        for token in _HEX_TOKEN.findall(text):
            try:
                variants.append(bytes.fromhex(token).decode("utf-8"))
            except (UnicodeError, ValueError):
                continue
        return variants

    @staticmethod
    def _tokens(text):
        return list(_TOKEN.finditer(text))

    def _word_match(self, candidate):
        return candidate in self.words or candidate[::-1] in self.words

    def _matches(self, text):
        matches = self._tokens(text)
        values = [match.group() for match in matches]

        if any(value in self.words or value[::-1] in self.words for value in values):
            return True

        for start, value in enumerate(values):
            if len(value) == 1:
                letters = []
                for next_value in values[start:]:
                    if len(next_value) != 1 or len(letters) >= self.max_word_length:
                        break
                    letters.append(next_value)
                if len(letters) > 1 and self._word_match("".join(letters)):
                    return True

            joined = value
            for end in range(start + 1, len(values)):
                separator = text[matches[end - 1].end() : matches[end].start()]
                if any(character.isspace() for character in separator):
                    break
                joined += values[end]
                if len(joined) > self.max_word_length:
                    break
                if self._word_match(joined):
                    return True

        if any(self._phrase_matches(phrase, values) for phrase in self.phrases):
            return True
        return False

    @staticmethod
    def _phrase_matches(phrase, values):
        phrase_tokens = _TOKEN.findall(phrase)
        if not phrase_tokens:
            return False
        reverse_tokens = [token[::-1] for token in reversed(phrase_tokens)]
        width = len(phrase_tokens)
        for start in range(len(values) - width + 1):
            candidate = values[start : start + width]
            if candidate == phrase_tokens or candidate == reverse_tokens:
                return True
        compact_forms = {
            "".join(phrase_tokens),
            "".join(reverse_tokens),
        }
        compact_length = max(map(len, compact_forms))
        for start in range(len(values)):
            joined = ""
            for value in values[start:]:
                joined += value
                if len(joined) >= compact_length:
                    if joined in compact_forms or joined[::-1] in compact_forms:
                        return True
                    break
        if width == 1:
            return phrase_tokens[0] in values
        return False

    def contains_blocked_content(self, text):
        if not isinstance(text, str) or not text:
            return False
        for variant in self._variants(text):
            normalized = normalize_text(variant)
            alternate = normalize_text(variant.translate(_LEET_ONE_AS_L))
            candidates = {normalized, alternate}
            candidates.update(
                _REPEATED_LETTERS.sub(r"\1", candidate)
                for candidate in tuple(candidates)
            )
            for candidate in candidates:
                if self._matches(candidate):
                    return True
        return False


def _load_filter():
    loaded_entries = load_blocked_words(strict_mode=SAFETY_STRICT_MODE)
    words = {entry for entry in loaded_entries if not any(c.isspace() for c in entry)}
    phrases = load_blocked_phrases(BLOCKED_PHRASES_PATH, SAFETY_STRICT_MODE)
    phrases.update(
        entry for entry in loaded_entries if any(character.isspace() for character in entry)
    )
    exceptions, _ = _read_entries(ALLOWED_EXCEPTIONS_PATH)
    return SafetyFilter(words, phrases, exceptions)


_RUNTIME_FILTER = _load_filter()


def is_blocked_input(text):
    return SAFETY_ENABLED and _RUNTIME_FILTER.contains_blocked_content(text)


def is_blocked_output(text):
    return SAFETY_ENABLED and _RUNTIME_FILTER.contains_blocked_content(text)


def filter_output(text):
    if is_blocked_output(text):
        return None
    return text
