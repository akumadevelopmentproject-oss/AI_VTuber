"""Vorsichtige Korrektur von Namen in STT-Text (nur nach der Input-Safety aufrufen).

1. Exakte Aliase aus data/stt_aliases.json (case-insensitive, Wortgruppen).
2. Bekannte Spielnamen (Steam-Cache, Brain, aktives Spiel) mit difflib auf
   Wortgruppen von 2-5 Wörtern. Nur sehr eindeutige Treffer werden ersetzt;
   unsichere Treffer führen zu einer Rückfrage statt zu einer Änderung.
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from config import settings
from core.game_knowledge import numeral_value
from core.memory import _normalize

ALIAS_PATH = Path(__file__).resolve().parent.parent / "data" / "stt_aliases.json"
FUZZY_RATIO = 0.88
ASK_RATIO = 0.74
TIE_MARGIN = 0.03
MIN_WORDS = 2
MAX_WORDS = 5
FILLER_TOKENS = {
    "und", "oder", "der", "die", "das", "ein", "eine", "ist", "in", "im", "von", "mit",
    "für", "fuer", "nicht", "kein", "zu", "auf", "an", "bei",
}

_alias_cache = {}


def _short(name):
    return name.split(":")[0].strip()


def tokens_of(text):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'’-]+", text, re.UNICODE)]


def load_aliases(path=None):
    """{normalisierte Phrase: Zielname}. Fehlende oder kaputte Datei = keine Aliase."""
    path = Path(path) if path else ALIAS_PATH
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return {}
    cached = _alias_cache.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    aliases = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            phrase = _normalize(key) if isinstance(key, str) else ""
            if phrase and isinstance(value, str) and value.strip():
                aliases[phrase] = value.strip()
    _alias_cache[path] = (stamp, aliases)
    return aliases


def known_names(brain=None, game_knowledge=None, active_game=None, extra_names=(), aliases=None):
    """{normalisierter Name: Anzeigename} aus Steam-Cache, Brain, aktivem Spiel und Alias-Zielen."""
    names = {}

    def add(display):
        for variant in {display, _short(display)}:
            key = _normalize(variant)
            if key:
                names.setdefault(key, variant)

    knowledge = game_knowledge or getattr(brain, "knowledge", None)
    try:
        for entry in knowledge.cache.values():
            add(entry["name"])
            for alias in entry.get("aliases", []):
                key = _normalize(alias)
                if key:
                    names.setdefault(key, alias.title())
    except Exception:
        pass
    try:
        for game in brain.games:
            add(game["game"])
    except Exception:
        pass
    if active_game:
        add(active_game)
    for extra in extra_names or ():
        if isinstance(extra, str) and extra:
            add(extra)
    for target in (aliases or {}).values():
        add(target)
    return names


def _trailing_numeral(parts):
    return numeral_value(parts[-1]) if len(parts) >= 3 else None


def _apply_numerals(text, names):
    """"Crusader Kings 3" / "Crusader Kings drei" -> "Crusader Kings III" (bei bekanntem Titel)."""
    tokens = tokens_of(text)
    norm = [_normalize(t[0]) for t in tokens]
    found = []
    for key, display in names.items():
        parts = key.split()
        value = _trailing_numeral(parts)
        if value is None:
            continue
        base = parts[:-1]
        for start in range(len(tokens) - len(base)):
            if norm[start:start + len(base)] == base and numeral_value(norm[start + len(base)]) == value:
                found.append((start, start + len(base) + 1, display))
    chosen = []
    for cand in sorted(found, key=lambda c: -(c[1] - c[0])):
        if not any(cand[0] < c[1] and c[0] < cand[1] for c in chosen):
            chosen.append(cand)
    return _apply(text, [(tokens[s][1], tokens[e - 1][2], d) for s, e, d in chosen])


def _apply(text, replacements):
    for start, end, display in sorted(replacements, reverse=True):
        text = text[: start] + display + text[end:]
    return text


def _apply_aliases(text, aliases):
    tokens = tokens_of(text)
    norm = [_normalize(t[0]) for t in tokens]
    found = []
    for phrase, target in aliases.items():
        parts = phrase.split()
        size = len(parts)
        for start in range(len(tokens) - size + 1):
            if norm[start:start + size] == parts:
                found.append((start, start + size, target))
    chosen = []
    for cand in sorted(found, key=lambda c: -(c[1] - c[0])):
        if not any(cand[0] < c[1] and c[0] < cand[1] for c in chosen):
            chosen.append(cand)
    return _apply(text, [(tokens[s][1], tokens[e - 1][2], target) for s, e, target in chosen])


def _related(a, b):
    """Ein Name ist Präfix des anderen ("Crusader Kings" / "Crusader Kings III")."""
    a, b = _normalize(a), _normalize(b)
    return a == b or a.startswith(b + " ") or b.startswith(a + " ")


def _candidates(tokens, norm, names):
    """(start, ende, Anzeigename, Ähnlichkeit) für Wortgruppen von 2-5 Wörtern."""
    result = []
    for key, display in names.items():
        parts = key.split()
        size = len(parts)
        if size >= 3 and numeral_value(parts[-1]) is None and all(len(p) >= 4 for p in parts[1:]):
            # Teilname ohne erstes Wort ("Breakout Infinite" für "Arena Breakout Infinite")
            partial = parts[1:]
            for start in range(len(tokens) - len(partial) + 1):
                window = norm[start:start + len(partial)]
                if any(w in FILLER_TOKENS for w in window):
                    continue
                ratio = 1.0 if window == partial else SequenceMatcher(None, " ".join(window), " ".join(partial)).ratio()
                if ratio >= FUZZY_RATIO:
                    result.append((start, start + len(partial), display, ratio))
        if size == 1 and len(key) >= 6:
            for start in range(len(tokens) - 1):
                if "".join(norm[start:start + 2]) == key:
                    result.append((start, start + 2, display, 1.0))
            continue
        if size < MIN_WORDS or size > MAX_WORDS:
            continue
        for length in range(max(MIN_WORDS, size - 1), min(MAX_WORDS, size + 1) + 1):
            for start in range(len(tokens) - length + 1):
                window = norm[start:start + length]
                if window == parts or any(w in FILLER_TOKENS for w in window):
                    continue
                if length == size + 1:
                    # ein eingeschobenes Wort ("arena black breakout")
                    for skip in range(1, size):
                        if (window[:skip] + window[skip + 1:] == parts
                                and len(window[skip]) >= 3 and window[skip].isalpha()):
                            result.append((start, start + length, display, 1.0))
                            break
                    else:
                        ratio = SequenceMatcher(None, " ".join(window), key).ratio()
                        if ratio >= ASK_RATIO:
                            result.append((start, start + length, display, ratio))
                    continue
                ratio = SequenceMatcher(None, " ".join(window), key).ratio()
                if ratio >= ASK_RATIO:
                    result.append((start, start + length, display, ratio))
    return result


def _fuzzy(text, names):
    """Gibt (Text, Rückfrage-Info oder None) zurück."""
    tokens = tokens_of(text)
    norm = [_normalize(t[0]) for t in tokens]
    exact = set()
    for key in names:
        parts = key.split()
        for start in range(len(tokens) - len(parts) + 1):
            if norm[start:start + len(parts)] == parts:
                exact.add((start, start + len(parts)))

    def overlaps(span, others):
        return any(span[0] < o[1] and o[0] < span[1] for o in others)

    candidates = [c for c in _candidates(tokens, norm, names) if not overlaps(c[:2], exact)]
    sure = [c for c in candidates if c[3] >= FUZZY_RATIO]
    chosen, ask = [], None
    for cand in sorted(sure, key=lambda c: (-(c[1] - c[0]), -c[3])):
        span = cand[:2]
        if overlaps(span, [c[:2] for c in chosen]):
            continue
        rivals = {
            c[2] for c in sure
            if c[:2] == span and c[2] != cand[2] and not _related(c[2], cand[2])
            and cand[3] - c[3] <= TIE_MARGIN
        }
        if rivals:
            options = sorted({cand[2], *rivals})[:3]
            ask = ask or (span, options)
            continue
        chosen.append(cand)
    if ask is None:
        taken = [c[:2] for c in chosen]
        unsure = [
            c for c in candidates
            if c[3] < FUZZY_RATIO and not overlaps(c[:2], taken)
            and SequenceMatcher(None, norm[c[0]], names_first(c[2])).ratio() >= 0.8
        ]
        if unsure:
            best = max(unsure, key=lambda c: c[3])
            rivals = [c for c in unsure if c[:2] == best[:2] and not _related(c[2], best[2])
                      and best[3] - c[3] <= TIE_MARGIN]
            options = sorted({best[2], *(c[2] for c in rivals)})[:3]
            ask = (best[:2], options)
    replacements = [(tokens[s][1], tokens[e - 1][2], display) for s, e, display, _ in chosen]
    question = None
    if ask:
        (start, end), options = ask
        variants = {
            option: _apply(text, replacements + [(tokens[start][1], tokens[end - 1][2], option)])
            for option in options
        }
        question = {"options": options, "variants": variants}
    return _apply(text, replacements), question


def names_first(display):
    return (_normalize(display).split() or [""])[0]


def analyze_stt_text(text, brain=None, game_knowledge=None, active_game=None, aliases_path=None,
                     extra_names=()):
    """(korrigierter Text, Rückfrage-Info oder None). Rückfrage = unsicherer Treffer."""
    if not isinstance(text, str) or not text.strip():
        return text, None
    aliases = load_aliases(aliases_path)
    result = _apply_aliases(text, aliases)
    names = known_names(brain, game_knowledge, active_game, extra_names, aliases)
    question = None
    if names:
        result = _apply_numerals(result, names)
        result, question = _fuzzy(result, names)
    if getattr(settings, "STT_DEBUG", False) and result != text:
        print(f"[STT FIX] {result}")
    return result, question


def correct_stt_text(text, brain=None, game_knowledge=None, active_game=None, aliases_path=None,
                     extra_names=()):
    """Korrigiert nur eindeutig erkennbare Namen. Safety muss VORHER gelaufen sein."""
    return analyze_stt_text(text, brain, game_knowledge, active_game, aliases_path, extra_names)[0]


def question_for(options):
    if len(options) == 1:
        return f"Meinst du {options[0]}?"
    return "Meinst du " + ", ".join(options[:-1]) + " oder " + options[-1] + "?"
