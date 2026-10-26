"""Aktives Gesprächsthema (nur im Arbeitsspeicher, keine Datei).

Merkt sich das zuletzt besprochene Spiel, damit "es", "das Spiel", "davon" ... im
nächsten Satz richtig zugeordnet werden, und korrigiert vorsichtig STT-Fehler
bei bekannten Spielnamen.
"""
import re

from core.memory import _normalize
from core.stt_correction import analyze_stt_text, question_for

MAX_IDLE_TURNS = 5
YES_WORDS = {"ja", "jap", "jo", "genau", "richtig", "stimmt", "korrekt", "yes", "jawohl"}
NO_WORDS = {"nein", "nee", "nö", "falsch", "nicht", "no"}
FILLER_TOKENS = {
    "und", "oder", "der", "die", "das", "ein", "eine", "ist", "in", "im", "von", "mit",
    "für", "fuer", "nicht", "kein", "zu", "auf", "an", "bei",
}

CUE = re.compile(
    r"(?:was\s+wei(?:ß|ss)t\s+du\s+(?:alles\s+)?(?:über|ueber|von|zu)"
    r"|wei(?:ß|ss)t\s+du\s+(?:was|etwas|irgendwas|irgendetwas)\s+(?:über|ueber|von|zu)"
    r"|kennst\s+du"
    r"|was\s+ist"
    r"|erz(?:ä|ae)hl\w*\s+(?:mir\s+)?(?:mal\s+)?(?:was|etwas|mehr)\s+(?:über|ueber|zu)"
    r"|wie\s+findest\s+du"
    r"|spielst\s+du"
    r"|hast\s+du\s+schon\s+(?:mal\s+)?von)"
    r"\s+(?:das\s+spiel\s+)?(?P<name>[^?!.,;:]+)",
    re.IGNORECASE,
)
NAME_TAIL = re.compile(
    r"\s+(?:eigentlich|schon|überhaupt|denn|so|gerade|f(?:ü|ue)r\s+(?:ein|ein\w+)\s+\w+|geh(?:ö|oe)rt)\s*$",
    re.IGNORECASE,
)
GAME_WORD = re.compile(r"\b(?:spiel|game)\b", re.IGNORECASE)
PLAYED_CUE = re.compile(
    r"\bhast\s+du\s+(?P<name>[^?!.,;:]+?)\s+(?:schon\s+)?(?:mal\s+)?(?:gespielt|gezockt)\b",
    re.IGNORECASE,
)

# Wörter, die auf das aktive Spiel verweisen
GAME_REFERENCE = re.compile(r"\b(?:das|dieses|dem|diesem)\s+spiel\b", re.IGNORECASE)
ABOUT_REFERENCE = re.compile(r"\b(?:dar(?:ü|ue)ber|davon)\b", re.IGNORECASE)
DAS_AFTER_VERB = re.compile(
    r"\b(?P<head>(?:ist|hat|findest du|magst du|kennst du|spielst du|mag ich|mag|liebe|finde|"
    r"spiele|zocke)\s+)das\b(?=\s*(?:eigentlich\s*)?(?:[?!.,]*\s*$|f(?:ü|ue)r\s+ein))",
    re.IGNORECASE,
)
DAS_START = re.compile(
    r"^(?P<lead>\s*)das(?=\s+(?:ist|war|sind|finde|mag|liebe|spiele|zocke|gef(?:ä|ae)llt|"
    r"klingt|wirkt|sieht)\b)",
    re.IGNORECASE,
)
ES_WORD = re.compile(r"\b(?P<es>es)\b", re.IGNORECASE)
ES_BLOCKERS = {"geht", "gibt", "gehts", "regnet", "schneit"}


def _short(name):
    return name.split(":")[0].strip()


def _tokens(text):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'’-]+", text, re.UNICODE)]


ROMAN_NUMBERS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9}
ROMAN_BY_NUMBER = {2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX"}
ORDINAL_WORDS = {
    "erste": 1, "ersten": 1, "erster": 1, "erstes": 1,
    "zweite": 2, "zweiten": 2, "zweiter": 2, "zweites": 2,
    "dritte": 3, "dritten": 3, "dritter": 3, "drittes": 3,
}
NUMBER_WORDS = {
    **{str(n): n for n in range(1, 10)},
    **ROMAN_NUMBERS,
    **ORDINAL_WORDS,
    "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "vierte": 4, "vierten": 4,
    "fünf": 5, "fünfte": 5, "fünften": 5, "sechs": 6,
}


def _trailing_number(name):
    parts = _normalize(name).split()
    if not parts:
        return None
    last = parts[-1]
    return int(last) if last.isdigit() else ROMAN_NUMBERS.get(last)


def _roman_variant(name):
    """"Crusader Kings 3" -> "Crusader Kings III" (sonst None)."""
    parts = name.split()
    if len(parts) > 1 and parts[-1].isdigit() and int(parts[-1]) in ROMAN_BY_NUMBER:
        return " ".join(parts[:-1] + [ROMAN_BY_NUMBER[int(parts[-1])]])
    return None


def _clean_name(raw):
    name = raw.strip()
    while True:
        stripped = NAME_TAIL.sub("", name).strip()
        if stripped == name:
            return name
        name = stripped


QUESTION_WORDS = (
    "was", "wer", "wie", "wo", "welche", "welcher", "welches", "wann", "warum", "wieso", "weshalb",
    "erzähl", "erklär", "kennst", "worum", "gibt",
)
UNVERIFIED_REPLIES = (
    "Da bin ich mir bei {name} gerade nicht sicher. Worum geht's da?",
    "Ich finde dazu gerade keinen eindeutigen Treffer.",
    "Zu {name} habe ich gerade keine sicheren Infos. Erzähl du mir mal!",
)


class TopicContext:
    def __init__(self):
        self._unverified_count = 0
        self.reset()

    def reset(self):
        self.game = None
        self.verified = False
        self.idle = 0
        self.current = None
        self.pending = None
        self.pending_ask = False
        self.clarification = None
        self.ambiguous = None
        self.stt_pending = None
        self.stt_question = None
    def normalize(self, text, brain):
        """STT-Namenskorrektur (core.stt_correction). Unsicheres setzt stt_question."""
        self.stt_question = None
        if not isinstance(text, str) or not text.strip():
            return text
        pending, self.stt_pending = self.stt_pending, None
        if pending:
            chosen = self._answer_stt_question(text, pending)
            if chosen:
                return chosen
        result, question = analyze_stt_text(
            text, brain, active_game=self.game, extra_names=[*(self.pending or []), *([self.game] if self.game else [])]
        )
        result = self._keep_active_game(result)
        if question and not self.pending:
            self.stt_pending = question
            self.stt_question = question_for(question["options"])
        return result

    def _keep_active_game(self, text):
        """Ein schlechter STT-Treffer ("Rosetta Kings") ersetzt nie ein aktives Spiel mit Ziffer."""
        game = self.game
        if not game or _trailing_number(game) is None:
            return text
        base = " ".join(game.split()[:-1])
        pattern = re.compile(rf"\b{re.escape(base)}\b(?!\s+(?:\d+|[IVXivx]+|zwei|drei|vier|fünf)\b)", re.IGNORECASE)
        return pattern.sub(game, text)

    @staticmethod
    def _answer_stt_question(reply, pending):
        """Die korrigierte Fassung der ursprünglichen Frage, wenn Miki bestätigt oder wählt."""
        options = pending["options"]
        padded = f" {_normalize(reply)} "
        named = [o for o in options if f" {_normalize(o)} " in padded]
        if len(named) == 1:
            return pending["variants"][named[0]]
        words = set(padded.split())
        if len(options) == 1 and words & YES_WORDS and not words & NO_WORDS:
            return pending["variants"][options[0]]
        return None

    def _find_known(self, text, brain):
        try:
            found = brain.knowledge.mentioned_in(text)
        except Exception:
            found = []
        if found:
            return found[0]["name"], True
        try:
            for game in brain.games:
                if brain._mentions(game, text):
                    return game["game"], bool(game.get("appid"))
        except Exception:
            pass
        return None, False

    def detect_game(self, text, brain):
        """(Spielname, verifiziert) oder (None, False). Erfindet nichts."""
        name, verified = self._find_known(text, brain)
        if name:
            return name, verified
        match = CUE.search(text) or PLAYED_CUE.search(text)
        if not match:
            return None, False
        candidate = _clean_name(match.group("name"))
        played = PLAYED_CUE.search(text) is not None
        try:
            if not brain._valid_name(candidate):
                return None, False
            info = brain.knowledge.find_game(candidate)
            roman = _roman_variant(candidate)
            if not info and roman:
                info = brain.knowledge.find_game(roman)
            exact = bool(info) and _normalize(candidate) in brain.knowledge.names_for(info)
            if not exact and (GAME_WORD.search(text) or played or len(_normalize(candidate)) >= 8):
                options = brain.knowledge.candidates(candidate)
                if len(options) > 1 and not (info and roman and _normalize(roman) in brain.knowledge.names_for(info)):
                    self.ambiguous = options
                    return None, False
        except Exception:
            return None, False
        if info:
            return info["name"], True
        if GAME_WORD.search(text) or played:
            return candidate, False
        return None, False

    # ---------- Bezüge ----------
    def substitute(self, text, game):
        """Ersetzt Verweise wie "das Spiel", "es", "davon" durch den Spielnamen."""
        result = GAME_REFERENCE.sub(game, text)
        result = ABOUT_REFERENCE.sub(
            lambda m: ("von " if m.group(0).lower() == "davon" else "über ") + game, result
        )
        result = DAS_AFTER_VERB.sub(lambda m: m.group("head") + game, result)
        result = DAS_START.sub(lambda m: m.group("lead") + game, result)

        def replace_es(match):
            before = result_text[: match.start()].split()
            if not before or before[-1].lower().strip(",") in ES_BLOCKERS:
                return match.group("es")
            return game

        result_text = result
        return ES_WORD.sub(replace_es, result)

    # ---------- Ablauf pro Nachricht ----------
    def set_game(self, name, verified):
        self.game = name
        self.verified = bool(verified)
        self.idle = 0
        self.current = name

    def _choose(self, text, options):
        """Wählt aus den Rückfrage-Optionen anhand von Zahl, Ordinalzahl oder Namen."""
        tokens = [_normalize(t[0]) for t in _tokens(text)]
        padded = f" {' '.join(tokens)} "
        contained = [o for o in options if f" {_normalize(o)} " in padded]
        if len(contained) == 1:
            return contained[0]
        if len(tokens) > 6:
            return None
        genre = [o for o in options if o.endswith("-Genre")]
        if genre and {"genre", "allgemein", "nein"} & set(tokens):
            return genre[0]
        wanted = {NUMBER_WORDS[t] for t in tokens if t in NUMBER_WORDS}
        matches = [o for o in options if _trailing_number(o) in wanted]
        if len(matches) == 1:
            return matches[0]
        if not any(_trailing_number(o) for o in options):
            for token in tokens:
                index = ORDINAL_WORDS.get(token)
                if index and index <= len(options):
                    return options[index - 1]
        return None

    @staticmethod
    def _question(options):
        words = [o.split() for o in options]
        shared = 0
        while all(len(w) > shared + 1 and w[shared].lower() == words[0][shared].lower() for w in words):
            shared += 1
        if shared:
            parts = [options[0]] + [" ".join(w[shared:]) for w in words[1:]]
        else:
            parts = list(options)
        joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " oder " + parts[-1]
        return f"Meinst du {joined}?"

    def process(self, text, brain):
        """Gibt den Text mit aufgelösten Bezügen zurück und pflegt das aktive Spiel."""
        self.current = None
        self.clarification = None
        self.ambiguous = None
        options, self.pending = self.pending, None
        if options:
            chosen = self._choose(text, options)
            if chosen and chosen.endswith("-Genre"):
                self.game = None
                self.verified = False
                return text
            if chosen:
                try:
                    info = brain.knowledge.find_game(chosen)
                except Exception:
                    info = None
                name = info["name"] if info else chosen
                self.set_game(name, bool(info))
                return f"Was ist {name} für ein Spiel?" if self.pending_ask else text
        if self.game is not None and self.idle >= MAX_IDLE_TURNS:
            self.game = None
        name, verified = self.detect_game(text, brain)
        if self.ambiguous:
            self.pending = self.ambiguous
            self.pending_ask = PLAYED_CUE.search(text) is None
            self.clarification = self._question(self.ambiguous)
            return text
        if name:
            self.set_game(name, verified)
            return text
        if self.game is None:
            return text
        resolved = self.substitute(text, self.game)
        if resolved != text:
            self.idle = 0
            self.current = self.game
            return resolved
        self.idle += 1
        return text

    def unverified_reply(self, user_text):
        """Feste Antwort, wenn nach einem Spiel OHNE verifizierte Daten gefragt wird (kein Raten)."""
        if not self.current or self.verified:
            return None
        padded = f" {_normalize(user_text)} "
        asks = "?" in user_text or any(f" {w} " in padded for w in QUESTION_WORDS)
        if not asks:
            return None
        self._unverified_count += 1
        options = UNVERIFIED_REPLIES
        return options[self._unverified_count % len(options)].format(name=self.current)

    def after_learn(self, learned):
        """Ein neu gelerntes Spiel wird zum aktiven Spiel."""
        if isinstance(learned, dict) and learned.get("type") == "game_memory" and learned.get("game"):
            self.set_game(learned["game"], bool(learned.get("appid")))

    def lookup_text(self, user_text):
        return f"{user_text} {self.current}" if self.current else user_text

    def prompt_block(self):
        if not self.game:
            return ""
        lines = [
            "ACTIVE TOPIC:",
            f'You and Miki are currently talking about the game "{self.game}". Words like "es", '
            '"das", "das Spiel", "davon" and "darüber" refer to it unless the message clearly '
            "means something else.",
        ]
        if not self.verified:
            lines.append(
                "No verified information about this game is available. Do not guess or hint at its "
                "genre, setting, developer or content, and do not say you heard anything about it. "
                'Say you are not sure ("Da bin ich mir gerade nicht sicher.") and ask Miki what it is about.'
            )
        return "\n".join(lines)
