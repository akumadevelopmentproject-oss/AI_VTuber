"""Natürliche Gesprächsführung: Wiederholungen vermeiden, Stilregeln, Kontext-Budget.

Alles lebt im Arbeitsspeicher; es wird nichts dauerhaft gespeichert.
"""
import re
from collections import deque

from core.memory import _normalize

MAX_RECENT = 10
GOAL_COOLDOWN = 5
GAG_COOLDOWN = 4
NAME_WINDOW = 3
QUESTION_WINDOW = 3
CONTEXT_BUDGET = 3600
MIN_GOAL_BUDGET = 500
LONG_REQUEST_CHARS = 110

ASSISTANT_SENTENCES = re.compile(
    r"(?:wie kann ich (?:dir|ihnen|euch) (?:heute |sonst |weiter )?helfen|"
    r"ich stehe (?:dir|ihnen) (?:gerne |gern )?(?:jederzeit )?zur verf(?:ü|ue)gung|"
    r"wenn du (?:noch )?(?:weitere|andere|mehr) fragen hast|"
    r"zögere nicht|z(?:ö|oe)gere nicht,? (?:mich )?zu fragen|"
    r"ich bin (?:nur )?(?:eine|ein) (?:ki|k(?:ü|ue)nstliche intelligenz|sprachmodell|assistent)|"
    r"als (?:ki|sprachmodell|assistent)|"
    r"nat(?:ü|ue)rlich! gerne helfe ich|gerne helfe ich (?:dir|ihnen)|"
    r"gibt es (?:sonst )?noch etwas,? (?:wobei|womit) ich (?:dir )?helfen)",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
META_TAG = re.compile(r"\[\s*[A-ZÄÖÜ][A-ZÄÖÜ _-]{2,24}\s*\]\s*")
ACTION_TEXT = re.compile(r"\*[^*\n]{1,30}\*\s*")
NEGATION = re.compile(r"\b(?:nie|niemals|nicht|noch nie|kein\w*|nichts)\b", re.IGNORECASE)
EXPERIENCE_CLAIM = re.compile(
    r"\bich\s+(?:habe|hab)\b[^.!?]*\b(?:gespielt|gezockt|gesehen|geschaut|gelesen|ausprobiert|"
    r"erlebt|besucht|gekauft|gegessen|getestet|angeschaut|recherchiert|geh(?:ö|oe)rt)\b"
    r"|\bich\s+(?:war|bin)\s+(?:gestern|letzte\w*|schon\s+mal|mal)\b"
    r"|\bich\s+war\s+(?:dort|da|damals)\b"
    r"|\bgestern\s+(?:habe|hab|war|bin)\s+ich\b"
    r"|\bwie\s+in\s+(?:den|dem)\s+(?:videos?|trailern?|streams?)\b",
    re.IGNORECASE,
)
IDENTITY_QUESTION = re.compile(
    r"\b(?:wer bist du|wie hei(?:ß|ss)t du|stell dich vor|was bist du|wer ist luna)\b"
)
PERSONAL_CUES = re.compile(
    r"\b(?:was wei(?:ß|ss)t du (?:alles )?(?:ü|ue)ber mich|was hast du dir gemerkt|"
    r"(?:passt|passen|gef(?:ä|ae)llt|gef(?:ä|ae)llen) (?:zu |mir|mich)|f(?:ü|ue)r mich|zu mir|"
    r"empfiehl\w*|vorschlag|vorschl(?:ä|ae)ge|was soll ich|welche[sr]? \w+ (?:passt|soll)|"
    r"was mag ich|was hasse ich|was feiere ich|meine (?:vorlieben|interessen|hobbys)|"
    r"kennst du mich|erinnerst du dich)\b"
)
DETAIL_CUES = re.compile(
    r"\b(?:erkl(?:ä|ae)r\w*|ausf(?:ü|ue)hrlich\w*|im detail|genau|schritt f(?:ü|ue)r schritt|"
    r"erz(?:ä|ae)hl\w* (?:mir )?(?:mehr|was)|beschreib\w*)\b"
)
STOPWORDS = {
    "der", "die", "das", "und", "ist", "ich", "du", "mit", "von", "für", "fuer", "ein", "eine", "mag",
    "liebt", "miki", "mikis", "mein", "meine", "wie", "was", "wer", "wo", "auch", "nicht", "gern",
    "gerne", "sehr", "über", "ueber", "hat", "habe", "bin", "bist", "dir", "mir", "mich", "dich",
    "luna", "wir", "ihr", "sie", "es", "zu", "im", "in", "auf", "an", "den", "dem", "des", "als",
}


def _words(text):
    return [w for w in _normalize(text).split() if len(w) >= 3 and w not in STOPWORDS]


def _overlap(a, b):
    """Wortstamm-Vergleich über die ersten 5 Zeichen (Katze/Katzen, Spiel/Spiele)."""
    stems = {w[:5] for w in _words(b)}
    return any(w[:5] in stems for w in _words(a))


def memory_relevant(memory_text, user_text):
    if not isinstance(memory_text, str) or not isinstance(user_text, str):
        return False
    normalized = _normalize(user_text)
    if PERSONAL_CUES.search(normalized):
        return True
    return _overlap(memory_text, user_text)


def filter_memories(texts, user_text, limit=6, personal_limit=20):
    normalized = _normalize(user_text) if isinstance(user_text, str) else ""
    cap = personal_limit if PERSONAL_CUES.search(normalized) else limit
    return [t for t in texts if memory_relevant(t, user_text)][:cap]


def strip_assistant_phrases(answer):
    """Entfernt Standard-Assistenten-Sätze; behält die Antwort, wenn sonst nichts übrig bliebe."""
    if not isinstance(answer, str) or not ASSISTANT_SENTENCES.search(answer):
        return answer
    sentences = SENTENCE_SPLIT.split(answer.strip())
    kept = [s for s in sentences if not ASSISTANT_SENTENCES.search(s)]
    return " ".join(kept).strip() or answer


def clean_output(answer):
    """Entfernt Meta-Tags ([SCHERZE]), Aktionen (*lacht*) und erfundene Erlebnisse."""
    if not isinstance(answer, str):
        return answer
    text = META_TAG.sub("", answer)
    text = ACTION_TEXT.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    sentences = SENTENCE_SPLIT.split(text)
    kept = [s for s in sentences if not _invented_experience(s)]
    if kept:
        text = " ".join(kept)
    return text.strip() or answer.strip()


def _invented_experience(sentence):
    if NEGATION.search(sentence):
        return False
    return bool(EXPERIENCE_CLAIM.search(sentence))


def _opening(answer, words=3):
    tokens = re.findall(r"[\w'-]+", answer, re.UNICODE)
    return " ".join(tokens[:words])


def _last_question(answer):
    sentences = [s for s in SENTENCE_SPLIT.split(answer.strip()) if s]
    for sentence in reversed(sentences):
        if sentence.endswith("?"):
            return sentence[:70]
    return ""


def _similar(a, b):
    first, second = set(_normalize(a).split()), set(_normalize(b).split())
    if not first or not second:
        return False
    return len(first & second) / len(first | second) >= 0.8


class ResponseHistory:
    def __init__(self, size=MAX_RECENT):
        self.items = deque(maxlen=size)

    def add(self, answer):
        if isinstance(answer, str) and answer.strip():
            self.items.append(answer.strip())

    def recent(self, count=None):
        items = list(self.items)
        return items if count is None else items[-count:]

    def is_duplicate(self, answer):
        return any(_similar(answer, old) for old in self.items)

    def mentions(self, topic, window):
        return any(_normalize(topic) and f" {_normalize(topic)} " in f" {_normalize(a)} " for a in self.recent(window))

    def used_name_recently(self):
        return any(re.search(r"\bMiki\b", a) for a in self.recent(NAME_WINDOW))

    def asked_recently(self):
        return any(a.rstrip().endswith("?") for a in self.recent(QUESTION_WINDOW))

    def openings(self):
        seen = []
        for answer in reversed(self.recent(5)):
            opening = _opening(answer)
            if opening and opening.casefold() not in (o.casefold() for o in seen):
                seen.append(opening)
        return seen

    def guidance_block(self):
        if not self.items:
            return ""
        lines = [f'- Do not start with "{o}..."' for o in self.openings()[:4]]
        for answer in reversed(self.recent(3)):
            question = _last_question(answer)
            if question:
                lines.append(f'- Do not ask a similar question again: "{question}"')
                break
        lines.append("- Do not mention the same user fact again unless it is relevant.")
        return (
            "RECENT RESPONSE GUIDANCE:\nAvoid repeating these recent phrases or ideas:\n"
            + "\n".join(lines)
        )


def style_block(user_text, history):
    normalized = _normalize(user_text) if isinstance(user_text, str) else ""
    detailed = len(user_text or "") > LONG_REQUEST_CHARS or bool(DETAIL_CUES.search(normalized))
    rules = [
        "You are Luna, a casual VTuber, not a generic support assistant. Never say things like "
        '"Wie kann ich dir helfen?", "Ich stehe dir gerne zur Verfügung", '
        '"Wenn du weitere Fragen hast" or "Ich bin eine KI".',
        "Speak natural, casual German like in a Twitch or Discord chat, always with informal "
        '"du". NEVER address Miki with "Sie", "Ihnen" or "Ihr". Do not force slang or invent idioms.',
        "Answer directly on the concrete content of Miki's message. Avoid empty filler such as "
        '"Das ist ein interessantes Thema", "Ich mag es, wenn man das Gefühl hat ...", '
        '"Wenn du willst, kann ich dir helfen" or "Was halten Sie davon?".',
        "Never output meta tags like [SCHERZE], [IRONISCH], [FRÖHLICH] or [TRAURIG] and no stage "
        "directions like *lacht* or *seufzt*. Express emotion through wording only.",
        "You are an AI character: never claim real experiences. No \"Ich habe X gespielt / gesehen / "
        'gelesen / ausprobiert", no "gestern ...", no "ich war dort". Give opinions about the '
        'concept instead ("Das wirkt auf mich ...", "Nach dem, was ich darüber weiß ...").',
        "Never invent facts about games, films or other things. If you are not sure, say so "
        '(for example "Da bin ich mir gerade nicht sicher.").',
        "Length: normally 1 to 3 short sentences."
        + (" Miki asked for details, so a longer answer is fine, without monologues." if detailed else ""),
        "Vary your wording naturally; do not reuse the same agreement phrase or sentence opening.",
        "Use the name Miki only occasionally (direct address, emotion, important answers), never in every sentence.",
        "Mood only changes tone; never state your mood, energy or values.",
    ]
    if history.used_name_recently():
        rules.append('Do not use the name "Miki" in this answer.')
    if not IDENTITY_QUESTION.search(normalized):
        rules.append('Do not introduce yourself or say "Ich bin Luna".')
    if history.asked_recently():
        rules.append("Do not end with a question this time.")
    else:
        rules.append("A follow-up question is optional; only ask if it truly fits.")
    return "STYLE GUIDANCE:\n" + "\n".join(f"- {r}" for r in rules)


def assemble_blocks(ranked, budget=CONTEXT_BUDGET):
    """ranked: Liste von Text-Blöcken nach Priorität (wichtigster zuerst). Gibt die passenden zurück."""
    chosen, used = [], 0
    for block in ranked:
        if not block:
            continue
        if used + len(block) > budget and chosen:
            continue
        chosen.append(block)
        used += len(block)
    return chosen
