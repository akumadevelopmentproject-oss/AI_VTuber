"""Automatische Erkennung allgemeiner Interessen und Vorlieben (ohne Sprachmodell).

Nur klare, persönliche Aussagen in der Ich-Form werden erkannt. Fragen,
Unsicheres, Situatives, Vages und sensible Themen werden verworfen.
"""
import re

from core.memory import LEADING_FILLER, QUESTION_START, UNCERTAIN, _normalize
from core.safety import is_blocked_input, is_blocked_output

INTEREST_TYPES = ("interest", "preference", "dislike", "favorite")
SENTIMENTS = ("very_positive", "positive", "neutral", "negative", "very_negative")

SENSITIVE = re.compile(
    r"passwor|kennwort|\bpin\b|\biban\b|kreditkarte|kontonummer|paypal|telefon|handynummer|"
    r"adresse|stra(?:ß|ss)e|\bwohn|e-?mail|krank|diagnos|diabet|depress|therapie|medikament|"
    r"tablette|schwanger|krebs|allergi|\barzt|\bärzt|\baerzt|gesundheit|behinder|"
    r"\bsex|porn|erotik|fetisch|nackt|religi|\bgott\b|islam|\bchrist(?:en|entum|lich)|muslim|"
    r"\bjuden?\b|j(?:ü|ue)disch|atheis|kirche|politik|partei|\bwahl|\bafd\b|\bcdu\b|\bspd\b|"
    r"\bfdp\b|rechtsextrem|linksextrem|suizid|selbstmord|drogen|schulden|gehalt|einkommen|"
    r"\d{5,}|@",
    re.IGNORECASE,
)

VAGUE_FIRST = {
    "das", "es", "dies", "diese", "dieses", "dich", "dir", "mich", "mir", "sie", "ihn", "ihm",
    "so", "was", "alles", "nichts", "etwas", "vieles", "man", "uns", "euch", "wenn", "weil",
    "dass", "ob", "zu", "mit", "in", "am", "bei", "auf", "an", "um", "und", "oder", "aber",
    "den", "dem", "der", "die", "ein", "eine", "einen", "nicht", "nur", "auch", "ja", "nein",
}
CLAUSE_WORDS = {
    "ich", "du", "dass", "weil", "wenn", "ob", "dich", "dir", "mich", "mir", "heute", "gestern",
    "morgen", "gerade", "eben", "vorhin", "nicht", "kein", "keine", "jetzt",
}
ARTICLES = {"der", "die", "das", "den", "dem", "ein", "eine", "einen", "meine", "diese"}
STRONG_TAIL = {"sehr", "total", "mega", "richtig", "voll", "wirklich", "so sehr", "extrem"}
TAIL = re.compile(
    r"\s+(?P<w>sehr|total|richtig|mega|echt|voll|wirklich|so sehr|extrem|auch|ziemlich|gerne|gern)$",
    re.IGNORECASE,
)
PAST = re.compile(r"\b(?:fr(?:ü|ue)her|damals)\b", re.IGNORECASE)
LABEL = re.compile(r"^[^\W\d_][\w\- äöüÄÖÜß'&/]*$", re.UNICODE)

STRONG_ADJ = {"geil", "genial", "großartig", "grossartig", "mega"}
STRONG_INT = {"richtig", "total", "voll", "mega", "absolut", "so"}
POS_ADJ = r"geil|super|toll|klasse|gro(?:ß|ss)artig|genial|cool|spannend|stark|sch(?:ö|oe)n|lustig|interessant"
NEG_ADJ = r"doof|bl(?:ö|oe)d|langweilig|schlecht|nervig|furchtbar|(?:ö|oe)de|schrecklich|mies"
INT = r"(?:(?P<i>richtig|total|voll|echt|wirklich|mega|absolut|so) )?"
VERBS = (
    r"h(?:ö|oe)re|schaue|sehe|lese|esse|trinke|koche|male|zeichne|backe|sammle|"
    r"mache|reise|tanze|singe|h(?:ö|oe)r|schau|lese"
)

CHANGE = re.compile(
    r"^ich (?:mochte|liebte) (?P<t>.+?)(?: fr(?:ü|ue)her| damals)?,? aber "
    r"(?:mittlerweile|inzwischen|heute|jetzt)(?: mag ich (?:sie|es|ihn|das))? nicht mehr$",
    re.IGNORECASE,
)
FAVORITE = re.compile(
    r"^mein lieblings-?(?P<k>[\wäöüß]+) ist (?:der |die |das )?(?P<t>.+)$", re.IGNORECASE
)
PATTERNS = [
    (re.compile(r"^ich (?:liebe|vergöttere) (?P<t>.+)$", re.IGNORECASE), "interest", "very_positive"),
    (re.compile(r"^ich hasse (?P<t>.+)$", re.IGNORECASE), "dislike", "very_negative"),
    (
        re.compile(r"^ich kann (?P<t>.+?) (?:gar |überhaupt )?nicht ausstehen$", re.IGNORECASE),
        "dislike",
        "very_negative",
    ),
    (re.compile(r"^ich feiere? (?P<t>.+)$", re.IGNORECASE), "interest", "very_positive"),
    (re.compile(r"^ich stehe auf (?P<t>.+)$", re.IGNORECASE), "interest", "very_positive"),
    (
        re.compile(r"^ich bin (?:ein )?(?:gro(?:ß|ss)er |riesiger )?fan von (?P<t>.+)$", re.IGNORECASE),
        "interest",
        "very_positive",
    ),
    (
        re.compile(
            r"^ich interessiere mich (?:sehr |total |besonders )?f(?:ü|ue)r (?P<t>.+)$", re.IGNORECASE
        ),
        "interest",
        "positive",
    ),
    (
        re.compile(rf"^ich (?:{VERBS}) (?:sehr |total |richtig )?(?:gern|gerne) (?P<t>.+)$", re.IGNORECASE),
        "preference",
        "positive",
    ),
    (
        re.compile(rf"^ich finde (?P<t>.+?) {INT}(?:{NEG_ADJ})$", re.IGNORECASE),
        "dislike",
        "negative",
    ),
    (
        re.compile(rf"^ich finde (?P<t>.+?) {INT}(?P<a>{POS_ADJ})$", re.IGNORECASE),
        "interest",
        "positive",
    ),
    (
        re.compile(r"^ich mag (?P<t>.+?) (?P<n>gar |überhaupt )?nicht(?: mehr)?$", re.IGNORECASE),
        "dislike",
        "negative",
    ),
    (re.compile(r"^ich mag (?P<t>.+)$", re.IGNORECASE), "interest", "positive"),
]


def _clean_topic(raw):
    topic = re.sub(r"\s+", " ", raw).strip(" .,!…\"'")
    strong = False
    while True:
        match = TAIL.search(topic)
        if not match:
            break
        strong = strong or match.group("w").lower() in STRONG_TAIL
        topic = topic[: match.start()].strip()
    tokens = topic.split(" ")
    if len(tokens) > 1 and tokens[0].lower() in ARTICLES:
        topic = " ".join(tokens[1:])
    return topic, strong


def _valid_topic(topic):
    if not topic or len(topic) > 40 or not LABEL.match(topic):
        return False
    tokens = _normalize(topic).split()
    if not tokens or len(tokens) > 4 or tokens[0] in VAGUE_FIRST:
        return False
    if any(token in CLAUSE_WORDS for token in tokens) or len(_normalize(topic)) < 3:
        return False
    if SENSITIVE.search(topic) or is_blocked_input(topic) or is_blocked_output(topic):
        return False
    return True


def extract_interest(user_text):
    """Gibt {"type","topic","sentiment","confidence"[,"category"]} oder None zurück."""
    if not isinstance(user_text, str):
        return None
    text = user_text.strip()
    if not text or len(text) > 200 or "?" in text or SENSITIVE.search(text):
        return None
    text = LEADING_FILLER.sub("", text).strip().rstrip(".!…").strip()
    normalized = _normalize(text)
    if not normalized or QUESTION_START.match(normalized) or UNCERTAIN.search(text):
        return None

    change = CHANGE.match(text)
    if change:
        topic, _ = _clean_topic(change.group("t"))
        if _valid_topic(topic):
            return {"type": "dislike", "topic": topic, "sentiment": "negative", "confidence": 0.9}
        return None
    if PAST.search(text):
        return None

    favorite = FAVORITE.match(text)
    if favorite:
        category = favorite.group("k")
        topic, _ = _clean_topic(favorite.group("t"))
        if category.lower().startswith(("spiel", "game")) or not _valid_topic(topic):
            return None
        if len(category) > 25 or SENSITIVE.search(category):
            return None
        return {
            "type": "favorite",
            "topic": topic,
            "category": category[0].upper() + category[1:],
            "sentiment": "very_positive",
            "confidence": 0.92,
        }

    for pattern, kind, sentiment in PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        topic, strong = _clean_topic(match.group("t"))
        if not _valid_topic(topic):
            return None
        groups = match.groupdict()
        confidence = 0.93
        if "a" in groups and groups.get("a"):
            confidence = 0.9
            if (groups.get("i") or "").lower() in STRONG_INT or groups["a"].lower() in STRONG_ADJ:
                sentiment = "very_positive"
        elif groups.get("n"):
            sentiment = "very_negative"
        elif strong and sentiment == "positive":
            sentiment = "very_positive"
        return {"type": kind, "topic": topic, "sentiment": sentiment, "confidence": confidence}
    return None
