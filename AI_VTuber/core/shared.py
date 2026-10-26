"""Erkennung gemeinsamer Gesprächsthemen und Running Gags (regelbasiert, ohne Modell)."""
import re

GAG_PRONUNCIATION = re.compile(
    r"\bdu (?:sprichst|sagst) (?P<t>[\w-]{2,30}) (?:\w+ ){0,2}"
    r"(?:komisch|falsch|lustig|seltsam|merkwürdig|witzig)(?: aus)?\b",
    re.IGNORECASE,
)
EXPLICIT_TOPIC = re.compile(
    r"\bwir (?:reden|sprechen|quatschen|labern) (?:\w+ )?(?:schon wieder|mal wieder|wieder|"
    r"immer noch|ständig|dauernd|oft) (?:\w+ )?(?:über|von) (?P<t>[^.!?,;]{2,40})",
    re.IGNORECASE,
)
SHARED_QUESTION = re.compile(
    r"\b(?:was haben wir (?:\w+ )?(?:schon )?(?:oft |öfter )?(?:\w+ )?besprochen|"
    r"wor(?:ü|ue)ber (?:reden|sprechen) wir (?:\w+ )?(?:oft|öfter|ständig|dauernd)|"
    r"was besprechen wir (?:oft|öfter)|unsere (?:insider|running gags?)|"
    r"haben wir (?:\w+ )?(?:insider|running gags?)|was ist unser (?:running gag|insider))\b"
)
GAG_QUESTION = re.compile(r"\b(?:insider|running gags?)\b")
SHARED_FORGET = re.compile(r"\b(?:insider|running gags?|gag)\b")
GENERIC_TOPIC_WORDS = {"aussprache", "gag", "insider", "running", "unser", "unseren", "über", "ueber"}


def extract_gag(user_text):
    """Gibt z.B. "YouTube-Aussprache" für "Du sprichst YouTube immer noch komisch aus" zurück."""
    match = GAG_PRONUNCIATION.search(user_text)
    if not match:
        return None
    word = match.group("t").strip("-")
    if word.casefold() in {"es", "das", "mich", "dich", "alles", "immer", "wieder"}:
        return None
    return f"{word}-Aussprache"


def extract_explicit_topic(user_text):
    match = EXPLICIT_TOPIC.search(user_text)
    if not match:
        return None
    topic = " ".join(match.group("t").split())
    if not topic or len(topic.split()) > 4:
        return None
    return topic
