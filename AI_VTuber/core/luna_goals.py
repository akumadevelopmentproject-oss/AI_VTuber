import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from core.interests import SENSITIVE as INTEREST_SENSITIVE
from core.memory import _normalize
from core.safety import is_blocked_input, is_blocked_output

GOALS_PATH = Path(__file__).resolve().parent.parent / "data" / "luna_goals.json"
GOAL_TYPES = ("learn_about_topic", "ask_miki_about_topic", "revisit_topic")
STATUSES = ("active", "completed", "abandoned")
MAX_ACTIVE = 8
MAX_PROMPT_GOALS = 2
MIN_MENTIONS = 3
LOW_CURIOSITY_MIN_MENTIONS = 4
CURIOSITY_THRESHOLD = 0.75
HINT_INTERVAL = 8
GOAL_COOLDOWN = 5
PRIORITY_STEP = 0.05
MAX_COUNTERS = 60
ASK_PROGRESS = 0.8
OTHER_PROGRESS = 0.4

DESCRIPTION = {
    "learn_about_topic": "{t} interessiert mich inzwischen ziemlich. Ich würde gern mehr darüber verstehen.",
    "ask_miki_about_topic": "Ich würde gern verstehen, was dir an {t} so gefällt.",
    "revisit_topic": "Über {t} würde ich irgendwann gern nochmal mit dir reden.",
}
GUIDANCE = {
    "learn_about_topic": "Understand more about {t} through conversation with Miki.",
    "ask_miki_about_topic": "Ask Miki what he likes about {t}.",
    "revisit_topic": "Come back to talking about {t} with Miki.",
}
REASONS = {
    "learn_about_topic": "Luna hat daran ein eigenes Interesse entwickelt",
    "ask_miki_about_topic": "Miki spricht häufig positiv darüber",
    "revisit_topic": "Das Thema kommt häufig gemeinsam vor",
}

QUESTIONS = re.compile(
    r"^(?:luna )?(?:was interessiert dich(?: \w+)?|was (?:m(?:ö|oe)chtest|willst) du noch (?:wissen|erfahren)|"
    r"hast du (?:irgendwelche |eigene |noch )?ziele|"
    r"wor(?:ü|ue)ber w(?:ü|ue)rdest du (?:gern|gerne) mehr (?:erfahren|wissen)|"
    r"was w(?:ü|ue)rdest du (?:gern|gerne) (?:mehr )?(?:wissen|erfahren))$"
)
ABANDON_A = re.compile(
    r"^(?:luna )?(?:bitte )?interessier(?:e)? dich (?:nicht mehr|nicht l(?:ä|ae)nger) (?:f(?:ü|ue)r|an|um) (?P<t>.+)$",
    re.IGNORECASE,
)
ABANDON_B = re.compile(
    r"^(?:luna )?(?:bitte )?(?:vergiss|l(?:ö|oe)sch\w*|streich\w*) (?:dein|deine|das) ziel(?:e)? "
    r"(?:(?:ü|ue)ber|zu|bei|f(?:ü|ue)r|von) (?P<t>.+)$",
    re.IGNORECASE,
)
REVISIT = re.compile(
    r"^(?:luna )?(?:lass|lasst) uns (?:sp(?:ä|ae)ter|irgendwann|nachher) (?:nochmal|noch mal|nochmals|wieder|mal)"
    r"(?: wieder)? (?:(?:ü|ue)ber|von) (?P<t>.+?) (?:reden|sprechen|quatschen)$",
    re.IGNORECASE,
)


def _now():
    return datetime.now().replace(microsecond=0).isoformat()


def _clamp(value, default):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return default
    return round(min(1.0, max(0.0, float(value))), 3)


def _key(topic):
    return _normalize(topic)


def _contains(text, topic):
    topic_n = _normalize(topic)
    return bool(topic_n) and f" {topic_n} " in f" {_normalize(text)} "


def _label(topic):
    if not isinstance(topic, str):
        return None
    topic = " ".join(topic.split()).strip(" .,;:!?\"'")
    if not topic or len(topic) > 40 or len(topic.split()) > 4 or re.search(r"\d{5,}", topic):
        return None
    if INTEREST_SENSITIVE.search(topic) or is_blocked_input(topic) or is_blocked_output(topic):
        return None
    return topic


def _clean_goal(item):
    if not isinstance(item, dict) or item.get("type") not in GOAL_TYPES:
        return None
    topic = _label(item.get("topic"))
    if not topic:
        return None
    reason = item.get("reason")
    now = _now()
    return {
        "type": item["type"],
        "topic": topic,
        "reason": reason[:120] if isinstance(reason, str) else REASONS[item["type"]],
        "priority": _clamp(item.get("priority"), 0.5),
        "status": item.get("status") if item.get("status") in STATUSES else "active",
        "progress": _clamp(item.get("progress"), 0.0),
        "hinted": item.get("hinted") is True,
        "created_at": item.get("created_at") if isinstance(item.get("created_at"), str) else now,
        "updated_at": item.get("updated_at") if isinstance(item.get("updated_at"), str) else now,
    }


def _empty():
    return {"goals": [], "mentions": {}, "conversations": 0, "last_hint_at": None}


class LunaGoals:
    def __init__(self, path=GOALS_PATH):
        self.path = Path(path)
        self._invalid = False
        self.data = self.load()
        if not self.path.exists():
            self.save()

    def load(self):
        data = _empty()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return data
        except (OSError, ValueError, UnicodeError):
            self._invalid = True
            return data
        if not isinstance(raw, dict):
            self._invalid = True
            return data
        seen = {}
        goals = raw.get("goals")
        for item in goals if isinstance(goals, list) else []:
            clean = _clean_goal(item)
            if clean:
                key = (_key(clean["topic"]), clean["status"] == "active")
                seen.setdefault(key, clean)
        data["goals"] = list(seen.values())
        mentions = raw.get("mentions")
        for key, value in (mentions.items() if isinstance(mentions, dict) else []):
            if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                data["mentions"][key] = value
        count = raw.get("conversations")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            data["conversations"] = count
        last = raw.get("last_hint_at")
        if isinstance(last, int) and not isinstance(last, bool) and last >= 0:
            data["last_hint_at"] = last
        return data

    def save(self):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self._invalid and self.path.exists():
                self.path.with_suffix(".json.bak").write_bytes(self.path.read_bytes())
                self._invalid = False
            descriptor, temporary = tempfile.mkstemp(prefix=".goals.", suffix=".tmp", dir=self.path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=4)
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

    @property
    def goals(self):
        return self.data["goals"]

    @property
    def active(self):
        return [g for g in self.goals if g["status"] == "active"]

    def find(self, topic, status=None):
        wanted = _key(topic)
        for goal in self.goals:
            if status and goal["status"] != status:
                continue
            if _key(goal["topic"]) == wanted:
                return goal
        return None

    def _match(self, topic):
        """Aktives Ziel zum Thema (auch bei Teilübereinstimmung)."""
        exact = self.find(topic, "active")
        if exact:
            return exact
        topic_n = _key(topic)
        if not topic_n:
            return None
        for goal in self.active:
            goal_n = _key(goal["topic"])
            if f" {topic_n} " in f" {goal_n} " or f" {goal_n} " in f" {topic_n} ":
                return goal
        return None

    def add_goal(self, goal_type, topic, reason=None, priority=0.5, force=False):
        topic = _label(topic)
        if goal_type not in GOAL_TYPES or not topic:
            return None
        existing = self.find(topic, "active")
        if existing:
            return self.update_goal(topic, priority=priority, reason=reason)
        closed = self.find(topic)
        if closed and not force:
            return None
        if len(self.active) >= MAX_ACTIVE:
            if not force:
                return None
            lowest = min(self.active, key=lambda g: g["priority"])
            lowest["status"] = "abandoned"
            lowest["updated_at"] = _now()
        now = _now()
        self.goals[:] = [g for g in self.goals if not (_key(g["topic"]) == _key(topic) and g["status"] != "active")]
        goal = {
            "type": goal_type,
            "topic": topic,
            "reason": (reason or REASONS[goal_type])[:120],
            "priority": _clamp(priority, 0.5),
            "status": "active",
            "progress": 0.2,
            "hinted": False,
            "created_at": now,
            "updated_at": now,
        }
        self.goals.append(goal)
        self.save()
        return goal

    def update_goal(self, topic, priority=None, progress=None, reason=None):
        goal = self.find(topic, "active")
        if not goal:
            return None
        if priority is not None:
            target = _clamp(priority, goal["priority"])
            delta = max(-PRIORITY_STEP, min(PRIORITY_STEP, target - goal["priority"]))
            goal["priority"] = _clamp(goal["priority"] + delta, goal["priority"])
        if progress is not None:
            goal["progress"] = max(goal["progress"], _clamp(progress, goal["progress"]))
        if isinstance(reason, str) and reason:
            goal["reason"] = reason[:120]
        goal["updated_at"] = _now()
        if goal["progress"] >= 1.0:
            return self.complete_goal(topic)
        self.save()
        return goal

    def complete_goal(self, topic):
        goal = self.find(topic, "active")
        if not goal:
            return None
        goal.update(status="completed", progress=1.0, hinted=False, updated_at=_now())
        self.save()
        return goal

    def abandon_goal(self, topic):
        goal = self._match(topic)
        if not goal:
            return None
        goal.update(status="abandoned", hinted=False, updated_at=_now())
        self.save()
        return goal

    def _candidates(self, user_text, brain):
        found = {}

        def add(topic, kind, importance=0.0, strength=0.0):
            topic = _label(topic)
            if not topic or not _contains(user_text, topic):
                return
            key = _key(topic)
            old = found.get(key)
            if old is None:
                found[key] = {"topic": topic, "kind": kind, "importance": importance, "strength": strength}
            else:
                old["importance"] = max(old["importance"], importance)
                old["strength"] = max(old["strength"], strength)

        positive = ("positive", "very_positive")
        for game in brain.games:
            if game["sentiment"] in positive:
                add(game["game"].split(":")[0].strip(), "ask_miki_about_topic")
        for item in brain.user_items:
            if item["sentiment"] in positive:
                add(item["topic"], "ask_miki_about_topic")
        for item in brain.interests:
            if item.get("strength", 0) >= 0.3:
                add(item["topic"], "learn_about_topic", strength=item["strength"])
        for item in brain.shared:
            add(item["topic"], "revisit_topic", importance=item["importance"])
        return found

    def _priority(self, count, candidate, curiosity):
        target = (
            0.3
            + 0.04 * min(count, 8)
            + 0.2 * candidate["importance"]
            + 0.2 * candidate["strength"]
            + 0.1 * curiosity
        )
        return _clamp(target, 0.5)

    def _resolve_hints(self, user_text):
        hinted = [g for g in self.active if g["hinted"]]
        if not hinted:
            return
        words = len(_normalize(user_text).split())
        answered = "?" not in user_text and words >= 3
        for goal in hinted:
            goal["hinted"] = False
            if answered:
                step = ASK_PROGRESS if goal["type"] == "ask_miki_about_topic" else OTHER_PROGRESS
                goal["progress"] = _clamp(goal["progress"] + step, goal["progress"])
                goal["updated_at"] = _now()
                if goal["progress"] >= 1.0:
                    goal.update(status="completed", progress=1.0)

    def observe(self, user_text, brain, curiosity=0.7):
        """Pro normaler, nicht blockierter Nachricht aufrufen (Safety liegt davor)."""
        if not isinstance(user_text, str) or not user_text.strip() or len(user_text) > 300:
            return False
        self.data["conversations"] += 1
        self._resolve_hints(user_text)
        if not INTEREST_SENSITIVE.search(user_text):
            needed = MIN_MENTIONS if curiosity >= CURIOSITY_THRESHOLD else LOW_CURIOSITY_MIN_MENTIONS
            counters = self.data["mentions"]
            for key, cand in self._candidates(user_text, brain).items():
                counters[key] = counters.get(key, 0) + 1
                if counters[key] < needed:
                    continue
                priority = self._priority(counters[key], cand, curiosity)
                if self.find(cand["topic"], "active"):
                    self.update_goal(cand["topic"], priority=priority)
                elif self.find(cand["topic"]) is None:
                    self.add_goal(cand["kind"], cand["topic"], REASONS[cand["kind"]], priority)
            if len(counters) > MAX_COUNTERS:
                keep = sorted(counters.items(), key=lambda kv: kv[1], reverse=True)[:MAX_COUNTERS]
                self.data["mentions"] = dict(keep)
        return self.save()

    def relevant_goals(self, user_text):
        matches = [g for g in self.active if _contains(user_text, g["topic"])]
        matches.sort(key=lambda g: g["priority"], reverse=True)
        return matches[:MAX_PROMPT_GOALS]

    def _hint_due(self):
        last = self.data["last_hint_at"]
        return last is None or self.data["conversations"] - last >= HINT_INTERVAL

    @staticmethod
    def _level(priority):
        return "high" if priority >= 0.7 else "medium" if priority >= 0.45 else "low"

    def prompt_block(self, user_text, is_blocked=None, recent=None):
        if not self._hint_due():
            return ""
        recent = (recent or [])[-GOAL_COOLDOWN:]
        goals = [
            g for g in self.relevant_goals(user_text)
            if not any(_contains(r, g["topic"]) for r in recent)
        ]
        lines = []
        used = []
        for goal in goals:
            guidance = GUIDANCE[goal["type"]].format(t=goal["topic"])
            if is_blocked is not None and is_blocked(guidance):
                continue
            lines.append(f"- Topic: {goal['topic']} | Goal: {guidance} | Priority: {self._level(goal['priority'])}")
            used.append(goal)
        if not lines:
            return ""
        for goal in used:
            goal["hinted"] = True
        self.data["last_hint_at"] = self.data["conversations"]
        self.save()
        return (
            "LUNA CURRENT CURIOSITY:\n"
            + "\n".join(lines)
            + "\nUse these only when naturally relevant. Do not force questions. "
            "Do not claim you performed actions outside this conversation "
            "(no research, reading, playing or watching)."
        )

    def handle_command(self, user_text):
        text = user_text.strip().rstrip(".!?").strip() if isinstance(user_text, str) else ""
        if not text:
            return None
        if QUESTIONS.match(_normalize(text)):
            return self._describe_goals()
        for pattern in (ABANDON_A, ABANDON_B):
            match = pattern.match(text)
            if match:
                topic = match.group("t").strip()
                goal = self.abandon_goal(topic)
                if goal:
                    return f"Okay, ich verfolge das Thema {goal['topic']} nicht weiter."
                return "Dazu habe ich gerade kein besonderes Ziel, Miki."
        match = REVISIT.match(text)
        if match:
            topic = _label(match.group("t"))
            if not topic:
                return None
            goal = self.add_goal(
                "revisit_topic", topic, "Miki möchte später nochmal darüber reden", 0.6, force=True
            )
            if goal:
                return f"Gern, wir reden später nochmal über {goal['topic']}."
        return None

    def _describe_goals(self):
        top = sorted(self.active, key=lambda g: g["priority"], reverse=True)[:MAX_PROMPT_GOALS]
        if not top:
            return "Gerade nichts Bestimmtes, aber ich bin neugierig, was du so erzählst."
        return " ".join(DESCRIPTION[g["type"]].format(t=g["topic"]) for g in top)


_DEFAULT_GOALS = None


def get_luna_goals():
    global _DEFAULT_GOALS
    if _DEFAULT_GOALS is None:
        _DEFAULT_GOALS = LunaGoals()
    return _DEFAULT_GOALS
