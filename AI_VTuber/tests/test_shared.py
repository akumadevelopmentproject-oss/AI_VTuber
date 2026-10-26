import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import state_isolation  # noqa: F401
from core import brain as brain_module
from core.brain import Brain
from core.game_knowledge import GameKnowledge
from core.memory import Memory
from core.shared import extract_explicit_topic, extract_gag

TEST_TERM = "blockedfixture"
GAME = {
    "should_store": True,
    "category": "gaming",
    "type": "game_preference",
    "game": "Arena Breakout",
    "sentiment": "very_positive",
    "confidence": 0.95,
}


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class FakeAI:
    def __init__(self):
        self.calls = []
        self.messages = []

    def chat(self, text):
        self.calls.append(text)
        return "normale Antwort"


class SharedTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.path = base / "brain.json"
        self.kn = GameKnowledge(base / "game_cache.json", online=False)
        self.brain = Brain(self.path, knowledge=self.kn)
        self.brain.store_candidate("Ich liebe Arena Breakout.", GAME)
        self.memory = Memory(base / "memory.json")

    def tearDown(self):
        self._dir.cleanup()

    def say(self, text, blocked=lambda t: False, brain=None):
        ai, tts = FakeAI(), FakeTTS()
        no_game = {"should_store": False}
        with patch.object(main, "get_memory", return_value=self.memory), patch.object(
            main, "get_brain", return_value=brain or self.brain
        ), patch.object(brain_module, "analyze_memory_candidate", return_value=no_game):
            with patch.object(main, "is_blocked_input", blocked):
                with patch.object(main, "filter_output", side_effect=lambda t: t):
                    with contextlib.redirect_stdout(io.StringIO()):
                        main.answer_user(ai, tts, text, "Luna")
        return ai, tts

    def kinds(self, brain=None):
        return {(m["type"], m["topic"]): m for m in (brain or self.brain).shared}

    def test_patterns(self):
        self.assertEqual(extract_gag("Du sprichst YouTube immer noch komisch aus."), "YouTube-Aussprache")
        self.assertIsNone(extract_gag("Ich spreche gerne."))
        self.assertEqual(extract_explicit_topic("Wir reden schon wieder über Arena Breakout."), "Arena Breakout")

    def test_three_mentions_create_shared_topic(self):
        self.say("Arena Breakout schon wieder.")
        self.say("Arena Breakout ist heute dran.")
        self.assertEqual(self.brain.shared, [])
        self.say("Wir reden schon wieder über Arena Breakout.")
        item = self.kinds()[("shared_topic", "Arena Breakout")]
        self.assertEqual(item["mentions"], 3)
        self.assertIn("öfter über Arena Breakout", item["summary"])
        self.say("Arena Breakout schon wieder.")
        self.assertEqual(len(self.brain.shared), 1)
        item = self.kinds()[("shared_topic", "Arena Breakout")]
        self.assertEqual(item["mentions"], 4)
        self.assertGreater(item["importance"], 0.66)

    def test_one_off_topic_not_stored(self):
        self.say("Heute war das Wetter schön.")
        self.say("Arena Breakout ist toll.")
        self.assertEqual(self.brain.shared, [])

    def test_running_gag(self):
        for _ in range(2):
            self.say("Du sprichst YouTube immer noch komisch aus.")
        self.assertEqual(self.brain.shared, [])
        self.say("Du sprichst YouTube schon wieder komisch aus.")
        gag = self.kinds()[("running_gag", "YouTube-Aussprache")]
        self.assertEqual(gag["mentions"], 3)
        self.assertIn("YouTube", gag["summary"])

    def test_blocked_and_sensitive_never_stored(self):
        blocked = lambda t: TEST_TERM in t
        for _ in range(4):
            ai, _ = self.say(f"Arena Breakout {TEST_TERM}", blocked=blocked)
            self.assertEqual(ai.calls, [])
        for _ in range(4):
            self.say("Arena Breakout und meine Diagnose Diabetes")
        self.assertEqual(self.brain.shared, [])
        self.assertEqual(self.brain.data["shared_candidates"], {})

    def test_restart_keeps_shared(self):
        for _ in range(3):
            self.say("Arena Breakout schon wieder.")
        again = Brain(self.path, knowledge=self.kn)
        self.assertIn(("shared_topic", "Arena Breakout"), self.kinds(again))
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(len(raw["shared_memories"]), 1)

    def test_questions(self):
        ai, tts = self.say("Worüber reden wir oft?")
        self.assertEqual(ai.calls, [])
        self.assertIn("noch nicht", tts.spoken[0])
        for _ in range(3):
            self.say("Arena Breakout schon wieder.")
        for question in ("Worüber reden wir oft?", "Was haben wir eigentlich schon oft besprochen?"):
            ai, tts = self.say(question)
            self.assertEqual(ai.calls, [])
            self.assertIn("Arena Breakout", tts.spoken[0])
        ai, tts = self.say("Haben wir irgendwelche Insider?")
        self.assertIn("noch nicht", tts.spoken[0])
        for _ in range(3):
            self.say("Du sprichst YouTube komisch aus.")
        ai, tts = self.say("Was ist unser Running Gag?")
        self.assertIn("YouTube", tts.spoken[0])
        self.assertNotIn("Arena", tts.spoken[0])

    def test_forget(self):
        for _ in range(3):
            self.say("Arena Breakout schon wieder.")
            self.say("Du sprichst YouTube komisch aus.")
        ai, tts = self.say("Vergiss unseren Insider über YouTube.")
        self.assertEqual(ai.calls, [])
        self.assertEqual(list(self.kinds()), [("shared_topic", "Arena Breakout")])
        self.say("Du sprichst YouTube komisch aus.")
        self.assertEqual(list(self.kinds()), [("shared_topic", "Arena Breakout")])

    def test_prompt_max_three_and_history(self):
        now = "2026-01-01T00:00:00"
        for name in ("Alpha", "Beta", "Gamma", "Delta"):
            self.brain.shared.append(
                {"type": "shared_topic", "topic": name, "summary": f"Miki und Luna reden öfter über {name}.",
                 "mentions": 5, "importance": 0.7, "created_at": now, "updated_at": now}
            )
        block = self.brain.prompt_block("Alpha Beta Gamma Delta")
        self.assertIn("SHARED HISTORY", block)
        self.assertEqual(block.count("reden öfter über"), 3)
        self.assertEqual(self.brain.prompt_block("Etwas ganz anderes").count("reden öfter über"), 0)
        self.assertIn("keine gemeinsamen realen Erlebnisse", self.brain.prompt_block("Alpha"))

    def test_invalid_shared_data(self):
        self.path.write_text(json.dumps({"shared_memories": 5, "shared_candidates": [1]}), encoding="utf-8")
        brain = Brain(self.path, knowledge=self.kn)
        self.assertEqual(brain.shared, [])


if __name__ == "__main__":
    unittest.main()
