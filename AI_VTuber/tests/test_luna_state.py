import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import state_isolation  # noqa: F401
from core import ai as ai_module
from core import brain as brain_module
from core.brain import Brain
from core.game_knowledge import GameKnowledge
from core.luna_state import MOODS, LunaState
from core.memory import Memory

TEST_TERM = "blockedfixture"


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


class StateTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.path = base / "luna_state.json"
        self.state = LunaState(self.path)
        self.brain = Brain(base / "brain.json", knowledge=GameKnowledge(base / "c.json", online=False))
        self.memory = Memory(base / "memory.json")

    def tearDown(self):
        self._dir.cleanup()

    def say(self, text, blocked=lambda t: False, state=None):
        ai, tts = FakeAI(), FakeTTS()
        no_game = {"should_store": False}
        with patch.object(main, "get_memory", return_value=self.memory), patch.object(
            main, "get_brain", return_value=self.brain
        ), patch.object(main, "get_luna_state", return_value=state or self.state), patch.object(
            brain_module, "analyze_memory_candidate", return_value=no_game
        ):
            with patch.object(main, "is_blocked_input", blocked):
                with patch.object(main, "filter_output", side_effect=lambda t: t):
                    with contextlib.redirect_stdout(io.StringIO()):
                        main.answer_user(ai, tts, text, "Luna")
        return ai, tts

    def test_defaults(self):
        d = self.state.data
        self.assertEqual(
            (d["mood"], d["energy"], d["curiosity"], d["familiarity_with_miki"], d["conversation_count"]),
            ("curious", 0.70, 0.80, 0.15, 0),
        )
        self.assertTrue(self.path.exists())

    def test_save_load_restart(self):
        for _ in range(7):
            self.say("Hallo Luna")
        again = LunaState(self.path)
        self.assertEqual(again.data["conversation_count"], 7)
        self.assertAlmostEqual(again.data["familiarity_with_miki"], 0.15 + 7 * 0.002, places=4)
        self.assertEqual(again.data["mood"], self.state.data["mood"])
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["conversation_count"], 7)

    def test_broken_files(self):
        for content in ("{kaputt", "[1,2]", "", '{"energy": "viel", "mood": "wild", "conversation_count": -4}'):
            self.path.write_text(content, encoding="utf-8")
            state = LunaState(self.path)
            self.assertEqual(state.data["mood"], "curious")
            self.assertEqual(state.data["conversation_count"], 0)
            self.assertEqual(state.data["energy"], 0.70)
        self.path.unlink()
        self.assertEqual(LunaState(self.path).data["conversation_count"], 0)

    def test_values_clamped(self):
        self.path.write_text(
            json.dumps({"energy": 5, "curiosity": -2, "familiarity_with_miki": 9, "mood": "sleepy"}),
            encoding="utf-8",
        )
        state = LunaState(self.path)
        self.assertEqual(state.data["energy"], 1.0)
        self.assertEqual(state.data["curiosity"], 0.0)
        self.assertEqual(state.data["familiarity_with_miki"], 1.0)
        for _ in range(300):
            state.register_conversation("Danke, super", True)
        for key in ("energy", "curiosity", "familiarity_with_miki"):
            self.assertTrue(0.0 <= state.data[key] <= 1.0)
        self.assertIn(state.data["mood"], MOODS)

    def test_count_and_familiarity_slow(self):
        for _ in range(10):
            self.say("Erzähl mir etwas.")
        self.assertEqual(self.state.data["conversation_count"], 10)
        self.assertAlmostEqual(self.state.data["familiarity_with_miki"], 0.17, places=3)

    def test_blocked_input_does_not_change_state(self):
        before = json.loads(self.path.read_text(encoding="utf-8"))
        blocked = lambda t: TEST_TERM in t
        ai, _ = self.say(f"{TEST_TERM} danke super", blocked=blocked)
        self.assertEqual(ai.calls, [])
        self.assertEqual(self.state.data["conversation_count"], 0)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), before)

    def test_mood_does_not_jump_each_message(self):
        moods = []
        for _ in range(10):
            self.say("Danke, das war super lustig.")
            moods.append(self.state.mood)
        self.assertEqual(moods[0:4], ["curious"] * 4)
        self.assertEqual(moods[4:9], [moods[4]] * 5)
        self.assertIn(moods[4], ("cheerful", "playful"))

    def test_energy_falls_and_recovers_after_pause(self):
        for _ in range(20):
            self.say("Hallo")
        low = self.state.data["energy"]
        self.assertLess(low, 0.70)
        self.assertGreaterEqual(low, 0.2)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        raw["last_updated"] = (datetime.now() - timedelta(hours=12)).isoformat()
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        self.assertAlmostEqual(LunaState(self.path).data["energy"], 0.70, places=2)

    def test_new_topic_raises_curiosity(self):
        before = self.state.data["curiosity"]
        self.say("Ich liebe Katzen.")
        self.assertGreater(self.state.data["curiosity"], before)

    def test_ollama_gets_state_context(self):
        with patch.object(ai_module, "get_luna_state", return_value=self.state), patch.object(
            ai_module, "get_memory", return_value=self.memory
        ), patch.object(ai_module, "get_brain", return_value=self.brain):
            messages = ai_module.AI()._messages_with_memory("Hallo")
        text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        self.assertIn("LUNA CURRENT STATE", text)
        self.assertIn("Mood: curious", text)
        self.assertIn("Familiarity with Miki: 0.15", text)
        self.assertIn("Behavior guidance", text)

    def test_state_questions(self):
        self.state.data["mood"] = "energetic"
        self.state.data["energy"] = 0.8
        for question in ("Wie geht es dir gerade?", "Wie ist deine Stimmung?", "Bist du heute müde?"):
            ai, tts = self.say(question)
            self.assertEqual(ai.calls, [])
            self.assertTrue(tts.spoken)
            for banned in ("0.8", "energy", "Wert"):
                self.assertNotIn(banned, tts.spoken[0])
        self.assertEqual(self.state.data["conversation_count"], 0)

    def test_state_questions_blocked_input_first(self):
        ai, tts = self.say("Wie geht es dir gerade?", blocked=lambda t: True)
        self.assertEqual((ai.calls, tts.spoken), ([], []))


if __name__ == "__main__":
    unittest.main()
