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
from core import ai as ai_module
from core import brain as brain_module
from core.brain import Brain
from core.game_knowledge import GameKnowledge
from core.luna_goals import MAX_ACTIVE, LunaGoals
from core.luna_state import LunaState
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


class FakeBrain:
    games = []
    interests = []
    shared = []

    def __init__(self, topics):
        self.user_items = [{"topic": t, "sentiment": "positive"} for t in topics]


class GoalTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.path = base / "luna_goals.json"
        self.goals = LunaGoals(self.path)
        self.state = LunaState(base / "luna_state.json")
        self.brain = Brain(base / "brain.json", knowledge=GameKnowledge(base / "c.json", online=False))
        self.memory = Memory(base / "memory.json")

    def tearDown(self):
        self._dir.cleanup()

    def say(self, text, blocked=lambda t: False):
        ai, tts = FakeAI(), FakeTTS()
        no_game = {"should_store": False}
        with patch.object(main, "get_memory", return_value=self.memory), patch.object(
            main, "get_brain", return_value=self.brain
        ), patch.object(main, "get_luna_state", return_value=self.state), patch.object(
            main, "get_luna_goals", return_value=self.goals
        ), patch.object(brain_module, "analyze_memory_candidate", return_value=no_game):
            with patch.object(main, "is_blocked_input", blocked):
                with patch.object(main, "filter_output", side_effect=lambda t: t):
                    with contextlib.redirect_stdout(io.StringIO()):
                        main.answer_user(ai, tts, text, "Luna")
        return ai, tts

    def topics(self, status="active"):
        return [g["topic"] for g in self.goals.goals if g["status"] == status]

    def test_no_goal_after_one_mention_goal_after_three(self):
        self.say("Ich liebe Katzen.")
        self.assertEqual(self.goals.goals, [])
        self.say("Katzen sind mein Ding.")
        self.assertEqual(self.goals.goals, [])
        self.say("Ich liebe Katzen wirklich.")
        self.assertEqual(self.topics(), ["Katzen"])
        goal = self.goals.goals[0]
        self.assertEqual(goal["type"], "ask_miki_about_topic")
        self.assertTrue(0.0 <= goal["priority"] <= 1.0)

    def test_low_curiosity_needs_more_mentions(self):
        self.state.data["curiosity"] = 0.5
        for _ in range(3):
            self.say("Ich liebe Katzen.")
        self.assertEqual(self.goals.goals, [])
        self.say("Ich liebe Katzen.")
        self.assertEqual(self.topics(), ["Katzen"])

    def test_no_duplicates(self):
        first = self.goals.add_goal("learn_about_topic", "Extraction Shooter", priority=0.5)
        self.goals.add_goal("revisit_topic", "extraction shooter", priority=0.9)
        self.assertEqual(len(self.goals.goals), 1)
        self.assertLessEqual(self.goals.goals[0]["priority"], 0.56)
        self.assertIs(first, self.goals.goals[0])

    def test_priority_changes_slowly(self):
        self.goals.add_goal("learn_about_topic", "Horror", priority=0.3)
        for _ in range(3):
            self.goals.update_goal("Horror", priority=1.0)
        self.assertAlmostEqual(self.goals.goals[0]["priority"], 0.45, places=2)

    def test_max_active(self):
        for i in range(MAX_ACTIVE + 3):
            self.goals.add_goal("revisit_topic", f"Thema{chr(97 + i)}")
        self.assertEqual(len(self.goals.active), MAX_ACTIVE)
        self.assertIsNone(self.goals.add_goal("revisit_topic", "Zusatz"))

    def test_restart_keeps_active_and_completed(self):
        self.goals.add_goal("learn_about_topic", "Anime")
        self.goals.add_goal("ask_miki_about_topic", "Rap")
        self.goals.complete_goal("Rap")
        again = LunaGoals(self.path)
        self.assertEqual([g["topic"] for g in again.active], ["Anime"])
        self.assertEqual(again.find("Rap")["status"], "completed")
        self.assertEqual(again.find("Rap")["progress"], 1.0)
        self.assertIn("goals", json.loads(self.path.read_text(encoding="utf-8")))

    def test_broken_file(self):
        for content in ("{kaputt", "[]", "", '{"goals": 5}', '{"goals": [1, {"type": "x"}]}'):
            self.path.write_text(content, encoding="utf-8")
            self.assertEqual(LunaGoals(self.path).goals, [])
        self.path.write_text("{kaputt", encoding="utf-8")
        goals = LunaGoals(self.path)
        goals.add_goal("revisit_topic", "Minecraft")
        self.assertEqual(len(LunaGoals(self.path).goals), 1)

    def test_sensitive_topics_create_no_goal(self):
        self.assertIsNone(self.goals.add_goal("learn_about_topic", "Diabetes"))
        self.assertIsNone(self.goals.add_goal("learn_about_topic", "Depression"))
        brain = FakeBrain(["Diabetes"])
        for _ in range(5):
            self.goals.observe("Diabetes Diabetes", brain, 0.9)
        self.assertEqual(self.goals.goals, [])

    def test_blocked_input_creates_no_goal(self):
        blocked = lambda t: TEST_TERM in t
        for _ in range(5):
            ai, _ = self.say(f"Ich liebe Katzen {TEST_TERM}", blocked=blocked)
            self.assertEqual(ai.calls, [])
        self.assertEqual(self.goals.goals, [])
        self.assertEqual(self.goals.data["conversations"], 0)
        self.assertEqual(self.goals.data["mentions"], {})

    def test_relevant_goal_goes_to_ollama(self):
        self.goals.add_goal("ask_miki_about_topic", "Arena Breakout", priority=0.8)
        with patch.object(ai_module, "get_luna_goals", return_value=self.goals), patch.object(
            ai_module, "get_luna_state", return_value=self.state
        ), patch.object(ai_module, "get_memory", return_value=self.memory), patch.object(
            ai_module, "get_brain", return_value=self.brain
        ):
            relevant = ai_module.AI()._messages_with_memory("Arena Breakout schon wieder")
            self.goals.data["last_hint_at"] = None
            other = ai_module.AI()._messages_with_memory("Wie spät ist es?")
        relevant_text = "\n".join(m["content"] for m in relevant)
        other_text = "\n".join(m["content"] for m in other)
        self.assertIn("LUNA CURRENT CURIOSITY", relevant_text)
        self.assertIn("Ask Miki what he likes about Arena Breakout", relevant_text)
        self.assertIn("Priority: high", relevant_text)
        self.assertNotIn("LUNA CURRENT CURIOSITY", other_text)
        self.assertNotIn("Arena Breakout", other_text)

    def test_prompt_max_two_and_no_offscreen_claims(self):
        for name in ("Alpha", "Beta", "Gamma", "Delta"):
            self.goals.add_goal("learn_about_topic", name)
        block = self.goals.prompt_block("Alpha Beta Gamma Delta")
        self.assertEqual(block.count("- Topic:"), 2)
        self.assertIn("Do not force questions", block)
        self.assertIn("Do not claim you performed actions outside this conversation", block)
        lowered = block.lower()
        for phrase in ("recherchiert", "gestern", "in meiner freizeit", "habe gespielt", "videos angeschaut"):
            self.assertNotIn(phrase, lowered)

    def test_hint_frequency(self):
        self.goals.add_goal("ask_miki_about_topic", "Arena Breakout")
        self.assertTrue(self.goals.prompt_block("Arena Breakout"))
        self.assertEqual(self.goals.prompt_block("Arena Breakout"), "")
        brain = FakeBrain([])
        for _ in range(7):
            self.goals.observe("Hallo", brain, 0.7)
        self.assertEqual(self.goals.prompt_block("Arena Breakout"), "")
        self.goals.observe("Hallo", brain, 0.7)
        self.assertTrue(self.goals.prompt_block("Arena Breakout"))

    def test_progress_and_completion_after_answer(self):
        self.goals.add_goal("ask_miki_about_topic", "Arena Breakout")
        self.goals.prompt_block("Arena Breakout")
        self.goals.observe("Mir gefällt das Looting dort richtig gut", FakeBrain([]), 0.7)
        goal = self.goals.find("Arena Breakout")
        self.assertEqual(goal["status"], "completed")
        self.assertEqual(LunaGoals(self.path).find("Arena Breakout")["status"], "completed")

    def test_question_is_no_answer(self):
        self.goals.add_goal("ask_miki_about_topic", "Arena Breakout")
        self.goals.prompt_block("Arena Breakout")
        self.goals.observe("Wie spät ist es eigentlich?", FakeBrain([]), 0.7)
        goal = self.goals.find("Arena Breakout")
        self.assertEqual((goal["status"], goal["progress"]), ("active", 0.2))
        self.assertFalse(goal["hinted"])

    def test_user_can_abandon_goal(self):
        self.goals.add_goal("ask_miki_about_topic", "Arena Breakout")
        self.goals.add_goal("learn_about_topic", "Horrorfilme")
        ai, tts = self.say("Interessier dich nicht mehr für Arena Breakout.")
        self.assertEqual(ai.calls, [])
        self.assertEqual(self.goals.find("Arena Breakout")["status"], "abandoned")
        ai, tts = self.say("Vergiss dein Ziel über Horrorfilme.")
        self.assertEqual(ai.calls, [])
        self.assertEqual(self.goals.find("Horrorfilme")["status"], "abandoned")
        self.assertEqual(self.goals.add_goal("revisit_topic", "Arena Breakout"), None)

    def test_revisit_command(self):
        ai, tts = self.say("Lass uns später nochmal über Minecraft reden.")
        self.assertEqual(ai.calls, [])
        goal = self.goals.find("Minecraft")
        self.assertEqual((goal["type"], goal["status"]), ("revisit_topic", "active"))
        self.assertIn("Minecraft", tts.spoken[0])

    def test_direct_questions(self):
        ai, tts = self.say("Was interessiert dich gerade?")
        self.assertEqual(ai.calls, [])
        self.assertIn("neugierig", tts.spoken[0])
        self.goals.add_goal("learn_about_topic", "Extraction-Shooter", priority=0.72)
        for question in (
            "Was interessiert dich gerade?",
            "Was möchtest du noch wissen?",
            "Hast du irgendwelche Ziele?",
            "Worüber würdest du gern mehr erfahren?",
        ):
            ai, tts = self.say(question)
            self.assertEqual(ai.calls, [])
            self.assertIn("Extraction-Shooter", tts.spoken[-1])
            self.assertNotIn("0.72", tts.spoken[-1])
            self.assertNotIn("priority", tts.spoken[-1].lower())

    def test_normal_question_still_goes_to_ollama(self):
        ai, _ = self.say("Wie spät ist es?")
        self.assertEqual(ai.calls, ["Wie spät ist es?"])


if __name__ == "__main__":
    unittest.main()
