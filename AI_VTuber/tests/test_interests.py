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
from core.interests import extract_interest
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


class InterestTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.path = base / "brain.json"
        self.kn = GameKnowledge(base / "game_cache.json", online=False)
        self.brain = Brain(self.path, knowledge=self.kn)
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

    def topics(self):
        return {m["topic"]: m for m in self.brain.user_items}

    def test_patterns(self):
        cases = {
            "Ich mag Horrorfilme.": ("Horrorfilme", "positive", "interest"),
            "Ich liebe Katzen.": ("Katzen", "very_positive", "interest"),
            "Ich höre gerne Rap.": ("Rap", "positive", "preference"),
            "Ich hasse scharfes Essen.": ("scharfes Essen", "very_negative", "dislike"),
            "Ich interessiere mich für Motorräder.": ("Motorräder", "positive", "interest"),
            "Ich feier Anime.": ("Anime", "very_positive", "interest"),
            "Ich stehe auf Horror.": ("Horror", "very_positive", "interest"),
            "Ich finde Fantasy richtig geil.": ("Fantasy", "very_positive", "interest"),
            "Ich mag Pizza nicht.": ("Pizza", "negative", "dislike"),
            "Mein Lieblingsessen ist Pizza.": ("Pizza", "very_positive", "favorite"),
        }
        for text, (topic, sentiment, kind) in cases.items():
            result = extract_interest(text)
            self.assertIsNotNone(result, text)
            self.assertEqual((result["topic"], result["sentiment"], result["type"]), (topic, sentiment, kind), text)

    def test_rejected_statements(self):
        for text in (
            "Mag ich Horrorfilme?",
            "Vielleicht mag ich Horrorfilme.",
            "Ich glaube, ich mag das.",
            "Der Film heute war gut.",
            "Horrorfilme sind Filme.",
            "Ich mag das.",
            "Das ist cool.",
            "Ich habe Diabetes.",
            "Ich hasse meine Krankheit.",
            "Ich mag Politik.",
            "Ich liebe mein Passwort.",
            "Ich mag es, wenn du lachst.",
            "Ich mag dich.",
            "Was mag ich?",
            "Ich mag Katzen, weil sie süß sind.",
            "Ich mochte Horrorfilme früher.",
        ):
            self.assertIsNone(extract_interest(text), text)

    def test_stored_through_main_flow_and_restart(self):
        for text in ("Ich liebe Katzen.", "Ich mag Horrorfilme."):
            ai, _ = self.say(text)
            self.assertEqual(ai.calls, [text])
        again = Brain(self.path, knowledge=self.kn)
        stored = {m["topic"]: m for m in again.user_items}
        self.assertEqual(stored["Katzen"]["sentiment"], "very_positive")
        self.assertEqual(stored["Horrorfilme"]["type"], "interest")
        self.assertEqual(stored["Horrorfilme"]["subject"], "Miki")
        self.assertEqual(stored["Horrorfilme"]["mentions"], 1)

    def test_unsafe_statements_not_stored(self):
        for text in ("Vielleicht mag ich Horrorfilme.", "Ich mag das.", "Ich habe Diabetes.", "Mag ich Horrorfilme?"):
            self.say(text)
        self.assertEqual(self.brain.user_items, [])
        self.assertNotIn("Diabetes", self.path.read_text("utf-8"))

    def test_blocked_input_never_stored(self):
        ai, _ = self.say(f"Ich liebe {TEST_TERM}.", blocked=lambda t: TEST_TERM in t)
        self.assertEqual((ai.calls, self.brain.user_items), ([], []))

    def test_duplicates_raise_mentions(self):
        self.say("Ich mag Horrorfilme.")
        self.say("Ich mag horrorfilme.")
        self.assertEqual(len(self.brain.user_items), 1)
        self.assertEqual(self.brain.user_items[0]["mentions"], 2)

    def test_opinion_change(self):
        self.say("Ich mag Horrorfilme.")
        self.say("Ich mochte Horrorfilme früher, aber mittlerweile nicht mehr.")
        item = self.brain.user_items[0]
        self.assertEqual((len(self.brain.user_items), item["sentiment"], item["type"]), (1, "negative", "dislike"))
        self.assertEqual(item["mentions"], 2)

    def test_favorite_replaced_per_category(self):
        self.say("Mein Lieblingsessen ist Pizza.")
        self.say("Mein Lieblingsessen ist Pasta.")
        self.assertEqual(len(self.brain.user_items), 1)
        self.assertEqual(self.brain.user_items[0]["topic"], "Pasta")

    def test_questions_answered_from_brain(self):
        for text in ("Ich liebe Katzen.", "Ich mag Horrorfilme.", "Ich hasse scharfes Essen."):
            self.say(text)
        ai, tts = self.say("Was mag ich?")
        self.assertEqual(ai.calls, [])
        self.assertIn("Katzen", tts.spoken[0])
        self.assertIn("Horrorfilme", tts.spoken[0])
        self.assertNotIn("scharfes", tts.spoken[0])
        _, tts = self.say("Was hasse ich?")
        self.assertIn("scharfes Essen", tts.spoken[0])
        _, tts = self.say("Was weißt du über meine Interessen?")
        self.assertIn("Katzen", tts.spoken[0])
        _, tts = self.say("Welche Sachen feier ich?")
        self.assertIn("Katzen", tts.spoken[0])
        ai, tts = self.say("Mag ich Horrorfilme?")
        self.assertEqual(ai.calls, [])
        self.assertTrue(tts.spoken[0].startswith("Ja"))
        _, tts = self.say("Mag ich Brokkoli?")
        self.assertIn("nicht gemerkt", tts.spoken[0])
        self.assertEqual(len(self.brain.user_items), 3)

    def test_empty_answer(self):
        _, tts = self.say("Was mag ich?")
        self.assertIn("noch nichts", tts.spoken[0])

    def test_forget_interest(self):
        self.say("Ich mag Horrorfilme.")
        self.say("Vergiss, dass ich Horrorfilme mag.")
        self.assertEqual(self.brain.user_items, [])

    def test_relevant_prompt_only(self):
        for text in ("Ich mag Horrorfilme.", "Ich finde Fantasy richtig geil.", "Ich liebe Katzen."):
            self.say(text)
        block = self.brain.prompt_block("Was soll ich heute schauen?")
        self.assertIn("Miki mag Horrorfilme", block)
        self.assertIn("Miki liebt Fantasy", block)
        specific = self.brain.prompt_block("Wie findest du Katzen?")
        self.assertIn("Katzen", specific)
        self.assertNotIn("Horror", specific)
        self.assertEqual(self.brain.prompt_block("Wie ist das Wetter?"), "")
        self.assertEqual(self.brain.prompt_block("Was soll ich schauen?", lambda t: True), "")

    def test_luna_interest_without_experience_claims(self):
        for _ in range(3):
            self.say("Ich mag Horrorfilme.")
        self.assertEqual(self.brain.interests[0]["topic"], "Horrorfilme")
        self.assertEqual(self.brain.interests[0]["strength"], 0.4)
        block = self.brain.prompt_block("Was soll ich heute schauen?")
        self.assertIn("nichts davon real erlebt", block)

    def test_known_game_stays_in_gaming_brain(self):
        self.brain.store_candidate(
            "Ich spiele COD urgerne.",
            {"should_store": True, "category": "gaming", "type": "game_preference",
             "game": "COD", "sentiment": "very_positive", "confidence": 0.95},
        )
        self.say("Ich mag COD nicht mehr.")
        self.assertEqual(self.brain.user_items, [])

    def test_invalid_user_memories_do_not_crash(self):
        self.path.write_text(
            json.dumps({"user_memories": [1, "x", {"topic": 5}, {"topic": "Katzen"}],
                        "game_memories": [], "luna_memories": []}),
            encoding="utf-8",
        )
        brain = Brain(self.path, knowledge=self.kn)
        self.assertEqual([m["topic"] for m in brain.user_items], ["Katzen"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
