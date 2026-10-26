import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import state_isolation  # noqa: F401
from core import ai as ai_module
from core.brain import Brain
from core.conversation import (
    ResponseHistory,
    assemble_blocks,
    filter_memories,
    strip_assistant_phrases,
    style_block,
)
from core.game_knowledge import GameKnowledge
from core.luna_goals import LunaGoals
from core.luna_state import LunaState
from core.memory import Memory


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.memory = Memory(base / "memory.json")
        self.brain = Brain(base / "brain.json", knowledge=GameKnowledge(base / "c.json", online=False))
        self.state = LunaState(base / "state.json")
        self.goals = LunaGoals(base / "goals.json")

    def tearDown(self):
        self._dir.cleanup()

    def build(self, ai, text):
        with patch.object(ai_module, "get_memory", return_value=self.memory), patch.object(
            ai_module, "get_brain", return_value=self.brain
        ), patch.object(ai_module, "get_luna_state", return_value=self.state), patch.object(
            ai_module, "get_luna_goals", return_value=self.goals
        ):
            messages = ai._messages_with_memory(text)
        return "\n".join(m["content"] for m in messages if m["role"] == "system")

    def test_identical_answer_detected(self):
        history = ResponseHistory()
        history.add("Ja, das stimmt wirklich ziemlich gut.")
        self.assertTrue(history.is_duplicate("Ja, das stimmt wirklich ziemlich gut!"))
        self.assertFalse(history.is_duplicate("Keine Ahnung, ich muss kurz nachdenken."))

    def test_history_keeps_last_ten(self):
        history = ResponseHistory()
        for i in range(15):
            history.add(f"Antwort Nummer {i} hier")
        self.assertEqual(len(history.recent()), 10)
        self.assertIn("14", history.recent()[-1])

    def test_same_opening_is_flagged_not_full_text(self):
        history = ResponseHistory()
        for tail in ("das ist cool.", "das war nett.", "das klingt gut."):
            history.add(f"Ja, ziemlich viel {tail} Ganz langer Rest den Ollama nicht nochmal sehen soll.")
        block = history.guidance_block()
        self.assertIn('Do not start with "Ja ziemlich viel..."', block)
        self.assertEqual(block.count("Ja ziemlich viel"), 1)
        self.assertNotIn("Ganz langer Rest", block)
        self.assertIn("Do not mention the same user fact again", block)

    def test_name_not_forced(self):
        history = ResponseHistory()
        self.assertNotIn('Do not use the name "Miki"', style_block("Hallo", history))
        history.add("Haha Miki, das war gut.")
        text = style_block("Hallo", history)
        self.assertIn('Do not use the name "Miki"', text)
        self.assertIn("only occasionally", text)

    def test_no_self_introduction_unless_asked(self):
        history = ResponseHistory()
        self.assertIn("Ich bin Luna", style_block("Wie war dein Tag?", history))
        self.assertIn('Do not introduce yourself', style_block("Wie war dein Tag?", history))
        self.assertNotIn("Do not introduce yourself", style_block("Wer bist du?", history))

    def test_memory_only_when_relevant(self):
        self.memory.add("Miki liebt Katzen.")
        ai = ai_module.AI()
        self.assertNotIn("Katzen", self.build(ai, "Wie geht es dir?"))
        self.assertIn("Katzen", self.build(ai, "Welches Haustier passt zu mir?"))
        self.assertIn("Katzen", self.build(ai, "Erzähl mir was über Katzen."))

    def test_filter_memories(self):
        texts = ["Miki liebt Katzen.", "Mikis Lieblingsspiel ist COD."]
        self.assertEqual(filter_memories(texts, "Wie spät ist es?"), [])
        self.assertEqual(filter_memories(texts, "Was weißt du über mich?"), texts)
        self.assertEqual(filter_memories(texts, "Spielst du COD?"), [texts[1]])

    def test_goal_not_reused_shortly_after(self):
        self.goals.add_goal("ask_miki_about_topic", "Arena Breakout", priority=0.8)
        ai = ai_module.AI()
        self.assertIn("LUNA CURRENT CURIOSITY", self.build(ai, "Arena Breakout schon wieder"))
        self.goals.data["last_hint_at"] = None
        ai.history.add("Was feierst du an Arena Breakout eigentlich so?")
        self.assertNotIn("LUNA CURRENT CURIOSITY", self.build(ai, "Arena Breakout schon wieder"))
        for _ in range(5):
            ai.history.add("Ganz anderes Thema heute.")
        self.goals.data["last_hint_at"] = None
        self.assertIn("LUNA CURRENT CURIOSITY", self.build(ai, "Arena Breakout schon wieder"))

    def test_shared_history_only_when_topic_fits(self):
        now = "2026-01-01T00:00:00"
        self.brain.shared.append(
            {"type": "shared_topic", "topic": "Arena Breakout", "summary": "Miki und Luna reden öfter über Arena Breakout.",
             "mentions": 5, "importance": 0.7, "created_at": now, "updated_at": now}
        )
        ai = ai_module.AI()
        self.assertIn("SHARED HISTORY", self.build(ai, "Arena Breakout schon wieder"))
        self.assertNotIn("SHARED HISTORY", self.build(ai, "Wie spät ist es?"))

    def test_running_gag_cooldown(self):
        now = "2026-01-01T00:00:00"
        self.brain.shared.append(
            {"type": "running_gag", "topic": "YouTube-Aussprache", "summary": "Miki macht sich über YouTube lustig.",
             "mentions": 4, "importance": 0.8, "created_at": now, "updated_at": now}
        )
        ai = ai_module.AI()
        self.assertIn("SHARED HISTORY", self.build(ai, "Sag mal YouTube"))
        ai.history.add("Ja, YouTube ist schwer auszusprechen.")
        self.assertNotIn("SHARED HISTORY", self.build(ai, "Sag mal YouTube"))
        for _ in range(4):
            ai.history.add("Etwas ganz anderes.")
        self.assertIn("SHARED HISTORY", self.build(ai, "Sag mal YouTube"))

    def test_assistant_phrases_removed(self):
        raw = "Klar, das geht. Wie kann ich dir helfen? Wenn du weitere Fragen hast, melde dich."
        self.assertEqual(strip_assistant_phrases(raw), "Klar, das geht.")
        self.assertEqual(strip_assistant_phrases("Ich bin eine KI und helfe."), "Ich bin eine KI und helfe.")
        self.assertEqual(strip_assistant_phrases("Haha, stimmt."), "Haha, stimmt.")
        block = style_block("Hallo", ResponseHistory())
        for phrase in ("Wie kann ich dir helfen?", "Ich stehe dir gerne zur Verfügung", "Ich bin eine KI"):
            self.assertIn(phrase, block)

    def test_question_limit(self):
        history = ResponseHistory()
        self.assertNotIn("Do not end with a question", style_block("Hallo", history))
        history.add("Und wie war dein Tag?")
        self.assertIn("Do not end with a question", style_block("Hallo", history))
        for _ in range(3):
            history.add("Aussage ohne Frage.")
        self.assertNotIn("Do not end with a question", style_block("Hallo", history))

    def test_length_guidance(self):
        history = ResponseHistory()
        self.assertIn("1 to 3 short sentences", style_block("Wie geht's?", history))
        self.assertNotIn("asked for details", style_block("Wie geht's?", history))
        self.assertIn("asked for details", style_block("Erkläre mir bitte ausführlich, wie das geht.", history))

    def test_context_budget_drops_lowest_priority(self):
        ranked = ["A" * 2000, "B" * 1500, "C" * 300]
        chosen = assemble_blocks(ranked, budget=2400)
        self.assertEqual(chosen[0][0], "A")
        self.assertNotIn("B" * 1500, chosen)
        self.assertEqual(chosen[-1][0], "C")

    def test_chat_records_history_and_strips(self):
        ai = ai_module.AI()

        class Resp:
            def __init__(self, text):
                self.text = text

            def read(self):
                import json
                return json.dumps({"message": {"content": self.text}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with patch.object(ai_module.urllib.request, "urlopen", return_value=Resp("Klar. Wie kann ich dir helfen?")), patch.object(
            ai_module, "get_memory", return_value=self.memory
        ), patch.object(ai_module, "get_brain", return_value=self.brain), patch.object(
            ai_module, "get_luna_state", return_value=self.state
        ), patch.object(ai_module, "get_luna_goals", return_value=self.goals):
            answer = ai.chat("Hallo")
        self.assertEqual(answer, "Klar.")
        self.assertEqual(ai.history.recent(), ["Klar."])

    def test_request_prompt_stays_reasonable(self):
        ai = ai_module.AI()
        text = self.build(ai, "Wie geht es dir?")
        extra = len(text) - len(ai.messages[0]["content"])
        self.assertLess(extra, 2400)


if __name__ == "__main__":
    unittest.main()
