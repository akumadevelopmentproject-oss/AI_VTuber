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
from core.luna_goals import LunaGoals
from core.luna_state import LunaState
from core.memory import Memory
from core.safety import BLOCKED_OUTPUT_MARKER

SAME = "Das klingt nach einem richtig guten Plan für heute Abend."
OTHER = "Hm, dann lass uns einfach loslegen und sehen was passiert."


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class Response:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({"message": {"content": self.text}}).encode()


class RetryTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        self.memory = Memory(base / "memory.json")
        self.brain = Brain(base / "brain.json", knowledge=GameKnowledge(base / "c.json", online=False))
        self.state = LunaState(base / "state.json")
        self.goals = LunaGoals(base / "goals.json")
        self.requests = []

    def tearDown(self):
        self._dir.cleanup()

    def run_chat(self, ai, text, replies, blocked_output=lambda t: False):
        queue = list(replies)

        def fake_urlopen(request, timeout=0):
            self.requests.append(json.loads(request.data))
            return Response(queue.pop(0))

        with patch.object(ai_module.urllib.request, "urlopen", fake_urlopen), patch.object(
            ai_module, "get_memory", return_value=self.memory
        ), patch.object(ai_module, "get_brain", return_value=self.brain), patch.object(
            ai_module, "get_luna_state", return_value=self.state
        ), patch.object(ai_module, "get_luna_goals", return_value=self.goals), patch.object(
            ai_module, "is_blocked_output", blocked_output
        ):
            return ai.chat(text)

    def ai_with_history(self):
        ai = ai_module.AI()
        ai.history.add(SAME)
        return ai

    def test_unique_answer_one_call(self):
        ai = ai_module.AI()
        self.assertEqual(self.run_chat(ai, "Hallo", [OTHER]), OTHER)
        self.assertEqual(len(self.requests), 1)

    def test_duplicate_retries_exactly_once_and_uses_second(self):
        ai = self.ai_with_history()
        answer = self.run_chat(ai, "Was machen wir?", [SAME, OTHER])
        self.assertEqual(answer, OTHER)
        self.assertEqual(len(self.requests), 2)
        retry = [m["content"] for m in self.requests[1]["messages"] if m["role"] == "system"]
        self.assertTrue(any(t.startswith("REWRITE REQUIRED:") for t in retry))
        first = [m["content"] for m in self.requests[0]["messages"] if m["role"] == "system"]
        self.assertFalse(any("REWRITE REQUIRED" in t for t in first))
        self.assertEqual(self.requests[1]["messages"][-1]["content"], "Was machen wir?")

    def test_first_draft_not_stored(self):
        ai = self.ai_with_history()
        draft = "Genau so ein Satz, der nur der verworfene Entwurf war, ja."
        ai.history.add(draft)
        self.run_chat(ai, "Was machen wir?", [draft, OTHER])
        self.assertEqual(ai.history.recent()[-1], OTHER)
        self.assertEqual(ai.history.recent().count(draft), 1)
        assistant = [m["content"] for m in ai.messages if m["role"] == "assistant"]
        self.assertEqual(assistant, [OTHER])

    def test_second_similar_no_third_call(self):
        ai = self.ai_with_history()
        answer = self.run_chat(ai, "Was machen wir?", [SAME, SAME])
        self.assertEqual(answer, SAME)
        self.assertEqual(len(self.requests), 2)

    def test_retry_failure_keeps_first_draft(self):
        ai = self.ai_with_history()
        answer = self.run_chat(ai, "Was machen wir?", [SAME])
        self.assertEqual(answer, SAME)
        self.assertEqual(len(self.requests), 2)

    def test_blocked_first_draft_is_not_retried(self):
        ai = self.ai_with_history()
        answer = self.run_chat(ai, "Hi", [SAME, OTHER], blocked_output=lambda t: t == SAME)
        self.assertIsNone(answer)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(ai.messages[-1]["content"], BLOCKED_OUTPUT_MARKER)

    def test_blocked_retry_is_blocked(self):
        ai = self.ai_with_history()
        answer = self.run_chat(ai, "Hi", [SAME, OTHER], blocked_output=lambda t: t == OTHER)
        self.assertIsNone(answer)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(ai.messages[-1]["content"], BLOCKED_OUTPUT_MARKER)
        self.assertNotIn(OTHER, ai.history.recent())

    def test_tts_gets_only_final_answer(self):
        ai = self.ai_with_history()
        tts = FakeTTS()
        queue = [SAME, OTHER]

        def fake_urlopen(request, timeout=0):
            return Response(queue.pop(0))

        with patch.object(main, "get_memory", return_value=self.memory), patch.object(
            main, "get_brain", return_value=self.brain
        ), patch.object(main, "get_luna_state", return_value=self.state), patch.object(
            main, "get_luna_goals", return_value=self.goals
        ), patch.object(ai_module, "get_memory", return_value=self.memory), patch.object(
            ai_module, "get_brain", return_value=self.brain
        ), patch.object(ai_module, "get_luna_state", return_value=self.state), patch.object(
            ai_module, "get_luna_goals", return_value=self.goals
        ), patch.object(brain_module, "analyze_memory_candidate", return_value={"should_store": False}), patch.object(
            ai_module.urllib.request, "urlopen", fake_urlopen
        ):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                main.answer_user(ai, tts, "Was machen wir?", "Luna")
        self.assertEqual(tts.spoken, [OTHER])
        self.assertNotIn(SAME, stream.getvalue())
        self.assertIn(OTHER, stream.getvalue())
        self.assertNotIn("[AI]", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
