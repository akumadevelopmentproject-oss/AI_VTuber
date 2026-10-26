import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import state_isolation  # noqa: F401
from config import settings
from core import ai as ai_module
from core import brain as brain_module
from core import game_knowledge as game_knowledge_module
from core.brain import Brain
from core.conversation import clean_output, style_block, ResponseHistory
from core.game_knowledge import GameKnowledge
from core.luna_goals import LunaGoals
from core.luna_state import LunaState
from core.memory import Memory
from core.topic_context import TopicContext

ARENA = {
    "appid": 2073620,
    "name": "Arena Breakout: Infinite",
    "genres": ["Action", "Free To Play"],
    "tags": ["Extraction Shooter", "FPS", "Tactical"],
    "developer": "MoreFun Studios",
    "publisher": "Level Infinite",
    "release_date": "Oct 2025",
    "source": "steam",
    "verified": True,
    "aliases": ["arena breakout"],
}


STEAM_ITEMS = [
    (203770, "Crusader Kings II"),
    (1158310, "Crusader Kings III"),
    (10, "Call of Duty: Modern Warfare"),
    (11, "Call of Duty: Black Ops 6"),
    (2073620, "Arena Breakout: Infinite"),
]


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


class QualityTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        base = Path(self._dir.name)
        knowledge = GameKnowledge(base / "cache.json", online=False)
        knowledge.cache["2073620"] = dict(ARENA, updated_at=datetime.now().isoformat(timespec="seconds"))
        self.memory = Memory(base / "memory.json")
        self.brain = Brain(base / "brain.json", knowledge=knowledge)
        self.state = LunaState(base / "state.json")
        self.goals = LunaGoals(base / "goals.json")
        self.requests = []

    def tearDown(self):
        self._dir.cleanup()

    def patches(self, replies=("Alles klar.",), analysis=None):
        queue = list(replies)

        def fake_urlopen(request, timeout=0):
            self.requests.append(json.loads(request.data))
            return Response(queue.pop(0) if len(queue) > 1 else queue[0])

        stack = contextlib.ExitStack()
        for module in (main, ai_module):
            stack.enter_context(patch.object(module, "get_memory", return_value=self.memory))
            stack.enter_context(patch.object(module, "get_brain", return_value=self.brain))
            stack.enter_context(patch.object(module, "get_luna_state", return_value=self.state))
            stack.enter_context(patch.object(module, "get_luna_goals", return_value=self.goals))
        stack.enter_context(patch.object(ai_module.urllib.request, "urlopen", fake_urlopen))
        stack.enter_context(
            patch.object(
                brain_module,
                "analyze_memory_candidate",
                return_value=analysis or {"should_store": False},
            )
        )
        stack.enter_context(patch.object(main, "filter_output", side_effect=lambda t: t))
        return stack

    def say(self, ai, text):
        tts = FakeTTS()
        with contextlib.redirect_stdout(io.StringIO()):
            main.answer_user(ai, tts, text, "Luna")
        return tts

    def chat_system_text(self, request):
        return "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")

    # ---------- Modelle ----------
    def test_settings_models(self):
        self.assertEqual(settings.CHAT_MODEL, "qwen3:4b-instruct-2507-q4_K_M")
        self.assertEqual(settings.FAST_MODEL, "qwen3:1.7b")
        self.assertFalse(hasattr(settings, "MODEL_NAME"))

    def test_chat_uses_chat_model_without_thinking(self):
        ai = ai_module.AI()
        with self.patches():
            ai.chat("Hallo")
        self.assertEqual(self.requests[0]["model"], settings.CHAT_MODEL)
        self.assertIs(self.requests[0]["think"], False)

    def test_chat_falls_back_to_fast_model(self):
        ai = ai_module.AI()
        calls = []

        def fake_urlopen(request, timeout=0):
            data = json.loads(request.data)
            calls.append(data["model"])
            if data["model"] == settings.CHAT_MODEL:
                raise urllib.error.HTTPError("url", 404, "not found", {}, None)
            return Response("Klar.")

        with self.patches(), patch.object(ai_module.urllib.request, "urlopen", fake_urlopen):
            answer = ai.chat("Hallo")
        self.assertEqual(answer, "Klar.")
        self.assertEqual(calls, [settings.CHAT_MODEL, "qwen3:1.7b"])
        self.assertEqual(ai.model, "qwen3:1.7b")

    def test_analysis_uses_fast_model(self):
        captured = []

        def fake_urlopen(request, timeout=0):
            captured.append(json.loads(request.data))
            return Response('{"should_store": false}')

        with patch.object(brain_module.urllib.request, "urlopen", fake_urlopen):
            brain_module.analyze_memory_candidate("Ich liebe Hades")
        self.assertEqual(captured[0]["model"], "qwen3:1.7b")
        self.assertIs(captured[0]["think"], False)

    # ---------- Spielerkennung ----------
    def test_arena_breakout_question_variants(self):
        for question in (
            "Weißt du was über Arena Breakout?",
            "Was weißt du über Arena Breakout?",
            "Kennst du Arena Breakout?",
            "Was ist Arena Breakout?",
            "Erzähl mir etwas über Arena Breakout.",
        ):
            reply = self.brain.handle_command(question)
            self.assertIsNotNone(reply, question)
            self.assertIn("Arena Breakout: Infinite", reply, question)
            self.assertNotIn("Brettspiel", reply)

    def test_references_with_active_game(self):
        topics = TopicContext()
        topics.process("Weißt du was über Arena Breakout?", self.brain)
        self.assertEqual(topics.game, "Arena Breakout: Infinite")
        resolved = topics.process("Was ist das eigentlich für ein Spiel?", self.brain)
        self.assertIn("Arena Breakout: Infinite", self.brain.handle_command(resolved))
        resolved = topics.process("Was für ein Genre hat das?", self.brain)
        self.assertIn("Extraction Shooter", self.brain.handle_command(resolved))
        resolved = topics.process("Wie findest du das Spiel?", self.brain)
        self.assertIn("Arena Breakout: Infinite", resolved)

    def test_normal_chat_gets_verified_information(self):
        ai = ai_module.AI()
        with self.patches():
            self.say(ai, "Wie findest du Arena Breakout?")
        system = self.chat_system_text(self.requests[-1])
        self.assertIn("VERIFIED GAME INFORMATION", system)
        self.assertIn("Extraction Shooter", system)
        self.assertIn("MoreFun Studios", system)
        self.assertIn("Do not invent", system) if "Do not invent" in system else self.assertIn("Do not contradict", system)

    def test_reference_question_gets_verified_information(self):
        ai = ai_module.AI()
        with self.patches():
            self.say(ai, "Weißt du was über Arena Breakout?")
            self.say(ai, "Wie findest du das Spiel?")
        system = self.chat_system_text(self.requests[-1])
        self.assertIn("VERIFIED GAME INFORMATION", system)
        self.assertIn("ACTIVE TOPIC", system)

    # ---------- STT-Korrektur ----------
    def test_stt_game_name_is_normalized(self):
        topics = TopicContext()
        self.assertEqual(
            topics.normalize("Weißt du was über Arena black Breakout?", self.brain),
            "Weißt du was über Arena Breakout?",
        )
        self.assertEqual(topics.normalize("Ich mag Arena Breakot", self.brain), "Ich mag Arena Breakout")
        self.assertEqual(topics.normalize("Arena Breakout ist gut", self.brain), "Arena Breakout ist gut")
        self.assertEqual(topics.normalize("Ich mag den Arena der Stadt", self.brain), "Ich mag den Arena der Stadt")

    def test_stt_normalization_runs_after_safety(self):
        ai = ai_module.AI()
        ai.topics = unittest.mock.MagicMock()
        with self.patches(), patch.object(main, "is_blocked_input", return_value=True):
            self.say(ai, "irgendwas Blockiertes")
        ai.topics.normalize.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_stt_corrected_name_reaches_ollama(self):
        ai = ai_module.AI()
        with self.patches():
            self.say(ai, "Wie findest du Arena black Breakout?")
        self.assertEqual(self.requests[-1]["messages"][-1]["content"], "Wie findest du Arena Breakout?")
        self.assertIn("VERIFIED GAME INFORMATION", self.chat_system_text(self.requests[-1]))

    # ---------- Aktives Spiel ----------
    def test_unknown_game_is_unverified(self):
        ai = ai_module.AI()
        with self.patches():
            self.say(ai, "Was ist Crusader Kings für ein Spiel?")
        self.assertEqual(ai.topics.game, "Crusader Kings")
        system = self.chat_system_text(self.requests[-1]) if self.requests else ""
        topic = ai.topics.prompt_block()
        self.assertIn("Do not guess", topic)
        self.assertIn("Crusader Kings", topic)

    def test_es_refers_to_active_game(self):
        topics = TopicContext()
        topics.process("Was ist Crusader Kings für ein Spiel?", self.brain)
        self.assertEqual(
            topics.process("Ich mag es eigentlich sehr.", self.brain),
            "Ich mag Crusader Kings eigentlich sehr.",
        )
        self.assertEqual(topics.process("Wie geht es dir?", self.brain), "Wie geht es dir?")
        self.assertEqual(
            topics.process("Erzähl mir mehr darüber.", self.brain), "Erzähl mir mehr über Crusader Kings."
        )

    def test_favorite_is_set_on_active_game(self):
        candidate = {
            "should_store": True,
            "category": "gaming",
            "type": "favorite_game",
            "game": "Crusader Kings",
            "sentiment": "very_positive",
            "confidence": 0.95,
        }
        ai = ai_module.AI()
        with self.patches(analysis=candidate):
            self.say(ai, "Was ist Crusader Kings für ein Spiel?")
            self.say(ai, "Das ist eines meiner Lieblingsspiele.")
        entries = {g["game"]: g for g in self.brain.games}
        self.assertIn("Crusader Kings", entries)
        self.assertTrue(entries["Crusader Kings"]["favorite"])
        self.assertEqual(entries["Crusader Kings"]["sentiment"], "very_positive")
        self.assertNotIn("es", {g["game"].lower() for g in self.brain.games})

    def test_active_game_expires(self):
        topics = TopicContext()
        topics.process("Was ist Crusader Kings für ein Spiel?", self.brain)
        for _ in range(5):
            topics.process("Wie ist das Wetter heute?", self.brain)
            self.assertEqual(topics.game, "Crusader Kings")
        topics.process("Wie ist das Wetter heute?", self.brain)
        self.assertIsNone(topics.game)

    # ---------- Ausgabe ----------
    def test_style_rules_cover_du_tags_experiences(self):
        block = style_block("Hallo", ResponseHistory())
        self.assertIn('NEVER address Miki with "Sie"', block)
        self.assertIn("[SCHERZE]", block)
        self.assertIn("never claim real experiences", block)
        self.assertIn("Da bin ich mir gerade nicht sicher", block)
        personality = (Path(main.__file__).parent / "config" / "personality.txt").read_text(encoding="utf-8")
        self.assertIn('niemals mit "Sie"', personality)

    def test_no_meta_tags(self):
        self.assertEqual(clean_output("[ SCHERZE ] Haha, okay *lacht* weiter."), "Haha, okay weiter.")
        self.assertEqual(clean_output("[IRONISCH] Na klar."), "Na klar.")

    def test_invented_experiences_removed(self):
        text = "Ich habe Crusader Kings gespielt. Das Konzept wirkt auf mich ziemlich spannend."
        self.assertEqual(clean_output(text), "Das Konzept wirkt auf mich ziemlich spannend.")
        self.assertEqual(clean_output("Gestern war ich dort. Klingt gut."), "Klingt gut.")
        self.assertEqual(clean_output("Ich habe es nie gespielt, klingt aber gut."), "Ich habe es nie gespielt, klingt aber gut.")
        self.assertEqual(clean_output("Ich habe den Film gesehen."), "Ich habe den Film gesehen.")

    def test_chat_cleans_model_output(self):
        ai = ai_module.AI()
        with self.patches(replies=("[SCHERZE] Ich habe Hunde gesehen. Hunde sind einfach loyaler.",)):
            answer = ai.chat("Ich finde Hunde viel cooler als Katzen.")
        self.assertEqual(answer, "Hunde sind einfach loyaler.")

    # ---------- Mehrdeutige Spielnamen ----------
    def steam(self, items=None):
        catalog = items or STEAM_ITEMS

        def fake_http(url, headers=None, timeout=0):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if "storesearch" in url:
                term = query["term"][0].lower()
                found = [{"type": "app", "id": i, "name": n} for i, n in catalog if term in n.lower()]
                return json.dumps({"items": found})
            if "appdetails" in url:
                appid = int(query["appids"][0])
                name = dict(catalog)[appid]
                data = {"type": "game", "name": name, "genres": [{"description": "Strategy"}],
                        "developers": ["Paradox Development Studio"], "publishers": [], "release_date": {}}
                return json.dumps({str(appid): {"success": True, "data": data}})
            return ""

        self.brain.knowledge.online = True
        return patch.object(game_knowledge_module, "_http_get", fake_http)

    def test_ambiguous_game_asks_instead_of_guessing(self):
        ai = ai_module.AI()
        with self.patches(), self.steam():
            tts = self.say(ai, "Was ist Crusader Kings für ein Spiel?")
        self.assertEqual(tts.spoken, ["Meinst du Crusader Kings II oder III?"])
        self.assertEqual(self.requests, [])
        self.assertIsNone(ai.topics.game)

    def test_ambiguous_choice_sets_active_game(self):
        for reply in ("Crusader Kings 3", "den dritten", "III", "drei", "Crusader Kings III"):
            ai = ai_module.AI()
            with self.patches(), self.steam():
                self.say(ai, "Was ist Crusader Kings für ein Spiel?")
                tts = self.say(ai, reply)
                self.assertEqual(ai.topics.game, "Crusader Kings III", reply)
                self.assertTrue(ai.topics.verified)
                self.assertIn("Crusader Kings III", tts.spoken[0], reply)
                self.assertIn("Paradox", tts.spoken[0], reply)
                follow = self.say(ai, "Was ist das für ein Spiel?")
                self.assertIn("Crusader Kings III", follow.spoken[0])
                self.assertNotIn("Crusader Kings II ", follow.spoken[0])

    def test_choice_second_game(self):
        topics = TopicContext()
        with self.steam():
            topics.process("Was ist Crusader Kings für ein Spiel?", self.brain)
            topics.process("die zweite", self.brain)
        self.assertEqual(topics.game, "Crusader Kings II")

    def test_ambiguity_not_resolved_by_unrelated_message(self):
        topics = TopicContext()
        with self.steam():
            topics.process("Was ist Crusader Kings für ein Spiel?", self.brain)
            topics.process("Wie ist das Wetter heute?", self.brain)
        self.assertIsNone(topics.game)
        self.assertIsNone(topics.pending)

    def test_call_of_duty_is_never_auto_selected(self):
        topics = TopicContext()
        with self.steam():
            topics.process("Kennst du Call of Duty?", self.brain)
        self.assertIsNone(topics.game)
        self.assertIn("Call of Duty", topics.clarification)
        self.assertLessEqual(len(topics.pending), 3)
        topics = TopicContext()
        topics.process("Call of Duty", self.brain)
        self.assertIsNone(topics.game)

    def test_unknown_game_is_not_ambiguous(self):
        topics = TopicContext()
        with self.steam():
            topics.process("Was ist Zzyzx Quest für ein Spiel?", self.brain)
        self.assertIsNone(topics.clarification)
        self.assertEqual(topics.game, "Zzyzx Quest")
        self.assertFalse(topics.verified)

    def test_opinion_vs_favorite_after_choice(self):
        opinion = {
            "should_store": True, "category": "gaming", "type": "game_preference",
            "game": "Crusader Kings III", "sentiment": "very_positive", "confidence": 0.95,
        }
        ai = ai_module.AI()
        with self.patches(analysis=opinion), self.steam():
            self.say(ai, "Was ist Crusader Kings für ein Spiel?")
            self.say(ai, "Crusader Kings 3")
            self.say(ai, "Ich mag es sehr.")
        entry = {g["game"]: g for g in self.brain.games}["Crusader Kings III"]
        self.assertFalse(entry["favorite"])
        self.assertEqual(entry["sentiment"], "very_positive")
        favorite = dict(opinion, type="favorite_game")
        with self.patches(analysis=favorite), self.steam():
            self.say(ai, "Das ist eines meiner Lieblingsspiele.")
        entry = {g["game"]: g for g in self.brain.games}["Crusader Kings III"]
        self.assertTrue(entry["favorite"])

    def test_arena_breakout_still_direct(self):
        topics = TopicContext()
        with self.steam():
            topics.process("Was ist Arena Breakout für ein Spiel?", self.brain)
        self.assertEqual(topics.game, "Arena Breakout: Infinite")
        self.assertIsNone(topics.clarification)


if __name__ == "__main__":
    unittest.main()
