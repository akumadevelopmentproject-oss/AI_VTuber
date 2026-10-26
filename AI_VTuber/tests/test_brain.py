import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
import state_isolation  # noqa: F401
from core import brain as brain_module
from core.ai import AI
from core.brain import Brain, analyze_memory_candidate, should_analyze
from core.game_knowledge import GameKnowledge
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


def cand(game, sentiment="positive", confidence=0.94, **extra):
    data = {
        "should_store": True,
        "category": "gaming",
        "type": "game_preference",
        "game": game,
        "sentiment": sentiment,
        "confidence": confidence,
    }
    data.update(extra)
    return data


def steam_entry(appid, name, tags=("Extraction Shooter", "FPS", "Multiplayer")):
    return {
        "appid": appid, "name": name, "genres": ["Action"], "tags": list(tags),
        "developer": "MoreFun Studios", "publisher": "MoreFun Studios",
        "short_description": "x", "release_date": "Sep 15, 2025",
        "steam_url": f"https://store.steampowered.com/app/{appid}/", "source": "steam",
        "verified": True, "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


class FakeSteam:
    """Simuliert die Steam-Store-Endpunkte; zählt Aufrufe."""

    def __init__(self, offline=False):
        self.offline = offline
        self.calls = []

    def __call__(self, url, headers=None, timeout=0):
        self.calls.append(url)
        if self.offline:
            raise OSError("offline")
        if "storesearch" in url:
            term = parse_qs(urlparse(url).query)["term"][0].lower()
            items = {
                "arena breakout": [{"type": "app", "name": "Arena Breakout: Infinite", "id": 2073620}],
                "call of duty": [
                    {"type": "app", "name": "Call of Duty: Black Ops 6", "id": 11},
                    {"type": "app", "name": "Call of Duty: Modern Warfare", "id": 12},
                ],
            }.get(term, [])
            return json.dumps({"total": len(items), "items": items})
        if "appdetails" in url:
            data = {
                "type": "game", "name": "Arena Breakout: Infinite",
                "genres": [{"id": "1", "description": "Action"}],
                "developers": ["MoreFun Studios"], "publishers": ["MoreFun Studios"],
                "short_description": "extraction shooter", "release_date": {"date": "Sep 15, 2025"},
            }
            return json.dumps({"2073620": {"success": True, "data": data}})
        tags = [{"tagid": 1, "name": "Free to Play"}, {"tagid": 2, "name": "Extraction Shooter"},
                {"tagid": 3, "name": "FPS"}]
        return f"<script>InitAppTagModal( 2073620, {json.dumps(tags)}, [], 'x' );</script>"


class SteamKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self._dir.name) / "game_cache.json"
        self.brain_path = Path(self._dir.name) / "brain.json"
        self.memory = Memory(Path(self._dir.name) / "memory.json")
        self.steam = FakeSteam()
        patcher = patch("core.game_knowledge._http_get", self.steam)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._dir.cleanup()

    def new(self):
        kn = GameKnowledge(self.cache_path, api_key="")
        return kn, Brain(self.brain_path, knowledge=kn)

    def say(self, brain, text, candidate=None, blocked=lambda t: False):
        ai, tts = FakeAI(), FakeTTS()
        with patch.object(main, "get_memory", return_value=self.memory), patch.object(
            main, "get_brain", return_value=brain
        ), patch.object(
            brain_module, "analyze_memory_candidate", return_value=candidate or {"should_store": False}
        ):
            with patch.object(main, "is_blocked_input", blocked):
                with patch.object(main, "filter_output", side_effect=lambda t: t):
                    with contextlib.redirect_stdout(io.StringIO()):
                        main.answer_user(ai, tts, text, "Luna")
        return ai, tts

    def test_resolves_arena_breakout_infinite(self):
        kn, _ = self.new()
        entry = kn.find_game("Arena Breakout")
        self.assertEqual(entry["appid"], 2073620)
        self.assertEqual(entry["name"], "Arena Breakout: Infinite")
        self.assertTrue(entry["verified"])
        self.assertEqual(entry["source"], "steam")
        self.assertIn("Extraction Shooter", entry["tags"])
        self.assertEqual(entry["developer"], "MoreFun Studios")
        self.assertNotIn("2D Shooter", json.dumps(entry))

    def test_cache_persists_and_avoids_second_lookup(self):
        kn, _ = self.new()
        kn.find_game("Arena Breakout")
        calls = len(self.steam.calls)
        kn2, _ = self.new()
        self.assertEqual(kn2.find_game("Arena Breakout")["appid"], 2073620)
        self.assertEqual(len(self.steam.calls), calls)
        self.assertIn("2073620", json.loads(self.cache_path.read_text("utf-8")))

    def test_offline_uses_cache_and_stale_cache(self):
        kn, _ = self.new()
        kn.find_game("Arena Breakout")
        kn.cache["2073620"]["updated_at"] = (datetime.now() - timedelta(days=90)).isoformat()
        self.steam.offline = True
        kn2 = GameKnowledge(self.cache_path, api_key="")
        kn2.cache = kn.cache
        entry = kn2.find_game("Arena Breakout")
        self.assertEqual(entry["appid"], 2073620)

    def test_stale_cache_refreshes_online(self):
        kn, _ = self.new()
        kn.find_game("Arena Breakout")
        kn.cache["2073620"]["updated_at"] = (datetime.now() - timedelta(days=90)).isoformat()
        kn.cache["2073620"]["tags"] = ["Alt"]
        kn.find_game("Arena Breakout")
        self.assertIn("Extraction Shooter", kn.cache["2073620"]["tags"])

    def test_offline_unknown_game_invents_nothing(self):
        self.steam.offline = True
        kn, brain = self.new()
        self.assertIsNone(kn.find_game("Arena Breakout"))
        self.assertEqual(kn.cache, {})
        _, tts = self.say(brain, "Was für ein Genre ist Arena Breakout?")
        self.assertEqual(tts.spoken, ["Das weiß ich gerade nicht sicher."])

    def test_ambiguous_game_is_not_assigned(self):
        kn, _ = self.new()
        self.assertIsNone(kn.find_game("Call of Duty"))
        self.assertEqual(kn.cache, {})

    def test_unknown_game_is_none(self):
        kn, _ = self.new()
        self.assertIsNone(kn.find_game("Voellig Unbekanntes Spiel"))

    def test_plays_is_stored_with_appid(self):
        kn, brain = self.new()
        self.say(
            brain, "Ich spiele Arena Breakout urgerne.",
            cand("Arena Breakout", "very_positive"),
        )
        entry = Brain(self.brain_path, knowledge=kn).games[0]
        self.assertEqual(entry["type"], "game_memory")
        self.assertIs(entry["plays"], True)
        self.assertEqual(entry["sentiment"], "very_positive")
        self.assertEqual(entry["appid"], 2073620)
        self.assertEqual(entry["game"], "Arena Breakout: Infinite")
        self.assertNotIn("genre", entry)
        self.assertIn("2073620", GameKnowledge(self.cache_path, online=False).cache)

    def test_old_wrong_genre_is_removed_and_normalized(self):
        old = {
            "user_memories": [],
            "game_memories": [{
                "type": "game_preference", "subject": "Miki", "game": "Arena Breakout",
                "sentiment": "very_positive", "confidence": 0.9, "mentions": 1,
                "created_at": "2026-10-14T15:57:25", "updated_at": "2026-10-14T15:57:25",
                "genre": "2D Shooter",
            }],
            "luna_memories": [],
        }
        self.brain_path.write_text(json.dumps(old), encoding="utf-8")
        kn, brain = self.new()
        self.assertNotIn("2D Shooter", self.brain_path.read_text("utf-8"))
        self.assertNotIn("genre", brain.games[0])
        brain.store_candidate("Ich spiele Arena Breakout urgerne.", cand("Arena Breakout", "very_positive"))
        self.assertEqual(len(brain.games), 1)
        self.assertEqual(brain.games[0]["game"], "Arena Breakout: Infinite")
        self.assertEqual(brain.games[0]["appid"], 2073620)
        self.assertEqual(brain.games[0]["mentions"], 2)

    def test_question_what_is_uses_game_knowledge(self):
        _, brain = self.new()
        ai, tts = self.say(brain, "Was ist Arena Breakout?")
        self.assertEqual(ai.calls, [])
        self.assertIn("Arena Breakout: Infinite", tts.spoken[0])
        self.assertIn("Extraction Shooter", tts.spoken[0])
        self.assertNotIn("2D", tts.spoken[0])

    def test_question_genre_uses_game_knowledge(self):
        _, brain = self.new()
        _, tts = self.say(brain, "Was für ein Genre ist Arena Breakout?")
        self.assertIn("Extraction Shooter", tts.spoken[0])

    def test_question_do_i_play_uses_brain(self):
        _, brain = self.new()
        self.say(brain, "Ich spiele Arena Breakout urgerne.", cand("Arena Breakout", "very_positive"))
        ai, tts = self.say(brain, "Spiele ich Arena Breakout?")
        self.assertEqual(ai.calls, [])
        self.assertIn("du spielst", tts.spoken[0])
        count = len(brain.games)
        self.assertEqual(count, 1)

    def test_question_topic_mixed(self):
        _, brain = self.new()
        self.say(brain, "Ich spiele Arena Breakout urgerne.", cand("Arena Breakout", "very_positive"))
        _, tts = self.say(brain, "Welche Extraction Shooter mag ich?")
        self.assertIn("Arena Breakout: Infinite", tts.spoken[0])
        _, tts = self.say(brain, "Was weißt du über Arena Breakout?")
        self.assertIn("Steam", tts.spoken[0])
        self.assertIn("Von dir weiß ich", tts.spoken[0])

    def test_blocked_input_no_steam_no_brain_no_cache(self):
        _, brain = self.new()
        blocked = lambda t: TEST_TERM in t
        ai, _ = self.say(
            brain, f"Ich spiele {TEST_TERM} urgerne.", cand(TEST_TERM, "very_positive"), blocked
        )
        self.assertEqual((ai.calls, brain.games, self.steam.calls), ([], [], []))
        self.assertEqual(json.loads(self.cache_path.read_text("utf-8")), {})

    def test_ollama_context_has_verified_info_and_user_memory(self):
        kn, brain = self.new()
        self.say(brain, "Ich spiele Arena Breakout urgerne.", cand("Arena Breakout", "very_positive"))
        block = brain.knowledge_block("Wie läuft Arena Breakout so?")
        self.assertIn("VERIFIED GAME INFORMATION:", block)
        self.assertIn("Name: Arena Breakout: Infinite", block)
        self.assertIn("Extraction Shooter", block)
        self.assertIn("Developer: MoreFun Studios", block)
        self.assertIn("Source: Steam", block)
        self.assertIn("USER MEMORY:", block)
        self.assertIn("Miki plays this game.", block)
        self.assertIn("very positive opinion", block)
        self.assertIn("Do not contradict or replace it with guessed information", block)
        self.assertEqual(brain.knowledge_block("Hallo wie geht es dir"), "")

    def test_unverified_game_in_context_forbids_guessing(self):
        self.steam.offline = True
        _, brain = self.new()
        brain.store_candidate("Ich spiele COD urgerne.", cand("COD", "very_positive"))
        block = brain.knowledge_block("Was denkst du über COD?")
        self.assertIn("UNVERIFIED GAME", block)
        self.assertIn("Do not guess", block)

    def test_api_key_never_printed(self):
        kn = GameKnowledge(self.cache_path, api_key="SECRETKEY123")
        out = io.StringIO()
        self.steam.offline = True
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            kn.find_game("Irgendein Spiel")
        self.assertNotIn("SECRETKEY123", out.getvalue())
        self.assertNotIn("SECRETKEY123", self.cache_path.read_text("utf-8"))


class BrainTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "brain.json"
        self.kn = GameKnowledge(Path(self._dir.name) / "game_cache.json", online=False)
        self.brain = Brain(self.path, knowledge=self.kn)
        self.memory = Memory(Path(self._dir.name) / "memory.json")

    def tearDown(self):
        self._dir.cleanup()

    def learn(self, text, candidate, brain=None):
        with patch.object(brain_module, "analyze_memory_candidate", return_value=candidate) as mock:
            result = (brain or self.brain).learn(text)
        return result, mock

    def answer(self, text, candidate=None, blocked=lambda t: False):
        ai, tts = FakeAI(), FakeTTS()
        candidate = candidate or {"should_store": False}
        with patch.object(main, "get_memory", return_value=self.memory), patch.object(
            main, "get_brain", return_value=self.brain
        ), patch.object(brain_module, "analyze_memory_candidate", return_value=candidate) as mock:
            with patch.object(main, "is_blocked_input", blocked):
                with patch.object(main, "filter_output", side_effect=lambda t: t):
                    with contextlib.redirect_stdout(io.StringIO()) as out:
                        main.answer_user(ai, tts, text, "Luna")
        return ai, tts, out.getvalue(), mock

    def test_empty_structure_created(self):
        self.assertEqual(
            json.loads(self.path.read_text("utf-8")),
            {
                "user_memories": [],
                "game_memories": [],
                "luna_memories": [],
                "shared_memories": [],
                "shared_candidates": {},
            },
        )

    def test_store_game(self):
        entry, _ = self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive"))
        self.assertEqual(entry["sentiment"], "very_positive")
        saved = json.loads(self.path.read_text("utf-8"))["game_memories"][0]
        self.assertEqual(saved["game"], "Arena Breakout")
        self.assertEqual(saved["subject"], "Miki")
        self.assertEqual(saved["mentions"], 1)

    def test_repeat_raises_mentions_and_opinion_updates(self):
        self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive"))
        self.learn("Ich spiele Arena Breakout urgerne.", cand("Arena Breakout", "positive"))
        self.assertEqual(len(self.brain.games), 1)
        self.assertEqual(self.brain.games[0]["mentions"], 2)
        self.learn("Arena Breakout ist langweilig geworden.", cand("Arena Breakout", "negative"))
        self.assertEqual(len(self.brain.games), 1)
        self.assertEqual(self.brain.games[0]["sentiment"], "negative")
        self.assertEqual(self.brain.games[0]["mentions"], 3)

    def test_question_not_stored(self):
        result, mock = self.learn("Was hältst du von Arena Breakout?", cand("Arena Breakout"))
        self.assertIsNone(result)
        mock.assert_not_called()

    def test_uncertain_not_stored(self):
        result, mock = self.learn("Vielleicht spiele ich COD.", cand("COD"))
        self.assertIsNone(result)
        mock.assert_not_called()

    def test_validation_rules(self):
        text = "Ich feier Arena Breakout mega."
        for bad in (
            cand("Arena Breakout", confidence=0.84),
            cand("Arena Breakout", confidence=True),
            cand("Arena Breakout", category="alltag") if False else {**cand("Arena Breakout"), "category": "alltag"},
            cand("Arena Breakout", sentiment="love"),
            cand("Fortnite"),
            {"should_store": False},
            {},
            None,
        ):
            result, _ = self.learn(text, bad)
            self.assertIsNone(result)
        self.assertEqual(self.brain.games, [])

    def test_sensitive_not_analyzed(self):
        self.assertFalse(should_analyze("Ich spiele COD, mein Passwort ist abc."))
        self.assertFalse(should_analyze("Wir reden über das Wetter."))
        self.assertTrue(should_analyze("Ich feier Arena Breakout mega."))

    def test_ollama_failure_is_safe(self):
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            self.assertEqual(analyze_memory_candidate("Ich mag COD"), {"should_store": False})

    def test_analyzer_request_format(self):
        captured = {}

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"message": {"content": json.dumps(cand("COD"))}}).encode()

        def fake(request, timeout=None):
            captured.update(json.loads(request.data.decode()))
            return Resp()

        with patch("urllib.request.urlopen", side_effect=fake):
            result = analyze_memory_candidate("Ich mag COD")
        self.assertTrue(result["should_store"])
        self.assertIs(captured["think"], False)
        self.assertEqual(captured["format"], "json")
        self.assertIs(captured["stream"], False)

    def test_blocked_input_never_learned(self):
        blocked = lambda t: TEST_TERM in t
        ai, tts, out, mock = self.answer(
            f"Ich feier {TEST_TERM} mega.", cand(TEST_TERM, "very_positive"), blocked
        )
        mock.assert_not_called()
        self.assertEqual((ai.calls, self.brain.games), ([], []))

    def test_main_flow_stores_and_still_asks_ollama(self):
        ai, tts, out, _ = self.answer(
            "Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive")
        )
        self.assertEqual(ai.calls, ["Ich feier Arena Breakout mega."])
        self.assertEqual(Brain(self.path, knowledge=self.kn).games[0]["game"], "Arena Breakout")

    def test_restart(self):
        self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive"))
        again = Brain(self.path, knowledge=self.kn)
        self.assertEqual(again.games[0]["sentiment"], "very_positive")

    def test_invalid_file_no_crash(self):
        self.path.write_text("{kaputt", encoding="utf-8")
        brain = Brain(self.path, knowledge=self.kn)
        self.assertEqual(brain.games, [])
        self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout"), brain)
        self.assertEqual(len(Brain(self.path, knowledge=self.kn).games), 1)

    def test_list_info_and_forget_commands(self):
        self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive"))
        self.learn("COD mag ich gar nicht.", cand("COD", "negative"))
        ai, tts, out, _ = self.answer("Welche Spiele mag ich?")
        self.assertEqual(ai.calls, [])
        self.assertIn("Arena Breakout", tts.spoken[0])
        self.assertIn("COD", tts.spoken[0])
        ai, tts, out, _ = self.answer("Was weißt du über Arena Breakout?")
        self.assertEqual(ai.calls, [])
        self.assertIn("Arena Breakout", tts.spoken[0])
        ai, tts, out, _ = self.answer("Vergiss, dass ich Arena Breakout mag.")
        self.assertEqual(ai.calls, [])
        self.assertEqual([g["game"] for g in self.brain.games], ["COD"])
        self.answer("Vergiss alles über COD.")
        self.assertEqual(self.brain.games, [])

    def test_memory_about_me_still_goes_to_memory(self):
        self.memory.add("Der Benutzer heißt Miki.")
        ai, tts, out, _ = self.answer("Was weißt du über mich?")
        self.assertIn("Miki", tts.spoken[0])

    def test_no_games_message(self):
        _, tts, _, _ = self.answer("Was weißt du über meine Spiele?")
        self.assertIn("noch nichts", tts.spoken[0])

    def test_relevant_prompt_selection(self):
        self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive"))
        self.learn("COD mag ich gar nicht.", cand("COD", "negative"))
        self.learn("Ich spiele Project Zomboid gerne.", cand("Project Zomboid", "positive"))
        general = self.brain.prompt_block("Was soll ich heute spielen?")
        self.assertIn("Miki mag Arena Breakout sehr", general)
        self.assertIn("Miki mag COD nicht", general)
        self.assertIn("Miki mag Project Zomboid", general)
        specific = self.brain.prompt_block("Wie findest du COD?")
        self.assertIn("COD", specific)
        self.assertNotIn("Arena", specific)
        self.assertEqual(self.brain.prompt_block("Wie ist das Wetter?"), "")

    def test_prompt_block_safety(self):
        self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive"))
        block = self.brain.prompt_block("Was soll ich spielen?", lambda t: True)
        self.assertEqual(block, "")

    def test_luna_interest(self):
        for appid, name, text in (
            (1, "Arena Breakout: Infinite", "Ich spiele Arena Breakout gerne."),
            (2, "Escape from Tarkov", "Ich mag Escape from Tarkov sehr."),
            (1, "Arena Breakout: Infinite", "Ich spiele Arena Breakout urgerne."),
        ):
            self.kn.cache[str(appid)] = steam_entry(appid, name)
            game = "Arena Breakout" if appid == 1 else name
            self.learn(text, cand(game))
        interest = self.brain.interests[0]
        self.assertEqual(interest["topic"], "Extraction Shooter")
        self.assertEqual(interest["strength"], 0.4)
        block = self.brain.prompt_block("Was soll ich heute spielen?")
        self.assertIn("Extraction Shooter", block)
        self.assertIn("nichts davon real erlebt", block)

    def test_ai_request_contains_brain_block(self):
        self.learn("Ich feier Arena Breakout mega.", cand("Arena Breakout", "very_positive"))
        ai = AI()
        with patch("core.ai.get_brain", return_value=self.brain), patch(
            "core.ai.get_memory", return_value=self.memory
        ):
            ai.messages.append({"role": "user", "content": "Was soll ich spielen?"})
            messages = ai._messages_with_memory("Was soll ich spielen?")
        blocks = [m["content"] for m in messages if m["role"] == "system"]
        self.assertTrue(any(b.startswith("Bekannte Spiele-Vorlieben") for b in blocks))
        self.assertEqual(len(ai.messages), 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
