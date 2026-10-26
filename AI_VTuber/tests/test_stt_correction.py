import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main
import state_isolation  # noqa: F401
from core import ai as ai_module
from core import stt_correction
from core.brain import Brain
from core.game_knowledge import GameKnowledge
from core.stt_correction import correct_stt_text, load_aliases
from core.topic_context import TopicContext

NOW = datetime.now().isoformat(timespec="seconds")


def entry(appid, name, aliases=()):
    return {"appid": appid, "name": name, "genres": ["Strategy"], "tags": [], "source": "steam",
            "verified": True, "aliases": list(aliases), "updated_at": NOW}


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class SttCorrectionTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.base = Path(self._dir.name)
        self.aliases = self.base / "aliases.json"
        self.aliases.write_text(json.dumps({
            "arena black breakout": "Arena Breakout",
            "arena breakout": "Arena Breakout",
            "crusader ging": "Crusader Kings",
            "crusader king": "Crusader Kings",
            "crusader kings": "Crusader Kings",
        }), encoding="utf-8")
        self.knowledge = GameKnowledge(self.base / "cache.json", online=False)
        self.knowledge.cache["1"] = entry(1, "Arena Breakout: Infinite", ["arena breakout"])
        self.knowledge.cache["2"] = entry(2, "Crusader Kings III")
        self.brain = Brain(self.base / "brain.json", knowledge=self.knowledge)
        self.requests = []

    def tearDown(self):
        self._dir.cleanup()

    def fix(self, text, **kwargs):
        kwargs.setdefault("aliases_path", self.aliases)
        return correct_stt_text(text, brain=self.brain, **kwargs)

    # ---------- Aliase ----------
    def test_default_alias_file_has_required_entries(self):
        aliases = load_aliases()
        self.assertEqual(aliases["arena black breakout"], "Arena Breakout")
        self.assertEqual(aliases["crusader ging"], "Crusader Kings")
        self.assertEqual(aliases["crusader king"], "Crusader Kings")

    def test_exact_alias_case_insensitive(self):
        self.assertEqual(self.fix("ich spiele arena black breakout gerne"), "ich spiele Arena Breakout gerne")
        self.assertEqual(self.fix("Arena BLACK Breakout"), "Arena Breakout")

    def test_crusader_variants(self):
        self.assertEqual(self.fix("Crusader ging"), "Crusader Kings")
        self.assertEqual(self.fix("Was ist Crusader King für ein Spiel?"), "Was ist Crusader Kings für ein Spiel?")
        self.assertEqual(self.fix("Crusader Kings"), "Crusader Kings")

    def test_new_alias_needs_only_the_file(self):
        self.assertEqual(self.fix("Ich mag Cyber Punk"), "Ich mag Cyber Punk")
        self.aliases.write_text(json.dumps({"cyber punk": "Cyberpunk 2077"}), encoding="utf-8")
        stt_correction._alias_cache.clear()
        self.assertEqual(self.fix("Ich mag Cyber Punk"), "Ich mag Cyberpunk 2077")

    def test_broken_alias_file_is_ignored(self):
        self.aliases.write_text("{kaputt", encoding="utf-8")
        stt_correction._alias_cache.clear()
        self.assertEqual(self.fix("Hallo Welt"), "Hallo Welt")

    # ---------- Fuzzy ----------
    def test_fuzzy_multiword_without_alias(self):
        missing = self.base / "none.json"
        self.assertEqual(self.fix("Ich mag Arena Breakot", aliases_path=missing), "Ich mag Arena Breakout")
        self.assertEqual(self.fix("Weißt du was über Arena black Breakout?", aliases_path=missing),
                         "Weißt du was über Arena Breakout?")
        self.assertEqual(self.fix("Crusader Kingz III", aliases_path=missing), "Crusader Kings III")

    def test_single_word_is_not_converted(self):
        for text in ("King", "Ich bin der King", "Kings", "crusader"):
            self.assertEqual(self.fix(text, aliases_path=self.base / "none.json"), text)

    def test_normal_words_unchanged(self):
        for text in (
            "Wie ist das Wetter heute?",
            "Ich mag den Arena der Stadt",
            "Heute gehe ich in die Arena und trinke Kaffee",
            "Was machen die Kings of Leon?",
        ):
            self.assertEqual(self.fix(text), text)

    def test_active_game_helps(self):
        missing = self.base / "none.json"
        self.assertEqual(self.fix("Wie findest du Dark Souls Remaster", aliases_path=missing),
                         "Wie findest du Dark Souls Remaster")
        self.assertEqual(
            self.fix("Ich mag Dark Sols Remaster", aliases_path=missing, active_game="Dark Souls Remaster"),
            "Ich mag Dark Souls Remaster",
        )

    def test_game_knowledge_param_without_brain(self):
        result = correct_stt_text("Ich mag Crusader Kingz III", game_knowledge=self.knowledge,
                                  aliases_path=self.base / "none.json")
        self.assertEqual(result, "Ich mag Crusader Kings III")

    # ---------- Unsicher / Nachfragen ----------
    def test_unsure_match_is_not_changed(self):
        missing = self.base / "none.json"
        text = "Ich mag Crusader Kingdom"
        self.assertEqual(self.fix(text, aliases_path=missing), text)
        result, question = stt_correction.analyze_stt_text(
            "Ich mag Crusader Kinds", self.brain, aliases_path=missing
        )
        self.assertEqual(result, "Ich mag Crusader Kinds")

    def test_unsure_match_asks_single_candidate(self):
        self.knowledge.cache["3"] = entry(3, "Crusader Kings II")
        topics = TopicContext()
        with patch.object(stt_correction, "ALIAS_PATH", self.base / "none.json"):
            text = topics.normalize("Was ist Crusaders Kinds für ein Spiel?", self.brain)
        self.assertEqual(text, "Was ist Crusaders Kinds für ein Spiel?")
        self.assertIsNotNone(topics.stt_question)
        self.assertTrue(topics.stt_question.startswith("Meinst du Crusader Kings"))

    def test_unsure_single_candidate_confirmation(self):
        self.knowledge.cache.pop("3", None)
        names = {"crusader kings": "Crusader Kings"}
        topics = TopicContext()
        with patch.object(stt_correction, "ALIAS_PATH", self.base / "none.json"), \
                patch.object(stt_correction, "known_names", return_value=names):
            topics.normalize("Was ist Crusadr Kinz für ein Spiel?", self.brain)
            self.assertEqual(topics.stt_question, "Meinst du Crusader Kings?")
            fixed = topics.normalize("Ja genau", self.brain)
        self.assertEqual(fixed, "Was ist Crusader Kings für ein Spiel?")
        self.assertIsNone(topics.stt_question)

    def test_unsure_single_candidate_rejection(self):
        names = {"crusader kings": "Crusader Kings"}
        topics = TopicContext()
        with patch.object(stt_correction, "ALIAS_PATH", self.base / "none.json"), \
                patch.object(stt_correction, "known_names", return_value=names):
            topics.normalize("Was ist Crusadr Kinz für ein Spiel?", self.brain)
            self.assertEqual(topics.normalize("Nein, anderes Thema", self.brain), "Nein, anderes Thema")
        self.assertIsNone(topics.stt_pending)

    def test_two_equally_good_names_ask_instead_of_correcting(self):
        names = {"blue moon": "Blue Moon", "blue room": "Blue Room"}
        with patch.object(stt_correction, "known_names", return_value=names):
            result, question = stt_correction.analyze_stt_text(
                "Ich mag Blue Roon", aliases_path=self.base / "none.json"
            )
        self.assertEqual(result, "Ich mag Blue Roon")
        self.assertIsNotNone(question)
        self.assertEqual(question["options"], ["Blue Moon", "Blue Room"])
        self.assertEqual(stt_correction.question_for(question["options"]), "Meinst du Blue Moon oder Blue Room?")

    def test_user_picks_one_of_two(self):
        names = {"blue moon": "Blue Moon", "blue room": "Blue Room"}
        topics = TopicContext()
        with patch.object(stt_correction, "ALIAS_PATH", self.base / "none.json"), \
                patch.object(stt_correction, "known_names", return_value=names):
            topics.normalize("Ich mag Blue Roon", self.brain)
            self.assertEqual(topics.stt_question, "Meinst du Blue Moon oder Blue Room?")
            self.assertEqual(topics.normalize("Blue Moon", self.brain), "Ich mag Blue Moon")

    # ---------- Reihenfolge / Safety ----------
    def test_safety_runs_before_correction(self):
        ai = ai_module.AI()
        ai.topics = MagicMock()
        with patch.object(main, "is_blocked_input", return_value=True), \
                patch.object(stt_correction, "correct_stt_text") as correction:
            main.answer_user(ai, FakeTTS(), "blockiert", "Luna")
        ai.topics.normalize.assert_not_called()
        correction.assert_not_called()

    def test_main_asks_question_without_calling_ollama(self):
        ai = ai_module.AI()
        ai.topics.normalize = MagicMock(return_value="x")
        ai.topics.stt_question = "Meinst du Crusader Kings?"
        tts = FakeTTS()
        with patch.object(main, "get_brain", return_value=self.brain), \
                patch.object(main, "filter_output", side_effect=lambda t: t), \
                patch.object(ai_module.urllib.request, "urlopen", side_effect=AssertionError("kein Ollama")), \
                contextlib.redirect_stdout(io.StringIO()):
            main.answer_user(ai, tts, "Crusadr Kinz", "Luna")
        self.assertEqual(tts.spoken, ["Meinst du Crusader Kings?"])

    # ---------- Debug ----------
    def test_debug_output_only_when_enabled(self):
        with patch.object(stt_correction.settings, "STT_DEBUG", False), \
                contextlib.redirect_stdout(io.StringIO()) as quiet:
            self.fix("Crusader ging")
        self.assertEqual(quiet.getvalue(), "")
        with patch.object(stt_correction.settings, "STT_DEBUG", True), \
                contextlib.redirect_stdout(io.StringIO()) as loud:
            self.fix("Crusader ging")
        self.assertIn("[STT FIX] Crusader Kings", loud.getvalue())
        self.assertIn("[STT FIX] Crusader Kings", loud.getvalue())


if __name__ == "__main__":
    unittest.main()
