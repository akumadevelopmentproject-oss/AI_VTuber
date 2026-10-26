import contextlib
import io
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main
import test_dialogue_quality as dq
from core import ai as ai_module
from core import game_knowledge as gk_module
from core.topic_context import TopicContext

NOW = datetime.now().isoformat(timespec="seconds")

CATALOG = [
    (203770, "Crusader Kings II"),
    (1158310, "Crusader Kings III"),
    (1091500, "Cyberpunk 2077"),
    (1, "Cyberpunk SFX Sound Pack"),
    (2, "Cyberpunk Brothel"),
    (3, "Cyberpunk 2077 Soundtrack"),
    (4, "Cyberpunk Inquisitor"),
    (2073620, "Arena Breakout: Infinite"),
]


class LiveFixTests(dq.QualityTests):
    def topics(self):
        return TopicContext()

    def fix(self, topics, text):
        with self.steam(CATALOG):
            return topics.normalize(text, self.brain)

    def test_rosetta_kings_3(self):
        self.assertEqual(self.fix(self.topics(), "Rosetta Kings 3"), "Crusader Kings III")
        self.assertEqual(self.fix(self.topics(), "Rosetta Kings drei"), "Crusader Kings III")

    def test_crusader_king_3(self):
        self.assertEqual(self.fix(self.topics(), "Crusader King 3"), "Crusader Kings III")
        self.assertEqual(self.fix(self.topics(), "Crusader Kings zwei"), "Crusader Kings II")

    def test_breakout_infinite_warfare(self):
        self.assertEqual(self.fix(self.topics(), "Breakout Infinite Warfare"), "Arena Breakout: Infinite")
        self.assertIn("Arena Breakout: Infinite", self.fix(self.topics(), "Was ist Breakout Infinite?"))

    def test_numeral_variants_same_title(self):
        from core.game_knowledge import numeral_variants
        self.assertIn("Crusader Kings III", numeral_variants("Crusader Kings 3"))
        self.assertIn("Crusader Kings III", numeral_variants("Crusader Kings drei"))
        self.assertIn("Crusader Kings II", numeral_variants("Crusader Kings zwei"))

    def test_cyberpunk_has_no_junk_candidates(self):
        with self.steam(CATALOG):
            options = self.brain.knowledge.candidates("Cyberpunk")
        self.assertEqual(options, ["Cyberpunk 2077", "das Cyberpunk-Genre"])
        self.assertEqual(TopicContext._question(options), "Meinst du Cyberpunk 2077 oder das Cyberpunk-Genre?")
        for bad in ("SFX", "Brothel", "Soundtrack", "Inquisitor"):
            self.assertFalse(any(bad in o for o in options))

    def test_cyberpunk_2077_exact(self):
        with self.steam(CATALOG):
            info = self.brain.knowledge.find_game("Cyberpunk 2077")
        self.assertEqual(info["name"], "Cyberpunk 2077")

    def test_exact_verification_with_numeral(self):
        with self.steam(CATALOG):
            info = self.brain.knowledge.find_game("Crusader Kings III")
            self.assertEqual((info["name"], info["appid"]), ("Crusader Kings III", 1158310))
            self.assertEqual(self.brain.knowledge.find_game("Crusader Kings 3")["appid"], 1158310)
        index = {i["canonical_name"]: i for i in self.brain.knowledge.canonical_index()}
        self.assertEqual(index["Crusader Kings III"]["appid"], 1158310)
        self.assertEqual(index["Crusader Kings III"]["normalized_name"], "crusader kings iii")

    def test_exact_title_beats_dlc_sharing_the_short_name(self):
        catalog = CATALOG + [(5, "Crusader Kings III: Royal Court"), (6, "Crusader Kings III: Northern Lords")]
        with self.steam(catalog):
            info = self.brain.knowledge.find_game("Crusader Kings III")
        self.assertEqual(info["appid"], 1158310)

    def test_pending_clarification_with_rosetta(self):
        for reply in ("Rosetta Kings 3", "Crusader Kings 3", "den dritten", "III", "drei"):
            ai = ai_module.AI()
            with self.patches(), self.steam(CATALOG):
                self.say(ai, "Was ist Crusader Kings für ein Spiel?")
                self.assertIsNotNone(ai.topics.pending, reply)
                self.say(ai, reply)
            self.assertEqual(ai.topics.game, "Crusader Kings III", reply)
            self.assertIsNone(ai.topics.pending, reply)
            self.assertIsNone(ai.topics.clarification, reply)

    def test_active_game_survives_bad_stt(self):
        topics = TopicContext()
        topics.set_game("Crusader Kings III", True)
        with self.steam(CATALOG):
            text = topics.normalize("Rosetta Kings", self.brain)
            topics.process(text, self.brain)
        self.assertEqual(topics.game, "Crusader Kings III")
        self.assertIsNone(topics.clarification)
        self.assertIsNone(topics.stt_question)

    def test_unverified_game_question_never_reaches_ollama(self):
        ai = ai_module.AI()
        ai.topics.set_game("Unbekannt Spiel", False)
        with self.patches():
            tts = self.say(ai, "Worum geht es in dem Spiel?")
        self.assertEqual(self.requests, [])
        self.assertEqual(len(tts.spoken), 1)

    def test_verified_game_answered_from_facts(self):
        ai = ai_module.AI()
        with self.patches():
            tts = self.say(ai, "Was ist Arena Breakout Infinite?")
        self.assertIn("MoreFun Studios", tts.spoken[0])


class WhisperFallbackTests(unittest.TestCase):
    def make(self, engine_error):
        from voice import stt as stt_module
        with patch.object(stt_module, "IMPORT_ERROR", None), \
                patch.object(stt_module, "sr", MagicMock(), create=True), \
                patch.object(stt_module.STT, "_calibrate", lambda self: None), \
                patch.object(stt_module, "WhisperEngine", side_effect=engine_error), \
                contextlib.redirect_stdout(io.StringIO()):
            return stt_module.STT()

    def test_init_failure_falls_back_to_google(self):
        self.assertEqual(self.make(RuntimeError("kaputt")).engine, "google")

    def test_runtime_failure_switches_to_google(self):
        from voice import stt as stt_module
        stt = self.make(RuntimeError("x"))
        stt.engine = "whisper"
        stt.whisper = MagicMock()
        stt.whisper.transcribe.side_effect = RuntimeError("boom")
        stt.recognizer = MagicMock()
        stt.recognizer.recognize_google.return_value = "hallo"
        with patch.object(stt_module, "sr", MagicMock(), create=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(stt._recognize(b"\0\0"), "hallo")
        self.assertEqual(stt.engine, "google")


if __name__ == "__main__":
    unittest.main()
