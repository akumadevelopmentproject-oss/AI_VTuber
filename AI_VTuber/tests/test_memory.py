import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
import state_isolation  # noqa: F401
from core.ai import AI
from core.brain import Brain
from core.game_knowledge import GameKnowledge
from core.memory import MAX_PROMPT_MEMORIES, Memory, extract_auto_memory

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


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "memory.json"
        self.memory = Memory(self.path)

    def tearDown(self):
        self._directory.cleanup()

    def run_answer(self, text, ai=None, tts=None, blocked=lambda text: False):
        ai = ai or FakeAI()
        tts = tts or FakeTTS()
        brain = Brain(
            Path(self._directory.name) / "brain.json",
            knowledge=GameKnowledge(Path(self._directory.name) / "game_cache.json", online=False),
        )
        no_store = {"should_store": False}
        with patch.object(main, "get_memory", return_value=self.memory), patch.object(
            main, "get_brain", return_value=brain
        ), patch("core.brain.analyze_memory_candidate", return_value=no_store):
            with patch.object(main, "is_blocked_input", blocked):
                with patch.object(main, "filter_output", side_effect=lambda t: t):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        main.answer_user(ai, tts, text, "Luna")
        return ai, tts, output.getvalue()

    def test_missing_file_is_created(self):
        self.assertTrue(self.path.is_file())
        self.assertEqual(json.loads(self.path.read_text("utf-8")), {"memories": []})

    def test_remember_confirms_without_ollama_and_speaks(self):
        ai, tts, output = self.run_answer("Merk dir, dass ich Miki heiße.")
        self.assertEqual(ai.calls, [])
        self.assertIn("Luna: Hab ich mir gemerkt, Miki.", output)
        self.assertEqual(tts.spoken, ["Hab ich mir gemerkt, Miki."])
        self.assertEqual(self.memory.memories[0]["text"], "Der Benutzer heißt Miki.")

    def test_memory_survives_restart(self):
        self.run_answer("Merk dir, dass mein Lieblingsspiel Minecraft ist.")
        restarted = Memory(self.path)
        self.assertEqual(len(restarted.memories), 1)
        self.assertIn("Minecraft", restarted.memories[0]["text"])

    def test_duplicates_are_ignored_case_insensitively(self):
        self.assertTrue(self.memory.add("Der Benutzer mag Katzen."))
        self.assertFalse(self.memory.add("der benutzer MAG katzen."))
        self.assertEqual(len(self.memory.memories), 1)

    def test_blocked_input_is_not_saved(self):
        blocked = lambda text: TEST_TERM in text
        ai, tts, output = self.run_answer(
            f"Merk dir, dass {TEST_TERM} wichtig ist.", blocked=blocked
        )
        self.assertEqual(self.memory.memories, [])
        self.assertEqual((ai.calls, tts.spoken, output), ([], [], ""))

    def test_what_do_you_know_about_me(self):
        self.memory.add("Der Benutzer heißt Miki.")
        for question in ("Was weißt du über mich?", "Was hast du dir gemerkt?", "Was weißt du von mir?"):
            ai, tts, output = self.run_answer(question)
            self.assertEqual(ai.calls, [])
            self.assertIn("Der Benutzer heißt Miki.", output)
            self.assertIn("Der Benutzer heißt Miki.", tts.spoken[0])

    def test_query_output_is_checked_by_output_safety(self):
        self.memory.add("Der Benutzer heißt Miki.")
        ai, tts = FakeAI(), FakeTTS()
        with patch.object(main, "get_memory", return_value=self.memory):
            with patch.object(main, "is_blocked_input", return_value=False):
                with patch.object(main, "filter_output", return_value=None):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        main.answer_user(ai, tts, "Was weißt du über mich?", "Luna")
        self.assertEqual((tts.spoken, output.getvalue()), ([], ""))

    def test_forget_single_memory(self):
        self.memory.add("Der Benutzer heißt Miki.")
        self.memory.add("Mikis Lieblingsspiel ist Minecraft.")
        self.run_answer("Vergiss, dass mein Lieblingsspiel Minecraft ist")
        self.assertEqual([m["text"] for m in self.memory.memories], ["Der Benutzer heißt Miki."])

    def test_forget_all_requires_confirmation(self):
        self.memory.add("Der Benutzer heißt Miki.")
        _, tts, _ = self.run_answer("Vergiss alles")
        self.assertEqual(tts.spoken, ["Willst du wirklich alle Erinnerungen löschen?"])
        self.assertEqual(len(self.memory.memories), 1)
        self.run_answer("Wie geht es dir?")
        self.assertEqual(len(self.memory.memories), 1)
        self.run_answer("Lösch deine Erinnerungen")
        self.run_answer("Ja")
        self.assertEqual(self.memory.memories, [])
        self.assertEqual(Memory(self.path).memories, [])

    def test_declined_confirmation_keeps_memories(self):
        self.memory.add("Der Benutzer heißt Miki.")
        self.run_answer("Vergiss alles")
        self.run_answer("Nein")
        self.assertEqual(len(self.memory.memories), 1)

    def test_invalid_or_empty_files_do_not_crash(self):
        for content in ("", "{kaputt", '{"memories": 5}', "[]", '{"memories": [1, {"text": 3}]}'):
            with self.subTest(content=content[:8]):
                self.path.write_text(content, encoding="utf-8")
                memory = Memory(self.path)
                self.assertEqual(memory.memories, [])
                memory.add("Der Benutzer mag Tee.")
                self.assertEqual(len(Memory(self.path).memories), 1)

    def test_at_most_twenty_memories_go_to_ollama(self):
        for number in range(30):
            self.memory.add(f"Erinnerung Nummer {number}.")
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"message":{"content":"hallo"}}'

        def fake_urlopen(request, timeout=0):
            captured.update(json.loads(request.data))
            return Response()

        ai = AI()
        with patch("core.ai.get_memory", return_value=self.memory):
            with patch("core.ai.urllib.request.urlopen", fake_urlopen):
                ai.chat("Was weißt du über mich?")
        system = [m["content"] for m in captured["messages"] if m["role"] == "system"]
        block = next(text for text in system if text.startswith("Bekannte Erinnerungen über Miki"))
        self.assertEqual(block.count("\n- "), MAX_PROMPT_MEMORIES)
        self.assertTrue(all("Bekannte Erinnerungen" not in m["content"] for m in ai.messages))

    def test_auto_patterns_extract_normalized_facts(self):
        expected = {
            "Mein Lieblingsspiel ist COD.": "Mikis Lieblingsspiel ist COD.",
            "Ich heiße Miki.": "Der Benutzer heißt Miki.",
            "Mein Name ist Miki": "Der Benutzer heißt Miki.",
            "Ich mag Minecraft.": "Der Benutzer mag Minecraft.",
            "Ich liebe Pizza": "Der Benutzer liebt Pizza.",
            "Ich hasse Spinat": "Der Benutzer hasst Spinat.",
            "Ich bevorzuge Tee": "Der Benutzer bevorzugt Tee.",
            "Ich spiele gerne Project Zomboid.": "Der Benutzer spielt gerne Project Zomboid.",
            "Ich spiele gern Minecraft": "Der Benutzer spielt gerne Minecraft.",
            "Meine Lieblingsfarbe ist Schwarz.": "Mikis Lieblingsfarbe ist Schwarz.",
        }
        for text, memory in expected.items():
            with self.subTest(text=text):
                self.assertEqual(extract_auto_memory(text), memory)

    def test_auto_ignores_questions_and_uncertain_statements(self):
        for text in (
            "Was hältst du von COD?",
            "COD ist heute im Angebot.",
            "Ich glaube vielleicht spiele ich später COD.",
            "Was ist dein Lieblingsspiel?",
            "Vielleicht mag ich das.",
            "Vielleicht ist COD mein Lieblingsspiel.",
            "Was ist mein Lieblingsspiel?",
            "Keine Ahnung.",
            "Ich mag das.",
            "Mag ich Minecraft?",
            "Ich mag Katzen, wenn sie ruhig sind.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_auto_memory(text))

    def test_auto_fact_is_saved_silently_and_still_goes_to_ollama(self):
        ai, tts, output = self.run_answer("Mein Lieblingsspiel ist COD.")
        self.assertEqual(ai.calls, ["Mein Lieblingsspiel ist COD."])
        self.assertNotIn("gemerkt", output)
        self.assertEqual(tts.spoken, ["normale Antwort"])
        restarted = Memory(self.path)
        self.assertEqual(
            [m["text"] for m in restarted.memories], ["Mikis Lieblingsspiel ist COD."]
        )
        _, _, output = self.run_answer("Was weißt du über mich?")
        self.assertIn("COD", output)

    def test_auto_duplicates_and_questions_are_not_saved(self):
        self.run_answer("Mein Lieblingsspiel ist COD.")
        self.run_answer("mein lieblingsspiel ist cod")
        self.run_answer("Was ist mein Lieblingsspiel?")
        self.run_answer("Vielleicht ist COD mein Lieblingsspiel.")
        self.assertEqual(len(self.memory.memories), 1)

    def test_blocked_input_is_not_auto_saved(self):
        blocked = lambda text: TEST_TERM in text
        self.run_answer(f"Ich mag {TEST_TERM}.", blocked=blocked)
        self.assertEqual(self.memory.memories, [])

    def test_explicit_remember_still_confirms(self):
        ai, tts, _ = self.run_answer("Merk dir, dass ich Katzen mag.")
        self.assertEqual(ai.calls, [])
        self.assertEqual(tts.spoken, ["Hab ich mir gemerkt, Miki."])
        self.assertEqual(len(self.memory.memories), 1)

    def test_integration_auto_fact_written_to_file_and_found_after_restart(self):
        ai = FakeAI()
        _, tts, _ = self.run_answer("Mein Lieblingsspiel ist COD.", ai=ai)
        self.assertEqual(ai.calls, ["Mein Lieblingsspiel ist COD."])
        on_disk = json.loads(self.path.read_text("utf-8"))
        self.assertEqual(
            [m["text"] for m in on_disk["memories"]], ["Mikis Lieblingsspiel ist COD."]
        )
        self.memory = Memory(self.path)
        _, tts, output = self.run_answer("Was weißt du über mich?")
        self.assertIn("Mikis Lieblingsspiel ist COD.", output)
        self.assertIn("COD", tts.spoken[0])

    def test_real_runtime_memory_uses_project_data_file(self):
        import core.memory as memory_module

        expected = Path(__file__).resolve().parents[1] / "data" / "memory.json"
        self.assertEqual(memory_module.MEMORY_PATH, expected)
        self.assertEqual(Memory.__init__.__defaults__[0], expected)

    def test_debug_output_only_when_enabled(self):
        with patch.object(main, "MEMORY_DEBUG", True):
            _, _, output = self.run_answer("Mein Lieblingsspiel ist COD.")
        self.assertIn("[MEMORY] gespeichert: True", output)
        _, _, output = self.run_answer("Ich mag Minecraft.")
        self.assertNotIn("[MEMORY]", output)

    def test_voice_exit_is_ignored_and_never_reaches_ollama_or_memory(self):
        phrases = (
            "Luna beenden", "okay Luna beenden", "ok Luna beenden", "okay, Luna beenden",
            "bitte Luna beenden", "also Luna beenden", "okay beenden",
            "Luna bitte beenden", "tschüss Luna", "okay tschüss",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(main.is_exit_command(phrase))
                ai, tts = FakeAI(), FakeTTS()

                class Stt:
                    def __init__(self):
                        self.items = iter((phrase,))

                    def listen(self):
                        try:
                            return next(self.items)
                        except StopIteration:
                            raise KeyboardInterrupt

                with patch.object(main, "get_memory", side_effect=AssertionError):
                    with patch.object(main, "is_blocked_input", side_effect=AssertionError):
                        with contextlib.redirect_stdout(io.StringIO()) as output:
                            with self.assertRaises(KeyboardInterrupt):
                                main.voice_loop(ai, tts, Stt())
                self.assertEqual((ai.calls, tts.spoken), ([], []))
                self.assertTrue(
                    output.getvalue().endswith("Beenden nur lokal mit Strg+C.\n")
                )

    def test_questions_about_quitting_are_not_exit(self):
        for phrase in ("Wie kann ich ein Programm beenden?", "Ich will das Spiel beenden"):
            self.assertFalse(main.is_exit_command(phrase))
        ai, _, _ = self.run_answer("Wie kann ich ein Programm beenden?")
        self.assertEqual(ai.calls, ["Wie kann ich ein Programm beenden?"])

    def test_normal_chat_still_goes_to_ollama(self):
        ai, tts, _ = self.run_answer("Erzähl mir etwas über das Wetter")
        self.assertEqual(ai.calls, ["Erzähl mir etwas über das Wetter"])
        self.assertEqual(tts.spoken, ["normale Antwort"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MemoryTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue())
    print("PASS" if result.wasSuccessful() else "FAIL")
    sys.exit(0 if result.wasSuccessful() else 1)
