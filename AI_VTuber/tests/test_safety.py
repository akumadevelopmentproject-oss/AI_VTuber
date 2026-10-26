import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main
import state_isolation  # noqa: F401
import core.safety as safety
from core.ai import AI
from core.safety import (
    SAFETY_DIRECTORY,
    SafetyFilter,
    load_blocked_phrases,
    load_blocked_words,
)

TEST_TERM = "blockedfixture"
TEST_PHRASE = "danger fixture"
TEST_EXCEPTION = "allowedfixture"


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.filter = SafetyFilter(
            blocked_words=(TEST_TERM, TEST_EXCEPTION),
            blocked_phrases=(TEST_PHRASE,),
            exceptions=(TEST_EXCEPTION,),
        )

    def test_ordinary_text_passes(self):
        self.assertFalse(self.filter.contains_blocked_content("Miki says hello."))

    def test_case_reverse_and_separated_variants_block(self):
        variants = (
            TEST_TERM.upper(),
            TEST_TERM[::-1],
            " ".join(TEST_TERM),
            ".".join(TEST_TERM),
            "-".join(TEST_TERM),
            "_".join(TEST_TERM),
            "b|0ck3df1xtur3",
            TEST_TERM.replace("e", "\u0435"),
            "blllockedfixture",
        )
        for variant in variants:
            with self.subTest(length=len(variant)):
                self.assertTrue(self.filter.contains_blocked_content(variant))
        self.assertTrue(SafetyFilter(("fulltest",)).contains_blocked_content("fu11test"))

    def test_phrase_is_checked(self):
        self.assertTrue(
            self.filter.contains_blocked_content("This is a danger fixture.")
        )
        self.assertTrue(
            self.filter.contains_blocked_content("d.a.n.g.e.r-f.i.x.t.u.r.e")
        )
        self.assertTrue(
            self.filter.contains_blocked_content("d a n g e r f i x t u r e")
        )

    def test_current_local_starter_lists_are_enforced(self):
        words = load_blocked_words()
        phrases = load_blocked_phrases()
        current_filter = SafetyFilter(words, phrases)
        self.assertGreater(len(words), 0)
        self.assertGreater(len(phrases), 0)
        self.assertFalse(current_filter.contains_blocked_content("I am gay and happy."))
        self.assertFalse(current_filter.contains_blocked_content("class assignment"))

    def test_allowed_exception_takes_precedence(self):
        self.assertFalse(
            self.filter.contains_blocked_content("allowedfixture is allowed.")
        )

    def test_all_owned_language_files_exist_and_load(self):
        self.assertTrue((SAFETY_DIRECTORY / "blocked_words" / "en.txt").is_file())
        self.assertTrue((SAFETY_DIRECTORY / "blocked_words" / "de.txt").is_file())
        self.assertTrue((SAFETY_DIRECTORY / "custom_words" / "en.txt").is_file())
        self.assertTrue((SAFETY_DIRECTORY / "custom_words" / "de.txt").is_file())
        self.assertGreater(len(load_blocked_words()), 0)

    def test_word_loader_ignores_comments_and_empty_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            word_file = directory / "custom.txt"
            word_file.write_text(
                "# local comments\n\nblockedfixture\n", encoding="utf-8"
            )
            self.assertEqual(
                load_blocked_words(word_file),
                {"blockedfixture"},
            )

    def test_generated_and_custom_directories_are_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated"
            custom = root / "custom"
            generated.mkdir()
            custom.mkdir()
            (generated / "en.txt").write_text("generatedfixture\n", encoding="utf-8")
            (custom / "de.txt").write_text("customfixture\n", encoding="utf-8")
            with patch.object(safety, "BLOCKED_WORDS_DIRECTORY", generated):
                with patch.object(safety, "CUSTOM_WORDS_DIRECTORY", custom):
                    loaded = load_blocked_words()
            self.assertIn("generatedfixture", loaded)
            self.assertIn("customfixture", loaded)

    def test_source_lists_are_merged_and_normalized(self):
        from tools.update_safety_lists import SOURCES, build_lists

        responses = {
            SOURCES["en"][0][1]: b"# comment\nNeutralFixture\n",
            SOURCES["en"][1][1]: b"neutralfixture\nSecondFixture\n",
            SOURCES["de"][0][1]: b"  Doppelt  \n# ignored\n",
        }
        generated, records = build_lists(lambda url: responses[url])
        self.assertEqual(generated["en"], ["NeutralFixture", "SecondFixture"])
        self.assertEqual(generated["de"], ["Doppelt"])
        self.assertEqual(len(records), 3)

    def test_failed_download_preserves_existing_generated_lists(self):
        import tools.update_safety_lists as updater

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "blocked_words"
            generated.mkdir()
            old_files = {
                generated / "en.txt": "old english\n",
                generated / "de.txt": "old german\n",
                root / "SOURCES.txt": "old metadata\n",
            }
            for path, content in old_files.items():
                path.write_text(content, encoding="utf-8")
            calls = 0

            def failed_fetch(_url):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("download failed")
                return b"newfixture\n"

            with patch.object(updater, "BLOCKED_WORDS_DIRECTORY", generated):
                with patch.object(updater, "SOURCES_PATH", root / "SOURCES.txt"):
                    with self.assertRaises(RuntimeError):
                        updater.update_lists(failed_fetch)
            for path, content in old_files.items():
                self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_updater_does_not_overwrite_custom_words(self):
        import tools.update_safety_lists as updater

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "blocked_words"
            custom = root / "custom_words"
            custom.mkdir()
            custom_file = custom / "en.txt"
            custom_file.write_text("ownedfixture\n", encoding="utf-8")
            responses = {
                source[1]: b"generatedfixture\n"
                for sources in updater.SOURCES.values()
                for source in sources
            }
            with patch.object(updater, "BLOCKED_WORDS_DIRECTORY", generated):
                with patch.object(updater, "SOURCES_PATH", root / "SOURCES.txt"):
                    updater.update_lists(lambda url: responses[url])
            self.assertEqual(
                custom_file.read_text(encoding="utf-8"), "ownedfixture\n"
            )

    def test_transformation_request_checks_candidate(self):
        self.assertTrue(
            self.filter.contains_blocked_content(
                "buchstabiere b l o c k e d f i x t u r e"
            )
        )

    def test_blocked_input_is_silent_and_never_reaches_ai(self):
        class FakeAI:
            def __init__(self):
                self.calls = []
                self.messages = []

            def chat(self, text):
                self.calls.append(text)
                return "safe response"

        class FakeTTS:
            def __init__(self):
                self.spoken = []

            def speak(self, text):
                self.spoken.append(text)

        fake_ai = FakeAI()
        fake_tts = FakeTTS()
        with patch.object(
            main, "is_blocked_input", self.filter.contains_blocked_content
        ):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main.answer_user(fake_ai, fake_tts, TEST_TERM[::-1], "Luna")
        self.assertEqual(fake_ai.calls, [])
        self.assertEqual(fake_tts.spoken, [])
        self.assertEqual(output.getvalue(), "")

    def test_text_mode_does_not_echo_blocked_input(self):
        class FakeAI:
            messages = []

            def chat(self, text):
                raise AssertionError

        class FakeTTS:
            def speak(self, text):
                raise AssertionError

        replies = iter((TEST_TERM, "exit"))
        with patch("main.getpass", side_effect=lambda prompt: next(replies)):
            with patch(
                "main.is_blocked_input",
                side_effect=lambda text: self.filter.contains_blocked_content(text),
            ):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main.text_loop(FakeAI(), FakeTTS())
        self.assertNotIn(TEST_TERM, output.getvalue())

    def test_blocked_output_does_not_reach_terminal_tts_or_history(self):
        class FakeAI:
            def __init__(self):
                self.messages = [{"role": "assistant", "content": TEST_TERM}]

            def chat(self, text):
                return TEST_TERM

        class FakeTTS:
            def __init__(self):
                self.spoken = []

            def speak(self, text):
                self.spoken.append(text)

        fake_ai = FakeAI()
        fake_tts = FakeTTS()
        with patch.object(main, "is_blocked_input", return_value=False):
            with patch(
                "main.filter_output",
                side_effect=lambda text: (
                    None if self.filter.contains_blocked_content(text) else text
                ),
            ):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main.answer_user(fake_ai, fake_tts, "hello", "Luna")
        self.assertNotIn(TEST_TERM, output.getvalue())
        self.assertEqual(fake_tts.spoken, [])
        self.assertEqual(fake_ai.messages[-1]["content"], "[BLOCKED_OUTPUT]")

    def test_ai_stores_marker_instead_of_blocked_answer(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return ('{"message":{"content":"' + TEST_TERM + '"}}').encode()

        ai = AI()
        with patch("core.ai.urllib.request.urlopen", return_value=FakeResponse()):
            with patch("core.ai.is_blocked_output", return_value=True):
                result = ai.chat("hello")
        self.assertIsNone(result)
        self.assertEqual(ai.messages[-1]["content"], "[BLOCKED_OUTPUT]")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SafetyTests)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream).run(suite)
    if not result.wasSuccessful():
        print(stream.getvalue())
    print("PASS" if result.wasSuccessful() else "FAIL")
    sys.exit(0 if result.wasSuccessful() else 1)
