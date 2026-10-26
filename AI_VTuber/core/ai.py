import json
import urllib.error
import urllib.request
from pathlib import Path

from config.settings import AI_DEBUG, CHAT_MODEL, FAST_MODEL, OLLAMA_NUM_PREDICT, OLLAMA_URL
from core.brain import get_brain
from core.conversation import (
    CONTEXT_BUDGET,
    MIN_GOAL_BUDGET,
    ResponseHistory,
    assemble_blocks,
    clean_output,
    filter_memories,
    strip_assistant_phrases,
    style_block,
)
from core.luna_goals import get_luna_goals
from core.luna_state import get_luna_state
from core.memory import get_memory
from core.safety import BLOCKED_OUTPUT_MARKER, is_blocked_output
from core.topic_context import TopicContext

PERSONALITY_PATH = Path(__file__).resolve().parent.parent / "config" / "personality.txt"
ERROR_TEXT = "Ich erreiche gerade mein KI-Modell nicht."
REWRITE_BLOCK = (
    "REWRITE REQUIRED:\n"
    "Your previous draft was too similar to a recent response.\n"
    "Answer the user's current message again using substantially different "
    "wording and sentence structure.\n"
    "Do not repeat the same opening, joke, question, or remembered fact.\n"
    "Keep the meaning relevant to the current user message."
)


def load_personality():
    try:
        return PERSONALITY_PATH.read_text(encoding="utf-8").strip()
    except OSError as error:
        print(f"Warnung: personality.txt konnte nicht geladen werden: {error}")
        return ""


class AI:
    def __init__(self):
        self.messages = [{"role": "system", "content": load_personality()}]
        self.history = ResponseHistory()
        self.topics = TopicContext()
        self.model = CHAT_MODEL

    def _messages_with_memory(self, user_text=""):
        # Kontext nur im Request, nicht im gespeicherten Verlauf.
        # Reihenfolge = Priorität: Spielwissen, Memories, Brain/Shared, State, Goals, Recent-Guidance
        recent = self.history.recent()
        ranked = []
        lookup_text = self.topics.lookup_text(user_text)

        def collect(factory):
            try:
                ranked.append(factory())
            except Exception:
                pass

        collect(lambda: get_brain().knowledge_block(lookup_text, is_blocked_output))
        collect(self.topics.prompt_block)

        def memory_block():
            texts = filter_memories(get_memory().prompt_memories(is_blocked_output), user_text)
            return "Bekannte Erinnerungen über Miki (nur nutzen, wenn passend):\n" + "\n".join(
                f"- {t}" for t in texts
            ) if texts else ""

        collect(memory_block)
        collect(lambda: get_brain().prompt_block(user_text, is_blocked_output, recent))
        collect(lambda: get_luna_state().prompt_block())
        if sum(len(b) for b in ranked if b) + MIN_GOAL_BUDGET <= CONTEXT_BUDGET:
            collect(lambda: get_luna_goals().prompt_block(user_text, is_blocked_output, recent))
        collect(self.history.guidance_block)

        blocks = [style_block(user_text, self.history)] + assemble_blocks(ranked)
        extra = [{"role": "system", "content": block} for block in blocks if block]
        return self.messages[:1] + extra + self.messages[1:]

    def _post(self, model, messages):
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_predict": OLLAMA_NUM_PREDICT},
        }
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["message"]["content"].strip()

    def _request(self, messages):
        """Ein Ollama-Aufruf mit CHAT_MODEL; ist es nicht installiert, wird FAST_MODEL benutzt."""
        try:
            return self._post(self.model, messages)
        except urllib.error.HTTPError as error:
            if error.code == 404 and self.model != FAST_MODEL:
                self.model = FAST_MODEL
                return self._post(self.model, messages)
            raise

    def chat(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        messages = self._messages_with_memory(user_text)

        try:
            answer = self._request(messages)
            if is_blocked_output(answer):
                self.messages.append(
                    {"role": "assistant", "content": BLOCKED_OUTPUT_MARKER}
                )
                return None
            answer = clean_output(strip_assistant_phrases(answer))
            if self.history.is_duplicate(answer):
                answer = self._retry(messages, answer)
                if answer is None:
                    self.messages.append(
                        {"role": "assistant", "content": BLOCKED_OUTPUT_MARKER}
                    )
                    return None
        except TimeoutError:
            return self._fail("Zeitüberschreitung bei Ollama.")
        except urllib.error.URLError as error:
            return self._fail(f"Ollama nicht erreichbar ({error.reason}). Läuft Ollama?")
        except ConnectionError as error:
            return self._fail(f"Verbindung zu Ollama unterbrochen: {error}")
        except (json.JSONDecodeError, KeyError, TypeError):
            return self._fail("Unerwartete Antwort von Ollama.")
        except Exception as error:
            return self._fail(f"Unerwarteter Fehler: {error}")

        self.messages.append({"role": "assistant", "content": answer})
        self.history.add(answer)
        return answer

    def _retry(self, messages, draft):
        """Genau ein Neuversuch. Gibt die Antwort zurück, None bei Safety-Block.

        Schlägt der Neuversuch technisch fehl, bleibt der (sichere) erste Entwurf.
        """
        if AI_DEBUG:
            print("[AI] duplicate response -> retry")
        retry_messages = messages[:-1] + [{"role": "system", "content": REWRITE_BLOCK}, messages[-1]]
        try:
            second = self._request(retry_messages)
        except Exception:
            return draft
        if not second:
            return draft
        if is_blocked_output(second):
            return None
        return clean_output(strip_assistant_phrases(second))

    def _fail(self, details):
        # Fehlgeschlagene Nutzernachricht aus dem Verlauf entfernen
        self.messages.pop()
        print(f"[Fehler] {details}")
        return ERROR_TEXT
