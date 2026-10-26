import argparse
import re
import sys
import time
import unicodedata
from getpass import getpass

from config.settings import MEMORY_DEBUG, VOICE_INPUT_ENABLED
from core.ai import AI
from core.public_chat import PublicChat, public_response
from core.brain import get_brain
from core.luna_goals import get_luna_goals
from core.luna_state import get_luna_state
from core.memory import extract_auto_memory, get_memory
from core.safety import (
    filter_output,
    is_blocked_input,
)
from voice.stt import STT
from voice.tts import TTS

EXIT_COMMANDS = {
    "exit",
    "quit",
    "bye",
    "tschüss",
    "tschuess",
    "luna beenden",
    "beenden",
    "programm beenden",
    "hör auf",
    "hoer auf",
}


EXIT_FILLERS = {"okay", "ok", "also", "so", "gut", "dann", "na", "ja", "hey", "mal"}
EXIT_COMMANDS |= {
    "tschüss luna",
    "tschuess luna",
    "luna tschüss",
    "luna tschuess",
    "bye luna",
    "luna bye",
    "luna exit",
    "exit luna",
    "luna quit",
}


def normalize_exit_text(text):
    text = unicodedata.normalize("NFC", text).casefold()
    words = re.findall(r"[^\W_]+", text, re.UNICODE)
    return " ".join(words)


def is_exit_command(text):
    if not isinstance(text, str):
        return False
    # "bitte" ist überall harmlos; andere Füllwörter nur an Anfang/Ende
    words = [word for word in normalize_exit_text(text).split() if word != "bitte"]
    while words and words[0] in EXIT_FILLERS:
        words.pop(0)
    while words and words[-1] in EXIT_FILLERS:
        words.pop()
    return " ".join(words) in EXIT_COMMANDS


def print_exit_message():
    print("Luna wird beendet.")
    print("Bis später, Miki!")


REPEAT_COMMAND = re.compile(
    r"^\s*(?:luna[\s,]+)?(?:"
    r"sprich(?:\s+(?:mir|es|das))?\s+nach"
    r"|kannst\s+du\s+(?:das\s+|es\s+|mir\s+)?nachsprechen"
    r"|wiederhole\s+genau"
    r"|sag(?:e)?\s+genau"
    r")\s*[:,;\-]?\s*(?P<text>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def extract_repeat_text(text):
    match = REPEAT_COMMAND.match(text)
    return match.group("text") if match else None


def answer_user(ai, tts, user_text, label):
    # Untrusted audience branch MUST precede all private topic/memory/goal calls.
    if isinstance(ai, PublicChat):
        answer = public_response(ai.chat(user_text))
        if answer is not None:
            print(f"{label}: {answer}\n")
            tts.speak(answer)
        return

    if is_blocked_input(user_text):
        return

    # Spielnamen-Korrektur und Bezüge ("es", "das Spiel") erst NACH der Input-Safety
    topics = getattr(ai, "topics", None)
    text, resolved = user_text, user_text
    if topics is not None:
        try:
            text = topics.normalize(user_text, get_brain())
            resolved = topics.process(text, get_brain())
        except Exception:
            text, resolved = user_text, user_text
    user_text = text

    stt_question = getattr(topics, "stt_question", None) if topics is not None else None
    if isinstance(stt_question, str) and stt_question:
        stt_question = filter_output(stt_question)
        if stt_question is None:
            return
        print(f"{label}: {stt_question}\n")
        tts.speak(stt_question)
        return

    clarification = getattr(topics, "clarification", None) if topics is not None else None
    if isinstance(clarification, str) and clarification:
        clarification = filter_output(clarification)
        if clarification is None:
            return
        print(f"{label}: {clarification}\n")
        tts.speak(clarification)
        return

    # Ziel-Befehle vor den Brain-Befehlen ("Vergiss dein Ziel ..." ist kein Brain-Vergessen)
    try:
        goal_reply = get_luna_goals().handle_command(user_text)
    except Exception:
        goal_reply = None
    if goal_reply is not None:
        goal_reply = filter_output(goal_reply)
        if goal_reply is None:
            return
        print(f"{label}: {goal_reply}\n")
        tts.speak(goal_reply)
        return

    # Brain-Befehle (Spiele anzeigen/vergessen) vor den allgemeinen Memory-Befehlen
    brain_reply = get_brain().handle_command(resolved)
    if brain_reply is not None:
        brain_reply = filter_output(brain_reply)
        if brain_reply is None:
            return
        print(f"{label}: {brain_reply}\n")
        tts.speak(brain_reply)
        return

    # Memory-Befehle erst nach der Input-Safety prüfen
    memory_reply = get_memory().handle(user_text)
    if memory_reply is not None:
        memory_reply = filter_output(memory_reply)
        if memory_reply is None:
            return
        print(f"{label}: {memory_reply}\n")
        tts.speak(memory_reply)
        return

    # Stimmungsfragen aus dem State; ändern den State nicht
    try:
        state_reply = get_luna_state().handle_command(user_text)
    except Exception:
        state_reply = None
    if state_reply is not None:
        state_reply = filter_output(state_reply)
        if state_reply is None:
            return
        print(f"{label}: {state_reply}\n")
        tts.speak(state_reply)
        return

    repeat_text = extract_repeat_text(user_text)
    if repeat_text:
        print(f"Luna wiederholt: {repeat_text}\n")
        tts.speak(repeat_text)
        return

    # Auto-Fakten still speichern; die Nachricht geht danach normal weiter
    auto_fact = extract_auto_memory(user_text)
    if auto_fact:
        saved = get_memory().add(auto_fact)
        if MEMORY_DEBUG:
            print(f"[MEMORY] erkannt: {auto_fact}")
            print(f"[MEMORY] gespeichert: {saved}")
            print(f"[MEMORY] Datei: {get_memory().path}")

    # Brain lernt nur Gaming und nur nach bestandener Input-Safety
    new_topic = False
    try:
        learned = get_brain().learn_all(resolved)
        new_topic = isinstance(learned, dict) and learned.get("mentions") == 1
        if topics is not None:
            topics.after_learn(learned)
    except Exception:
        pass
    try:
        get_luna_state().register_conversation(resolved, new_topic)
    except Exception:
        pass
    try:
        get_luna_goals().observe(resolved, get_brain(), get_luna_state().data["curiosity"])
    except Exception:
        pass

    print(f"Du: {user_text}")
    unverified = (
        topics.unverified_reply(user_text) if topics is not None and hasattr(topics, "unverified_reply") else None
    )
    if unverified:
        unverified = filter_output(unverified)
        if unverified is None:
            return
        print(f"{label}: {unverified}\n")
        tts.speak(unverified)
        return
    started = time.perf_counter()
    raw_answer = ai.chat(user_text)
    elapsed = time.perf_counter() - started
    answer = filter_output(raw_answer)
    if answer is None:
        if ai.messages and ai.messages[-1].get("role") == "assistant":
            ai.messages[-1]["content"] = "[BLOCKED_OUTPUT]"
        return

    print(f"{label}: {answer}")
    print(f"[KI: {elapsed:.1f}s]\n")
    # speak() blockiert, bis Luna fertig gesprochen hat
    tts.speak(answer)


def voice_loop(ai, tts, stt):
    print("Luna gestartet - Sprachmodus")
    print("Sprich einfach mit Luna.")
    print("Beenden nur lokal mit Strg+C. Gesprochene Befehle schalten Luna nicht aus.\n")

    while True:
        user_text = stt.listen()
        if not user_text:
            continue

        # A microphone does not authenticate its speaker; stream audio could say this.
        if is_exit_command(user_text):
            print("Beenden nur lokal mit Strg+C.")
            continue

        answer_user(ai, tts, user_text, "Luna")


def text_loop(ai, tts):
    print("AI VTuber gestartet")
    print(
        'Schreibe "Luna beenden", "tschüss", "bye", "exit", "quit" '
        'oder "beenden", um zu beenden.\n'
    )

    while True:
        try:
            user_text = getpass("Du: ").strip()
        except EOFError:
            break

        if not user_text:
            continue
        # Exit wird immer vor dem Safety-Filter geprüft
        if is_exit_command(user_text):
            print_exit_message()
            break

        answer_user(ai, tts, user_text, "KI")


def run(private=False):
    # Public is the safe default, including local microphone input during a stream.
    ai = AI() if private else PublicChat()
    tts = TTS(public=not private)
    if private:
        print("PRIVATMODUS: Nicht im Stream benutzen. Private Erinnerungen sind hier verfügbar.")
    else:
        print("STREAMMODUS: Kein privates Gedächtnis; Zuschauer erhalten keine Steuerrechte.")
    print("Lokaler Not-Aus: Strg+C im Luna-Fenster, alternativ Fenster schließen.")

    if VOICE_INPUT_ENABLED:
        stt = STT()
        if stt.available:
            voice_loop(ai, tts, stt)
            return
        print("Spracheingabe nicht verfügbar - wechsle in den Textmodus.\n")

    text_loop(ai, tts)


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Luna: standardmäßig öffentlicher Streammodus")
    parser.add_argument('--private', action='store_true',
                        help='Privates Gedächtnis aktivieren; nicht für Streams verwenden')
    args = parser.parse_args()
    try:
        run(private=args.private)
    except KeyboardInterrupt:
        print()
        print_exit_message()


if __name__ == "__main__":
    main()
