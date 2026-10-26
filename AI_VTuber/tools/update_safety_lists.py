import os
import tempfile
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFETY_DIRECTORY = PROJECT_ROOT / "data" / "safety"
BLOCKED_WORDS_DIRECTORY = SAFETY_DIRECTORY / "blocked_words"
SOURCES_PATH = SAFETY_DIRECTORY / "SOURCES.txt"
MAX_SOURCE_BYTES = 5 * 1024 * 1024
REQUEST_TIMEOUT = 30

SOURCES = {
    "en": (
        (
            "LDNOOBW",
            "https://raw.githubusercontent.com/LDNOOBW/"
            "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en",
            "LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words",
            "en",
            "CC BY 4.0",
        ),
        (
            "google-profanity-words",
            "https://raw.githubusercontent.com/coffee-and-fun/"
            "google-profanity-words/master/data/en.txt",
            "coffee-and-fun/google-profanity-words",
            "data/en.txt",
            "MIT",
        ),
    ),
    "de": (
        (
            "LDNOOBW",
            "https://raw.githubusercontent.com/LDNOOBW/"
            "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/de",
            "LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words",
            "de",
            "CC BY 4.0",
        ),
    ),
}


def parse_entries(data):
    text = data.decode("utf-8-sig")
    entries = {}
    for line in text.splitlines():
        entry = unicodedata.normalize("NFKC", line).strip()
        if not entry or entry.startswith("#") or entry.startswith("//"):
            continue
        entry = " ".join(entry.split())
        if entry:
            entries.setdefault(entry.casefold(), entry)
    return sorted(entries.values(), key=str.casefold)


def fetch_source(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AI-VTuber-Manual-Safety-List-Updater"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            data = response.read(MAX_SOURCE_BYTES + 1)
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError("Eine Safety-Quelle konnte nicht geladen werden.") from error
    if len(data) > MAX_SOURCE_BYTES:
        raise RuntimeError("Eine Safety-Quelle überschreitet das Größenlimit.")
    return data


def build_lists(fetch=fetch_source):
    generated = {}
    source_records = []
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for language, sources in SOURCES.items():
        entries = {}
        for name, url, repository, source_file, license_name in sources:
            for entry in parse_entries(fetch(url)):
                entries.setdefault(entry.casefold(), entry)
            source_records.append(
                (name, repository, source_file, retrieved_at, license_name)
            )
        generated[language] = sorted(entries.values(), key=str.casefold)
    return generated, source_records


def _stage_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def atomic_replace_many(contents):
    staged = {}
    originals = {}
    replaced = []
    try:
        for path, content in contents.items():
            originals[path] = path.read_bytes() if path.exists() else None
            staged[path] = _stage_file(path, content)
        for path, temporary_path in staged.items():
            os.replace(temporary_path, path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            original = originals[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                rollback = _stage_file(path, original.decode("utf-8"))
                os.replace(rollback, path)
        raise
    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)


def render_source_document(records):
    lines = [
        "Manually updated local moderation data.",
        "Runtime filtering reads local files only; no startup download is performed.",
        "",
    ]
    for name, repository, source_file, retrieved_at, license_name in records:
        lines.extend(
            (
                f"Source: {name}",
                f"Repository: https://github.com/{repository}",
                f"File: {source_file}",
                f"Retrieved: {retrieved_at}",
                f"License: {license_name}",
                "",
            )
        )
    return "\n".join(lines)


def update_lists(fetch=fetch_source):
    generated, source_records = build_lists(fetch)
    contents = {
        BLOCKED_WORDS_DIRECTORY / f"{language}.txt": (
            "\n".join(entries) + ("\n" if entries else "")
        )
        for language, entries in generated.items()
    }
    contents[SOURCES_PATH] = render_source_document(source_records)
    atomic_replace_many(contents)
    return {language: len(entries) for language, entries in generated.items()}


def main():
    try:
        counts = update_lists()
    except (RuntimeError, OSError, UnicodeError) as error:
        print(f"Safety-Listen nicht aktualisiert: {error}")
        return 1

    print(f"English entries: {counts['en']}")
    print(f"German entries: {counts['de']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
