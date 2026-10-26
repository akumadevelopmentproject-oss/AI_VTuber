"""Objektive Spieleinformationen aus Steam (Cache in data/game_cache.json).

Das lokale Sprachmodell liefert hier nie Fakten: Genres, Tags und Entwickler
stammen ausschließlich von Steam. Ohne Steam-Treffer gibt es keine Fakten.
"""
import json
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import STEAM_WEB_API_KEY
from core.memory import _normalize
from core.safety import is_blocked_input

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "game_cache.json"
CACHE_MAX_AGE_DAYS = 30
HTTP_TIMEOUT = 8
MISS_TTL = 600
OFFLINE_TTL = 60
MAX_TAGS = 10
MIN_QUERY_LENGTH = 3
USER_AGENT = "Mozilla/5.0 (Luna-VTuber)"
SEARCH_URL = "https://store.steampowered.com/api/storesearch/"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"
APP_URL = "https://store.steampowered.com/app/{appid}/"
APP_LIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
MAX_APP_LIST_PAGES = 8
MAX_CANDIDATES = 3
# Soundtracks, Asset-/Soundpacks, DLC, Demos, Mods und Erwachsenen-/Parodie-Titel sind keine Hauptspiele
NOT_A_GAME = re.compile(
    r"\b(?:dlc|soundtracks?|ost|sfx|sound\s*packs?|sounds?|asset\s*packs?|assets?|demo|bundle|packs?|"
    r"expansions?|beta|playtest|artbook|art\s*book|upgrade|mods?|wallpapers?|skins?|"
    r"brothel|hentai|porn\w*|nsfw|erotic\w*|xxx|nude|nudes|sexy)\b",
    re.IGNORECASE,
)
ROMANS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
NUMBER_WORDS_DE = ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
TITLE_YEAR = re.compile(r"^(?:\d{4}|\d{1,2}|" + "|".join(ROMANS) + r")$")


def numeral_value(token):
    """1-9 für "3", "iii", "drei" (sonst None)."""
    token = token.casefold()
    if token.isdigit() and 1 <= int(token) <= 9:
        return int(token)
    if token in ROMANS:
        return ROMANS.index(token) + 1
    if token in NUMBER_WORDS_DE:
        return NUMBER_WORDS_DE.index(token) + 1
    return None


def numeral_variants(name):
    """"Crusader Kings 3" -> ["Crusader Kings III", "Crusader Kings drei"] (ohne den Namen selbst)."""
    parts = name.split()
    if len(parts) < 2:
        return []
    value = numeral_value(parts[-1])
    if value is None:
        return []
    forms = [str(value), ROMANS[value - 1].upper(), NUMBER_WORDS_DE[value - 1]]
    return [" ".join(parts[:-1] + [form]) for form in forms if form.casefold() != parts[-1].casefold()]


def is_junk_title(title):
    return bool(NOT_A_GAME.search(title))

GENERIC_TAGS = {
    "free to play", "multiplayer", "singleplayer", "single player", "co op", "coop",
    "online co op", "pvp", "action", "adventure", "indie", "casual", "early access",
    "great soundtrack", "atmospheric", "massively multiplayer", "realistic",
    "gun customization", "first person", "third person",
}


def _http_get(url, headers=None, timeout=HTTP_TIMEOUT):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _short(name):
    return name.split(":")[0]


def _is_stale(entry):
    try:
        updated = datetime.fromisoformat(entry.get("updated_at", ""))
    except (TypeError, ValueError):
        return True
    return datetime.now() - updated > timedelta(days=CACHE_MAX_AGE_DAYS)


def _plain_extension(title, query):
    """Titel = Suchwort + Zahl/Jahr oder Untertitel nach Doppelpunkt ("Cyberpunk 2077").

    "Cyberpunk Brothel" (zusätzliches freies Wort) zählt nicht.
    """
    rest = _normalize(title).split()[len(query.split()):]
    if not rest:
        return True
    if all(TITLE_YEAR.match(token) for token in rest):
        return True
    return _normalize(_short(title)) == query


def _numeral_order(name):
    last = (_normalize(name).split() or [""])[-1]
    romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
    if last.isdigit():
        return int(last)
    return romans.index(last) + 1 if last in romans else 0


def _match_levels(query, names_by_key):
    """(exakte Treffer, Präfix-Treffer) als Schlüsselmengen."""
    q = _normalize(query)
    q_tokens = q.split()
    exact, short, prefix = set(), set(), set()
    for key, names in names_by_key.items():
        for name in names:
            n = _normalize(name)
            if not n:
                continue
            if n == q:
                exact.add(key)
            elif _normalize(_short(name)) == q:
                short.add(key)
            elif n.split()[: len(q_tokens)] == q_tokens:
                prefix.add(key)
    # Voller Titel schlägt "Name vor dem Doppelpunkt" (sonst gewinnen DLCs wie "CK III: Royal Court")
    return (exact or short), prefix


def _match_group(query, names_by_key):
    """Alle Schlüssel der besten Trefferstufe (genau vor Präfix), sonst leer."""
    exact, prefix = _match_levels(query, names_by_key)
    return exact or prefix


def _unique_match(query, names_by_key):
    """Liefert den eindeutigen Schlüssel oder None (bei Mehrdeutigkeit lieber nichts).

    names_by_key: {key: [Namen]}. Genauer Treffer (voller Name, Name vor dem
    Doppelpunkt, Alias) schlägt Präfix-Treffer. Ein einzelnes Wort ("Cyberpunk")
    wählt nie allein per Präfix ein Spiel aus.
    """
    exact, prefix = _match_levels(query, names_by_key)
    if exact:
        return next(iter(exact)) if len(exact) == 1 else None
    if len(prefix) == 1 and len(_normalize(query).split()) >= 2:
        return next(iter(prefix))
    return None


class GameKnowledge:
    def __init__(self, path=CACHE_PATH, online=True, api_key=None):
        self.path = Path(path)
        self.online = online
        self.api_key = STEAM_WEB_API_KEY if api_key is None else api_key
        self._misses = {}
        self._offline_until = 0.0
        self._app_list = None
        self._search_items = {}
        self.cache = self._load()
        if not self.path.exists():
            self._save()

    def _load(self):
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}
        if not isinstance(loaded, dict):
            return {}
        return {
            str(key): value
            for key, value in loaded.items()
            if isinstance(value, dict) and isinstance(value.get("name"), str)
        }

    def _save(self):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".game_cache.", suffix=".tmp", dir=self.path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.cache, stream, ensure_ascii=False, indent=4)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
            return True
        except OSError:
            return False
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)

    # ---------- Cache ----------
    def get(self, appid):
        return self.cache.get(str(appid)) if appid is not None else None

    def _names(self, entry):
        return [entry["name"], *entry.get("aliases", [])]

    def canonical_index(self):
        """Lokaler Index der bekannten Steam-Spiele (aus dem Cache, wächst mit jeder Verifikation)."""
        return [
            {
                "appid": entry.get("appid"),
                "canonical_name": entry["name"],
                "normalized_name": _normalize(entry["name"]),
                "aliases": sorted({_normalize(a) for a in entry.get("aliases", []) if _normalize(a)}),
            }
            for entry in self.cache.values()
        ]

    def lookup_canonical(self, name):
        """Exakter Treffer im Index (Name, Name vor dem Doppelpunkt oder Alias), sonst None."""
        wanted = _normalize(name)
        hits = {}
        for item in self.canonical_index():
            keys = {item["normalized_name"], _normalize(_short(item["canonical_name"])), *item["aliases"]}
            if wanted in keys:
                hits[item["appid"]] = item
        if len(hits) != 1:
            return None
        return self.get(next(iter(hits)))

    def _cache_lookup(self, name):
        for term in [name, *numeral_variants(name)]:
            entry = self.lookup_canonical(term)
            if entry is not None:
                return entry
            key = _unique_match(term, {k: self._names(v) for k, v in self.cache.items()})
            if key:
                return self.cache.get(key)
        return None

    def names_for(self, entry):
        """Alle Namen, mit denen ein Spiel im Text gemeint sein kann (eindeutig)."""
        names = {_normalize(n) for n in self._names(entry)}
        names.add(_normalize(_short(entry["name"])))
        names.discard("")
        return names

    def mentioned_in(self, text):
        """Cache-Einträge, deren Name im Text vorkommt (nur eindeutige Kurznamen)."""
        padded = f" {_normalize(text)} "
        owners = {}
        for key, entry in self.cache.items():
            for name in self.names_for(entry):
                owners.setdefault(name, set()).add(key)
        found = {}
        for name, keys in owners.items():
            if len(keys) == 1 and f" {name} " in padded:
                found[next(iter(keys))] = self.cache[next(iter(keys))]
        return list(found.values())

    # ---------- Mehrdeutigkeit ----------
    def candidates(self, name, limit=MAX_CANDIDATES):
        """Mehrere plausible Spiele zu einem Namen (AMBIGUOUS), sonst leere Liste.

        Leer heißt UNKNOWN (nichts Sinnvolles) oder eindeutig (dann findet find_game es).
        """
        if not isinstance(name, str) or is_blocked_input(name):
            return []
        query = _normalize(name)
        if len(query) < MIN_QUERY_LENGTH or len(query) > 80:
            return []
        found = {}
        cached = {k: self._names(v) for k, v in self.cache.items()}
        exact_cached, prefix_cached = _match_levels(name, cached)
        for key in exact_cached or prefix_cached:
            found[key] = self.cache[key]["name"]
        items = self._search_items.get(query)
        if items is None and self.online and time.monotonic() >= self._offline_until:
            try:
                self._search_appid(name)
            except Exception:
                self._offline_until = time.monotonic() + OFFLINE_TTL
            items = self._search_items.get(query)
        exact_items, prefix_items = _match_levels(name, items or {})
        single_word = len(query.split()) == 1
        for key in exact_items or prefix_items:
            title = items[key][0]
            if is_junk_title(title):
                continue
            if single_word and not exact_items and not _plain_extension(title, query):
                continue
            found.setdefault(key, title)
        names = []
        for title in found.values():
            if _normalize(title) not in {_normalize(n) for n in names}:
                names.append(title)
        names = [
            n for n in names
            if not any(o != n and _normalize(n).startswith(_normalize(o) + " ")
                       and ":" in n and n.split(":")[0].strip() == o for o in names)
        ]
        names.sort(key=lambda n: (len(_normalize(n).split()), _numeral_order(n)))
        if single_word and not (exact_cached or exact_items) and len(names) == 1:
            # ein einziges Hauptspiel zu einem lockeren Einzelwort: nicht raten, nachfragen
            return [names[0], f"das {name.strip().capitalize()}-Genre"]
        return names[:limit] if len(names) > 1 else []

    # ---------- Suche ----------
    def find_game(self, name, allow_online=True):
        """Gibt den verifizierten Steam-Eintrag zurück oder None. Erfindet nichts."""
        if not isinstance(name, str):
            return None
        query = _normalize(name)
        if len(query) < MIN_QUERY_LENGTH or len(query) > 80 or is_blocked_input(name):
            return None

        entry = self._cache_lookup(name)
        if entry is not None:
            if _is_stale(entry) and allow_online:
                self._refresh(entry)
            return entry
        if not (allow_online and self.online):
            return None

        now = time.monotonic()
        if now < self._offline_until or now - self._misses.get(query, -MISS_TTL) < MISS_TTL:
            return None

        entry = self._lookup_online(name)
        if entry is None:
            self._misses[query] = now
            return None
        entry = self._store(entry, query)
        return entry

    def _store(self, entry, query):
        key = str(entry["appid"])
        existing = self.cache.get(key, {})
        aliases = set(existing.get("aliases", []))
        if query != _normalize(entry["name"]):
            aliases.add(query)
        entry["aliases"] = sorted(aliases)
        self.cache[key] = entry
        self._save()
        return entry

    def _refresh(self, entry):
        try:
            fresh = self._fetch_entry(entry["appid"])
        except Exception:
            fresh = None
        if fresh is None:
            return
        fresh["aliases"] = entry.get("aliases", [])
        self.cache[str(fresh["appid"])] = fresh
        self._save()
        entry.clear()
        entry.update(fresh)

    def _lookup_online(self, name):
        # Exakter Titel zuerst; Zahl, römische Zahl und Zahlwort gelten als derselbe Titel
        for term in [name, *numeral_variants(name)]:
            try:
                appid = self._search_appid(term)
            except Exception:
                self._offline_until = time.monotonic() + OFFLINE_TTL
                return None
            if appid is None:
                continue
            try:
                return self._fetch_entry(appid)
            except Exception:
                self._offline_until = time.monotonic() + OFFLINE_TTL
                return None
        return None

    def _search_appid(self, name):
        params = urllib.parse.urlencode({"term": name, "l": "english", "cc": "US"})
        data = json.loads(_http_get(f"{SEARCH_URL}?{params}"))
        asks_for_extra = bool(NOT_A_GAME.search(name))
        items = {
            str(item["id"]): [item["name"]]
            for item in data.get("items", [])
            if isinstance(item, dict)
            and item.get("type") == "app"
            and isinstance(item.get("name"), str)
            and isinstance(item.get("id"), int)
            and (asks_for_extra or not is_junk_title(item["name"]))
        }
        key = _unique_match(name, items)
        self._search_items[_normalize(name)] = items
        if key is None and self.api_key and not items:
            key = self._search_with_key(name)
        return int(key) if key else None

    def _search_with_key(self, name):
        """Optionaler Zusatzweg über die offizielle Steam Web API (Key bleibt intern)."""
        try:
            if self._app_list is None:
                apps, last = {}, 0
                for _ in range(MAX_APP_LIST_PAGES):
                    params = urllib.parse.urlencode(
                        {"key": self.api_key, "max_results": 50000, "last_appid": last,
                         "include_games": "true"}
                    )
                    response = json.loads(_http_get(f"{APP_LIST_URL}?{params}", timeout=20))
                    page = response.get("response", {})
                    for app in page.get("apps", []):
                        apps[str(app["appid"])] = [app["name"]]
                    if not page.get("have_more_results"):
                        break
                    last = page.get("last_appid", 0)
                self._app_list = apps
            return _unique_match(name, self._app_list)
        except Exception:
            return None

    def _fetch_entry(self, appid):
        params = urllib.parse.urlencode({"appids": appid, "l": "english", "cc": "US"})
        data = json.loads(_http_get(f"{DETAILS_URL}?{params}"))
        block = data.get(str(appid), {})
        info = block.get("data") if block.get("success") else None
        if not isinstance(info, dict) or info.get("type") != "game":
            return None
        name = info.get("name")
        if not isinstance(name, str) or not name.strip() or is_blocked_input(name):
            return None
        genres = [g["description"] for g in info.get("genres", []) if isinstance(g, dict) and g.get("description")]
        return {
            "appid": int(appid),
            "name": name.strip(),
            "genres": genres,
            "tags": self._fetch_tags(appid),
            "developer": ", ".join(info.get("developers") or []) or None,
            "publisher": ", ".join(info.get("publishers") or []) or None,
            "short_description": info.get("short_description") or None,
            "release_date": (info.get("release_date") or {}).get("date") or None,
            "steam_url": APP_URL.format(appid=appid),
            "source": "steam",
            "verified": True,
            "updated_at": _now(),
        }

    def _fetch_tags(self, appid):
        try:
            page = _http_get(
                APP_URL.format(appid=appid) + "?l=english",
                headers={"Cookie": "birthtime=0; wants_mature_content=1; Steam_Language=english"},
            )
            match = re.search(r"InitAppTagModal\(\s*\d+,\s*(\[.*?\])\s*,", page, re.DOTALL)
            tags = json.loads(match.group(1)) if match else []
            return [t["name"] for t in tags if isinstance(t, dict) and t.get("name")][:MAX_TAGS]
        except Exception:
            return []

    # ---------- Ausgabe ----------
    @staticmethod
    def topics(entry):
        """Verifizierte Genres und Tags ohne Dubletten."""
        seen, result = set(), []
        for value in [*entry.get("tags", []), *entry.get("genres", [])]:
            key = _normalize(value)
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result

    @staticmethod
    def specific_tags(entry):
        return [t for t in entry.get("tags", []) if _normalize(t) not in GENERIC_TAGS]

    def has_topic(self, entry, topic):
        wanted = _normalize(topic)
        return bool(wanted) and any(_normalize(t) == wanted for t in self.topics(entry))

    def is_known_topic(self, topic):
        return any(self.has_topic(entry, topic) for entry in self.cache.values())

    def fact_text(self, entry):
        topics = ", ".join(self.topics(entry)[:6]) or "unbekannt"
        parts = [f"Laut Steam ist {entry['name']} ein Spiel"]
        if entry.get("developer"):
            parts[0] += f" von {entry['developer']}"
        return f"{parts[0]}. Genres und Tags: {topics}."

    def genre_text(self, entry):
        topics = self.topics(entry)[:6]
        if not topics:
            return "Das weiß ich gerade nicht sicher."
        return f"Laut Steam: {', '.join(topics)}."

    def prompt_lines(self, entry):
        topics = ", ".join(self.topics(entry)) or "unknown"
        lines = [f"Name: {entry['name']}", f"Genres/Tags: {topics}"]
        if entry.get("developer"):
            lines.append(f"Developer: {entry['developer']}")
        if entry.get("publisher") and entry.get("publisher") != entry.get("developer"):
            lines.append(f"Publisher: {entry['publisher']}")
        if entry.get("release_date"):
            lines.append(f"Release: {entry['release_date']}")
        lines.append("Source: Steam")
        return lines


_DEFAULT = None


def get_knowledge():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GameKnowledge()
    return _DEFAULT
