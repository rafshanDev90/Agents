#P1: arXiv search client — first real stage of the paper pipeline
'''
The Problem: The agent cannot research what it cannot find. Papers have to
come from SOMEWHERE, and we need that "somewhere" to be (a) free, (b) keyless,
(c) polite (no bans), and (d) deterministic enough to test offline.

The Fix: arXiv's Atom API (no key, generous limits) wrapped in two lessons:

  1. POLITENESS IS CODE. arXiv's docs ask for >=3s between calls from the
     same client. We enforce that in one place (a module-level meter) so
     every future caller inherits it.

  2. CACHE Sits BETWEEN THE MODEL AND THE API. arXiv itself states results
     "do not change until new articles are added - no need to call more than
     once a day". So identical queries are served from SQLite within 24h.
     This is our rate-limit armor from the research phase (the 429 lesson).

Url contract (from the official manual):
    https://export.arxiv.org/api/query?search_query=all:"topic"
        &start=0&max_results=20&sortBy=relevance&sortOrder=descending
Response: Atom 1.0 XML — <feed><entry>…</entry>… per paper.

OFFLINE TEST: paper_demo.py parses a FIXED Atom document (no network), so
the parsing logic is proven even with zero internet.
'''
from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL = 3.0          # seconds between live calls (arXiv's ask)
CACHE_TTL = 24 * 3600             # arXiv updates daily at most — serves fresh
CACHE_DB = Path(__file__).resolve().parent.parent / "sessions" / "papers.db"

# Atom namespaces from the manual (they never change unless arXiv moves)
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}

_last_call = 0.0                  # module state = one politeness meter per process


# ─────────────────────────────────────────────────── politeness + cache

def _throttle() -> None:
    """Sleep just enough to keep >=3s between live arXiv calls (P1 lesson)."""
    global _last_call
    gap = time.monotonic() - _last_call
    if _last_call and gap < ARXIV_MIN_INTERVAL:
        time.sleep(ARXIV_MIN_INTERVAL - gap)
    _last_call = time.monotonic()


class QueryCache:
    """SQLite table (source, query) -> parsed JSON + timestamp.

    Key insight for P1's exit question: the cache is not a speed feature;
    it is the barrier that lets the AGENT (which loves to retry queries)
    stay inside arXiv's rate limits. Miss -> live call; hit -> instant.
    """

    def __init__(self, db_path: Path = CACHE_DB):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row   # same convention as rag_store.py
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS query_cache ("
            " source TEXT NOT NULL, query TEXT NOT NULL,"
            " fetched_at INTEGER NOT NULL, response TEXT NOT NULL,"
            " PRIMARY KEY (source, query))"
        )
        self.conn.commit()

    def get(self, source: str, url: str, now: int) -> str | None:
        row = self.conn.execute(
            "SELECT response, fetched_at FROM query_cache"
            " WHERE source = ? AND query = ?", (source, url)
        ).fetchone()
        if row is None or now - int(row["fetched_at"]) > CACHE_TTL:
            return None                       # miss (or stale) -> caller refetches
        return row["response"]

    def put(self, source: str, url: str, now: int, response: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO query_cache (source, query,"
                " fetched_at, response) VALUES (?, ?, ?, ?)",
                (source, url, now, response),
            )


# ──────────────────────────────────────────────────── query + parse (P1 core)

def build_query(topic: str, max_results: int = 20, year_from: int | None = None,
                sort_by: str = "relevance") -> str:
    """Turn a plain-language topic into the API's query syntax.

    `all:"..."` searches every field with phrase semantics (manual §5.1).
    IMPORTANT: the manual writes operators as `+AND+`, but that is just a
    URL-encoding of spaces. If you pre-embed `+` here, urlencode double-
    encodes it (%2B = literal plus) and arXiv's parser breaks. We therefore
    build the expression with real spaces and let urlencode quote it once.
    year_from adds the submittedDate range filter (manual §5.1, format
    YYYYMMDDTTTT GMT; 0000 = midnight).
    """
    phrase = " ".join(topic.replace('"', " ").split())
    search = f'all:"{phrase}"'
    if year_from:
        search += " AND submittedDate:[%d0000 TO 210001010000]" % year_from
    params = urllib.parse.urlencode({
        "search_query": search, "start": 0, "max_results": max_results,
        "sortBy": sort_by, "sortOrder": "descending",
    })
    return f"{ARXIV_API}?{params}"


def parse_atom(xml_text: str) -> list[dict]:
    """Atom XML -> plain dicts, one per paper entry.

    Every field below is validated against the manual's entry schema
    (§3.3.2): id/title/summary/authors/links/categories. Text is
    whitespace-collapsed because arXiv wraps titles across lines.
    Quirk (manual §3.4): query ERRORS arrive as a pseudo-entry whose id
    points at arxiv.org/api/errors — skip it here; search_arxiv lifts it
    into an exception via _feed_error().
    """
    entries = []
    root = ET.fromstring(xml_text)
    for entry in root.findall("atom:entry", NS):
        raw_id = entry.findtext("atom:id", "", NS)          # .../abs/2401.12345v2
        if raw_id.startswith("http://arxiv.org/api/errors"):
            continue                                        # error pseudo-entry
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]            # strip the URL prefix
        pdf_link = entry.find("atom:link[@title='pdf']", NS)
        primary = entry.find("arxiv:primary_category", NS)
        entries.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(entry.findtext("atom:title", "", NS).split()),
            "abstract": " ".join(entry.findtext("atom:summary", "", NS).split()),
            "year": int(entry.findtext("atom:published", "0000", NS)[:4] or 0),
            "authors": [a.findtext("atom:name", "", NS)
                        for a in entry.findall("atom:author", NS)],
            "primary_category": primary.get("term") if primary is not None else None,
            "categories": [c.get("term") for c in entry.findall("atom:category", NS)],
            "pdf_url": (pdf_link.get("href").replace("http://", "https://")
                        if pdf_link is not None else None),
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return entries


def search_arxiv(topic: str, max_results: int = 20, year_from: int | None = None,
                 cache: QueryCache | None = None, sort_by: str = "relevance"
                 ) -> list[dict]:
    """The one-line story: build URL -> cache hit? done. -> miss: throttle,
    fetch, validate, parse, cache, return papers. Raises on transport errors
    (caller decides retry vs fallback — P2's job)."""
    url = build_query(topic, max_results, year_from, sort_by)
    cache = cache or QueryCache()
    now = int(time.time())

    if (cached := cache.get("arxiv", url, now)) is not None:
        return json.loads(cached)                      # instant, zero live calls

    _throttle()
    with urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "senior-agent-learning/0.1 (local research project)"
    }), timeout=60) as response:                       # 200 or raise
        xml_text = response.read().decode("utf-8")

    papers = parse_atom(xml_text)
    if root_error := _feed_error(xml_text):            # malformed id → Atom error entry
        raise RuntimeError(f"arXiv query error: {root_error}")
    cache.put("arxiv", url, now, json.dumps(papers))
    return papers


def _feed_error(xml_text: str) -> str | None:
    root = ET.fromstring(xml_text)
    entry = root.find("atom:entry", NS)
    if entry is not None and entry.findtext("atom:id", "", NS).startswith(
            "http://arxiv.org/api/errors"):
        return entry.findtext("atom:summary", "unknown error", NS)
    return None
