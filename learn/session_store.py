#L6 (evolved): Professional session persistence — SQLite
'''
The Problem: JSON sessions are rewritten whole and non-atomically on every
turn (a crash mid-write corrupts them), cannot be queried ("list my sessions",
"which one covered X?"), and accumulate SDK serialization noise forever.

The Fix: SessionStore over stdlib sqlite3 (zero new dependencies):
  * atomic transactions          — crash-safe saves
  * append-only incremental save — only NEW messages are written per turn
  * sessions table               — list / resume / delete with metadata
  * FTS5 index                   — ready for /session search (phase 4)
  * embeddings can go in a column later — SQLite becomes the vector store
    without adding a vector DB server
'''

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from session import agentSession

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "sessions" / "agent.db"

# Columns we persist. Everything else Ollama ignores anyway (measured:
# reasoning/annotations/audio/refusal cost 0 tokens on input), except
# `reasoning`, which we keep for learning/debug visibility but do NOT count
# toward the token budget.
_CANONICAL_FIELDS = ("role", "content", "tool_calls", "tool_call_id", "name", "reasoning")


def estimate_tokens(text: str) -> int:
    """Cheap ~chars/4 heuristic — good enough for budgeting, not billing."""
    return len(text) // 4


def canonicalize(message: Any) -> Dict[str, Any]:
    """Reduce a history entry (dict or SDK message object) to stable fields."""
    if hasattr(message, "model_dump"):
        message = message.model_dump()
    if not isinstance(message, dict):
        raise TypeError(f"Cannot persist message of type {type(message)!r}")

    entry: Dict[str, Any] = {
        "role": message.get("role", "assistant"),
        "content": message.get("content") or "",
    }

    tool_calls = message.get("tool_calls")
    if tool_calls:
        slim_calls = []
        for call in tool_calls:
            call = call or {}
            function = call.get("function") or {}
            slim_calls.append(
                {
                    "id": call.get("id"),
                    "type": "function",
                    "function": {
                        "name": function.get("name"),
                        "arguments": function.get("arguments") or "{}",
                    },
                }
            )
        entry["tool_calls"] = slim_calls

    if entry["role"] == "tool":
        entry["tool_call_id"] = message.get("tool_call_id")
        entry["name"] = message.get("name") or message.get("tool_name")

    if message.get("reasoning"):
        entry["reasoning"] = message["reasoning"]

    return entry


def estimate_entry_tokens(entry: Dict[str, Any]) -> int:
    """Prompt-token estimate. `reasoning` excluded — Ollama ignores it on input."""
    tokens = estimate_tokens(entry.get("content") or "")
    if entry.get("tool_calls"):
        tokens += estimate_tokens(json.dumps(entry["tool_calls"]))
    return tokens


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


def _auto_title(entries: List[Dict[str, Any]]) -> str:
    for entry in entries:
        if entry["role"] == "user":
            content = " ".join(entry["content"].split())
            return content[:47] + "…" if len(content) > 48 else content
    return ""


class SessionStore:
    """SQLite-backed replacement for the JSON save/load loop."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.fts_enabled = False
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id             TEXT PRIMARY KEY,
                title          TEXT NOT NULL DEFAULT '',
                created_at     INTEGER NOT NULL,
                updated_at     INTEGER NOT NULL,
                turn_count     INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                session_id    TEXT NOT NULL,
                seq           INTEGER NOT NULL,
                role          TEXT NOT NULL,
                content       TEXT NOT NULL DEFAULT '',
                tool_calls    TEXT,
                tool_call_id  TEXT,
                tool_name     TEXT,
                reasoning     TEXT,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (session_id, seq)
            );
            """
        )
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5("
                "session_id UNINDEXED, seq UNINDEXED, content)"
            )
            self.fts_enabled = True
        except sqlite3.OperationalError:  # SQLite built without FTS5
            self.fts_enabled = False
        self.conn.commit()

    # ------------------------------------------------------------------ write

    def save(self, session: agentSession) -> None:
        """Incremental, atomic save: only messages not yet in the DB are written."""
        session_id = session.session_id
        entries = [canonicalize(message) for message in session.history]
        now = int(time.time())

        with self.conn:
            row = self.conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at,"
                    " turn_count, token_estimate) VALUES (?, ?, ?, ?, 0, 0)",
                    (session_id, _auto_title(entries), now, now),
                )
            elif not row["title"]:
                auto_title = _auto_title(entries)
                if auto_title:
                    self.conn.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        (auto_title, session_id),
                    )

            next_seq = self._next_seq(session_id)
            for seq, entry in enumerate(entries[next_seq:], start=next_seq):
                self._insert_message(session_id, seq, entry, now)

            turn_count = sum(1 for entry in entries if entry["role"] == "user")
            token_total = sum(estimate_entry_tokens(entry) for entry in entries)
            self.conn.execute(
                "UPDATE sessions SET updated_at = ?, turn_count = ?,"
                " token_estimate = ? WHERE id = ?",
                (now, turn_count, token_total, session_id),
            )

    def _next_seq(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS next FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["next"])

    def _insert_message(self, session_id: str, seq: int, entry: Dict[str, Any], now: int) -> None:
        tool_calls = json.dumps(entry["tool_calls"]) if entry.get("tool_calls") else None
        tokens = estimate_entry_tokens(entry)
        self.conn.execute(
            "INSERT INTO messages (session_id, seq, role, content, tool_calls,"
            " tool_call_id, tool_name, reasoning, token_estimate)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                seq,
                entry["role"],
                entry["content"],
                tool_calls,
                entry.get("tool_call_id"),
                entry.get("name"),
                entry.get("reasoning"),
                tokens,
            ),
        )
        if self.fts_enabled:
            index_text = (entry["content"] or "").strip() or (tool_calls or "")
            if index_text:
                self.conn.execute(
                    "INSERT INTO messages_fts (session_id, seq, content) VALUES (?, ?, ?)",
                    (session_id, seq, index_text),
                )

    # ------------------------------------------------------------------- read

    def load(self, session_id: str) -> agentSession:
        session = agentSession(session_id=session_id)
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
        if rows:
            session.history = [self._rehydrate(row) for row in rows]
        return session

    @staticmethod
    def _rehydrate(row: sqlite3.Row) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"role": row["role"], "content": row["content"]}
        if row["tool_calls"]:
            entry["tool_calls"] = json.loads(row["tool_calls"])
        if row["tool_call_id"]:
            entry["tool_call_id"] = row["tool_call_id"]
        if row["tool_name"]:
            entry["name"] = row["tool_name"]
        if row["reasoning"]:
            entry["reasoning"] = row["reasoning"]
        return entry

    def list_sessions(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, s.turn_count,"
            " s.token_estimate,"
            " (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count"
            " FROM sessions s ORDER BY s.updated_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def resolve(self, identifier: str) -> List[str]:
        """Exact id, else case-insensitive prefix matches (newest first)."""
        if not identifier:
            return []
        rows = self.conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        ids = [row["id"] for row in rows]
        if identifier in ids:
            return [identifier]
        needle = identifier.lower()
        return [session_id for session_id in ids if session_id.lower().startswith(needle)]

    def exists(self, session_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------ search

    def search(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Full-text search across ALL sessions.

        FTS5 (bm25-ranked) when available; plain LIKE across messages as a
        fallback. Every word is quoted so user input can't inject FTS5
        operators (AND/OR/NEAR/…) — a literal keyword search.
        """
        terms = [term for term in re.findall(r"\w+", query) if term]
        if not terms:
            return []

        if self.fts_enabled:
            match = " ".join(f'"{term}"' for term in terms)
            rows = self.conn.execute(
                "SELECT messages_fts.session_id AS session_id, messages_fts.seq AS seq,"
                " m.role AS role, messages_fts.content AS content, s.title AS title"
                " FROM messages_fts"
                " JOIN messages m ON m.session_id = messages_fts.session_id"
                " AND m.seq = messages_fts.seq"
                " JOIN sessions s ON s.id = messages_fts.session_id"
                " WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        else:
            conditions = " AND ".join("LOWER(m.content) LIKE ?" for _ in terms)
            rows = self.conn.execute(
                "SELECT m.session_id, m.seq, m.role, m.content, s.title"
                " FROM messages m JOIN sessions s ON s.id = m.session_id"
                f" WHERE {conditions} ORDER BY m.session_id, m.seq LIMIT ?",
                [f"%{term.lower()}%" for term in terms] + [limit],
            ).fetchall()
        return [dict(row) for row in rows]

    # --------------------------------------------------------- create / delete

    def create(self, title: str = "") -> agentSession:
        base = _slug(title) or "session"
        session_id = base
        while self.exists(session_id):
            session_id = f"{base}-{uuid.uuid4().hex[:4]}"
        now = int(time.time())
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at,"
                " turn_count, token_estimate) VALUES (?, ?, ?, ?, 0, 0)",
                (session_id, title.strip(), now, now),
            )
        return agentSession(session_id=session_id)

    def fresh(self) -> agentSession:
        """A fresh start: an in-memory draft session that save() persists
        only once the draft actually contains a conversation."""
        return agentSession(session_id=f"draft-{uuid.uuid4().hex[:4]}")

    def delete(self, session_id: str) -> bool:
        if not self.exists(session_id):
            return False
        with self.conn:
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            if self.fts_enabled:
                self.conn.execute(
                    "DELETE FROM messages_fts WHERE session_id = ?", (session_id,)
                )
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return True

    # ---------------------------------------------------------------- migration

    def import_legacy_json(self, sessions_dir: Optional[Path] = None) -> int:
        """Idempotently import old JSON sessions (files are left untouched)."""
        sessions_dir = Path(sessions_dir) if sessions_dir else self.db_path.parent
        imported = 0
        for path in sorted(sessions_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                print(f"  skipping {path.name}: {error}")
                continue
            session_id = data.get("session_id") or path.stem
            if self.exists(session_id):
                continue
            legacy = agentSession(session_id=session_id)
            if isinstance(data.get("history"), list):
                legacy.history = data["history"]
            self.save(legacy)
            imported += 1
        return imported

    def close(self) -> None:
        self.conn.close()
