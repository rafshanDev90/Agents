#R1–R3: RAG from scratch — chunking, embeddings, similarity search
'''
The Problem: The LLM knows nothing about your papers. Fine-tuning could inject
knowledge into weights, but it is slow, expensive, and frozen at train time —
a paper fetched today could not be queried until you retrained.

The Fix: Retrieval-Augmented Generation, built from stdlib only.
The pipeline has two halves that are usually confused:

    INDEXING (offline)                     QUERY (online)
    documents                              question
      -> chunk_text     (R1)                 -> get_embeddings (R2)
      -> get_embeddings (R2)                 -> similarity search (R3)
      -> SQLite vector store                 -> top-k chunks -> prompt -> LLM

Under the hood there is no magic:
  * an embedding is just a fixed-length list of floats (nomic-embed-text: 768
    of them) produced by a neural net, arranged so that texts about the same
    idea end up with vectors that point in similar DIRECTIONS
  * similarity = cosine: dot(a,b) / (||a|| * ||b||), i.e. angle between the
    vectors. Magnitude is ignored on purpose (long chunks shouldn't win just
    for having more words)
  * the "vector database" is only that: a table of blobs + a loop computing
    cosine. At our scale (hundreds of chunks) a full scan in pure Python
    beats any specialized server in simplicity and is instant.
'''
from __future__ import annotations

import array
import heapq
import json
import math
import re
import sqlite3
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable, Iterable

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"   # ollama pull nomic-embed-text (768 dims, 8k ctx)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "sessions" / "rag.db"


# ─────────────────────────────────────────────────────────────── R1: chunking

def chunk_text(text: str, max_chars: int = 1200, overlap_words: int = 40) -> list[str]:
    """Split long text into retrieval-sized pieces (R1).

    WHY chunk at all?
      1. Embeddings compress a text to ONE vector — a 10-page document gets
         1 vector for ~50 different ideas, all blurred together. A 2-3
         paragraph chunk carries ~1 idea, so its vector is sharp.
      2. At query time we pay context-window rent per retrieved chunk. Small
         chunks let us fetch only the relevant paragraph, not the whole paper.

    WHY overlap?
      Ideas that sit at the edge of a chunk boundary get split in half and
      then neither half matches well on its own. Repeating the tail
      (~40 words) of the previous chunk at the head of the next makes the
      boundary idea appear complete in AT LEAST one chunk.
    """
    text = " ".join(text.split())          # collapse all whitespace, we rewrap below
    if not text:
        return []
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = _split_long_paragraph(paragraph, max_chars)
        for piece in pieces:
            if len(current) + len(piece) + 1 <= max_chars:
                current = f"{current} {piece}".strip()
            else:
                if current:
                    chunks.append(current)
                tail = _tail_words(current, overlap_words)
                current = f"{tail} {piece}".strip()      # the overlap bridge
    if current:
        chunks.append(current)
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    # heuristic paragraph signals: numbered/lettered section markers, headings
    markers = re.split(
        r"(?=(?:\d+\.\d*\s)|(?:[A-Z]{3,}))", text
    ) if "\n" in text else [text]
    return [m for m in markers if m.strip()]


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    label = paragraph.strip("\"' ").split(" ")[0]
    if not label.rstrip("0123456789.").isupper() or len(paragraph) <= max_chars:
        return [paragraph]
    # split oversized paragraph on the last complete sentence that fits
    sentences = re.split(r"(?<=[.!?]) +", paragraph)
    pieces, buf = [], ""
    for sentence in sentences:
        if len(buf) + len(sentence) + 1 > max_chars and buf:
            pieces.append(buf.strip())
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf.strip())
    return pieces


def _tail_words(text: str, count: int) -> str:
    words = text.split()
    return " ".join(words[-count:]) if count and words else ""


# ──────────────────────────────────────────────── R2: embeddings (vectors)

def get_embeddings(texts: list[str], model: str = EMBED_MODEL,
                   base_url: str = OLLAMA) -> list[list[float]]:
    """Turn texts into vectors via Ollama's native batch endpoint (R2).

    An embedding model is a net trained with a contrastive objective:
    pairs of texts about the same topic are pulled together, unrelated
    texts pushed apart. Nobody decided the meaning of dimension #412 —
    the directions are emergent, but relative DISTANCES are meaningful.

    POST /api/embed {"model":..., "input": [t1, t2, ...]} -> {"embeddings":[[...]]}
    Batching all chunks in one call avoids per-request HTTP overhead.
    Fails loudly with one clear message when Ollama is down — callers decide
    whether that is fatal (indexing) or skippable (search fallback).
    """
    payload = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(
        f"{base_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode())
    embeddings = data.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise RuntimeError(f"Ollama returned {len(embeddings or [])} embeddings "
                           f"for {len(texts)} inputs")
    return embeddings


def normalize(vector: list[float]) -> list[float]:
    """Scale a vector to length 1.0 (R2).

    After this, cosine similarity collapses to a plain dot product:
    dot(â, b̂) = cos(angle). We normalize AT INDEXING TIME and normalize the
    query too — so the per-search math is simpler AND we store fewer reasons
    to make mistakes at query time.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else vector


def cosine(a: list[float], b: list[float]) -> float:
    """cos(angle) between two vectors (R3) — kept for teaching/benchmarking.

    search() uses the dot product because our stored vectors are normalized;
    this explicit version is what you compare against in the exercise.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────── R3: the "vector database" is a table

class RagStore:
    """SQLite as a vector store — like session_store.py planned.

    The embedding lives in the row as a BLOB of float32. Why not JSON?
    768 floats as JSON text ≈ 12 KB; as float32 BLOB ≈ 3 KB — 4× smaller,
    no parse cost. array('f') does the bytes<->floats conversion for us.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS rag_chunks ("
            " id INTEGER PRIMARY KEY,"
            " source TEXT NOT NULL,"
            " section TEXT,"
            " seq INTEGER NOT NULL,"
            " dim INTEGER NOT NULL,"
            " content TEXT NOT NULL,"
            " embedding BLOB NOT NULL)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------ write

    def add_chunks(self, source: str, chunks: Iterable[str],
                   embeddings: list[list[float]], section: str = "") -> int:
        """Normalize, pack to float32 BLOBs, insert. One transaction = atomic."""
        rows = []
        for seq, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            if len(vector) != len(embeddings[0]):
                raise ValueError(f"embedding dim mismatch inside batch: "
                                 f"{len(vector)} vs {len(embeddings[0])}")
            unit = array.array("f", normalize(vector))
            rows.append((source, section, seq, len(unit), chunk, unit.tobytes()))
        with self.conn:
            self.conn.executemany(
                "INSERT INTO rag_chunks (source, section, seq, dim, content,"
                " embedding) VALUES (?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM rag_chunks").fetchone()["n"]

    # ------------------------------------------------------------------- read

    def search(self, query_embedding: list[float], top_k: int = 5,
               source: str | None = None) -> list[dict]:
        """Full-scan cosine search, pure Python (R3).

        Brutally honest truth about "vector databases": at this scale the
        DB is a plain SELECT and the similarity math is a loop. Specialized
        engines only matter once you have millions of vectors and need
        approximate search (ANN/HNSW) — we are nowhere near that.
        Returns rows ordered best-first, each with its cosine score.
        """
        unit_query = normalize(query_embedding)
        if all(x == 0.0 for x in unit_query):
            # a zero query vector is orthogonal to everything (cosine = 0/undefined)
            # -> silent arbitrary results. Fail loudly instead (garbage-in guard).
            raise ValueError(
                "query embedding is all-zero — query text produced no meaning "
                "(out-of-vocab fake embedder? empty string?)"
            )
        sql = ("SELECT id, source, section, seq, dim, content, embedding"
               " FROM rag_chunks")
        params: tuple = ()
        if source:
            sql += " WHERE source = ?"
            params = (source,)
        scored = []
        for row in self.conn.execute(sql, params):
            stored = array.array("f")
            stored.frombytes(row["embedding"])
            # dot of two normalized vectors == cosine (see normalize())
            score = sum(a * b for a, b in zip(unit_query, stored))
            scored.append({
                "id": row["id"], "source": row["source"], "section": row["section"],
                "seq": row["seq"], "score": round(score, 4),
                "content": row["content"],
            })
        return heapq.nlargest(top_k, scored, key=lambda hit: hit["score"])

    def close(self) -> None:
        self.conn.close()
