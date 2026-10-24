#R1–R3 lab: see RAG's three stages working, offline or real

"""Run with Ollama down  ->  uses a deterministic FakeEmbedder (L9 pattern):
the pipeline logic is proven, only vector QUALITY improves with the real model.
Run with `--real` (Ollama up + `ollama pull nomic-embed-text`)  ->  real vectors.

What to notice in either mode:
  * chunks of the same DOC about the same IDEA score highest (that's retrieval)
  * scores are cosines in [0,1] for normalized positive-ish vectors
  * the "vector database" step is literally a table scan + dot products
"""

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent))

from rag_store import RagStore, chunk_text, get_embeddings, normalize, cosine

# ─────────────────────────────────────────── fake embedder (offline teaching)

_VOCAB = [
    "chunk", "vector", "embed", "retrieval", "rag", "sqlite", "paper",
    "agent", "tool", "pdf", "citation", "compare", "cache", "token",
    "context", "quantiz", "prompt", "model", "search", "injection",
    "window", "session", "truncat", "limit", "cosine", "similar",
    "angle", "score",
]


def fake_embedding(text: str, dim: int | None = None) -> list[float]:
    """Deterministic bag-of-concepts vector.

    Teaches what an embedding IS at the crudest level: a text gets a vote in
    one coordinate per concept it mentions. A real embedding model does the
    same PLUS captures paraphrase/synonyms — same pipeline, better vectors.
    """
    dim = dim or len(_VOCAB)
    vec = [0.0] * dim
    for token in text.lower().split():
        for i, concept in enumerate(_VOCAB):
            if i < len(vec) and concept in token:
                vec[i] += 1.0
    return vec


class FakeEmbedder:
    """Same interface as rag_store.get_embeddings — L9's fake-model lesson:
    swap implementations, the retrieval logic under test never changes."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [fake_embedding(t) for t in texts]


def build_embedder(use_real: bool) -> Callable[[list[str]], list[list[float]]]:
    if not use_real:
        return FakeEmbedder()
    def real(texts):
        return get_embeddings(texts)
    return real


# ─────────────────────────────────────────────────── toy corpus (mini-papers)

CORPUS = {
    "paper:chunking-guide": (
        "CHUNKING GUIDE. Good chunk sizes make retrieval work. Large chunks blur "
        "many ideas into one embedding vector, so the vector cannot point at any "
        "single idea, and similarity between conceptually different chunks "
        "becomes noise. Small chunks keep each idea sharp but lose the surrounding "
        "context, so a chunk about a method no longer mentions the paper it "
        "belongs to. Overlap between consecutive chunks lets an idea that falls "
        "on the boundary stay whole in at least one chunk. A common size is a few "
        "chunks of two to four paragraphs each with a short overlap tail. "
        "SMOKE TEST. Measure retrieval quality, not just run it: label ten "
        "questions, ask which chunk should top the list, log the hits, and "
        "regression-test after every chunker change."
    ),
    "paper:vectors-and-search": (
        "VECTOR SEARCH BASICS. An embedding maps text to a list of numbers. The "
        "similarity of two texts is the cosine of the angle between their "
        "embedding vectors, ignoring vector length. Normalizing vectors to unit "
        "length turns cosine into a plain dot product, which is what the search "
        "loop computes. SQLITERAG. A vector store can be a plain SQLite table of "
        "chunks and float blobs; full table scan cosine is instant at small "
        "scale. PAPER CONTEXT TRICK. Store the source and section with every "
        "chunk so an answer can say which paper a claim came from."
    ),
    "paper:agent-loop": (
        "AGENT LOOP NOTES. An agent loop calls a model, executes tool calls such "
        "as search or file tools, appends the results to the conversation window, "
        "and repeats until the model answers in plain text. Token budget matters: "
        "tool outputs must be truncated before they enter the context window, "
        "otherwise retrieval quality drops for the model reading the prompt."
    ),
}

QUERIES = [
    # query                                  # chunk that SHOULD win
    ("how large should a chunk be?  ",       "paper:chunking-guide"),
    ("how does cosine similarity work?",     "paper:vectors-and-search"),
    ("why truncate tool outputs for agent prompts?", "paper:agent-loop"),
]


# ─────────────────────────────────────────────────────────────────────── demo

def main() -> None:
    use_real = "--real" in sys.argv
    embed = build_embedder(use_real)
    mode = "REAL nomic-embed-text" if use_real else "FAKE embedder (offline)"
    print(f"═══ RAG lab ═══ mode: {mode}\n")

    store = RagStore(db_path=Path("/tmp/opencode/rag_demo.db"))
    store.conn.execute("DELETE FROM rag_chunks")     # fresh rebuild each run
    store.conn.commit()
    print(f"db cleared for a fresh run\n")

    # ── INDEXING: docs -> chunks (R1) -> vectors (R2) -> sqlite (R3)
    # small max_chars so the toy docs split into several chunks -> overlap
    # and multi-chunk retrieval are visible in the demo
    CORPUS_CHUNKS: dict[str, list[str]] = {
        source: chunk_text(text, max_chars=350, overlap_words=25)
        for source, text in CORPUS.items()
    }
    for source, chunks in CORPUS_CHUNKS.items():
        print(f"[R1] {source}: {len(chunks)} chunks")
        embeddings = embed(chunks)
        print(f"[R2]    embedded {len(chunks)} texts (dim={len(embeddings[0])})")
        store.add_chunks(source, chunks, embeddings)

    # ── sanity check: cosine() vs the dot the index uses (analytic identity)
    a, b = embed(["cosine angle between two vectors",
                  "a small sqlite database of paper notes"])[0:2]
    assert abs(sum(x * y for x, y in zip(normalize(a), normalize(b)))
               - cosine(a, b)) < 1e-9, "dot(normalized) != cosine — check R3!"
    print("[R3-check] dot(normalized vectors) == cosine() ✓\n")

    # ── QUERY: embed question, full-scan, top-k
    print("── top-1 retrieval per query ──")
    hits_total = 0
    for query, expected_source in QUERIES:
        qvec = embed([query])[0]
        (top,) = store.search(qvec, top_k=1)
        hit = top["source"] == expected_source
        hits_total += hit
        print(f"Q: {query.strip():<45} -> {top['source']:<26}"
              f" score={top['score']:.3f}  {'✓' if hit else '✗ miss'}")

    print(f"\nretrieval: {hits_total}/{len(QUERIES)} exact top-1 hits")
    print(f"chunks in \"vector db\": {store.count()}  (a table, not a server)")


if __name__ == "__main__":
    main()
