# RAG Curriculum — Agent Learning Track, Series R

> **PAUSED** — switch to `PIPELINE_CURRICULUM.md` (P-track) first.
> Papers will feed the RAG store, so the P-track is the dependency-ordered
> path. R sessions resume at T5 once P-track reaches P7.

Protocol (same as PROGRESS.md): **one concept → read the code → run it → one
exercise → exit question → next topic.** One topic per session. No skipping.

Status key: ✓ covered · ⇒ next up · ☐ later

| #  | Topic | You will be able to answer | Code surface | Status |
|----|-------|----------------------------|--------------|--------|
| T0 | Map of RAG | "What gets built offline vs online, and where in *our* repo does each piece live?" | none (diagrams) | ✓ today |
| T1 | Chunking deep-dive | "Why not 1 chunk = 1 paper? What does overlap actually rescue?" | `chunk_text()` (~60 lines) | ✓ written, needs walkthrough |
| T2 | Embeddings | "What IS a vector here? What do the 768 numbers mean? What does normalize change?" | `get_embeddings`, `normalize`, `cosine` (~70 lines) | ✓ written, needs walkthrough |
| T3 | Vector store | "Why is SQLite a vector DB? What breaks at 10M vectors?" | `RagStore.add_chunks`, BLOB trick (~50 lines) | ✓ written, needs walkthrough |
| T4 | Retrieval (query path) | "What actually happens when you search? Why heap not sort?" | `RagStore.search` (~30 lines) | ✓ written, needs walkthrough |
| T5 | Augmentation (RAG's second half) | "How do retrieved chunks become part of the prompt?" | NEW: `build_context()` + agent loop hookup (≤80 lines) | ☐ next build |
| T6 | Hybrid search | "When does keyword FTS5 beat vectors? How do I merge two ranked lists?" | NEW: wire FTS5 next to cosine | ☐ |
| T7 | Retrieval quality & reranking | "Why do good embedders beat fake ones? What does a reranker add?" | `--real` lab + rerank experiment | ☐ |
| T8 | Paper ingestion | "How does a PDF become chunks?" | NEW: `pdf_tools.py` (PyMuPDF → sections → `RagStore`) | ☐ |
| T9 | End-to-end RAG tool in the agent | "What tool does the model call and what does it get back?" | NEW: `search_docs` in `main.py` registry | ☐ |
| T10 | Evaluation | "Is my retrieval good or lucky? How do I regression-test it?" | NEW: golden questions + hit-rate table (L9 style) | ☐ |
| T11 | Scale & caching | "What changes at 100k chunks? Why batch embedding calls?" | read-only ANN tour + `paper_query_cache` reuse | ☐ |

## Session format per topic
1. Concept (in chat, no code pasting walls)
2. Read: the ONE function (~≤80 lines) with line refs
3. Run: the demo/lab command that shows it
4. Exercise: prediction → run → compare (does the code do what you expected?)
5. Exit question: answer aloud before moving on
6. Score in "Notes / rubric scores" (PROGRESS.md style) → tick the box here

## Where we are now
T0 covered informally. T1–T4 code exists (written during yesterday's sprint) —
we walk it slowly, one topic per session, starting with **T1 chunking**.
New code only enters at T5.
