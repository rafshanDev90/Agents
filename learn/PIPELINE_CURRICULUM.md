# Paper Pipeline Curriculum — Agent Learning Track, Series P

Protocol: **one concept → read one function (≤80 lines) → run it → exercise →
exit question.** One topic per session. RAG track (RAG_CURRICULUM.md) is
PAUSED until the P-track reaches P7 — papers feed RAG, not the other way.

Notation: ✓ covered · ⇒ next · ☐ later

| #  | Topic | You will be able to answer | Code surface | Status |
|----|-------|----------------------------|--------------|--------|
| P0 | Pipeline map | "What are the stages, the state machine, and what does each existing module provide?" | none (diagrams only) | ⇒ now |
| P1 | arXiv search client | "How do I query a real API politely (Atom XML, paging, 3s spacing) and cache it?" | `paper_engine.py`: `search_arxiv` + cache (~70 lines) | ✓ code + offline lab PASSING (`PIPELINE_P1.md`) — walkthrough pending |
| P2 | Metadata enrichment + fallbacks | "Why merge 3 sources? What is a 429? How do I back off?" | `search_s2`, `search_openalex`, merge+dedup (~70 lines) | ☐ |
| P3 | Download machinery | "What guards stop a bad PDF (wrong type, huge, scanned)? How does resume work?" | `download_pdf` + guards (~60 lines) | ☐ |
| P4 | PDF → text | "What is a text layer? When is a PDF unparseable?" | `pdf_to_text` via PyMuPDF (~40 lines) | ☐ |
| P5 | Section splitting | "How do heuristics find Intro/Method/Results without an LLM?" | `split_sections` (~70 lines) + fallback | ☐ |
| P6 | Card extraction (6-field) | "How does a model fill a JSON schema HONESTLY (quote or flag)?" | `extract_card` + CARD_SCHEMA (~75 lines, two-stage) | ☐ |
| P7 | Papers table + state machine | "Why is status a column? How does resume-on-crash fall out of it?" | `paper_store.py` (~70 lines) | ☐ |
| P8 | Agent wiring | "How do tools enter main.py's registry and why raise MAX_TOOL_ROUNDS?" | `add tools + tool_schemas.json` (~60 lines) | ☐ |
| P9 | Digest + dynamic N | "How does the agent decide how many papers?" | `write_digest` + saturation logic in prompt (~80 lines) | ☐ |
| P10 | Evaluation gate | "How do I prove zero hallucinated citations and fit in token budget?" | golden-run script (L9 style, ~60 lines) | ☐ |

Dependency chain: P1→P2→P3→P4→P5→P6→P7 are strictly sequential.
P8 needs P6+P7. P9 needs P8. P10 is the exit exam of the whole track.

## The P0 map (covered now — concepts only)

    ┌── OFFLINE STAGES (per paper, can crash, must resume) ──────────────┐
    │  metadata (P1–P2)      PDF bytes (P3–P4)       understanding (P6) │
    │  title/abstract/year → download+parse → sections → JSON card      │
    └──────────────┬─────────────────────────────────────────────────────┘
                   ▼
    papers table (P7): status = meta → parsed → carded | failed
    (resume = query WHERE status != 'carded'; a crash mid-run leaves
     earlier work done — same philosophy as your incremental SQLite saves)

    ┌── ONLINE (agent turns, in main.py loop) ───────────────────────────┐
    │  search_papers → model plans N → fetch_paper × N (each = offline   │
    │  stages above) → compare_papers(cards only) → write_digest         │
    └─────────────────────────────────────────────────────────────────────┘

    KEY RULE: full paper text exists ONLY offline/in the DB. The model's
    chat context sees metadata (~50 tok), cards (~250 tok), and — on
    demand — one section. Never a whole PDF.

## Where we are now
P0 map done. P1 code written + offline lab passing — see `PIPELINE_P1.md`
for the walkthrough, exercise, and rubric. Next session: your `--real` run
+ exercise review, then **P2 — metadata enrichment + fallbacks**.
