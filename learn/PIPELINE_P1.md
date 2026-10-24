# P1 — arXiv Search Client (lesson doc)

Code: `paper_engine.py` (~190 lines with docs; ~80 executable)
Test: `paper_demo.py` — fully offline, deterministic. Status: PASSING.

## What P1 gives the pipeline
One function the agent will call as a tool later (P8):
`search_arxiv(topic, max_results=20, year_from=None)` → list of paper dicts:
`{arxiv_id, title, abstract, year, authors, primary_category, categories,
pdf_url, abs_url}`. This is the **metadata tier** — cheap (~50 tok/paper),
enough for the model to triage which papers deserve `fetch_paper`.

## The four pieces, walkthrough order

1. **`build_query()`** — plain English → API URL. Body: `all:"phrase"` +
   optional `submittedDate:[20250000 TO …]` + `urlencode()` quoting.
   🔹 Trap we hit live: the manual writes operators as `+AND+`. That plus is
   just an encoded SPACE. Pre-embedding `+` makes urlencode double-encode it
   (`%2B` = literal plus) and arXiv's parser breaks. Rule: build with real
   spaces, let urlencode quote once.
2. **`parse_atom()`** — stdlib `xml.etree` + the three namespaces from the
   manual. Whitespace-collapses titles (arXiv hard-wraps them). Skips the
   error pseudo-entry (§3.4) so bad results can't masquerade as papers.
3. **`_throttle()`** — a module-level meter enforcing arXiv's ≥3s spacing.
   The `sleep` only runs on the live path — cache hits never wait.
4. **`QueryCache`** — SQLite table `(source, url) → JSON response + timestamp`,
   TTL 24h (arXiv says results must not change more often than daily —
   caching 24h is not an approximation, it's their contract).

## Why the cache sits between ADDITION and API (exit question)
The agent loop RETRIES and REFINES queries by nature — it will ask the same
search again while planning. Without the cache every retry is a live call and
the 429s return. The cache is not a speed feature; it's the barrier that
makes polite behavior automatic. **Evidence from the lab**: 2nd identical
query = 0.07ms, transport calls stayed at 1 across a full rerun.

## How to run
```bash
uv run python learn/paper_demo.py            # offline: fixture XML, fake transport
uv run python learn/paper_demo.py --real     # on your PC: live arXiv + real throttle
```
Note: `--real` also demonstrates `_throttle()` delaying the 2nd call ~3s.

## Exercise (do before P2)
Prediction first, then run `--real` twice in a row:
1. Will you see the same papers both times?
2. Will the 2nd run be faster or slower? By how much?
3. If you open `sessions/papers.db` with `sqlite3 query_cache` — how many rows?

## What P1 does NOT do (deliberately — that's P2+)
- No Semantic Scholar / OpenAlex (P2: enrichment + 429 fallback chain)
- No downloads (P3), no parsing (P4), no cards (P6)
- `arxiv_id` still carries the version suffix (`2501.12345v2`); dedup across
  sources will need base-id stripping — P2 problem

## Rubric self-check (answer before claiming P1 done)
- [ ] Can you explain why `--real`'s 2nd run has ZERO transport calls?
- [ ] Could you add `sortBy=submittedDate` support without touching parsing?
- [ ] What breaks if we deleted the `row_factory` line? (it broke today)
