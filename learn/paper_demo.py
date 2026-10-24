#P1 lab: prove the arXiv client offline — parsing, query building, cache

"""No network needed by default: a FIXED Atom document stands in for the API
(the L9 fake-model lesson, applied to an HTTP boundary). LocaleArn with
`--real` + internet: the same code path runs against live arXiv.

Observations the demo prints:
  * what raw query this topic becomes (quote handling, date filter)
  * what parse_atom extracts per paper (the metadata tier, ~50 tok/paper)
  * cache round-trip: 2nd lookup of the same URL is instant + zero live calls
  * a malformed-id Atom error entry is REJECTED with a readable message
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from paper_engine import (ARXIV_API, QueryCache, build_query, parse_atom,
                          search_arxiv)

# ───────────────────────────────────────────────── fixture (manual's schema)

FIXTURE = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query: search_query=all:"agentic rag"</title>
  <opensearch:totalResults>2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2501.12345v2</id>
    <published>2026-09-14T17:00:00-04:00</published>
    <title>
      ReAct-RAG:
      pipelines for
      agentic retrieval
    </title>
    <summary>  We study  pipelines   that let an agent decide
        when to   retrieve. </summary>
    <author><name>J. Devs</name></author>
    <author><name>A. Junior</name></author>
    <link href="http://arxiv.org/abs/2501.12345v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2501.12345v2" rel="related" title="pdf"
          type="application/pdf"/>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.IR" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2502.99999v1</id>
    <published>2025-01-01T00:00:00-05:00</published>
    <title>Paper two</title>
    <summary>Another abstract.</summary>
    <author><name>B. Author</name></author>
  </entry>
</feed>"""

FIXTURE_ERROR = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_99</id>
    <title>Error</title><summary>incorrect id format for 99</summary></entry>
</feed>"""


def main() -> None:
    use_real = "--real" in sys.argv
    print("═══ P1 lab ═══ arXiv search client (offline unless --real)\n")

    # ① query building — plain topic -> API syntax (assert the DECODED form,
    #    because the wire form is urlencoded: spaces -> '+')
    from urllib.parse import parse_qs, urlparse
    url = build_query('agentic "rag" pipelines', max_results=20, year_from=2025)
    decoded = parse_qs(urlparse(url).query)["search_query"][0]
    print(f"[1] built url: {url}\n    decoded query: {decoded}\n")
    assert decoded == ('all:"agentic rag pipelines" AND'
                       " submittedDate:[20250000 TO 210001010000]"), decoded
    assert "%2B" not in url, "operator '+' must NOT be double-encoded"

    # ② parsing offline — the metadata tier shape
    papers = parse_atom(FIXTURE)
    p = papers[0]
    print(f"[2] fixture parse: {len(papers)} papers")
    for field in ("arxiv_id", "title", "year", "authors",
                  "primary_category", "pdf_url"):
        print(f"    {field:<18}: {p[field]}")
    assert p["arxiv_id"] == "2501.12345v2"
    assert "ReAct-RAG: pipelines" in p["title"], "multiline title collapsed"
    assert p["pdf_url"] and p["pdf_url"].endswith("2501.12345v2")
    assert len(papers[1]["categories"]) == 0, "entry without categories parses"

    # ③ arXiv's error-as-Atom pattern -> rejected loudly, not silently
    assert parse_atom(FIXTURE_ERROR) == []       # parser skips the error entry
    from paper_engine import _feed_error
    err = _feed_error(FIXTURE_ERROR)
    print(f"[3] error-entry detection: '{err}'\n")
    assert err and "incorrect id" in err

    # ④ cache + throttle behavior. COLD cache required: an old tmp DB would
    #    make call #1 a hit and the miss-vs-hit comparison meaningless.
    db_path = Path("/tmp/opencode/papers_p1.db")
    db_path.unlink(missing_ok=True)
    cache = QueryCache(db_path=db_path)

    if use_real:
        # REAL path: real transport, two DISTINCT topics -> two LIVE calls.
        # The 3s politeness meter stretches the 2nd one; the 3rd (repeat of
        # #1) is a cache hit with neither network nor sleep.
        import urllib.request  # noqa: F401  (real urlopen stays patched-in)

        def timed(topic):
            t = time.perf_counter()
            papers = search_arxiv(topic, cache=cache, year_from=2025,
                                  max_results=5)
            return papers, time.perf_counter() - t

        p1, dt1 = timed("agentic retrieval pipelines backend")
        p2, dt2 = timed("chunked retrieval evaluation")
        p3, dt3 = timed("agentic retrieval pipelines backend")   # repeat

        for name, papers, dt in (("call#1 (cold)", p1, dt1),
                                 ("call#2 (throttled live)", p2, dt2),
                                 ("call#3 (cache hit)", p3, dt3)):
            print(f"[4] {name:<24}: {len(papers)} papers in {dt:.2f}s")
        print("    sample of what real metadata looks like:")
        for paper in p1[:2]:
            print(f"    - [{paper['year']}] {paper['title'][:70]}")
        rows = cache.conn.execute(
            "SELECT COUNT(*) AS n FROM query_cache").fetchone()["n"]
        print(f"    query_cache rows: {rows} (2 unique URLs cached)")
        assert p1 and p2 and p3, "live arXiv returned nothing"
        assert p3 == p1, "repeat query must return identical cached papers"
        assert dt3 < 0.5, "cache hit must not wait for network nor sleep"
        # dt2 is network latency + whatever politeness sleep remained; both
        # vary by network, so we OBSERVE it rather than assert on it.
        print(f"    observation: 2nd live call {dt2:.2f}s vs repeat {dt3:.2f}s"
              " — the gap is network + the ~3s meter you'd see in a trace")
        print("pass: politeness + cache armor hold on live arXiv")
        return

    # OFFLINE path: same pipeline, fixture replaces the transport so the
    # logic is provable with zero internet.
    calls = []
    import urllib.request as ur
    real_urlopen = ur.urlopen

    def fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        class R:
            def read(self): return FIXTURE.encode()
            def __enter__(self): return self
            def __exit__(self, *a): pass
        print(f"    (live call #{len(calls)} -> {request.full_url[:60]}...)")
        return R()

    ur.urlopen = fake_urlopen
    t0 = time.perf_counter()
    got1 = search_arxiv("agentic rag pipelines", cache=cache, year_from=2025)
    live_ms = (time.perf_counter() - t0) * 1000
    got2 = search_arxiv("agentic rag pipelines", cache=cache, year_from=2025)
    hit_ms = (time.perf_counter() - t0 - 0) * 1000 - live_ms
    ur.urlopen = real_urlopen

    print(f"[4] cache: 1st call -> {len(got1)} papers in {live_ms:.2f}ms"
          f" (live transport calls so far: {len(calls)})")
    print(f"    2nd  call -> {len(got2)} papers in {hit_ms:.2f}ms"
          f" (live transport calls total: {len(calls)})")
    assert len(calls) == 1, "second identical query must be a pure cache hit"
    assert got1 == got2

    print("\nPASS: query building, Atom parsing, error rejection, cache armor")
    if not use_real:
        print("(add --real on your PC for the live arXiv call + 3s throttle demo)")


if __name__ == "__main__":
    main()
