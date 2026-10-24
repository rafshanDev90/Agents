#P3: PDF download engine — URL construction, fallback mirrors, byte-level guards
'''
The Problem we hit live: arxiv.org is 404 on papers that DO exist.
Root cause trace (from the session):
  https://arxiv.org/2307.15872v1.pdf          <- what we sent   (404!)
  https://arxiv.org/pdf/2307.15872v1          <- what arXiv serves
Two mistakes compounded:
  1. the /pdf/ path segment was dropped
  2. the ".pdf" suffix was appended — arXiv RETIRED that form on /pdf/
Rule (boundary lesson): never hand-assemble PDF urls from the id when the
API/GUI hands you one. The Atom entry's <link title="pdf"> IS the canonical
url (that is how parse_atom fills paper["pdf_url"] today).

The Fix, three layers:
  A. URL CONTRACT
     - normalize any caller input (accept "2307.15872v1", "/abs/23...v1",
       "https://arxiv.org/abs/23...v1")
     - prefer an explicitly provided pdf_url (API ground truth)
     - fallback chain: primary -> export.arxiv.org mirror (arXiv's own
       recommendation for programmatic traffic).
  B. BYTE-LEVEL GUARDS (Content-Type can lie; magic bytes cannot)
     - must start with b"%PDF-"
     - hard cap 15MB, streamed through a .part file then atomically renamed     (crash mid-download never leaves a corrupt "good-looking" file)
  C. FAILURE SEMANTICS
     - HTTP 404 on mirrors => the paper genuinely has no open PDF (withdrawn
       or abstract-only) — that is NOT the same as "network down" and the
       caller may want different recovery for each.
'''
import os
import re
import time
import urllib.error
import urllib.request

DOWNLOAD_DIR = "downloaded_papers"
MAX_FILE_SIZE = 15 * 1024 * 1024   # 15MB hard cap ( papers exceed? guard, verify)
DELAY_SECONDS = 3.0
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _normalize_id(arxiv_id: str) -> str:
    """Accepts several caller shapes, returns the canonical bare id."""
    id_ = (arxiv_id or "").strip()
    id_ = re.sub(r"^https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/", "", id_)
    id_ = re.sub(r"^(?:/abs/|abs/)", "", id_)
    id_ = re.sub(r"\.pdf$", "", id_)
    return id_


def _candidate_urls(arxiv_id: str, pdf_url: str = "") -> list[str]:
    """Every acceptable source for this PDF, most-authoritative first."""
    urls = []
    if pdf_url and "/pdf/" in pdf_url:                       # API-provided truth
        urls.append(pdf_url)
    urls.append(f"https://arxiv.org/pdf/{arxiv_id}")         # canonical GUI form
    urls.append(f"https://export.arxiv.org/pdf/{arxiv_id}")  # programmatic mirror
    # de-dup, keep order
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def download_paper_pdf(arxiv_id: str, pdf_url: str = "") -> str:
    """Download a paper PDF. Returns the local path, or "" on failure.

    Failure semantics for the caller (main.py):
      ""        = guards rejected it or no valid source
      localpath = success; file is ATOMIC (only ever seen fully written)
    """
    arxiv_id = _normalize_id(arxiv_id)
    if not arxiv_id:
        print("❌ [Bad input] no arxiv id after normalization")
        return ""

    output_path = os.path.join(DOWNLOAD_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(output_path):
        print(f"💡 [Skipping] {arxiv_id}.pdf already cached locally.")
        return output_path

    time.sleep(DELAY_SECONDS)   # politeness: roughly one fetch per 3s
    last_error = ""
    for url in _candidate_urls(arxiv_id, pdf_url):
        try:
            print(f"📡 [Downloading] {url}")
            request = urllib.request.Request(url, headers={
                "User-Agent": "PaperAgentPipeline/1.0"
            })
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "")
                data = response.read(MAX_FILE_SIZE + 1)   # cap +1 = overflows tell us

            if "application/pdf" not in content_type.lower():
                last_error = f"Content-Type was '{content_type}', not a PDF"
                print(f"❌ Guard: {last_error}")
                continue

            if len(data) > MAX_FILE_SIZE:
                last_error = f"file exceeds {MAX_FILE_SIZE // (1024*1024)}MB cap"
                print(f"❌ Guard: {last_error}")
                break                                     # no mirror will shrink it
            if not data.startswith(b"%PDF-"):
                # Content-Type said PDF but bytes disagree — trust the BYTES
                last_error = "missing %PDF- magic bytes (HTML error page? scan?)"
                print(f"❌ Guard: {last_error}")
                continue

            part_path = output_path + ".part"
            with open(part_path, "wb") as f:
                f.write(data)
            os.replace(part_path, output_path)            # atomic: fs guarantees
            print(f"✅ Success: saved {output_path} ({len(data)/(1024*1024):.1f} MB)")
            return output_path

        except urllib.error.HTTPError as error:
            last_error = f"HTTP {error.code}"
            print(f"❌ [{url}] failed: {last_error}")
            if error.code == 404:
                continue                                  # next mirror may serve it
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            print(f"❌ [{url}] failed: {last_error}")

    print(f"❌ Failed to download {arxiv_id}; last error: {last_error}\n"
          f"   💡 if this was HTTP 404 on both mirrors, the paper likely has"
          f"   no open PDF (withdrawn or abstract-only) — handle it as such.")
    return ""


if __name__ == "__main__":
    # LIVE usage: on a PC with internet — the paper-form example from the session
    path = download_paper_pdf("/abs/2307.15872v1")        # proves normalization
    print("pipeline result:", path or "failed")
