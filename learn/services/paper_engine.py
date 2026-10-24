import os
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# Constants enforcing the P1 rate-limiting and caching protocol
CACHE_FILE = "arxiv_cache.json"
DELAY_SECONDS = 3.0
NAMESPACE = {'atom': 'http://www.w3.org/2005/Atom'}

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def search_arxiv(query_str: str, max_results: int = 5, start: int = 0) -> list:
    """Queries the arXiv Atom API politely and returns parsed paper metadata dictionaries."""
    cache = load_cache()
    cache_key = f"{query_str}_{max_results}_{start}"

    if cache_key in cache:
        print("💡 [Cache Hit] Returning cached results.")
        return cache[cache_key]

    print(f"📡 [API Request] Fetching from arXiv... Enforcing polite {DELAY_SECONDS}s wait.")
    time.sleep(DELAY_SECONDS)

    # Constructing standard URL query parameters
    params = {
        'search_query': query_str,
        'start': start,
        'max_results': max_results
    }
    base_url = "https://export.arxiv.org/api/query?"
    url = base_url + urllib.parse.urlencode(params)

    try:
        # Fetching raw Atom XML response
        response = urllib.request.urlopen(url).read()
        root = ET.fromstring(response)

        papers = []
        # Finding all entry elements within the XML tree structure
        for entry in root.findall('atom:entry', NAMESPACE):
            paper_id = entry.find('atom:id', NAMESPACE).text.split('/abs/')[-1]
            title = entry.find('atom:title', NAMESPACE).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', NAMESPACE).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', NAMESPACE).text[:10] # Extract YYYY-MM-DD

            # The API's own PDF <link> is ground truth for download targets;
            # hand-built URLs is exactly what caused the 404 session bug.
            pdf_url = None
            for link in entry.findall('atom:link', NAMESPACE):
                if link.get('title') == 'pdf':
                    pdf_url = (link.get('href') or '').replace('http://', 'https://')

            papers.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "published_year": published,
                "pdf_url": pdf_url
            })
            
        # Commit to local cache storage
        cache[cache_key] = papers
        save_cache(cache)
        return papers
        
    except Exception as e:
        print(f"❌ Error encountered fetching from arXiv: {e}")
        return []
if __name__ == "__main__":
    # Example usage
    results = search_arxiv('agentic "rag" pipelines', max_results=5)
    for paper in results:
        print(f"{paper['id']}: {paper['title']} ({paper['published_year']})")