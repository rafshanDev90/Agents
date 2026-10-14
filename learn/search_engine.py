from duckduckgo_search import DDGS
from rich.console import Console

console = Console()

class SearchEngine:
    def __init__(self):
        self.ddgs = DDGS()
        console.print("[bold green]DuckDuckGo Search Engine Initialized[/bold green]")

    def search(self, query, max_results=10):
        results = []
        for result in self.ddgs.text(query, max_results=max_results):
            results.append({
                'title': result['title'],
                'url': result['href'],
                'snippet': result['body']
            })
        return results