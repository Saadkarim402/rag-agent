import logging
import re
import urllib.parse
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class DuckDuckGoSearchTool:
    """Tool to search the web using DuckDuckGo's HTML interface.

    Allows fetching real-time online search snippets with zero API keys required.
    """

    def __init__(self, max_results: int = 3) -> None:
        self.max_results = max_results
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Queries DuckDuckGo and returns a list of result dicts:

        {"title": str, "snippet": str, "url": str}.
        """
        if not query:
            return []

        # Clean search query and URL-encode
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        try:
            response = requests.get(url, headers=self.headers, timeout=8)
            response.raise_for_status()
            html = response.text

            # Parse results by splitting individual search result result__body containers
            blocks = html.split('<div class="result results_links results_links_deep web-result')
            if len(blocks) <= 1:
                blocks = html.split('<div class="web-result')

            results = []
            for block in blocks[1 : self.max_results + 1]:
                # Extract title, url, and snippet
                href_match = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)

                if href_match:
                    raw_url = href_match.group(1)
                    parsed_url = urllib.parse.urlparse(raw_url)
                    actual_url = raw_url
                    # Decode DDG internal redirect links if present
                    if "uddg=" in parsed_url.query:
                        qs = urllib.parse.parse_qs(parsed_url.query)
                        actual_url = qs.get("uddg", [raw_url])[0]

                    title = re.sub(r"<[^>]+>", "", href_match.group(2)).strip()
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()

                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": actual_url
                    })

            logger.info(f"[WEB SEARCH] Retrieved {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
