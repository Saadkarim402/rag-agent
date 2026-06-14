import logging
import re
import urllib.parse
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class DuckDuckGoSearchTool:
    """Tool to search the web using DuckDuckGo with a transparent fallback to the

    Wikipedia search API if rate-limited or blocked.
    """

    def __init__(self, max_results: int = 3) -> None:
        self.max_results = max_results
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def _wikipedia_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback to the Wikipedia Search API if DDG scraping fails."""
        logger.info(f"[WIKIPEDIA FALLBACK] Querying Wikipedia for: '{query}'")
        encoded_query = urllib.parse.quote_plus(query)
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={encoded_query}&utf8=&format=json&srlimit={self.max_results}"
        )
        try:
            response = requests.get(url, headers=self.headers, timeout=6)
            response.raise_for_status()
            data = response.json()
            search_items = data.get("query", {}).get("search", [])
            
            results = []
            for item in search_items:
                title = item.get("title", "Wikipedia Article")
                snippet = item.get("snippet", "")
                # Clean html tags from snippet
                clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                # URL encode the title for the Wikipedia link
                safe_title = urllib.parse.quote(title.replace(" ", "_"))
                url_link = f"https://en.wikipedia.org/wiki/{safe_title}"

                results.append({
                    "title": title,
                    "snippet": clean_snippet,
                    "url": url_link
                })
            return results
        except Exception as e:
            logger.error(f"Wikipedia search fallback failed: {e}")
            return []

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Queries DuckDuckGo first, falling back to Wikipedia search if blocked or zero results."""
        if not query:
            return []

        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        try:
            response = requests.get(url, headers=self.headers, timeout=6)
            # If rate-limited or blocked, status code might be 202, 403, 503, etc.
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo search returned status code {response.status_code}. Using Wikipedia fallback.")
                return self._wikipedia_fallback(query)

            html = response.text
            blocks = html.split('<div class="result results_links results_links_deep web-result')
            if len(blocks) <= 1:
                blocks = html.split('<div class="web-result')

            results = []
            for block in blocks[1 : self.max_results + 1]:
                href_match = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
                snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)

                if href_match:
                    raw_url = href_match.group(1)
                    parsed_url = urllib.parse.urlparse(raw_url)
                    actual_url = raw_url
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

            # If DDG returned zero results, use Wikipedia fallback
            if not results:
                logger.warning("DuckDuckGo returned 0 results. Triggering Wikipedia fallback.")
                return self._wikipedia_fallback(query)

            logger.info(f"[WEB SEARCH] Retrieved {len(results)} results from DuckDuckGo for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}. Using Wikipedia fallback.")
            return self._wikipedia_fallback(query)
