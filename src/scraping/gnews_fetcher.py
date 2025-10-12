import asyncio
from gnews import GNews
from datetime import datetime
import httpx
import base64, re
from bs4 import BeautifulSoup

# --- Utility: try to decode article URL from Google News RSS/Read links ---
def extract_real_article_from_rss(url: str) -> str:
    """
    Extract the real article URL from a news.google.com/rss/articles or
    news.google.com/read link without making a request.
    (This works only if Google actually encodes it in Base64; otherwise fallback.)
    """
    m = re.search(r'/(?:articles|read)/([A-Za-z0-9_-]+)', url)
    if not m:
        return url
    token = m.group(1)
    # Strip common prefixes
    token = token.replace('CBMi', '').replace('CBM', '')
    # Fix base64 padding
    padding = '=' * (-len(token) % 4)
    try:
        decoded = base64.b64decode(token + padding).decode('utf-8', errors='ignore')
        m2 = re.search(r'https?://[^\s]+', decoded)
        if m2:
            return m2.group(0)
    except Exception:
        pass
    return url

class GNewsFetcher:
    """ 
    A class to fetch news articles using the GNews API, supporting parallel searches and configurable parameters.
    Attributes:
        country (str): The country code for news search (default: "ZA").
        max_results (int): Maximum number of articles to fetch per search (default: 20).
        start_date (datetime | None): The earliest date for articles (default: datetime(2000, 1, 1)).
        end_date (datetime | None): The latest date for articles (default: datetime(2025, 1, 1)).
        gnews (GNews): Instance of the GNews API client.
    Methods:
        __init__(country="ZA", max_results=20, language="en", start_date=datetime(2000, 1, 1), end_date=datetime(2025, 1, 1)):
            Initializes the GNewsFetcher with the specified configuration.
        update_config(country=None, max_results=None, start_date=None, end_date=None):
            Updates the fetcher configuration and reinitializes the GNews client.
        async _fetch_single(query, visited_urls, delay=1.0):
            Fetches articles for a single search-country query, resolving URLs and adding metadata.
            Ensures URLs are not revisited and applies an optional delay between requests.
        async get_bundle_search_parallel(search_country_queries, delay=1.0):
            Runs multiple searches in parallel for different countries, rate-limiting URL resolution within each search.
            Returns a flattened list of articles with metadata.
        add_metadata(article, search_country_query):
            Adds country and search metadata to an article dictionary.
    """
    def __init__(
        self,
        country: str = "ZA",
        max_results: int = 20,
        language: str = "en",
        start_date: datetime | None = datetime(2018, 1, 1),
        end_date: datetime | None = datetime(2025, 1, 1),
        visited_urls: set[str] = set(),
    ) -> None:
        self.country = country
        self.max_results = max_results
        self.start_date = start_date
        self.end_date = end_date
        self.gnews = GNews(
            language=language,
            country=country,
            max_results=max_results,
            start_date=start_date,
            end_date=end_date,
        )
        self.visited_urls = visited_urls # To track visited URLs across searches
    def update_config(self, country=None, max_results=None, start_date=None, end_date=None):
        if country:
            self.country = country
        if max_results:
            self.max_results = max_results
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        self.gnews = GNews(
            language="en",
            country=self.country,
            max_results=self.max_results,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    async def _fetch_single(self, query: dict[str, str], delay: float = 1.0) -> list[dict[str, str]]:
        """
        Fetch and resolve articles for a single search-country query.
        """
        def fetch_sync():
            return self.gnews.get_news(query["search"])

        search_result = await asyncio.to_thread(fetch_sync)
        results = []

        # Sequentially resolve each URL with optional delay
        for article in search_result:
            raw_url = article.get("link") or article.get("url")
            if raw_url not in self.visited_urls:
                article["url"] = raw_url
                self.add_metadata(article, query)
                results.append(article)
                self.visited_urls.add(raw_url)

        return results

    async def get_bundle_search_parallel(self, search_country_queries: list[dict[str, str]], delay: float = 1.0) -> list[dict[str, str]]:
        """
        Run multiple searches in parallel (one per country) but rate-limit
        URL resolution inside each search.
        """
        tasks = [self._fetch_single(query, delay=delay) for query in search_country_queries]
        results_list = await asyncio.gather(*tasks)
        return [article for sublist in results_list for article in sublist]

    def add_metadata(self, article: dict[str, str], search_country_query: dict[str, str]) -> None:
        article["country"] = search_country_query["country"]
        article["search"] = search_country_query["search"]