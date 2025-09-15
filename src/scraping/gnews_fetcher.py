from gnews import GNews
from datetime import datetime

class GNewsFetcher:
    def __init__(self, country : str ="ZA", max_results : int =20,
                 language : str ="en",
                 start_date : datetime | None = datetime(2000,1,1),
                 end_date : datetime | None =datetime(2025,1,1)) -> None:
        self.country = country
        self.max_results = max_results
        self.start_date = start_date
        self.end_date = end_date
        self.gnews = GNews(language=language, country=country,
                           max_results=max_results,
                           start_date=start_date,
                           end_date=end_date)

    def update_config(self, country=None, max_results=None, start_date=None, end_date=None):
        if country:
            self.country = country
        if max_results:
            self.max_results = max_results
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        self.gnews = GNews(language="en",
                           country=self.country,
                           max_results=self.max_results,
                           start_date=self.start_date,
                           end_date=self.end_date)

    def get_single_search(self, search_country_query: dict[str, str]) -> list[dict[str, str]]:
        search_result = self.gnews.get_news(search_country_query["search"])
        clean_results = []
        for article in search_result:
            self.add_metadata(article, search_country_query)
            clean_results.append(article)
        return clean_results

    def get_bundle_search(self, search_country_queries: list[dict[str, str]], visited_urls: list[str]) -> list[dict[str, str]]:
        filtered_results = []
        for query in search_country_queries:
            if query["country"] != self.country:
                self.update_config(country=query["country"])
            search_result = self.gnews.get_news(query["search"])
            for article in search_result:
                # Prefer article['link'] if available
                real_url = article.get("link") or article.get("url")
                if real_url not in visited_urls:
                    article["url"] = real_url  # overwrite 'url' with real publisher link
                    self.add_metadata(article, query)
                    filtered_results.append(article)
                    visited_urls.append(real_url)
        return filtered_results

    def add_metadata(self, article: dict[str, str], search_country_query: dict[str, str]) -> None:
        article["country"] = search_country_query["country"]
        article["search"] = search_country_query["search"]
