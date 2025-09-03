# LOADING AND IMPORTING LIBRARIES
from gnews_fetcher import GNewsFetcher
import logic_parser
from news_boy import BrowserSim
import utils

import os, json

print("loading configurations...")
# LOADING CONFIGURATIONS
io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] #TESTING
metrics = io["metrics"]    
searches = io["searches"]
countries = io["countries"]
countries_names = list(countries.keys())
countries_codes = list(countries.values())
years = io["years"]

config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
gnews_filter = config['gnews_filter']
max_results = config['max results']
page_timeout = config['page timeout']
min_page_text_length = config['min page text length']
skip_words = config['website text skip words']

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
news_instruction = prompts["instructions"]["news_instruction"]
format_instruction = prompts["instructions"]["format_instruction"]
output_format = prompts["formats"]["output_format"]

print("generating searches...")
# GENERATING SEARCHES and initialising url tracker for efficiency
gnews_searches = utils.generate_search_queries(
    google_search_templates=searches,
    country_names=countries_names,
    search_metrics=metrics,
    years=years
)
print(f"Generated {len(gnews_searches)} searches.")
visited_urls = []

print("generating prompts...")
# GENERATE PROMPTS FOR PARSING AND FILTERING
news_instruction = utils.generate_prompt_text(news_instruction + format_instruction, metrics, output_format)

print("fetching news articles...")
# FETCHING URLS
news_agent = GNewsFetcher(country=countries[gnews_searches[0]["country"]], max_results=max_results)
google_news_articles = news_agent.get_bundle_search(search_country_queries=gnews_searches, visited_urls=visited_urls)
print(f"Fetched a whole {len(google_news_articles)} articles...")

print("fetching website data...")
# BROWSING AND GETTING FULL TEXT
accessed_articles = []
browser = BrowserSim(page_wait=page_timeout, min_text_length=min_page_text_length, skip_words=skip_words)
browser.start()
for article in google_news_articles:
    full_text = browser.get_page(article["url"])
    if full_text is None:
        continue
    article["full_text"] = full_text
    accessed_articles.append(article)
browser.end()
print(f"Accessed {len(accessed_articles)} articles with full text.")

print("parsing articles...")
parsing_agent = logic_parser.TextParser()
parsing_agent.configure_parsing(None, news_instruction, metrics)
parsed_articles = [] 

# PARSE ARTICLES
for article in accessed_articles:
    # url = article["url"]
    # if url in visited_urls:
    #     continue
    # visited_urls.append(article["url"])

    full_text = article.get("full_text", "")
    agent_response = parsing_agent.parse_and_format(full_text)
    parsed_articles.append(agent_response) if agent_response is not None else None
