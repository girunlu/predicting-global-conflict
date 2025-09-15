# LOADING AND IMPORTING LIBRARIES
from gnews_fetcher import GNewsFetcher
import logic_parser
from news_boy import BrowserSim
import utils
import os, json, datetime

t0 = utils.log_time()

print("loading configurations...")
# LOADING CONFIGURATIONS
io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] # TESTING
metric_data = io["metrics"]
search_format = io["search format"]
country_data = io["countries"]
countries_names = list(country_data.keys())
countries_codes = list(country_data.values())
years = io["years"]
excluded_terms = io["exclusions"]

config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
max_results = config['max results']
page_timeout = config['page timeout']
min_page_text_length = config['min page text length']
skip_words = config['website text skip words']
year_chunk_length = config["search year chunk length"]
article_start = config["article start trim"]
article_end = config["article end trim"]

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
news_instruction = prompts["instructions"]["news_instruction"]
format_instruction = prompts["instructions"]["format_instruction"]
output_format = prompts["formats"]["output_format"]

print("generating searches...")
# GENERATING SEARCHES and initialising url tracker for efficiency
gnews_searches = utils.generate_search_queries(
    search_format = search_format,
    countries = country_data,
    metrics = metric_data,
    years = years,
    exclusions = excluded_terms,
    year_chunk_length = year_chunk_length
)
print(f"Generated {len(gnews_searches)} searches...")
visited_urls = []

print("generating prompts...")
# GENERATE PROMPTS FOR PARSING AND FILTERING
metrics_desc = {m["title"]: m["description"] for m in metric_data}

news_instruction = utils.generate_prompt_text(
    news_instruction + format_instruction,
    metrics_desc,
    output_format
)

t0 = utils.log_time(t0, "Loading config")

print("fetching news articles... - limited to 30 for testing")
# FETCHING URLS
news_agent = GNewsFetcher(country=gnews_searches[0]["country"], max_results=max_results)
google_news_articles = news_agent.get_bundle_search(search_country_queries=gnews_searches, visited_urls=visited_urls)[:30]
print(f"Fetched a whole {len(google_news_articles)} articles...")

t1 = utils.log_time(t0, "Fetching gnews headlines")

print("fetching website data...")
# BROWSING AND GETTING FULL TEXT
browser = BrowserSim(page_wait=page_timeout, min_text_length=300, skip_words=skip_words)
browser.start()
accessed_articles = []

for article in google_news_articles:  # test first 5
    text = browser.get_page(article["url"])
    if text:
        article["full_text"] = text
        accessed_articles.append(article)

browser.end()
print(f"Got {len(accessed_articles)} articles with full text.")

print("preprocessing articles...")
# PREPROCESSING AND PARSING ARTICLES

t2 = utils.log_time(t1, "Browsing sites w/ selenium")
# Prepare keyword mapping from your metric data
metric_keywords = {m["title"]: m["rich search"].replace("(", "").replace(")", "").split(" OR ") 
                   for m in metric_data}

PreFilter = logic_parser.SpacyParser(metric_keywords)

print("preprocessing articles...")
preprocessed_articles = []
for article in accessed_articles:
    full_text = utils.trim_text(article.get("full_text", ""), words_start=article_start, words_end=article_end)
    if PreFilter.preprocess_text(full_text):
        preprocessed_articles.append(article)  # keep only relevant articles

print(f"Kept {len(preprocessed_articles)} / {len(accessed_articles)} articles after pre-filtering.")
t3 = utils.log_time(t2, "Preprocessing articles")

print("parsing articles...")
allowed_metrics = [m["title"] for m in metric_data]
parsing_agent = logic_parser.TextParser()
parsing_agent.configure_parsing(None, news_instruction, allowed_metrics)
parsed_articles = []

# PARSE ARTICLES
for article in accessed_articles:
    full_text = article.get("full_text", "")
    agent_response = parsing_agent.parse_and_format(full_text)
    if agent_response:
        parsed_articles.append(agent_response)

t4 = utils.log_time(t3, "Parsing articles with llm")

# FLATTEN RESPONSES
flattened_data = []
for article_response in parsed_articles:
    if article_response is None:
        continue
    for entry in article_response:
        flattened_data.append(entry)

# SAVE TO CSV
output_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_dir, exist_ok=True)
utils.save_to_csv_flat(flattened_data, allowed_metrics, countries_names, years, output_dir=output_dir)

end = utils.log_time(t4, "Saving to CSV")

utils.log_time(store=True, store_data={
    "comment": "multi-threading and trimming text",
    "num_countries": len(countries_names),
    "num_metrics": len(allowed_metrics),
    "num_years": len(years),
    "num_searches": len(gnews_searches),
    "num_articles_fetched": len(google_news_articles),
    "num_articles_accessed": len(accessed_articles),
    "avg_article_length": int(sum(len(a.get("full_text", "")) for a in accessed_articles) / len(accessed_articles)) if accessed_articles else 0,
    "num_articles_filtered" : len(preprocessed_articles),
    "avg_article_length_filtered": int(sum(len(a.get("full_text", "")) for a in preprocessed_articles) / len(preprocessed_articles)) if preprocessed_articles else 0,
    "tokens used" : len(preprocessed_articles) * int(sum(len(a.get("full_text", "")) for a in preprocessed_articles) / len(preprocessed_articles)) / 4,
    "llm_model": "gpt-3.5",
    "max_results": max_results
})
