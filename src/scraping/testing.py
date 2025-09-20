# --------------------------
# IMPORTING LIBRARIES
# --------------------------
from gnews_fetcher import GNewsFetcher
import os, json, asyncio
from news_boy import AsyncPlaywrightBrowser
from utils import (
    chunk_and_clean_text, remove_repeated_phrase_from_text,
    log_time, generate_search_queries, generate_prompt_text,
    trim_text, save_articles_json, save_to_csv_flat
)
from logic_parser import AsyncTextParser, AsyncKeywordFilter
# --------------------------
# LOADING CONFIGS AND DATA
# --------------------------
t = log_time(label="start")
print("loading configurations...")

io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["official"]
metric_data = io["metrics"]
search_format = io["search format"]
country_data = io["countries"]
country_names = list(country_data.keys())
country_codes = list(country_data.values())
years = io["years"]
excluded_terms = io["exclusions"]

config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
max_results = config['max results']
page_timeout = config['page timeout']
min_page_text_length = config['min page text length']
skip_words = config['website text skip words']
year_chunk_length = config["search year chunk length"]
fuzzy_match_threshold = config.get("fuzzy keyword similarity", 0.5)

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
news_instruction = prompts["instructions"]["news_instruction"]
format_instruction = prompts["instructions"]["format_instruction"]
output_format = prompts["formats"]["output_format"]

for country, code in country_data.items():
    print("HEY")