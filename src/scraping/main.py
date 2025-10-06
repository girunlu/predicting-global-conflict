# --------------------------
# IMPORTING LIBRARIES
# --------------------------
from gnews_fetcher import GNewsFetcher
import os, json, asyncio
from news_boy import AsyncPlaywrightBrowser
from utils import (
    chunk_and_clean_text, remove_repeated_phrase_from_text,
    log_time, generate_search_queries, generate_prompt_text,
    trim_text, save_articles_json, save_to_csv_flat,
    clean_text, list_files, load_articles_json
)
from logic_parser import AsyncTextParser, AsyncKeywordFilter
# --------------------------
# LOADING CONFIGS AND DATA
# --------------------------
print("loading configurations...")

io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["official"]
metric_data = io["metrics"]
search_format = io["search format"]
country_data = io["countries"] #TODO FIX
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
    # --------------------------
    # GENERATE SEARCH QUERIES
    # --------------------------
    print("generating searches...")
    gnews_searches = generate_search_queries(
        search_format=search_format,
        country_name = country,
        country_code = code,
        metrics=metric_data,
        years=years,
        exclusions=excluded_terms,
        year_chunk_length=year_chunk_length
    )
    print(f"Generated {len(gnews_searches)} searches...")

    metrics_desc = {m["title"]: m["description"] for m in metric_data}
    news_instruction = generate_prompt_text(
        news_instruction + format_instruction,
        metrics_desc,
        output_format
    )

    # --------------------------
    # FETCH ARTICLES FROM GNEWS
    # --------------------------
    print("fetching news articles...")

    # Create a GNewsFetcher instance
    news_agent = GNewsFetcher(max_results=max_results, country="US", language="en")

    # Fetch all searches in parallel
    google_news_articles = asyncio.run(news_agent.get_bundle_search_parallel(gnews_searches))

    print(f"Fetched {len(google_news_articles)} gnews articles...")
    # --------------------------
    # SCRAPE FULL ARTICLES WITH PLAYWRIGHT
    # --------------------------
    async def scrape_articles_parallel(articles, page_wait, min_text_length, skip_words, n_contexts=5, batch_size=50):
        if not articles:
            return []

        pbrowser = AsyncPlaywrightBrowser(
            page_wait=page_wait,
            min_text_length=min_text_length,
            skip_words=skip_words,
            n_contexts=n_contexts,
            max_concurrent_tasks=batch_size  # <-- limits running pages
        )
        await pbrowser.start()

        accessed_articles = []

        for i in range(0, len(articles), batch_size):
            batch = articles[i:i+batch_size]
            tasks = []
            for j, article in enumerate(batch):
                context_id = j % n_contexts
                tasks.append(pbrowser.get_page_text(article["url"], context_id=context_id))

            results = await asyncio.gather(*tasks)
            for article, text in zip(batch, results):
                if text and len(text) >= min_text_length:
                    text = remove_repeated_phrase_from_text(text.lower(), min_words_in_phrase=2)
                    clean_chunks = chunk_and_clean_text(text, chunk_size=50, max_nontext_ratio=0.3)
                    article["full_text"] = " ".join(clean_chunks)
                    accessed_articles.append(article)

        await pbrowser.end()
        return accessed_articles

    accessed_articles = asyncio.run(
        scrape_articles_parallel(
            google_news_articles,
            page_wait=page_timeout,
            min_text_length=min_page_text_length,
            skip_words=skip_words,
            n_contexts=5
        ),
    )
    print(f"Got {len(accessed_articles)} articles with full text.")

    save_articles_json(accessed_articles, filename=f"{country} scraped articles.json",updir = "src/scraping", subdir="outputs/web/raw")
    # --------------------------
    # ASYNC NLP PREPROCESSING + PARSING
    # --------------------------
    print("preprocessing and parsing articles in parallel...")

    # Prepare metric keywords mapping
    metric_keywords = {m["title"]: m["rich search"].replace("(", "").replace(")", "").split(" OR ")
                    for m in metric_data}

    # Initialize async parsers
    async_filter = AsyncKeywordFilter(metric_keywords, max_workers=5, min_score=int(fuzzy_match_threshold*100))
    async_parser = AsyncTextParser(max_workers=5)
    async_parser.configure_parsing(None, news_instruction, [m["title"] for m in metric_data])

    async def process_articles(batch_size=50):
        results = []

        # Preprocess in batches
        for i in range(0, len(accessed_articles), batch_size):
            batch = accessed_articles[i:i + batch_size]

            # Trim text and prepare tasks
            prefilter_tasks = []
            for a in batch:
                trimmed_text = trim_text(a['full_text'][:max(10000, len(a['full_text']))], words_start=50, words_end=1500)
                a['full_text'] = trimmed_text
                prefilter_tasks.append(async_filter.preprocess_text(trimmed_text))

            prefilter_results = await asyncio.gather(*prefilter_tasks)
            filtered_batch = [a for a, keep in zip(batch, prefilter_results) if keep]

            # Parse in batches
            parse_tasks = [async_parser.parse_and_format(a['full_text']) for a in filtered_batch]
            parse_results = await asyncio.gather(*parse_tasks)

            for article, parsed in zip(filtered_batch, parse_results):
                if parsed:
                    article['parsed'] = parsed
                    results.append(article)

        return results

    preprocessed_and_parsed_articles = asyncio.run(process_articles())

    print(f"Processed {len(preprocessed_and_parsed_articles)} articles after NLP and LLM parsing.")
    # --------------------------
    # FLATTEN AND SAVE
    # --------------------------
    flattened_data = []
    for article in preprocessed_and_parsed_articles:
        for entry in article.get("parsed", []):
            flattened_data.append(entry)

    save_articles_json(flattened_data, filename=f"{country} llm outputs.json",updir = "src/scraping", subdir="outputs/llm")

    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    save_to_csv_flat(flattened_data, [m["title"] for m in metric_data], country_names, years, output_dir=output_dir)

    t = log_time(t, "Saving to CSV")
    # --------------------------
    # LOGGING
    # --------------------------
    log_time(store=True,name=country, store_data={
        "comment": "all good man, sped this up a lot",
        "num_countries": len(country_names),
        "num_metrics": len(metric_data),
        "num_years": len(years),
        # "num_searches": len(gnews_searches),
        # "num_articles_fetched": len(google_news_articles),
        "num_articles_accessed": len(accessed_articles),
        # "num_articles_filtered": len(preprocessed_and_parsed_articles),
        "llm_model": "gpt-3.5",
        "max_results": max_results
    })

    print("all done!")