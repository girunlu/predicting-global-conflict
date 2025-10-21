# --------------------------
# IMPORTING LIBRARIES
# --------------------------
from gnews_fetcher import GNewsFetcher
import os, json, asyncio
from news_boy import AsyncPlaywrightBrowser
from helpers import (
    chunk_and_clean_text, remove_repeated_phrase_from_text,
    generate_search_queries, generate_prompt_text,
    trim_text, save_articles_json, save_to_csv_flat,
    clean_text, load_articles_json, list_files
)
from logic_parser import AsyncTextParser, AsyncKeywordFilter
# --------------------------
# LOADING CONFIGS AND DATA
# --------------------------
print("loading configurations...")
io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] #TODO
metric_data = io["metrics"]
search_format = io["search format"]
country_data : dict = io["countries"] #TODO FIX
accepted_countries = list(json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["official"]["countries"].keys())
country_names = list(country_data.keys())
country_codes = list(country_data.values())
years = io["years"]
excluded_terms = io["exclusions"]
print("loaded io.json")
#-----------------------------
config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
max_results = config['max results']
page_timeout = config['page timeout']
min_page_text_length = config['min page text length']
skip_words = config['website text skip words']
year_chunk_length = config["search year chunk length"]
fuzzy_match_threshold = config.get("fuzzy keyword similarity", 0.5)
task_list = config.get("task list", [
        "gnews fetch",
        "web scrape",
        "fuzzy filter",
        "openai classify",
        "save results"
    ])
save_points = config.get("save points", [
        "gnews fetch",
        "web scrape",
        "openai classify"
    ])
parallel_filter_tabs = config.get("max parallel filter tabs", 50)
browser_instances = config.get("max browser instances", 5)
browser_tabs = config.get("max browser tabs", 75)
print("loaded config.json")
#-----------------------------
prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
news_instruction = prompts["instructions"]["news_instruction"]
format_instruction = prompts["instructions"]["format_instruction"]
output_format = prompts["formats"]["output_format"]
metrics_desc = {m["title"]: m["description"] for m in metric_data} # giving LLM context
metric_keywords = {m["title"]: m["rich search"].replace("(", "").replace(")", "").split(" OR ")
                for m in metric_data} #for filtering
news_instruction = generate_prompt_text(
    news_instruction + format_instruction,
    metrics_desc,
    output_format
)
#-----------------------------
# TRYING TO SAVE SOME TIME
visited_urls = set()
print("loaded prompts.json\n\n")
# MAKING SURE CONFIGS ARE VALID
if not task_list:
    raise ValueError("No tasks specified in config.json - go through the readme and fix this :)")
if not save_points:
    raise ValueError("No save points specified in config.json - go through the readme and fix this :)")
# LOOPING THROUGH COUNTRIES
for country, code in country_data.items():
    if "gnews fetch" in task_list:
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
        # --------------------------
        # FETCH ARTICLES FROM GNEWS
        # --------------------------
        print("fetching news articles...")
        # Create a GNewsFetcher instance
        news_agent = GNewsFetcher(max_results=max_results, country="US", language="en", visited_urls=visited_urls)
        # Fetch all searches in parallel
        google_news_articles = asyncio.run(news_agent.get_bundle_search_parallel(gnews_searches))
        # SAVING ARTICLES IF REQUESTED
        if "gnews fetch" in save_points:
            save_articles_json(google_news_articles, filename=f"{country} gnews articles.json",updir = "src/scraping", lowdir="outputs/web/gnews")
        print(f"Scraped {len(google_news_articles)} gnews articles for {country}...\n\n")
    # IF NOT GNEWS FETCH, DOUBLE CHECKING THEY AREN'T NEEDED IN FUTURE STEPS
    elif "web scrape" in task_list:
        print("loading previously fetched articles...")
        files = list_files("outputs/web/gnews")
        # MAKES SURE THERE ARE FILES
        if not files:
            raise ValueError(f"No previous gnews articles found for {country} - cannot proceed to web scrape step.")
        google_news_articles = load_articles_json(f"{country} gnews articles.json",updir = "src/scraping", lowdir="outputs/web/gnews")
        # MAKES SURE THERE ARE RELEVANT FILES
        if not google_news_articles:
            raise ValueError(f"No previous gnews articles found for {country} - cannot proceed to web scrape step. Start again with the web scraping")
        print(f"Fetched {len(google_news_articles)} gnews articles for {country}...")

    if "web scrape" in task_list:
        # --------------------------
        # SCRAPE FULL ARTICLES WITH PLAYWRIGHT
        # --------------------------
        async def scrape_articles_parallel(articles, page_wait, min_text_length, skip_words, n_contexts=browser_instances, batch_size=browser_tabs):
            """
            Asynchronously scrape a list of article URLs using AsyncPlaywrightBrowser.

            Parameters:
            - articles: list of article dicts (each must contain a "url" key)
            - page_wait: time to wait for page load / network idle on each page
            - min_text_length: minimum number of characters required to accept scraped text
            - skip_words: list of words/phrases used by the browser scraping to ignore irrelevant text
            - n_contexts: number of browser contexts to create (parallel browser instances)
            - batch_size: max number of pages to open/handle in each batch (limits concurrency)

            Returns:
            - accessed_articles: list of article dicts that had valid "full_text" extracted
            """
            # If there are no articles to process, return early.
            if not articles:
                raise ValueError("No articles provided for scraping - cannot proceed to web scrape step.")

            # Create and configure the Playwright browser wrapper.
            pbrowser = AsyncPlaywrightBrowser(
            page_wait=page_wait,
            min_text_length=min_text_length,
            skip_words=skip_words,
            n_contexts=n_contexts,
            max_concurrent_tasks=batch_size  # <-- limits running pages
            )
            # Start the browser and initialize contexts.
            await pbrowser.start()

            accessed_articles = []

            # Process articles in batches to control concurrency and memory usage.
            for i in range(0, len(articles), batch_size):
                batch = articles[i:i+batch_size]
                tasks = []
                # For each article in the batch, schedule a page scrape task.
                # We distribute pages across contexts by cycling context IDs.
                for j, article in enumerate(batch):
                    context_id = j % n_contexts
                    tasks.append(pbrowser.get_page_text(article["url"], context_id=context_id))

            # Await all page scrapes in the current batch.
            results = await asyncio.gather(*tasks)

            # Post-process each scraped text result.
            for article, text in zip(batch, results):
                # Only keep pages that returned text and meet the minimum length.
                if text and len(text) >= min_text_length:
                    # Normalize and remove repeated boilerplate phrases.
                    text = remove_repeated_phrase_from_text(text.lower(), min_words_in_phrase=3)
                    # Split into clean chunks, remove junk, then rejoin for full_text.
                    clean_chunks = clean_text(chunk_and_clean_text(text, chunk_size=50, max_nontext_ratio=0.3))
                    article["full_text"] = " ".join(clean_chunks)
                    accessed_articles.append(article)

            # Cleanly shut down the browser and contexts when done.
            await pbrowser.end()
            return accessed_articles

        # Run the async scraping function from the synchronous main flow.
        accessed_articles = asyncio.run(
            scrape_articles_parallel(
            google_news_articles,
            page_wait=page_timeout,
            min_text_length=min_page_text_length,
            skip_words=skip_words,
            n_contexts=browser_instances,
            batch_size=browser_tabs
            )
        )
        print(f"Got {len(accessed_articles)} articles with full text.")

        if "web scrape" in save_points:
            save_articles_json(accessed_articles, filename=f"{country} scraped articles.json",updir = "src/scraping", lowdir="outputs/web/scrape")

    elif "openai classify" in task_list:
        print("loading previously scraped articles...")
        files = list_files("outputs/web/scrape")
        # MAKES SURE THERE ARE FILES
        if not files:
            raise ValueError(f"No previous scraped articles found for {country} - cannot proceed to fuzzy filter step.")
        accessed_articles = load_articles_json(f"{country} scraped articles.json",updir = "src/scraping", lowdir="outputs/web/scrape")
        # MAKES SURE THERE ARE RELEVANT FILES
        if not accessed_articles:
            raise ValueError(f"No previous scraped articles found for {country} - cannot proceed to fuzzy filter step. Start again with the web scraping")
        print(f"Fetched {len(accessed_articles)} scraped articles for {country}...")
    
    if "openai classify" in task_list:
        # --------------------------
        # ASYNC NLP PREPROCESSING + PARSING
        # --------------------------
        print("preprocessing and parsing articles in parallel...")

        # Initialize async filter
        async_filter = AsyncKeywordFilter(metric_keywords, max_workers=5, min_score=int(fuzzy_match_threshold*100))

        async def preprocess_articles(batch_size=parallel_filter_tabs):
            """
            Asynchronously preprocess articles in batches and apply the async keyword filter.

            - Trims article text to a manageable length.
            - Runs the async_filter.preprocess_text() on each trimmed text (returns truthy/falsey keep flag).
            - Collects and returns only the articles that pass the prefilter.

            Parameters:
            - batch_size: how many articles to process concurrently (controls asyncio.gather size).

            Returns:
            - filtered_articles: list of article dicts that passed the prefilter.
            """
            filtered_articles = []

            # Process the accessed_articles list in batches to limit concurrency and memory use.
            for i in range(0, len(accessed_articles), batch_size):
                batch = accessed_articles[i:i + batch_size]
                prefilter_tasks = []

            # Prepare prefilter tasks for each article in the batch.
            for a in batch:
                # Trim the article full_text to a bounded length for faster preprocessing.
                trimmed_text = trim_text(a['full_text'][:max(10000, len(a['full_text']))], words_start=50, words_end=1500)
                # Replace the full_text in the article with the trimmed version to save memory downstream.
                a['full_text'] = trimmed_text

                # Queue the async prefilter call.
                prefilter_tasks.append(async_filter.preprocess_text(trimmed_text))

            # Await all prefilter tasks for this batch.
            prefilter_results = await asyncio.gather(*prefilter_tasks)

            # Keep only the articles where the corresponding prefilter result evaluated true.
            filtered_batch = [a for a, keep in zip(batch, prefilter_results) if keep]

            # Extend the running list of filtered articles.
            filtered_articles.extend(filtered_batch)

            return filtered_articles

        filtered_articles = asyncio.run(preprocess_articles())

        print(f"Filtered down to {len(filtered_articles)} articles after preprocessing.")
        # --------------------------
        # ASYNC LLM PARSING
        # --------------------------
        print("parsing articles with LLM in parallel...")

        async_parser = AsyncTextParser(max_workers=browser_instances)
        async_parser.configure_parsing(None, news_instruction, [m["title"] for m in metric_data])

        async def parse_articles(batch_size=50):
            results = []
            for i in range(0, len(filtered_articles), batch_size):
                batch = filtered_articles[i:i + batch_size]
                parse_tasks = [async_parser.parse_and_format(a['full_text']) for a in batch]
                parse_results = await asyncio.gather(*parse_tasks)
                for article, parsed in zip(batch, parse_results):
                    if parsed:
                        article['parsed'] = parsed
                        results.append(article)
            return results

        preprocessed_and_parsed_articles = asyncio.run(parse_articles())

        print(f"Processed {len(preprocessed_and_parsed_articles)} articles after NLP and LLM parsing.")
        # --------------------------
        # FLATTEN AND SAVE
        # --------------------------
        flattened_data = []
        for article in preprocessed_and_parsed_articles:
            for entry in article.get("parsed", []):
                flattened_data.append(entry)

        if "openai classify" in save_points:
            save_articles_json(flattened_data, filename=f"{country} llm outputs.json",updir = "src/scraping", lowdir="outputs/llm")

        output_dir = os.path.join(os.getcwd(), "outputs","csv", "cleaned")
        os.makedirs(output_dir, exist_ok=True)
        save_to_csv_flat(flattened_data, [m["title"] for m in metric_data], accepted_countries, years, output_dir=output_dir)

print("all done!")