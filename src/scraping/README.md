# Overview

This project collects news via GNews, scrapes full article text with Playwright, filters and parses content with LLMs, and outputs structured metric-date records. It is built for batch processing across countries and metrics.

# High-level workflow

1. Load configs (`config.json`, `io.json`, `prompts.json`) and helper modules.
2. Build search queries for country × metric × year chunks.
3. Fetch search results from Google News using `GNewsFetcher` (parallel per search).
4. Resolve article URLs and de-duplicate.
5. Scrape full article HTML/text with `AsyncPlaywrightBrowser` (concurrent contexts, semaphore).
6. Clean, chunk and prefilter texts using `utils` functions and `AsyncKeywordFilter`.
7. Send filtered text to OpenAI via `AsyncTextParser` to extract metric/date findings.
8. Flatten parsed results and save outputs (`llm outputs`, `outputs/`, JSON files).
9. Log run metadata.

# Key modules & responsibilities

* `gnews_fetcher.GNewsFetcher`

  * Uses `gnews` client to run search queries.
  * `get_bundle_search_parallel(searches)`: runs per-country searches concurrently.
  * `_fetch_single(...)`: fetches and attaches basic metadata; avoids revisiting URLs.
  * `extract_real_article_from_rss()` helper decodes Google RSS/read tokens when possible.

* `news_boy.AsyncPlaywrightBrowser`

  * Controls Playwright browser lifecycle and multiple contexts.
  * `start()` / `end()` manage browser and contexts.
  * `resolve_final_url(page, url)`: handles redirects (Google RSS, etc.).
  * `get_page_text(url, context_id)`: scrapes visible paragraph/div text, enforces `min_text_length`, skip-words, and concurrency via semaphore.

* `logic_parser.AsyncTextParser`

  * Wraps OpenAI responses (blocking SDK calls run in a ThreadPoolExecutor).
  * `configure_parsing(summary, extraction, allowed_metrics)` sets LLM instructions.
  * `parse_text()` and `summarise_text()` call OpenAI; `format_response()` normalizes/validates JSON output (expects list of `{country, metric, dates}`).
  * Filters dates to `mm-yyyy` format (expects mm-yyyy entries).

* `logic_parser.AsyncKeywordFilter`

  * Pre-filters article text using keyword lists (exact match then fuzzy via `rapidfuzz`).
  * `preprocess_text()` is run in executor for concurrency.

* `utils` (assumed)

  * `chunk_and_clean_text`, `remove_repeated_phrase_from_text`, `trim_text`, `save_articles_json`, `save_to_csv_flat`, `list_files`, `load_articles_json`, `generate_search_queries`, `generate_prompt_text`, `log_time` etc.
  * Responsible for cleaning, chunking, building prompts, saving JSON/CSV, and logging.

# Configuration & prompts

* `config.json` — runtime behavior: max results, page timeout, min page text length, skip words, year chunk length, fuzzy similarity, batch size.
* `io.json` — metrics definitions, search format, countries mapping, years, exclusions.
* `prompts.json` — `news_instruction`, `format_instruction`, `output_format` for LLM guidance.

# Data flow (detailed)

1. Read initial list of processed search files from `processed web data`.
2. For each file (country batch):

   * Load `accessed_articles` created earlier or from prior steps.
   * Generate `gnews_searches` via `generate_search_queries`.
   * `GNewsFetcher.get_bundle_search_parallel(gnews_searches)` → list of `article` dicts (title, link, published).
   * `scrape_articles_parallel(...)` uses `AsyncPlaywrightBrowser` in batches to fetch `full_text` for each article (applies trimming, dedupe, chunking).
   * Save scraped articles to `to parse/` via `save_articles_json`.
   * Build `metric_keywords` from `metric_data`.
   * Prefilter each article using `AsyncKeywordFilter.preprocess_text` (concurrent).
   * Parse filtered articles via `AsyncTextParser.parse_and_format` (concurrent).
   * Collect `parsed` results, flatten into records, save to `llm outputs` and `outputs/` (CSV optional).
   * Log run summary with `log_time(store=True, ...)`.

# Concurrency and rate control

* GNews searches are parallelized via `asyncio.gather`.
* Playwright scraping:

  * `n_contexts` parallel browser contexts.
  * `max_concurrent_tasks` enforced by an `asyncio.Semaphore`.
  * Batch processing (`batch_size`) limits how many pages are opened/gathered at once.
* OpenAI calls are blocking and run inside a `ThreadPoolExecutor` (executor size configurable via `max_workers`).
* Keyword prefiltering also uses the executor to avoid blocking event loop.

# Important behaviours & validations

* `AsyncTextParser.format_response` expects LLM output to be valid JSON list; it attempts `json.loads` then `ast.literal_eval`.
* Date validation uses regex `^\d{2}-\d{4}$` — only mm-yyyy accepted; otherwise entry discarded.
* If parser isn't configured (`configure_parsing`), calling parse/summarise raises `ValueError`.
* `GNewsFetcher` does NOT fetch article content — only search results and links; URL resolution is handled by Playwright.

# Files produced

* Scraped articles JSON: `to parse/{country} scraped articles.json`
* LLM outputs: `llm outputs/{original_file_name}` (JSON)
* Aggregated CSV (optional): `outputs/*.csv`

# Setup and runtime requirements

* Python 3.11+ (code uses `|` union types).
* System:

  * Playwright (and browsers) installed: `pip3 install playwright` and `playwright install`.
  * `gnews` library.
  * `openai` Python SDK compatible with `OpenAI(...).responses.create` usage or the installed SDK matching code.
  * `rapidfuzz`, `beautifulsoup4`, `httpx`, `python-dotenv`.
* Environment variables:

  * `OPENAI_API_KEY` in `.env` or environment.
* Recommended: run in virtualenv, install dependencies via `py -m pip install -r requirements.txt`.

# Common failure modes & mitigations

* Playwright browser crashes or times out:

  * Increase `page_wait` and `max_task_time`.
  * Reduce `max_concurrent_tasks` or `batch_size`.
* LLM returns non-JSON or unexpected format:

  * Tighten `format_instruction` and include exact schema examples.
  * Add retries and logging of raw LLM outputs for failed parses.
* Dates missing or not mm-yyyy:

  * Either adjust `format_instruction` to force mm-yyyy or extend `format_response` to accept year-only and normalize downstream.
* Google redirect tokens not decoded:

  * `extract_real_article_from_rss()` is best-effort; fallback to Playwright resolving final URL.

# How to run (concise)

1. Ensure env and Playwright installed:

   ```
   py -m pip install -r requirements.txt
   py -m playwright install
   ```
2. Set `OPENAI_API_KEY` in `.env`.
3. Place `io.json`, `config.json`, `prompts.json` next to main script.
4. Run the main script from project root:

   ```
   python main_script.py
   ```

   (Or the filename containing the orchestrating code.)

# Recommended improvements (brief)

* Centralize all concurrency/batching settings into `config.json`.
* Add robust retry/backoff for Playwright navigation and OpenAI calls.
* Persist raw LLM responses for debugging parse failures.
* Add unit tests for `format_response`, `extract_real_article_from_rss`, and `resolve_final_url`.
* Add a CLI entrypoint to run single-country or single-search for easier testing.

# Contact points in code (where to edit behavior)

* `config.json` — tuning thresholds and batch sizes.
* `prompts.json` — change LLM instructions and schema.
* `GNewsFetcher` — adjust `max_results` or country defaults.
* `AsyncPlaywrightBrowser` — tweak timeouts and `max_concurrent_tasks`.
* `AsyncTextParser.format_response` — change date validation or accepted formats.
