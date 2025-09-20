import os, json, time, os, ast, re
from datetime import datetime
from dateutil import parser
import pandas as pd
from pathlib import Path

def generate_search_queries(
    search_format: str,
    country_name: str, 
    country_code: str,
    metrics: list[dict],
    years: list[str],
    exclusions: str = "",
    year_chunk_length: int = 3
) -> list[dict]:
    """
    Generates rich search queries using metrics, countries, and chunked years.
    """
    def chunk_years(years_list, chunk_size):
        for i in range(0, len(years_list), chunk_size):
            yield years_list[i:i + chunk_size]

    queries_to_search = []

    for metric in metrics:
        metric_title = metric["title"]
        metric_rich = metric["rich search"]

        for year_group in chunk_years(years, year_chunk_length):
            if len(year_group) > 1:
                year_part = "(" + " OR ".join(year_group) + ")"
            else:
                year_part = year_group[0]

            query = (
                search_format
                .replace("[metric]", metric_rich)
                .replace("[country]", country_name)
                .replace("[year]", year_part)
            )

            if exclusions:
                query = f"{query} {exclusions}"

            queries_to_search.append({
                "search": query,
                "country": country_name,     # human-readable name
                "country_code": country_code, # ISO code for GNews
                "metric": metric_title,
                "years": year_group
            })

    return queries_to_search

def display_article_results(articles : list[dict]):
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Published Date: {article['published date']}")
        print(f"URL: {article['url']}")
        print(f"Parsed: {article.get('parsed_response', 'N/A')}\n")

def generate_prompt_text(any_text: str, metrics: dict, examples: list[dict] | None) -> str:
    """
    Replaces placeholders in the prompt with metric definitions and examples.
    """
    # Build metric definitions text
    metrics_definitions = "\n".join(
        [f"- {name}: {definition}" for name, definition in metrics.items()]
    )

    result = any_text.replace("[all metrics]", metrics_definitions)

    if examples is not None:
        result = result.replace("[examples]", json.dumps(examples, indent=2))

    return result

def save_articles_json(articles, filename="articles.json", subdir="testing/outputs"):
    """
    Saves any list of dicts to a JSON file with a timestamp.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), subdir)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{timestamp}_{filename}")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(articles)} items to {filepath}")
    return filepath

def load_articles_json(filename="accessed_articles.json"):
    filepath = os.path.join(os.getcwd(), "testing", "outputs", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_to_csv(
    data: list[dict],
    metrics: list[str],
    countries: list[str],
    years: list[int],
    output_dir: str = "outputs",
    date_format: str = "%m-%Y",
):
    """
    Saves parsed article data into CSV files, one per metric,
    using a fixed list of years to generate month columns.
    """
    print("Saving results to CSV...")

    # Generate all month strings from years
    all_months_str = []
    for year in years:
        for month in range(1, 13):
            all_months_str.append(f"{month:02d}-{year}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Build one DataFrame per metric
    for metric in metrics:
        df_metric = pd.DataFrame(0, index=countries, columns=all_months_str)

        for entry in data:
            if entry.get("metric") != metric:
                continue
            country = entry.get("country")
            if country not in countries:
                continue
            for date_str in entry.get("dates", []):
                try:
                    dt = parser.parse(date_str)
                    month_str = dt.strftime(date_format)
                    if month_str in df_metric.columns:
                        df_metric.loc[country, month_str] = 1
                except Exception as e:
                    print(f"Skipping unparseable date {date_str}: {e}")

        df_metric = df_metric.reset_index().rename(columns={"index": "country"})
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(output_dir, f"{metric.replace(' ', '_')}_{timestamp}.csv")
        df_metric.to_csv(filename, index=False)
        print(f"Saved metric '{metric}' to {filename}")

def save_to_csv_flat(
    data: list[dict],
    metrics: list[str],
    countries: list[str],
    years: list[int],
    output_dir: str = "outputs",
    date_format: str = "%m-%Y",
):
    """
    Saves parsed article data into a single CSV with columns:
    date | country | metric1 | metric2 | ...
    Each row represents one (date, country) combination.
    """
    print("Saving flattened results to CSV...")

    # Generate all month strings from years
    all_months_str = []
    for year in years:
        for month in range(1, 13):
            all_months_str.append(f"{month:02d}-{year}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare a list of rows
    rows = []
    for country in countries:
        for month in all_months_str:
            row = {"date": month, "country": country}
            # Initialize all metrics to 0
            for metric in metrics:
                row[metric] = 0
            rows.append(row)

    df = pd.DataFrame(rows)

    # Fill in 1s for metrics that occurred
    for entry in data:
        country = entry.get("country")
        metric = entry.get("metric")
        if country not in countries or metric not in metrics:
            continue
        for date_str in entry.get("dates", []):
            try:
                dt = parser.parse(date_str)
                month_str = dt.strftime(date_format)
                # Set the metric to 1 for this country and date
                df.loc[(df["country"] == country) & (df["date"] == month_str), metric] = 1
            except Exception as e:
                print(f"Skipping unparseable date {date_str}: {e}")

    # Save to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_dir, f"all_metrics_flat_{timestamp}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved flattened metrics CSV to {filename}")

# Globals to keep state between calls
_TIMING_LOGS = []
_RUN_START = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_RUN_START_TIME = time.perf_counter()

def log_time(start=None, label: str = "None", store: bool = False, store_data: dict = None, save_dir: str = "testing/outputs", name = None):
    """
    Tracks and stores execution time for script sections.

    Args:
        start: start time from time.perf_counter()
        label: name of the step (e.g., 'Fetching news')
        store: if True, finalizes run logs with metadata and writes JSON
        store_data: optional metadata (dict) to save with logs
        save_dir: directory to save timing logs

    Returns:
        tuple: (end_time, log_entry) if measuring a section
    """
    global _TIMING_LOGS, _RUN_START, _RUN_START_TIME

    # Measure time for a section
    if not store:
        if start is None:
            return time.perf_counter()  # fresh timer
        end = time.perf_counter()
        duration = end - start
        log_entry = {"label": label, "duration_seconds": round(duration, 3)}
        _TIMING_LOGS.append(log_entry)
        print(f"{label} took {duration:.2f} seconds.\n")
        return end

    # Finalize and save logs
    else:
        total_runtime = time.perf_counter() - _RUN_START_TIME
        for entry in _TIMING_LOGS:
            entry["percent_of_total"] = round(
                (entry["duration_seconds"] / total_runtime) * 100, 2
            )

        result = {
            "run_started": _RUN_START,
            "total_runtime_seconds": round(total_runtime, 2),
            "steps": _TIMING_LOGS,
            "metadata": store_data or {},
        }

        # Save to file
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filepath = save_path / f"{name} timing_summary_{_RUN_START}.json" if name else save_path / f"timing_summary_{_RUN_START}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Timing summary saved to {filepath}\n")
        return result

def trim_text(text, words_start : int = 20, words_end : int = 1500):
    words = text.split()
    return " ".join(words[words_start:max(len(words),words_end)])

def remove_repeated_phrase_from_text(text, min_words_in_phrase=2, max_words_to_check=200):
    words = text.split()
    n = len(words)

    # Only consider the first N words for detecting repeated phrase
    check_words = words[:max_words_to_check]
    max_len = len(check_words) // 2

    repeated_phrase = None

    # Find the largest repeated consecutive phrase in the first N words
    for length in range(max_len, min_words_in_phrase - 1, -1):
        i = 0
        while i + 2*length <= len(check_words):
            phrase = check_words[i:i+length]
            next_phrase = check_words[i+length:i+2*length]
            if phrase == next_phrase:
                repeated_phrase = phrase
                break
            i += 1
        if repeated_phrase:
            break

    # If a repeated phrase was found, remove all its occurrences in the whole text
    if repeated_phrase:
        phrase_len = len(repeated_phrase)
        i = 0
        cleaned_words = []
        while i <= n - phrase_len:
            if words[i:i+phrase_len] == repeated_phrase:
                i += phrase_len  # skip this repeated phrase
            else:
                cleaned_words.append(words[i])
                i += 1
        # append remaining words
        cleaned_words.extend(words[i:])
        return " ".join(cleaned_words)
    else:
        return text
    
def chunk_and_clean_text(text, chunk_size=50, max_nontext_ratio=0.3):
    """
    Split text into chunks of chunk_size and skip chunks with too many non-text characters.
    
    Parameters:
    - text: str, raw scraped text
    - chunk_size: int, approx number of characters per chunk
    - max_nontext_ratio: float, max allowed ratio of non-alphanumeric characters
    
    Returns:
    - List of clean text chunks
    """
    # Normalize whitespace
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]

        # Count non-alphanumeric characters
        nontext_count = len(re.findall(r'[^a-zA-Z0-9\s]', chunk))
        ratio = nontext_count / max(1, len(chunk))

        if ratio <= max_nontext_ratio:
            # chunk is mostly text
            chunks.append(chunk.strip())

    return " ".join(chunks)


