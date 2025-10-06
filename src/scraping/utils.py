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
    Generates a list of search query dictionaries for news or data scraping, combining metrics, country information, and chunked years.
    Args:
        search_format (str): A template string for the search query, containing placeholders: [metric], [country], [year].
        country_name (str): The human-readable name of the country to include in the query.
        country_code (str): The ISO country code, useful for APIs like GNews.
        metrics (list[dict]): A list of metric dictionaries, each containing at least "title" and "rich search" keys.
        years (list[str]): A list of years (as strings) to include in the queries.
        exclusions (str, optional): Additional exclusion terms to append to each query. Defaults to "".
        year_chunk_length (int, optional): Number of years to group together in each query chunk. Defaults to 3.
    Returns:
        list[dict]: A list of dictionaries, each representing a search query with keys:
            - "search": The generated search query string.
            - "country": The country name.
            - "country_code": The ISO country code.
            - "metric": The metric title.
            - "years": The list of years included in this query chunk.
    Example:
        >>> generate_search_queries(
                search_format="[metric] in [country] during [year]",
                country_name="France",
                country_code="FR",
                metrics=[{"title": "GDP", "rich search": "Gross Domestic Product"}],
                years=["2020", "2021", "2022"],
                exclusions="-sports",
                year_chunk_length=2
        [
            {
                "search": "Gross Domestic Product in France during (2020 OR 2021) -sports",
                "country": "France",
                "country_code": "FR",
                "metric": "GDP",
                "years": ["2020", "2021"]
            },
            {
                "search": "Gross Domestic Product in France during 2022 -sports",
                "country": "France",
                "country_code": "FR",
                "metric": "GDP",
                "years": ["2022"]
            }
        ]
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
    Generates a prompt text by replacing placeholders in the input string with metric definitions and example data.
    Args:
        any_text (str): The input string containing placeholders such as '[all metrics]' and '[examples]'.
        metrics (dict): A dictionary where keys are metric names and values are their definitions.
        examples (list[dict] | None): A list of example dictionaries to be inserted into the prompt, or None if not provided.
    Returns:
        str: The formatted prompt text with placeholders replaced by metric definitions and examples.
    Placeholders:
        - '[all metrics]': Replaced with a formatted list of metric definitions.
        - '[examples]': Replaced with a JSON-formatted string of example data (if provided).
    """
    # Build metric definitions text
    metrics_definitions = "\n".join(
        [f"- {name}: {definition}" for name, definition in metrics.items()]
    )

    result = any_text.replace("[all metrics]", metrics_definitions)

    if examples is not None:
        result = result.replace("[examples]", json.dumps(examples, indent=2))

    return result

def save_articles_json(articles, filename, updir, lowdir=None, capture_time=True):
    """
    Saves a list of article dictionaries to a JSON file, with optional timestamp and directory structure.
        List of dictionaries representing articles to save.
        Name of the parent directory where the file will be saved.
        Name of the subdirectory inside `updir` where the file will be saved. If None, file is saved directly in `updir`.
    capture_time : bool, optional
        If True, prepends a timestamp to the filename. Default is True.
    Returns
    -------
    str
        The full path to the saved JSON file.
    Raises
    ------
    ValueError
        If both `updir` and `lowdir` are None.
    Notes
    -----
    - Creates directories if they do not exist.
    - Prints the number of items saved and the file path.
    """
    if updir is None and lowdir is None:
        raise ValueError("Either updir or lowdir must be provided")

    # Make path
    if lowdir is None:
        output_dir = os.path.join(os.getcwd(), updir)
    else:
        output_dir = os.path.join(os.getcwd(), updir, lowdir)

    os.makedirs(output_dir, exist_ok=True)

    if capture_time:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join(output_dir, f"{timestamp}_{filename}")
    else:
        filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(articles)} items to {filepath}")
    return filepath

def load_articles_json(filename, updir, lowdir = None):
    """
    Loads a JSON file containing articles from a specified directory path.

    Args:
        filename (str): The name of the JSON file to load.
        updir (str): The upper directory path where the file is located.
        lowdir (str, optional): An optional lower directory within updir. Defaults to None.

    Raises:
        ValueError: If both updir and lowdir are None.

    Returns:
        dict or list: The parsed JSON content from the file.

    Example:
        articles = load_articles_json("articles.json", "data")
        articles = load_articles_json("articles.json", "data", "2024")
    """
    if updir is None and lowdir is None:
        raise ValueError("Either updir or lowdir must be provided")
    if lowdir is None:
        filepath = os.path.join(os.getcwd(), updir, filename)
    else:
        filepath = os.path.join(os.getcwd(), updir, lowdir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_to_csv_flat(
    data: list[dict],
    metrics: list[str],
    countries: list[str],
    years: list[int],
    output_dir: str = "outputs",
    date_format: str = "%m-%Y",
):
    """
    Saves parsed article data into a flattened CSV file, where each row represents a unique (date, country) combination,
    and each metric is a column indicating its occurrence (1) or absence (0) for that month and country.
    Parameters
    ----------
    data : list of dict
        List of parsed article data entries. Each entry should contain 'country', 'metric', and 'dates' keys.
        - 'country': str, country name.
        - 'metric': str, metric name.
        - 'dates': list of str, date strings representing when the metric occurred.
    metrics : list of str
        List of metric names to include as columns in the output CSV.
    countries : list of str
        List of country names to include as rows in the output CSV.
    years : list of int
        List of years to generate month-year combinations for each country.
    output_dir : str, optional
        Directory where the output CSV file will be saved. Defaults to "outputs".
    date_format : str, optional
        Format string for month-year representation in the CSV. Defaults to "%m-%Y".
    Returns
    -------
    None
        The function saves the resulting DataFrame to a CSV file in the specified output directory.
        The filename includes a timestamp for uniqueness.
    Notes
    -----
    - The CSV columns are: 'date', 'country', followed by one column for each metric.
    - Each metric column contains 1 if the metric occurred for the country in that month, otherwise 0.
    - Unparseable dates in the input data are skipped with a warning.
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
    Logs and measures execution time for code sections, and optionally saves timing summaries to disk.
    Parameters:
        start (float, optional): The starting time (from time.perf_counter()) for measuring duration. If None and store is False, returns a fresh timer.
        label (str, optional): Label for the timed section. Default is "None".
        store (bool, optional): If True, finalizes and saves timing logs to disk. If False, logs a single timing entry.
        store_data (dict, optional): Additional metadata to include in the saved timing summary. Default is None.
        save_dir (str, optional): Directory where timing summary will be saved. Default is "testing/outputs".
        name (str, optional): Optional prefix for the timing summary filename.
    Returns:
        float: If store is False and start is None, returns a fresh timer (time.perf_counter()).
        float: If store is False and start is provided, returns the end time (time.perf_counter()) after logging the duration.
        dict: If store is True, returns the timing summary dictionary after saving it to disk.
    Side Effects:
        - Appends timing logs to the global _TIMING_LOGS list.
        - Prints timing information to stdout.
        - Saves timing summary as a JSON file in the specified directory.
    Example usage:
        start = log_time(label="Step 1")
        # ... code to time ...
        log_time(start, label="Step 1")
        # After all steps:
        log_time(store=True, store_data={"experiment": "test"}, name="run1")
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
    """
    Trims a given text by extracting a subset of words between specified start and end indices.

    Args:
        text (str): The input text to be trimmed.
        words_start (int, optional): The starting index (inclusive) of the word to begin extraction. Defaults to 20.
        words_end (int, optional): The ending index (exclusive) of the word to stop extraction. Defaults to 1500.

    Returns:
        str: A string containing the words from `words_start` up to (but not including) `words_end`.
    """
    words = text.split()
    return " ".join(words[words_start:min(len(words),words_end)])

def remove_repeated_phrase_from_text(text, min_words_in_phrase=2, max_words_to_check=200):
    """
    Removes the largest repeated consecutive phrase from the input text.
    This function searches for the largest sequence of consecutive words (phrase) 
    that is repeated at least twice within the first `max_words_to_check` words of the text.
    If such a repeated phrase is found, all its occurrences are removed from the entire text.
    Args:
        text (str): The input text to process.
        min_words_in_phrase (int, optional): The minimum number of words in a phrase to consider as repeated. Defaults to 2.
        max_words_to_check (int, optional): The maximum number of words from the start of the text to check for repeated phrases. Defaults to 200.
    Returns:
        str: The text with all occurrences of the detected repeated phrase removed. 
             If no repeated phrase is found, returns the original text.
    """
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
    Splits the input text into chunks of a specified size, removes chunks with a high ratio of non-text characters, and returns the cleaned text.
    Args:
        text (str): The input text to be chunked and cleaned.
        chunk_size (int, optional): The size of each chunk in characters. Defaults to 50.
        max_nontext_ratio (float, optional): The maximum allowed ratio of non-alphanumeric characters in a chunk. Chunks exceeding this ratio are discarded. Defaults to 0.3.
    Returns:
        str: The cleaned text, consisting of valid chunks joined by spaces.
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

# compiled for speed
_WSEP = '<<WSEP>>'                 # temporary placeholder for word separators
_re_newlines = re.compile(r'\n\s*\n+')   # collapse many newlines -> paragraph break
_re_multi_space = re.compile(r' {2,}')   # 2+ spaces -> word separator
_re_spaced_letters = re.compile(r'(?:[A-Za-z0-9](?: [A-Za-z0-9]){1,})')  # "a b c" etc.
_re_space_before_punct = re.compile(r'\s+([.,:;?!%])')

def clean_text(text: str) -> str:
    """
    Cleans and normalizes a given text string by performing the following operations:
    - Returns the input unchanged if it is empty or None.
    - Normalizes all newline characters to '\n'.
    - Replaces tab characters and non-breaking spaces with regular spaces.
    - Collapses sequences of two or more newlines into exactly two, preserving paragraph breaks.
    - Marks sequences of two or more spaces with a placeholder to preserve word separation.
    - Merges spaced letters within each chunk (e.g., "a b c" becomes "abc").
    - Restores placeholders to single spaces, ensuring word separators are single spaces.
    - Removes spaces before punctuation (e.g., "word ," becomes "word,").
    - Strips leading and trailing whitespace from each line and from the entire text.
    Args:
        text (str): The input text string to be cleaned.
    Returns:
        str: The cleaned and normalized text string.
    """

    if not text:
        return text

    # normalize newlines & common whitespace
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\t', ' ')
    text = text.replace('\u00A0', ' ')  # non-breaking space -> normal

    # collapse >=2 newlines into exactly two (preserve paragraph breaks)
    text = _re_newlines.sub('\n\n', text)

    # mark 2+ spaces (word separators) with a placeholder so we don't lose them
    text = _re_multi_space.sub(_WSEP, text)

    # merge spaced letters inside each chunk (won't cross the placeholder)
    # e.g. "a b c" -> "abc"
    text = _re_spaced_letters.sub(lambda m: m.group(0).replace(' ', ''), text)

    # restore placeholders to single space (word separators become single spaces)
    text = text.replace(_WSEP, ' ')

    # remove spaces before punctuation like "word ," -> "word,"
    text = _re_space_before_punct.sub(r'\1', text)

    # strip leading/trailing whitespace for each line and global
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text.strip()

def list_files(directory):
    """Return a list of file names in the given directory (non-recursive)."""
    actual_dir = os.path.join(os.getcwd(), directory)
    if not os.path.exists(actual_dir):
        raise ValueError(f"Directory {actual_dir} does not exist.")
    return [f for f in os.listdir(actual_dir) if os.path.isfile(os.path.join(actual_dir, f))]

def save_to_master_csv_bulk(
    data: list[dict],
    metrics: list[str],
    years: list[int],
    file_name: str,
    output_file: str = "outputs/master_raw.csv",
    date_format: str = "%m-%Y",
):
    """
    Saves bulk data entries to a master CSV file, appending new rows for valid country-metric-date combinations.
    This function processes a list of data entries, filters them by allowed metrics and valid date ranges,
    and writes the results to a specified CSV file. Dates are parsed and formatted according to the given
    date_format, and only entries within the specified years and up to the current month are included.
    Parameters:
        data (list[dict]): List of data entries, each containing 'country', 'metric', and 'dates'.
        metrics (list[str]): List of allowed metric names (case-insensitive).
        years (list[int]): List of years to include in the output.
        file_name (str): Name of the source file to record in the output.
        output_file (str, optional): Path to the output CSV file. Defaults to "outputs/master_raw.csv".
        date_format (str, optional): Format string for output dates. Defaults to "%m-%Y".
    Returns:
        None
    Side Effects:
        - Creates the output directory if it does not exist.
        - Appends or writes rows to the specified CSV file.
        - Prints the number of rows appended.
        - Prints a message for any unparseable date encountered.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Build allowed months dynamically
    start_year = min(years)
    end_year = datetime.now().year
    end_month = datetime.now().month

    all_months_str = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == end_year and month > end_month:
                break
            all_months_str.append(f"{month:02d}-{year}")

    rows = []
    for entry in data:
        country = entry.get("country")
        metric = entry.get("metric").lower().strip()
        if metric not in metrics:
            continue

        for date_str in entry.get("dates", []):
            try:
                dt = parser.parse(date_str)
                month_str = dt.strftime(date_format)

                if month_str not in all_months_str:
                    continue

                rows.append({
                    "country": country,
                    "metric": metric,
                    "date": month_str,
                    "source_file": file_name
                })

            except Exception as e:
                print(f"Skipping unparseable date {date_str}: {e}")

    if not rows:
        return

    df_new = pd.DataFrame(rows)

    # Append mode
    if os.path.exists(output_file):
        df_new.to_csv(output_file, mode="a", header=False, index=False)
    else:
        df_new.to_csv(output_file, mode="w", header=True, index=False)

    print(f"Appended {len(rows)} rows to {output_file}")
