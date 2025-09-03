import os
from datetime import datetime
from dateutil import parser
import pandas as pd
import json

def generate_search_queries(google_search_templates : list[str], 
    country_names : list[str], 
    search_metrics : list[str], 
    years : list[str]) -> list[dict]:
    '''
    # Outputs
    Array of dictionary with search query and desired country
    '''
    queries_to_search = []
    for search in google_search_templates:
        for country in country_names:
            search_c = search.replace("[country]", country)
            for metric in search_metrics:
                search_m = search_c.replace("[metric]", metric)
                for year in years:
                    search_y = search_m.replace("[year]", year)
                    queries_to_search.append({"search": search_y,
                                         "country": country
                                        })
    return queries_to_search

def display_article_results(articles : list[dict]):
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Published Date: {article['published date']}")
        print(f"URL: {article['url']}")
        print(f"Parsed: {article.get('parsed_response', 'N/A')}\n")

def generate_prompt_text(any_text : str, metrics : list[str], examples : list[dict] | None) -> str:
    if examples is None:
        return any_text.replace("[all metrics]", ", ".join(metrics))
    return any_text.replace("[all metrics]", ", ".join(metrics)).replace("[examples]", str(examples))

def save_articles_json(articles, filename="accessed_articles.json"):
    output_dir = os.path.join(os.getcwd(),"testing", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(articles)} articles to {filepath}")

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