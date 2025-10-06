# Scraping Module Overview

This module handles the collection and processing of news data for predicting global conflict. Before diving in, please review the following files to understand their roles:

## Key Files

1. **prompts.json**  
    Contains instructions and supplementary prompts for the OpenAI API, currently focused on preliminary scraping from GNews.

2. **io.json**  
    Manages all input/output operations between the internet and the OpenAI API, used for both testing and production.

3. **logic_parser.py**  
    Provides basic interaction with the OpenAI API.

4. **gnews_fetcher.py**  
    Fetches news articles from the GNews API and performs initial formatting.

5. **news_boy.py**  
    (Refer to source for details.)

## Main Workflow

After familiarizing yourself with the above files, proceed to `main.py`, which orchestrates the scraping process:

1. Extracts scraping-, parsing-, and configuration-related information from `io.json`, `prompts.json` and `config.json`.
2. Generates search queries and retrieves articles from GNews.
3. Retrives full articles using PlayWright.
3. Filters results using RapidFuzz and the OpenAI API based on provided queries.
4. Converts filtered results into a binary CSV containing date-time and country data.

## How to Run

1. Execute `install_requirements.bat` to install dependencies.
2. Run `main.py` to start data collection and processing.

---
For further details, consult the source code and configuration files.