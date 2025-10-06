# get modules
import os, re, json, ast, asyncio
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from rapidfuzz import fuzz

# load environment variables for OpenAI
load_dotenv()

class AsyncTextParser:
    """
    Asynchronous text parser for extracting structured information from text using OpenAI's GPT models.
    Attributes:
        ai (OpenAI): Instance of the OpenAI API client.
        model (str): The model name to use for text parsing (default: "gpt-3.5-turbo").
        executor (ThreadPoolExecutor): Executor for running blocking API calls in separate threads.
        summary_instruction (str): Instruction for summarizing text (set via configure_parsing).
        extraction_instruction (str): Instruction for extracting information (set via configure_parsing).
        allowed_metrics (list[str]): List of allowed metrics for filtering parsed results.
    Methods:
        __init__(model="gpt-3.5-turbo", max_workers=5):
            Initializes the parser with the specified model and thread pool size.
        async get_chatgpt_response(instruction: str, prompt: str):
            Sends a prompt to the OpenAI API asynchronously and returns the response text.
        configure_parsing(summary: str, extraction: str, allowed_metrics: list[str] = []):
            Configures the parser with summary and extraction instructions, and allowed metrics.
        async summarise_text(text: str):
            Summarizes the given text using the configured summary instruction.
        async parse_text(text: str):
            Extracts information from the given text using the configured extraction instruction.
        format_response(response: str):
            Cleans and parses the response string into a list of dictionaries with 'country', 'metric', and 'dates'.
            Filters entries based on allowed metrics and valid date formats.
        async parse_and_format(text):
            Parses the text and formats the response into structured data.
    Raises:
        ValueError: If the parser is not configured before calling summarise_text or parse_text.
    """
    def __init__(self, model="gpt-3.5-turbo", max_workers=5):
        self.ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def get_chatgpt_response(self, instruction: str, prompt: str):
        loop = asyncio.get_running_loop()
        # Wrap blocking call in executor
        return await loop.run_in_executor(
            self.executor,
            lambda: self.ai.responses.create(
                model=self.model,
                instructions=instruction,
                input=prompt,
            ).output_text
        )

    def configure_parsing(self, summary: str, extraction: str, allowed_metrics: list[str] = []):
        self.summary_instruction = summary
        self.extraction_instruction = extraction
        self.allowed_metrics = [m.lower() for m in allowed_metrics]

    async def summarise_text(self, text: str):
        if not hasattr(self, 'summary_instruction'):
            raise ValueError("Parser not configured.")
        prompt = text[:min(len(text), 4000 * 4)]
        return await self.get_chatgpt_response(self.summary_instruction, prompt)

    async def parse_text(self, text: str):
        if not hasattr(self, 'extraction_instruction'):
            raise ValueError("Parser not configured.")
        prompt = text[:min(len(text), 3500 * 4)]
        return await self.get_chatgpt_response(self.extraction_instruction, prompt)

    def format_response(self, response: str):
        if not response or not isinstance(response, str):
            return None

        response_cleaned = response.lower().strip()
        if "no" in response_cleaned[:min(10, len(response_cleaned))]:
            return None

        # Remove ```json ... ``` wrappers
        response = re.sub(r"^```[a-zA-Z]*\n?", "", response)
        response = re.sub(r"\n?```$", "", response).strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(response)
            except Exception:
                return None

        if not isinstance(data, list):
            return None

        formatted_response = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            country = str(entry.get("country", "")).strip()
            metric = str(entry.get("metric", "")).strip().lower()
            dates = entry.get("dates", [])

            if metric not in self.allowed_metrics:
                continue

            valid_dates = [d for d in dates if isinstance(d, str) and re.match(r"^\d{2}-\d{4}$", d)]

            if country and metric and valid_dates:
                formatted_response.append({
                    "country": country,
                    "metric": metric,
                    "dates": valid_dates
                })

        return formatted_response if formatted_response else None

    async def parse_and_format(self, text):
        response = await self.parse_text(text)
        return self.format_response(response)

class AsyncKeywordFilter:
    '''
    AsyncKeywordFilter provides asynchronous keyword filtering with fuzzy matching.
    Attributes:
        metric_mapping (dict[str, list[str]]): Mapping of metric names to lists of keywords.
        executor (ThreadPoolExecutor): Thread pool for running synchronous preprocessing in background threads.
        min_score (int): Minimum fuzzy match score (0-100) for keyword matching.
    Args:
        metric_keywords (dict[str, list[str]]): Dictionary mapping metrics to their associated keywords.
        max_workers (int, optional): Maximum number of threads for concurrent processing. Defaults to 5.
        min_score (int, optional): Minimum fuzzy match threshold (0-100). Defaults to 80.
    Methods:
        preprocess_text(text: str) -> bool:
            Asynchronously checks if any keyword matches the input text, using exact or fuzzy matching.
        _preprocess_sync(text: str) -> bool:
            Synchronously checks if any keyword matches the input text, using exact or fuzzy matching.
    '''
    def __init__(self, metric_keywords: dict[str, list[str]], max_workers: int = 5, min_score: int = 80):
        self.metric_mapping = metric_keywords
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.min_score = min_score

    async def preprocess_text(self, text: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._preprocess_sync, text)

    def _preprocess_sync(self, text: str) -> bool:
        text_lower = text.lower()

        for metric, keywords in self.metric_mapping.items():
            for kw in keywords:
                # simple exact match first (fast)
                if kw.lower() in text_lower:
                    return True
                # fuzzy match fallback
                for word in text_lower.split():
                    if fuzz.partial_ratio(kw.lower(), word) >= self.min_score:
                        return True
        return False