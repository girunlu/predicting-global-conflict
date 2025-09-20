import os, re, json, ast, asyncio
from openai import OpenAI
from dotenv import load_dotenv
import spacy
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from rapidfuzz import fuzz

# Load environment and initialize OpenAI
load_dotenv()

class AsyncTextParser:
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
        prompt = text[:min(len(text), 4000 * 4)]
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
    def __init__(self, metric_keywords: dict[str, list[str]], max_workers: int = 5, min_score: int = 80):
        """
        metric_keywords: dict of metric -> list of keywords
        max_workers: number of threads
        min_score: fuzzy match threshold (0-100)
        """
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


# ---------------------------
# Example usage
# ---------------------------
async def main():
    keywords = {
        "conflict": ["war", "battle", "invasion", "military"],
        "disaster": ["earthquake", "flood", "hurricane"]
    }
    filterer = AsyncKeywordFilter(keywords, max_workers=4)
    texts = [
        "The military forces launched an invasion yesterday.",
        "I baked a cake today.",
        "Floods have displaced thousands of people."
    ]

    tasks = [filterer.preprocess_text(t) for t in texts]
    results = await asyncio.gather(*tasks)
    print(results)  # [True, False, True]

# asyncio.run(main())
