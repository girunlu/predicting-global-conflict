import os, re, json, ast
from openai import OpenAI
from dotenv import load_dotenv

# Load environment and initialize OpenAI
load_dotenv()

class TextParser:
    def __init__(self, model="gpt-3.5-turbo"):
        self.ai= OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def get_chatgpt_response(self, instruction : str, prompt : str):
        response = self.ai.responses.create(
            model=self.model,
            instructions=instruction,
            input=prompt,
        )
        return response.output_text
    
    def configure_parsing(self, summary : str, extraction : str, allowed_metrics : list[str] = []):
        self.summary_instruction = summary
        self.extraction_instruction = extraction
        self.allowed_metrics = [m.lower() for m in allowed_metrics]

    def summarise_text(self, text : str): 
        if not hasattr(self, 'summary_instruction'):
            raise ValueError("Parser not configured. Please run configure_parsing() first.")
        
        full_prompt = f"{text[:min(len(text),4000*4)]}"
        return self.get_chatgpt_response(self.summary_instruction, full_prompt)

    def parse_text(self, text : str):
        if not hasattr(self, 'extraction_instruction'):
            raise ValueError("Parser not configured. Please run configure_parsing() first.")
        
        full_prompt = f"{text[:min(len(text),4000*4)]}"
        return self.get_chatgpt_response(self.extraction_instruction, full_prompt)

    def format_response(self, response):
        if not response or not isinstance(response, str):
            return None

        response_cleaned = response.lower().strip()
        if "no" in response_cleaned[:min(10, len(response_cleaned))]:
            return None

        # --- Clean wrappers like ```json ... ``` ---
        response = response.strip()
        if response.startswith("```"):
            # remove ```json ... ``` or ``` ... ```
            response = re.sub(r"^```[a-zA-Z]*\n?", "", response)
            response = re.sub(r"\n?```$", "", response).strip()

        data = None
        # --- Try parsing as JSON ---
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try parsing as Python literal (handles single quotes, etc.)
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

            # 🚫 Drop entries with metrics not in allowed list
            if metric not in self.allowed_metrics:
                continue

            # Validate dates (only mm-yyyy or yyyy allowed)
            valid_dates = [
                d for d in dates
                if isinstance(d, str) and (
                    re.match(r"^\d{2}-\d{4}$", d) or re.match(r"^\d{4}$", d)
                )
            ]

            if country and metric and valid_dates:
                formatted_response.append({
                    "country": country,
                    "metric": metric,
                    "dates": valid_dates
                })

        return formatted_response if formatted_response else None


    def parse_and_format(self, text):
        response = self.parse_text(text)
        formatted = self.format_response(response)
        # print(f"Response: {response}")
        # print(f"Formatted: {formatted}")
        return formatted