import utils, logic_parser
import os, json, datetime

io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] #TESTING
metrics = io["metrics"] 
years = io["years"]
countries = io["countries"]
country_names = list(countries.keys())

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
news_instruction = prompts["instructions"]["news_instruction"]
format_instruction = prompts["instructions"]["format_instruction"]
output_format = prompts["formats"]["output_format"]
news_instruction = utils.generate_prompt_text(news_instruction + format_instruction, metrics, output_format)

print("parsing articles...")
# EXTRACTING INFORMATION FROM ARTICLES
parsing_agent = logic_parser.TextParser()
parsing_agent.configure_parsing(None, news_instruction, metrics)

articles = utils.load_articles_json()
output_dir = os.path.join(os.getcwd(), "testing", "outputs")
os.makedirs(output_dir, exist_ok=True)

raw_responses_file = os.path.join(output_dir, f"llm_responses_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

parsed_articles = []
with open(raw_responses_file, "w", encoding="utf-8") as f:
    for article in articles:
        full_text = article.get("full_text", "")
        agent_response = parsing_agent.parse_text(full_text)  # Get raw response
        f.write(f"Article Title: {article.get('title', 'N/A')}\n")
        f.write(f"Article Text: {full_text}\n")
        f.write(f"LLM Response:\n{agent_response}\n")
        f.write("-" * 80 + "\n")

        formatted_response = parsing_agent.format_response(agent_response)
        if formatted_response:
            parsed_articles.append(formatted_response)

# Flatten parsed_articles
flattened_data = []
for article_response in parsed_articles:
    if article_response is None:
        continue
    for entry in article_response:
        flattened_data.append(entry)

# Save to CSV
utils.save_to_csv_flat(flattened_data, metrics, country_names, years, output_dir = output_dir)