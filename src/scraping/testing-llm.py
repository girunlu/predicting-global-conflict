import utils, logic_parser
import os, json

io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] #TESTING
metrics = io["metrics"] 

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
news_instruction = prompts["instructions"]["news_instruction"]
news_instruction = utils.generate_instructions(news_instruction, metrics)
# summary_instruction = prompts["instructions"]["summary_instruction"]
# summary_instruction = utils.generate_instructions(summary_instruction, metrics)

print("parsing articles...")
# EXTRACTING INFORMATION FROM ARTICLES
parsing_agent = logic_parser.TextParser()
parsing_agent.configure_parsing(summary_instruction, news_instruction)

articles = utils.load_articles_json()

for article in articles[:5]:
    full_text = article.get("full_text", "")
    print(f"Text: {full_text[:500]}...\n\n")
    summary_text = parsing_agent.summarise_text(full_text)
    print(f"Summary: {summary_text}\n\n")
    parsing_agent.parse_and_format(summary_text)
    print("\n\n---\n\n")
