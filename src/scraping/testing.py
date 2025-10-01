# --------------------------
# IMPORTING LIBRARIES
# --------------------------
import os, json, asyncio
import utils
# --------------------------
# LOADING CONFIGS AND DATA
# --------------------------
io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["official"]
metric_data = io["metrics"]
metrics = [m["title"] for m in metric_data]
print(metrics)
years = io["years"]

files = utils.list_files("llm outputs")
countries = [" ".join(c.split("_")[2].split()[:-2]).strip() for c in files]

for file in files:
    name = " ".join(file.split("_")[2].split()[:-2]).strip()

    data = utils.load_articles_json(file, "llm outputs")

    utils.save_to_master_csv(
        data=data,
        metrics=metrics,
        years=years,
        file_name=name,
        )