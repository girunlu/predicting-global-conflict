# Merge Methodology

## Why

model_data.csv only has past conflict events as predictors. master_raw.csv has risk indicators (elections, coups, ethnic tension, etc.) scraped from news. Merging them adds contextual features the model otherwise lacks.

## How

`utils/risk_merge.py` contains `RiskIndicatorMerger`. It does 5 things:

1. Extracts ISO3 country codes from model_data's `matched_admin1_id` (e.g. `"AFG - Kabul"` -> `"AFG"`)
2. Maps master_raw's `source_file` column to ISO3 codes via pycountry + manual overrides
3. Counts mentions per (country, month, metric) and pivots to 7 wide columns
4. Left joins on `(country_code, month_year)` — since master_raw is country-level while model_data is admin-1 level (provinces/states), the same risk values get broadcast to every admin-1 region within that country
5. Lags all 7 risk columns by 1 month (t-1) to prevent data leakage

## Key Decisions

**source_file over country column** — `source_file` has 236 clean country names. `country` has 399 messy values (cities, orgs, encoding issues). source_file answers "what was researched about this country" which is what we want.

**Counts over binary** — A country mentioned 12 times for ethnic tension in a month is different from 1 mention. Counts preserve that.

**Lagged by 1 month** — Matches the existing pipeline where all predictors are (t-1). Prevents the model from seeing same-month data at forecast time.

## Result

7 new predictor columns added: `risk_contested_election (t-1)`, `risk_crop_failure (t-1)`, `risk_economic_concern (t-1)`, `risk_ethnic_tension (t-1)`, `risk_military_coup (t-1)`, `risk_natural_disaster (t-1)`, `risk_political_assassination (t-1)`.

## Performance

No consistent MAE improvement on the top 10 most active regions. Expected — those regions have strong autoregressive signal already. The risk indicators are leading signals that should help more for emerging conflict and lower-activity regions.

## Files

| File | Role |
|---|---|
| `utils/risk_merge.py` | `RiskIndicatorMerger` class (importable module) |
| `merge.ipynb` | Performance comparison notebook |
| `data/processed/model_data.csv` | ACLED processed features |
| `data/raw/master_raw.csv` | Scraped risk indicators |
