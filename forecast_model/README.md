# ACLED Conflict Forecasting

Forecasts **monthly conflict event counts** at the **Admin-1 level** using ACLED data enriched with risk indicators, macroeconomic variables, public holidays, and engineered features. All pipelines — data preparation, single-region forecasting, and multi-model benchmarking — run through a single entry point: `main.py`.

---

## Project Structure

```
forecast_model/
├── main.py                  # Single entry point for all modes
├── config/
│   └── settings.py          # Predictor / target / feature group definitions
├── models/
│   └── simple_model.py      # Random Forest training & evaluation
├── utils/
│   ├── preprocessing.py     # Dataset build pipelines (baseline & enriched)
│   ├── data_cleaning.py     # Feature engineering (lags, rolling avgs, interactions)
│   ├── risk_merge.py        # CAST risk indicator integration
│   ├── evaluators.py        # Model evaluation utilities (MAE, MAPE, splits)
│   ├── ablation.py          # Multi-model ablation benchmark logic
│   ├── visualization.py     # Plotting utilities
│   ├── map_admin_regions.py # Spatial admin-1 neighbour mapping
│   └── features/
│       ├── worldbank.py     # World Bank macro & income-level features
│       └── holidays.py      # Public holiday calendar features
├── data/
│   ├── raw/                 # Source data (ACLED, World Bank, shapefiles, …)
│   └── processed/           # Built datasets (auto-generated, not committed)
├── notebooks/               # Exploratory analysis (data_an, merge, models, ablation)
└── outputs/
    └── figures/             # Forecast plots & ablation heatmaps
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/predicting-global-conflict.git
cd forecast_model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Raw Data

Place the following files in `data/raw/` (available from the [project Google Drive](https://drive.google.com/drive/folders/1qG9lFDUKTZW2kG6erbAqRSJhdm5l1255?usp=sharing)):

| File | Description |
| ---- | ----------- |
| `1997-01-01-2025-07-03.csv` | ACLED event-level conflict data |
| `master_raw.csv` | CAST early-warning risk indicators |
| `combined_indicators.csv` | World Bank annual macro indicators |
| `holidays_raw.csv` | Public holiday calendar (233 countries) |
| `boundaries/ne_10m_admin_1_states_provinces/` | Admin-1 shapefiles |

---

## Running the Pipeline

Everything runs through `main.py`. There are four modes.

### 1. Build the baseline dataset (30 features)

```bash
python main.py
```

Loads from disk if already built. Add `--clean-data` to force a rebuild from raw files.

```bash
python main.py --clean-data
```

Saves to: `data/processed/model_data.csv`

### 2. Build the enriched dataset (51 features)

```bash
python main.py --enrich
python main.py --enrich --clean-data   # force rebuild
```

Adds on top of the baseline in sequence:

- **Risk indicators** — 7 CAST early-warning signals, lagged t-1
- **Macro indicators** — inflation, youth unemployment, income inequality (prior-year), income level code
- **Holiday features** — monthly public holiday count, lagged t-1
- **Engineered features** — lag-2s, rolling averages, cross-type interactions

Saves to: `data/processed/model_data_risk_macro_holidays_engineered.csv`

### 3. Forecast a single region

```bash
python main.py --region "UKR - Donetsk" --event "Battles"
```

With enriched features:

```bash
python main.py --region "UKR - Donetsk" --event "Battles" --enrich
```

With `--enrich`, the model trains on all 51 features. Without it, the baseline 30 features are used.

**Available event types:** `Battles`, `Explosions/Remote violence`, `Violence against civilians`

Outputs:

- MAE and MAPE printed to console
- Forecast plot saved to `outputs/figures/`
- Feature importance plot saved to `outputs/figures/`

### 4. Run the multi-model ablation benchmark

```bash
python main.py --ablation
```

Evaluates **5 models × 5 cumulative feature sets × 3 targets × top-10 regions = 750 evaluations** with a 6-month chronological holdout.

| Model | Notes |
| ----- | ----- |
| RF (Random Forest) | Project baseline |
| LGBM-Poisson | Poisson loss for discrete counts |
| LGBM-Tweedie (p=1.5) | CAST methodology — handles zero-inflation and heavy tail |
| XGBoost | Strong tabular baseline |
| GBR (sklearn) | Gradient boosting sanity check |

| Feature set | Contents |
| ----------- | -------- |
| Baseline | 30 ACLED predictors |
| +Risk | + 7 CAST risk indicators |
| +Macro | + inflation, unemployment, Gini, income level |
| +Holidays | + monthly holiday count |
| +Engineered | + lag-2s, rolling averages, co-escalation interaction |

Options:

```bash
python main.py --ablation --top-n 20       # evaluate top-20 regions (default: 10)
python main.py --ablation --holdout 12     # 12-month holdout (default: 6)
python main.py --ablation --clean-data     # rebuild data then run ablation
```

Outputs:

- MAE tables printed to console
- Heatmap saved to `outputs/figures/ablation_heatmap.png`
- Raw results saved to `outputs/ablation_results.csv`

---

## Dataset Details

### Feature groups

| Group | Count | Source |
| ----- | ----- | ------ |
| Event lags (t-1) | 8 | ACLED — Battles, Remote violence, Protests, Riots, etc. |
| Neighbour spillover (t-1) | 6 | ACLED — summed events across adjacent admin-1 regions |
| Temporal controls | 16 | Linear trend, year, month/quarter dummies |
| Risk indicators (t-1) | 7 | CAST — crop failure, ethnic tension, military coup, etc. |
| Macro indicators | 4 | World Bank — inflation, youth unemployment, Gini, income level |
| Holiday features | 2 | Public holiday calendar — count + binary flag, lagged |
| Engineered features | 9 | Lag-2s, 3-month rolling averages, organized violence aggregate, co-escalation interaction |
| **Total predictors** | **51** | |
| Targets | 3 | Battles, Explosions/Remote violence, Violence against civilians |

### Leakage prevention

All features are lagged or shifted to ensure the model only sees information available at prediction time:

- Event and risk features: **t-1** (previous month)
- Macro indicators: **prior-year** values (publication lag)
- Holiday counts: **t-1** (previous month)

### Train / test split

A **6-month chronological holdout** is used for all evaluation. Data is never shuffled. Sample weights apply exponential recency decay (more recent months weighted higher during training).

---

## Configuration

`config/settings.py` is the single source of truth for all column names:

```python
predictors       # 30 baseline feature names
targets          # 3 target event type names
macro_features   # 4 World Bank feature names (model-ready)
holiday_features # 2 holiday feature names (model-ready)
```

---

## Modeling Details

The forecasting model (`--region/--event` mode) uses a **Random Forest Regressor** with:

- 100 trees, `random_state=42`
- Recency-based sample weights (exponential decay)
- 6-month chronological holdout for evaluation

For more comprehensive benchmarking across model architectures and feature sets, use `--ablation`.

---

## License

MIT License.
