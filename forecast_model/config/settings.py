predictors = [
    'Battles (t-1)',
    'Explosions/Remote violence (t-1)',
    'Protests (t-1)',
    'Riots (t-1)',
    'Strategic developments (t-1)',
    'Violence against civilians (t-1)',
    'Excessive force against protesters (t-1)',
    'Agreement (t-1)',
    'Explosions/Remote violence_neighbours (t-1)',
    'Strategic developments_neighbours (t-1)',
    'Protests_neighbours (t-1)',
    'Violence against civilians_neighbours (t-1)',
    'Battles_neighbours (t-1)',
    'Riots_neighbours (t-1)',
    'linear_month_trend',
    'year',
    'month_2',
    'month_3',
    'month_4',
    'month_5',
    'month_6',
    'month_7',
    'month_8',
    'month_9',
    'month_10',
    'month_11',
    'month_12',
    'quarter_2',
    'quarter_3',
    'quarter_4',
    ]

# importance_weight is a recency-based sample weight for training, NOT a model feature.
# It is read directly from the dataframe in evaluators.py (sample_weight parameter).
# Including it as a predictor would leak temporal ordering into the model — the model
# would learn from "when" the data was collected rather than from the conflict signals.

targets = [
    'Battles',
    'Explosions/Remote violence',
    'Violence against civilians'
    ]

# ── Enrichment feature groups ──────────────────────────────────────────────────
# Raw column names produced by the enrichment functions (before transformation).
# These are intermediate values — do NOT use directly as model features.
holiday_raw_cols = ['holiday_count_month', 'is_holiday_month']
macro_raw_cols   = ['inflation', 'youth_unemployment', 'income_inequality', 'income_level_code']

# Leakage-safe versions ready for use as model features.
# holiday_raw_cols are lagged 1 month; macro indicators use prior-year values (_py).
# income_level_code is a stable structural attribute — safe to use as-is.
holiday_features = ['n_holidays_lag1', 'is_holiday_month_lag1']
macro_features   = ['inflation_py', 'youth_unemployment_py', 'income_inequality_py',
                    'income_level_code']

# Risk indicator feature names are detected at runtime from column patterns
# (startswith 'risk_', endswith '(t-1)') since they depend on master_raw.csv content.
# Engineered feature names are defined in utils/data_cleaning.build_enhanced_features.