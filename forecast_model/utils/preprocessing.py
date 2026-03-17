import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from utils import data_cleaning, map_admin_regions
from utils.risk_merge import RiskIndicatorMerger
from utils.features.holidays import add_holiday_features
from utils.features.worldbank import add_worldbank_features
from utils.fetch_world_bank_data import WorldBankDataFetcher
from config import settings

_ROOT         = Path(__file__).parent.parent
_BASELINE_OUT = _ROOT / "data/processed/model_data.csv"
_ENRICHED_OUT = _ROOT / "data/processed/model_data_risk_macro_holidays_engineered.csv"
_RAW_CSV      = _ROOT / "data/raw/1997-01-01-2025-07-03.csv"
_BOUNDARIES   = _ROOT / "data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"


def _build_combined():
    """
    Shared internal step: load raw ACLED, build event/neighbour tables,
    add lagged columns, time features, and importance weights.

    Returns (combined, gdf) where combined has MultiIndex (matched_admin1_id, month_year).
    """
    df = pd.read_csv(_RAW_CSV)
    df = df[df['year'] >= 2018].copy()
    df['date'] = pd.to_datetime(df['event_date'], format='%d %B %Y')
    df['month_year'] = df['date'].dt.to_period('M').astype(str)

    gdf = gpd.read_file(_BOUNDARIES)
    df_neighbours = map_admin_regions.add_admin1_neighbors(df, gdf)
    df_neighbours = df_neighbours.dropna(subset=['matched_admin1_id']).copy()

    neighbour_data = data_cleaning.summarise_neighbour_events(df_neighbours)
    event_data     = data_cleaning.get_monthly_events(df_neighbours)
    subevent_data  = data_cleaning.get_monthly_subevents(
        df_neighbours, ['Excessive force against protesters', 'Agreement']
    )

    # Get World Bank data
    wb_dir = "data/raw/world_bank"
    indicators_path = os.path.join(wb_dir, "combined_indicators.csv")
    metadata_path = os.path.join(wb_dir, "country_metadata.csv")

    if not os.path.exists(indicators_path) or not os.path.exists(metadata_path):
        wb = WorldBankDataFetcher()
        countries = wb.get_countries()
        data = wb.get_all_indicators()

        wb.save_data(data, countries, wb_dir)
        countries.to_csv(metadata_path, index=False)

    combined = pd.concat([event_data, subevent_data], axis=1).join(neighbour_data, how='left')
    combined = data_cleaning.add_lagged_columns(combined)
    combined = data_cleaning.add_time_trend_features(combined)
    combined = data_cleaning.add_importance_weights(combined)
    return combined, gdf


def prepare_data_pipeline(clean_data: bool = False) -> pd.DataFrame:
    """
    Baseline pipeline: 30 ACLED predictors + targets + importance_weight.
    Saves to data/processed/model_data.csv.
    """
    if not clean_data and _BASELINE_OUT.exists():
        print("Loading baseline data from disk...")
        return pd.read_csv(_BASELINE_OUT, index_col=[0, 1])

    print("Building baseline dataset...")
    combined, _ = _build_combined()

    keep = settings.predictors + settings.targets + ['importance_weight']
    model_data = combined[[c for c in keep if c in combined.columns]]

    _BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(_BASELINE_OUT)
    print(f"Saved: {_BASELINE_OUT}")
    return model_data


def prepare_enriched_pipeline(
    clean_data: bool = False,
    master_raw_csv: str = "data/raw/master_raw.csv",
    indicators_csv: str = "data/raw/world_bank/combined_indicators.csv",
    metadata_csv:   str = "data/raw/world_bank/country_metadata.csv",
    holidays_csv:   str = "data/raw/holidays_raw.csv",
) -> pd.DataFrame:
    """
    Full enrichment pipeline — adds on top of the baseline in sequence:
      1. Risk indicators (CAST signals, lagged t-1)
      2. World Bank macro indicators (prior-year, anti-leakage)
      3. Holiday features (lagged t-1)
      4. Engineered features (lag-2, rolling averages, interactions)

    Also saves the baseline (model_data.csv) as a side effect so both
    datasets are always available on disk.

    Saves to data/processed/model_data_risk_macro_holidays_engineered.csv.
    """
    if not clean_data and _ENRICHED_OUT.exists():
        print("Loading enriched data from disk...")
        return pd.read_csv(_ENRICHED_OUT)

    print("Building enriched dataset...")
    combined, gdf = _build_combined()

    # ── Save baseline as a side effect ────────────────────────────────────
    keep = settings.predictors + settings.targets + ['importance_weight']
    baseline = combined[[c for c in keep if c in combined.columns]]
    _BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(_BASELINE_OUT)
    print(f"  Saved baseline: {_BASELINE_OUT}")

    # ── 1. Risk indicators (RiskIndicatorMerger reads from disk) ──────────
    print("  Merging risk indicators...")
    merger = RiskIndicatorMerger(lag=1)
    df = merger.merge(_BASELINE_OUT, master_raw_csv)
    # df is now a flat DataFrame; set MultiIndex for the feature functions
    df = df.set_index(["matched_admin1_id", "month_year"])

    # ── 2. World Bank macro indicators ────────────────────────────────────
    print("  Adding macro indicators...")
    df = add_worldbank_features(df, gdf,
                                indicators_path=indicators_csv,
                                metadata_path=metadata_csv)
    df = df.sort_index()
    # Prior-year shift: raw col → _py col (year Y uses year Y-1's value).
    # WB values are broadcast yearly so shift(12) lands on the correct prior year.
    raw_to_py = {r: p for r, p in
                 zip(['inflation', 'youth_unemployment', 'income_inequality'],
                     ['inflation_py', 'youth_unemployment_py', 'income_inequality_py'])}
    for raw_col, py_col in raw_to_py.items():
        if raw_col in df.columns:
            shifted = df.groupby(level='matched_admin1_id')[raw_col].shift(12)
            # Year-specific median imputation (matches feature_engineering_3.ipynb):
            # missing values get the median for that year across all regions, then
            # a global median as final fallback for years with no data at all.
            year_key = df.index.get_level_values('month_year').str[:4].astype(int)
            year_medians = shifted.groupby(year_key).transform('median')
            df[py_col] = shifted.fillna(year_medians).fillna(shifted.median())

    # income_level_code is a stable structural attribute (no prior-year shift needed).
    # Impute territories/unclassified regions with the global median (matches notebook).
    if 'income_level_code' in df.columns:
        median_level = df['income_level_code'].median()
        df['income_level_code'] = df['income_level_code'].fillna(median_level)

    # ── 3. Holiday features ────────────────────────────────────────────────
    print("  Adding holiday features...")
    df = add_holiday_features(df, gdf, holidays_path=holidays_csv)
    # Sort so that positional shift(1) below lands on the correct prior month.
    df = df.sort_index()
    # Lag 1 month — consistent with t-1 design; column names match settings.holiday_features
    for raw_col, lag_col in zip(settings.holiday_raw_cols, settings.holiday_features):
        if raw_col in df.columns:
            df[lag_col] = (
                df.groupby(level='matched_admin1_id')[raw_col]
                .shift(1).fillna(0).astype(int)
            )

    # ── 4. Engineered features ─────────────────────────────────────────────
    print("  Building engineered features...")
    df = data_cleaning.build_enhanced_features(df.reset_index())

    _ENRICHED_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ENRICHED_OUT, index=False)
    print(f"  Saved enriched: {_ENRICHED_OUT}")
    return df


def filter_admin1_data(df, admin1_region):
    return df.loc[admin1_region]
