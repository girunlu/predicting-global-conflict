import os
import pandas as pd
import geopandas as gpd
from utils import data_cleaning, map_admin_regions
from utils.risk_merge import RiskIndicatorMerger
from utils.features.holidays import add_holiday_features
from utils.features.worldbank import add_worldbank_features
from config import settings

_BASELINE_OUT = "data/processed/model_data.csv"
_ENRICHED_OUT = "data/processed/model_data_risk_macro_holidays_engineered.csv"
_RAW_CSV      = "data/raw/1997-01-01-2025-07-03.csv"
_BOUNDARIES   = "data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"


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

    neighbour_data = data_cleaning.summarise_neighbour_events(df_neighbours)
    event_data     = data_cleaning.get_monthly_events(df_neighbours)
    subevent_data  = data_cleaning.get_monthly_subevents(
        df_neighbours, ['Excessive force against protesters', 'Agreement']
    )

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
    if not clean_data and os.path.exists(_BASELINE_OUT):
        print("Loading baseline data from disk...")
        return pd.read_csv(_BASELINE_OUT, index_col=[0, 1])

    print("Building baseline dataset...")
    combined, _ = _build_combined()

    keep = settings.predictors + settings.targets + ['importance_weight']
    model_data = combined[[c for c in keep if c in combined.columns]]

    os.makedirs(os.path.dirname(_BASELINE_OUT), exist_ok=True)
    model_data.to_csv(_BASELINE_OUT)
    print(f"Saved: {_BASELINE_OUT}")
    return model_data


def prepare_enriched_pipeline(
    clean_data: bool = False,
    master_raw_csv: str = "data/raw/master_raw.csv",
    indicators_csv: str = "data/raw/world_bank_data/combined_indicators.csv",
    metadata_csv:   str = "data/raw/world_bank_data/country_metadata.csv",
    holidays_csv:   str = "data/raw/holidays.csv",
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
    if not clean_data and os.path.exists(_ENRICHED_OUT):
        print("Loading enriched data from disk...")
        return pd.read_csv(_ENRICHED_OUT)

    print("Building enriched dataset...")
    combined, gdf = _build_combined()

    # ── Save baseline as a side effect ────────────────────────────────────
    keep = settings.predictors + settings.targets + ['importance_weight']
    baseline = combined[[c for c in keep if c in combined.columns]]
    os.makedirs(os.path.dirname(_BASELINE_OUT), exist_ok=True)
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
            df[py_col] = shifted.fillna(shifted.median())

    # ── 3. Holiday features ────────────────────────────────────────────────
    print("  Adding holiday features...")
    df = add_holiday_features(df, gdf, holidays_path=holidays_csv)
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

    os.makedirs(os.path.dirname(_ENRICHED_OUT), exist_ok=True)
    df.to_csv(_ENRICHED_OUT, index=False)
    print(f"  Saved enriched: {_ENRICHED_OUT}")
    return df


def filter_admin1_data(df, admin1_region):
    return df.loc[admin1_region]
