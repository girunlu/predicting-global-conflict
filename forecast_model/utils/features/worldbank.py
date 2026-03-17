import os
import requests
import pandas as pd

# Ordinal encoding matching feature_engineering_3.ipynb:
# LIC=1, LMC=2, UMC=3, HIC=4  (API short codes)
INCOME_ORDER_API = {'LIC': 1, 'LMC': 2, 'UMC': 3, 'HIC': 4}

# Full-name mapping for metadata CSV files (same ordinal scale)
INCOME_LEVEL_MAP = {
    "Low income": 1,
    "Lower middle income": 2,
    "Upper middle income": 3,
    "High income": 4,
}


def _fetch_income_levels_from_api() -> dict:
    """
    Fetch World Bank income group classifications for all countries.

    Returns {iso3_code: income_level_ordinal} where 1=LIC, 2=LMC, 3=UMC, 4=HIC.
    Returns empty dict on network failure (caller handles missing values).
    """
    url = "https://api.worldbank.org/v2/country/all?format=json&per_page=300"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        countries = r.json()[1]   # index 0 is pagination metadata
        mapping = {}
        for c in countries:
            iso3  = c.get('id', '')
            level = c.get('incomeLevel', {}).get('id', '')
            if len(iso3) == 3 and level in INCOME_ORDER_API:
                mapping[iso3] = INCOME_ORDER_API[level]
        print(f"  Fetched income levels for {len(mapping)} countries from World Bank API")
        return mapping
    except Exception as e:
        print(f"  World Bank API unavailable ({e}) — income_level_code will be imputed")
        return {}


def _load_country_metadata(metadata_path: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path).copy()

    # Standardize join key
    meta = meta.rename(columns={"id": "country_iso3"})

    # Drop aggregate pseudo-entities
    meta = meta[meta["income_level"].fillna("") != "Aggregates"]
    meta = meta[meta["region_name"].fillna("") != "Aggregates"]

    # Clean ISO3
    meta = meta[meta["country_iso3"].notna()]
    meta["country_iso3"] = meta["country_iso3"].astype(str).str.upper().str.strip()

    # Income level code — same ordinal scale as API fetch
    meta["income_level_code"] = meta["income_level"].map(INCOME_LEVEL_MAP)

    return meta[["country_iso3", "income_level_code"]]


def _indicators_yearly_wide_to_monthly(indicators_path: str) -> pd.DataFrame:

    ind = pd.read_csv(indicators_path).copy()

    # Drop aggregate rows (no usable ISO3)
    ind = ind[ind["countryiso3code"].notna()].copy()
    ind = ind.rename(columns={"countryiso3code": "country_iso3"})
    ind["country_iso3"] = ind["country_iso3"].astype(str).str.upper().str.strip()

    # Detect year-suffixed columns: <feature>_<YYYY>
    year_cols = [
        c for c in ind.columns
        if "_" in c and c.split("_")[-1].isdigit() and len(c.split("_")[-1]) == 4
    ]

    long = ind.melt(
        id_vars=["country_iso3"],
        value_vars=year_cols,
        var_name="feature_year",
        value_name="value",
    )

    parts = long["feature_year"].str.rsplit("_", n=1, expand=True)
    long["feature"] = parts[0]
    long["year"] = parts[1].astype(int)

    # Pivot to (country_iso3, year, feature) and fill sparse indicators
    # (e.g. Gini is measured every few years) with within-country bfill then ffill
    annual = (
        long.pivot_table(
            index=["country_iso3", "year"],
            columns="feature",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    annual.columns.name = None

    # Extend to 2025 so prior-year values are available for all model months
    all_combos = pd.DataFrame(
        [(iso3, yr)
         for iso3 in annual["country_iso3"].unique()
         for yr in range(2010, 2026)],
        columns=["country_iso3", "year"],
    )
    annual = all_combos.merge(annual, on=["country_iso3", "year"], how="left")
    annual = annual.sort_values(["country_iso3", "year"])

    # Within each country: bfill then ffill to fill gaps in sparse series (e.g. Gini)
    feature_cols = [c for c in annual.columns if c not in ["country_iso3", "year"]]
    for col in feature_cols:
        annual[col] = annual.groupby("country_iso3")[col].transform(
            lambda x: x.bfill().ffill()
        )

    # Broadcast each annual value to all 12 months of that year
    months = pd.DataFrame({"month": list(range(1, 13))})
    annual = annual.merge(months, how="cross")
    annual["month_year"] = (
        annual["year"].astype(str)
        + "-"
        + annual["month"].astype(str).str.zfill(2)
    )

    wide = annual.drop(columns=["year", "month"])
    return wide


def add_worldbank_features(
    combined: pd.DataFrame,
    gdf,
    indicators_path: str = "data/raw/world_bank/combined_indicators.csv",
    metadata_path: str = "data/raw/world_bank/country_metadata.csv",
) -> pd.DataFrame:

    # Extract ISO3 directly from the matched_admin1_id prefix ("ISO3 - region name").
    # Previously used a GDF name_en reverse-lookup which silently failed for 214 regions
    # whose names changed after fix_france/fix_libya (e.g. "FRA - Auvergne-Rhone-Alpes"
    # vs original GDF departments). Direct extraction is robust to any name changes.
    _ACLED_TO_ISO3 = {'PSX': 'PSE', 'SDS': 'SSD'}  # ACLED-specific overrides

    idx = combined.index.to_frame(index=False)
    idx["matched_admin1_id"] = idx["matched_admin1_id"].astype(str)
    idx["month_year"] = pd.to_datetime(idx["month_year"], errors="coerce").dt.strftime("%Y-%m")
    idx["adm0_a3"] = (
        idx["matched_admin1_id"].str.split(" - ").str[0]
        .str.upper().str.strip()
        .replace(_ACLED_TO_ISO3)
    )

    # Build WB monthly indicators
    wb_monthly = _indicators_yearly_wide_to_monthly(indicators_path)

    # Drop aggregates like AFE/ARB/etc (keep real ISO3 only)
    wb_monthly = wb_monthly[wb_monthly["country_iso3"].astype(str).str.len() == 3].copy()
    wb_monthly["month_year"] = wb_monthly["month_year"].astype(str)

    # ── Income level code ───────────────────────────────────────────────────
    # Prefer local metadata CSV; fall back to World Bank API (matches notebook logic)
    if metadata_path and os.path.exists(metadata_path):
        meta = _load_country_metadata(metadata_path)
        wb_monthly = wb_monthly.merge(meta, on="country_iso3", how="left")
    else:
        income_map = _fetch_income_levels_from_api()
        if income_map:
            wb_monthly["income_level_code"] = wb_monthly["country_iso3"].map(income_map)

    # Merge WB onto idx using adm0_a3 (country ISO3) + month_year
    enriched = idx.merge(wb_monthly, left_on=["adm0_a3", "month_year"], right_on=["country_iso3", "month_year"], how="left")

    # Attach back to combined (robust: merge on explicit columns)
    wb_cols = ["inflation", "youth_unemployment", "income_inequality", "income_level_code"]
    wb_cols = [c for c in wb_cols if c in enriched.columns]

    # features keyed by (matched_admin1_id, month_year)
    feat = enriched[["matched_admin1_id", "month_year"] + wb_cols].copy()
    feat["matched_admin1_id"] = feat["matched_admin1_id"].astype(str)
    feat["month_year"] = feat["month_year"].astype(str)

    # normalize combined keys BEFORE merge
    tmp = combined.reset_index().copy()
    tmp["matched_admin1_id"] = tmp["matched_admin1_id"].astype(str)
    tmp["month_year"] = pd.to_datetime(tmp["month_year"], errors="coerce").dt.strftime("%Y-%m")

    out = tmp.merge(feat, on=["matched_admin1_id", "month_year"], how="left")
    out = out.set_index(["matched_admin1_id", "month_year"])

    return out
