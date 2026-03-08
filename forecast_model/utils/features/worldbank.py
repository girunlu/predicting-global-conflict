import pandas as pd

INCOME_LEVEL_MAP = {
    "Low income": -2,
    "Lower middle income": -1,
    "Upper middle income": 1,
    "High income": 2,
}


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

    # Income level code
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

    # Broadcast each (country, year) to all months
    months = pd.DataFrame({"month": list(range(1, 13))})
    long = long.merge(months, how="cross")
    long["month_year"] = long["year"].astype(str) + "-" + long["month"].astype(str).str.zfill(2)

    # Pivot back wide
    wide = (
        long.pivot_table(
            index=["country_iso3", "month_year"],
            columns="feature",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def add_worldbank_features(
    combined: pd.DataFrame,
    gdf,
    indicators_path: str = "data/raw/combined_indicators.csv",
    metadata_path: str = None,
) -> pd.DataFrame:

 
    admin1_to_iso3 = gdf[["adm0_a3", "name_en"]].dropna().copy()
    admin1_to_iso3["matched_admin1_id"] = (
        admin1_to_iso3["adm0_a3"].astype(str).str.upper().str.strip()
        + " - "
        + admin1_to_iso3["name_en"].astype(str).str.strip()
    )
    admin1_to_iso3["adm0_a3"] = admin1_to_iso3["adm0_a3"].astype(str).str.upper().str.strip()
    admin1_to_iso3 = admin1_to_iso3[["matched_admin1_id", "adm0_a3"]].drop_duplicates("matched_admin1_id")

    idx = combined.index.to_frame(index=False)
    idx["matched_admin1_id"] = idx["matched_admin1_id"].astype(str)
    idx["month_year"] = pd.to_datetime(idx["month_year"], errors="coerce").dt.strftime("%Y-%m")

    idx = idx.merge(admin1_to_iso3, on="matched_admin1_id", how="left")



    # normalize month_year to YYYY-MM
    idx["month_year"] = pd.to_datetime(idx["month_year"], errors="coerce").dt.strftime("%Y-%m")

    # Build WB monthly indicators + income level code
    wb_monthly = _indicators_yearly_wide_to_monthly(indicators_path)

    # Drop aggregates like AFE/ARB/etc (keep real ISO3 only)
    wb_monthly = wb_monthly[wb_monthly["country_iso3"].astype(str).str.len() == 3].copy()
    wb_monthly["month_year"] = wb_monthly["month_year"].astype(str)

    if metadata_path and __import__("os").path.exists(metadata_path):
        meta = _load_country_metadata(metadata_path)
        wb_monthly = wb_monthly.merge(meta, on="country_iso3", how="left")

    # Merge WB onto idx using adm0_a3 (country ISO3) + month_year
    enriched = idx.merge(wb_monthly, left_on=["adm0_a3", "month_year"], right_on=["country_iso3", "month_year"],how="left")

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
