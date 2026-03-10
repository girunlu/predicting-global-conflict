import pandas as pd

# change 1

def load_holidays_monthly(holidays_path: str = "data/raw/holidays_raw.csv") -> pd.DataFrame:
    hol = pd.read_csv(holidays_path)

    required = {"Country", "Date", "Holiday"}
    missing = required - set(hol.columns)
    if missing:
        raise ValueError(
            f"[holidays] Missing columns {missing} in {holidays_path}. Found: {list(hol.columns)}"
        )

    hol["Date"] = pd.to_datetime(hol["Date"], errors="coerce")
    hol = hol.dropna(subset=["Country", "Date"]).copy()

    hol["month_year"] = hol["Date"].dt.strftime("%Y-%m-01")

    hol_month = (
        hol.groupby(["Country", "month_year"], as_index=False)
           .agg(holiday_count_month=("Holiday", "size"))
    )
    hol_month["is_holiday_month"] = (hol_month["holiday_count_month"] > 0).astype(int)

    return hol_month

def add_iso3_to_holidays(hol_month: pd.DataFrame, country_map: pd.DataFrame) -> pd.DataFrame:
    
    cm = country_map[["admin", "adm0_a3"]].dropna().drop_duplicates().copy()
    cm["admin_key"] = cm["admin"].astype(str).str.strip().str.lower()

    out = hol_month.copy()
    out["Country_key"] = out["Country"].astype(str).str.strip().str.lower()

    out = out.merge(cm[["admin_key", "adm0_a3"]], left_on="Country_key", right_on="admin_key", how="left")
    out = out.rename(columns={"adm0_a3": "country_iso3"})
    out = out.drop(columns=["Country_key", "admin_key"])


    alias_to_iso3 = {
        "bahamas": "BHS",
        "bailiwick of jersey": "JEY",
        "curacao": "CUW",
        "democratic republic of congo": "COD",
        "republic of congo": "COG",
        "french guiana": "GUF",
        "guadeloupe": "GLP",
        "guinea-bissau": "GNB",
        "hong kong": "HKG",
        "macau": "MAC",
        "martinique": "MTQ",
        "mayotte": "MYT",
        "micronesia": "FSM",
        "north macedonia": "MKD",
        "reunion": "REU",
        "saint helena, ascension and tristan da cunha": "SHN",
        "saint-barthelemy": "BLM",
        "saint-martin": "MAF",
        "serbia": "SRB",
        "south sudan": "SSD",
        "tanzania": "TZA",
        "united states": "USA",
        "vatican city": "VAT",
        "virgin islands, u.s.": "VIR",
        "eswatini": "SWZ",
    }

    missing_mask = out["country_iso3"].isna()
    out.loc[missing_mask, "country_iso3"] = (
        out.loc[missing_mask, "Country"]
          .astype(str).str.strip().str.lower()
          .map(alias_to_iso3)
    )

    return out

def add_holiday_features(combined, gdf, holidays_path: str = "data/raw/holidays_raw.csv"):
    country_map = gdf[["admin", "adm0_a3"]].dropna().drop_duplicates()

    hol_month = load_holidays_monthly(holidays_path)
    hol_month = add_iso3_to_holidays(hol_month, country_map)
    hol_month["month_year"] = hol_month["month_year"].astype(str)

    hol_month = hol_month[["country_iso3", "month_year", "holiday_count_month", "is_holiday_month"]].copy()

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
    idx["month_year"] = pd.to_datetime(idx["month_year"], errors="coerce").dt.strftime("%Y-%m-01")

    idx = idx.merge(admin1_to_iso3, on="matched_admin1_id", how="left")


    idx = idx.merge(hol_month, left_on=["adm0_a3", "month_year"], right_on=["country_iso3", "month_year"],how="left")
    idx["holiday_count_month"] = idx["holiday_count_month"].fillna(0).astype(int)
    idx["is_holiday_month"] = idx["is_holiday_month"].fillna(0).astype(int)


    feat = idx[["matched_admin1_id", "month_year", "holiday_count_month", "is_holiday_month"]].copy()
    feat["matched_admin1_id"] = feat["matched_admin1_id"].astype(str)
    feat["month_year"] = feat["month_year"].astype(str)

    tmp = combined.reset_index().copy()
    tmp["matched_admin1_id"] = tmp["matched_admin1_id"].astype(str)
    tmp["month_year"] = pd.to_datetime(tmp["month_year"], errors="coerce").dt.strftime("%Y-%m-01")

    out = tmp.merge(feat, on=["matched_admin1_id", "month_year"], how="left")
    out = out.set_index(["matched_admin1_id", "month_year"])

    return out



