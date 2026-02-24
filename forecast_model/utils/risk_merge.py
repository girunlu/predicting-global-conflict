import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pycountry
from config import settings


class RiskIndicatorMerger:
    """
    Merges country-level risk indicators from master_raw.csv into the
    admin-1-level model_data.csv.

    Uses source_file as the country key, counts mentions per metric,
    and lags by 1 month to match the pipeline's (t-1) design.

    Usage:
        merger = RiskIndicatorMerger()
        df = merger.merge("data/processed/model_data.csv", "data/raw/master_raw.csv")
        enhanced_preds = merger.get_enhanced_predictors()
    """

    _MANUAL_ISO3 = {
        "Ivory Coast": "CIV", "Cote d'Ivoire": "CIV", "Côte d'Ivoire": "CIV",
        "eSwatini": "SWZ", "Eswatini": "SWZ", "Swaziland": "SWZ",
        "Kosovo": "XKX", "Palestine": "PSE", "State of Palestine": "PSE",
        "Taiwan": "TWN", "North Korea": "PRK", "South Korea": "KOR",
        "Russia": "RUS", "Iran": "IRN", "Syria": "SYR", "Venezuela": "VEN",
        "Bolivia": "BOL", "Tanzania": "TZA", "Vietnam": "VNM",
        "Laos": "LAO", "Moldova": "MDA", "North Macedonia": "MKD",
        "Turkiye": "TUR", "Türkiye": "TUR", "Turkey": "TUR",
        "Czech Republic": "CZE", "Democratic Republic of Congo": "COD",
        "Republic of Congo": "COG", "Republic of the Congo": "COG",
        "DR Congo": "COD", "DRC": "COD", "Congo": "COG",
        "Cape Verde": "CPV", "Cabo Verde": "CPV",
        "Micronesia": "FSM", "Brunei": "BRN", "East Timor": "TLS",
        "Timor-Leste": "TLS", "Myanmar": "MMR", "Burma": "MMR",
        "Curacao": "CUW", "Curaçao": "CUW",
        "Reunion": "REU", "Réunion": "REU",
        "Macau": "MAC", "Hong Kong": "HKG",
    }

    def __init__(self, lag: int = 1):
        self.lag = lag
        self.risk_cols_ = []
        self.risk_cols_lagged_ = []
        self._iso3_cache = {}

    def merge(self, model_data_path: str, master_raw_path: str) -> pd.DataFrame:
        """Full pipeline: load, transform, merge, lag."""
        model_df = self._load_model_data(model_data_path)
        risk_df = self._load_and_transform_risk(master_raw_path)
        merged = self._join(model_df, risk_df)
        merged = self._add_lag(merged)
        return merged

    def get_enhanced_predictors(self) -> list:
        """Original predictors + lagged risk columns."""
        return settings.predictors + self.risk_cols_lagged_

    def _load_model_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["country_code"] = df["matched_admin1_id"].str.split(" - ").str[0]
        return df

    def _load_and_transform_risk(self, path: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        raw["iso3"] = raw["source_file"].apply(self._to_iso3)
        raw["month_year"] = raw["date"].apply(self._normalize_date)
        raw = raw.dropna(subset=["iso3", "month_year"])

        counts = (
            raw.groupby(["iso3", "month_year", "metric"])
            .size()
            .reset_index(name="count")
        )

        pivot = counts.pivot_table(
            index=["iso3", "month_year"],
            columns="metric",
            values="count",
            fill_value=0,
        ).reset_index()
        pivot.columns.name = None

        metric_cols = [c for c in pivot.columns if c not in ("iso3", "month_year")]
        rename_map = {c: f"risk_{c.replace(' ', '_')}" for c in metric_cols}
        pivot = pivot.rename(columns=rename_map)
        self.risk_cols_ = list(rename_map.values())

        return pivot

    def _join(self, model_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
        merged = model_df.merge(
            risk_df,
            left_on=["country_code", "month_year"],
            right_on=["iso3", "month_year"],
            how="left",
        ).drop(columns=["iso3"], errors="ignore")

        merged[self.risk_cols_] = merged[self.risk_cols_].fillna(0).astype(int)
        return merged

    def _add_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["matched_admin1_id", "month_year"])

        self.risk_cols_lagged_ = []
        for col in self.risk_cols_:
            lag_name = f"{col} (t-{self.lag})"
            df[lag_name] = (
                df.groupby("matched_admin1_id")[col]
                .shift(self.lag)
                .fillna(0)
                .astype(int)
            )
            self.risk_cols_lagged_.append(lag_name)

        return df

    def _to_iso3(self, name) -> str | None:
        if pd.isna(name):
            return None
        name = name.strip()
        if name in self._iso3_cache:
            return self._iso3_cache[name]

        code = self._MANUAL_ISO3.get(name)
        if code is None:
            try:
                code = pycountry.countries.lookup(name).alpha_3
            except LookupError:
                try:
                    results = pycountry.countries.search_fuzzy(name)
                    code = results[0].alpha_3 if results else None
                except Exception:
                    code = None

        self._iso3_cache[name] = code
        return code

    @staticmethod
    def _normalize_date(date_str) -> str | None:
        if pd.isna(date_str):
            return None
        parts = str(date_str).strip().split("-")
        if len(parts) == 2:
            mm, yyyy = parts
            return f"{yyyy}-{mm.zfill(2)}-01"
        return None
