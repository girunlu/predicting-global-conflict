"""
utils/evaluators.py
───────────────────
Model training, evaluation, and comparison utilities for the conflict
forecasting pipeline.

All functions operate on a flat DataFrame (not multi-indexed) with at least:
  matched_admin1_id  — admin-1 region string, e.g. "UKR - Donetsk"
  month_year         — date string, YYYY-MM-DD format
  importance_weight  — recency-based sample weight (from data_cleaning.py)
  <predictor cols>   — feature columns
  <target cols>      — Battles | Explosions/Remote violence | Violence against civilians

Design principle: every function here is stateless and side-effect free
(no file I/O, no plt.show). Notebooks orchestrate; this module computes.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ── Region selection ───────────────────────────────────────────────────────

def find_top_regions(df: pd.DataFrame, targets: list, n: int = 10) -> list:
    """
    Return the n admin-1 regions with the highest total target event counts
    summed across the full date range and across all target columns.

    We focus evaluation on high-activity regions because:
      (a) prediction errors matter most where conflict is most intense, and
      (b) low-activity regions are dominated by zeros, making MAE trivially low.

    Parameters
    ----------
    df      : flat DataFrame with matched_admin1_id and target columns.
    targets : list of target column names to sum.
    n       : number of top regions to return.
    """
    available = [t for t in targets if t in df.columns]
    return (
        df.groupby("matched_admin1_id")[available]
        .sum().sum(axis=1)
        .sort_values(ascending=False)
        .head(n).index.tolist()
    )


# ── Random Forest evaluator ────────────────────────────────────────────────

class ModelEvaluator:
    """
    Trains a RandomForestRegressor on a single region's time series and
    evaluates it on a fixed held-out window at the end of the series.

    The evaluator supports comparing multiple predictor sets (e.g. baseline
    vs risk-enhanced) under identical conditions: same train/test split,
    same hyperparameters, same importance weighting.

    Time-series integrity is maintained by always splitting chronologically
    (first N - holdout_months rows = train, last holdout_months = test).
    No cross-validation is used because conflict data has strong temporal
    autocorrelation that would leak if folds were shuffled.

    Parameters
    ----------
    n_estimators    : Number of trees. 100 is a good default; increase for
                      stability at the cost of runtime.
    holdout_months  : Test window length in months. 6 gives ~half a year of
                      out-of-sample evaluation per region.
    random_state    : Seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100, holdout_months: int = 6,
                 random_state: int = 42):
        self.n_estimators   = n_estimators
        self.holdout_months = holdout_months
        self.random_state   = random_state

    def evaluate(self, df: pd.DataFrame, region: str, target: str,
                 predictors: list, label: str = "") -> dict | None:
        """
        Train and evaluate for one (region, target, predictor-set).

        Missing predictor columns are silently skipped — this allows the same
        call signature when comparing feature sets of different sizes.

        Returns a result dict including raw predictions (for plotting) and
        per-feature importances, or None if the region has too few rows.
        """
        region_df = (
            df[df["matched_admin1_id"] == region]
            .sort_values("month_year")
            .copy()
        )
        if len(region_df) < self.holdout_months + 6:
            return None

        train = region_df.iloc[: -self.holdout_months]
        test  = region_df.iloc[-self.holdout_months :]

        avail   = [p for p in predictors if p in train.columns]
        X_train = train[avail].fillna(0)
        X_test  = test[avail].fillna(0)
        y_train = train[target].fillna(0)
        y_test  = test[target].fillna(0)
        weights = train["importance_weight"].fillna(1)

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        rf.fit(X_train, y_train, sample_weight=weights)
        y_pred = rf.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        nz  = y_test != 0
        mape = (np.mean(np.abs((y_test[nz] - y_pred[nz]) / y_test[nz])) * 100
                if nz.any() else np.nan)

        return {
            "region":              region,
            "label":               label,
            "mae":                 round(mae, 2),
            "mape":                round(mape, 2) if not np.isnan(mape) else np.nan,
            "n_predictors":        len(avail),
            "train_rows":          len(train),
            "test_rows":           len(test),
            "y_test":              y_test.values,
            "y_pred":              y_pred,
            "feature_importances": pd.Series(
                rf.feature_importances_, index=avail
            ).sort_values(ascending=False),
        }


# ── Comparison helpers ─────────────────────────────────────────────────────

def run_comparison(df: pd.DataFrame, regions: list, target: str,
                   baseline_preds: list, enhanced_preds: list,
                   evaluator: ModelEvaluator) -> pd.DataFrame:
    """
    Run baseline vs enhanced evaluation across all regions for one target.

    Both models are trained on the same df — the enhanced model simply uses
    additional predictor columns that exist in df. This ensures any MAE
    difference is attributable to the new features, not data differences.

    Returns a tidy DataFrame with columns: region, label, mae, mape, y_test,
    y_pred, feature_importances (one row per region × label).
    """
    records = []
    for region in regions:
        for label, preds in [("Baseline", baseline_preds), ("Enhanced", enhanced_preds)]:
            result = evaluator.evaluate(df, region, target, preds, label)
            if result is not None:
                records.append(result)
    return pd.DataFrame(records)


def build_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a run_comparison() result into a side-by-side MAE/MAPE table.

    Rows are sorted by MAE Diff descending so regions where the Enhanced
    model improves most appear at the top. A positive MAE Diff means
    Enhanced is better (baseline MAE minus enhanced MAE).
    """
    mae  = results_df.pivot(index="region", columns="label", values="mae")
    mape = results_df.pivot(index="region", columns="label", values="mape")
    table = pd.DataFrame({
        "MAE Baseline":  mae["Baseline"],
        "MAE Enhanced":  mae["Enhanced"],
        "MAE Diff":      mae["Baseline"] - mae["Enhanced"],
        "MAE % Change":  ((mae["Enhanced"] - mae["Baseline"]) / mae["Baseline"] * 100).round(1),
        "MAPE Baseline": mape["Baseline"],
        "MAPE Enhanced": mape["Enhanced"],
    })
    return table.sort_values("MAE Diff", ascending=False)


# ── Generic multi-model evaluator ─────────────────────────────────────────

def evaluate_model(df: pd.DataFrame, region: str, target: str,
                   predictors: list, model_factory, holdout: int = 6) -> dict | None:
    """
    Generic single-model evaluator for the multi-model ablation benchmark.

    Unlike ModelEvaluator (which is RF-specific and stores predictions for
    plotting), this function accepts any sklearn-compatible model factory —
    a zero-argument callable that returns a fresh, unfitted model instance.
    Sample weights are passed to fit() when the model supports them (LGBM,
    RF, GBR do; XGBoost requires a keyword argument that is handled via try/except).

    Returns {'mae': float, 'mape': float} or None if insufficient data.

    Usage example
    -------------
    import lightgbm as lgb
    factory = lambda: lgb.LGBMRegressor(objective='tweedie',
                                        tweedie_variance_power=1.5,
                                        n_estimators=200, verbose=-1)
    result = evaluate_model(df, "UKR - Donetsk", "Battles", preds, factory)
    """
    region_df = (
        df[df["matched_admin1_id"] == region]
        .sort_values("month_year")
        .copy()
    )
    if len(region_df) < holdout + 6:
        return None

    avail   = [p for p in predictors if p in region_df.columns]
    train   = region_df.iloc[:-holdout]
    test    = region_df.iloc[-holdout:]
    X_train = train[avail].fillna(0)
    X_test  = test[avail].fillna(0)
    y_train = train[target].fillna(0)
    y_test  = test[target].fillna(0)
    weights = train.get("importance_weight", pd.Series(1.0, index=train.index)).fillna(1)

    model = model_factory()
    try:
        model.fit(X_train, y_train, sample_weight=weights)
    except TypeError:
        model.fit(X_train, y_train)

    y_pred = np.maximum(0, model.predict(X_test))
    mae    = mean_absolute_error(y_test, y_pred)
    nz     = y_test != 0
    mape   = (np.mean(np.abs((y_test[nz] - y_pred[nz]) / y_test[nz])) * 100
              if nz.any() else np.nan)

    return {
        "mae":  round(mae, 2),
        "mape": round(mape, 2) if not np.isnan(mape) else np.nan,
    }
