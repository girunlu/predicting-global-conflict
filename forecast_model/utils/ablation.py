"""
utils/ablation.py
-----------------
Multi-model ablation benchmark — reusable library version of model_comparison_4.ipynb.

Public API
----------
MODELS            : dict of model-name -> factory callable
build_feature_sets: build cumulative feature-set dict from an enriched DataFrame
run_ablation      : run the full evaluation loop, return results DataFrame
print_results     : print MAE tables to stdout
"""

import os
from itertools import product as iproduct

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from config import settings
from utils.evaluators import find_top_regions, evaluate_model
from utils.visualization import plot_ablation_heatmap


# ── Model definitions (mirrors model_comparison_4.ipynb) ──────────────────────
MODELS = {
    "RF": lambda: RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "LGBM-Poisson": lambda: lgb.LGBMRegressor(
        objective="poisson", n_estimators=200, learning_rate=0.05,
        num_leaves=31, random_state=42, verbose=-1, n_jobs=-1,
    ),
    "LGBM-Tweedie": lambda: lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=200, learning_rate=0.05,
        num_leaves=31, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1,
    ),
    "XGBoost": lambda: xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=200, learning_rate=0.05,
        max_depth=6, random_state=42, verbosity=0, n_jobs=-1,
    ),
    "GBR": lambda: GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42,
    ),
}


def build_feature_sets(df: pd.DataFrame) -> dict:
    """Build cumulative feature sets from the enriched dataframe."""
    risk_features = [
        c for c in df.columns
        if c.startswith("risk_") and c.endswith("(t-1)")
    ]
    if not risk_features:
        print("WARNING: No risk_ (t-1) columns found in dataframe. "
              "Ensure the enriched pipeline was run with master_raw.csv. "
              "'+Risk', '+Macro', '+Holidays', '+Engineered' sets will equal 'Baseline'.")
    engineered_features = [
        f for f in [
            "Battles (t-2)", "Explosions/Remote violence (t-2)",
            "Violence against civilians (t-2)",
            "organized_violence (t-1)", "is_active (t-1)", "battles_x_remote (t-1)",
            "Battles_3mo_avg (t-1)", "Remote_3mo_avg (t-1)", "VaC_3mo_avg (t-1)",
        ]
        if f in df.columns
    ]

    return {
        "Baseline":    settings.predictors,
        "+Risk":       settings.predictors + risk_features,
        "+Macro":      settings.predictors + risk_features + settings.macro_features,
        "+Holidays":   settings.predictors + risk_features + settings.macro_features + settings.holiday_features,
        "+Engineered": settings.predictors + risk_features + settings.macro_features + settings.holiday_features + engineered_features,
    }


def run_ablation(df: pd.DataFrame, top_n: int = 10, holdout: int = 6) -> pd.DataFrame:
    """
    Run the full ablation loop and return a flat DataFrame of results.

    Columns: model, feature_set, target, region, mae, mape
    """
    feature_sets = build_feature_sets(df)
    top_regions  = find_top_regions(df, settings.targets, n=top_n)

    combos = list(iproduct(MODELS.keys(), feature_sets.keys(), settings.targets, top_regions))
    print(f"Running {len(combos)} evaluations "
          f"({len(MODELS)} models x {len(feature_sets)} feature sets x "
          f"{len(settings.targets)} targets x {len(top_regions)} regions)...")

    records = []
    for i, (model_name, fs_name, target, region) in enumerate(combos):
        result = evaluate_model(
            df, region, target,
            feature_sets[fs_name],
            MODELS[model_name],
            holdout=holdout,
        )
        if result is not None:
            records.append({
                "model":       model_name,
                "feature_set": fs_name,
                "target":      target,
                "region":      region,
                **result,
            })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} done...")

    print(f"Done. {len(records)} results collected.")
    return pd.DataFrame(records)


def print_results(ablation_df: pd.DataFrame, feature_sets: dict) -> None:
    """Print MAE tables matching model_comparison_4.ipynb output."""
    fs_names = list(feature_sets.keys())

    print("\n" + "=" * 80)
    print("MEAN MAE -- ALL REGIONS + TARGETS COMBINED")
    print("=" * 80)
    pivot = (
        ablation_df.groupby(["model", "feature_set"])["mae"]
        .mean()
        .unstack("feature_set")[fs_names]
        .round(2)
    )
    print(pivot.to_string())

    print("\n" + "=" * 80)
    print("PER-TARGET BREAKDOWN")
    print("=" * 80)
    for target in settings.targets:
        sub = ablation_df[ablation_df["target"] == target]
        pivot_t = (
            sub.groupby(["model", "feature_set"])["mae"]
            .mean()
            .unstack("feature_set")
            .reindex(columns=fs_names)
            .round(2)
        )
        print(f"\n{'-'*50}")
        print(f"Target: {target}")
        print(f"{'-'*50}")
        print(pivot_t.to_string())

        if "RF" in pivot_t.index and "Baseline" in pivot_t.columns:
            rf_base  = pivot_t.loc["RF", "Baseline"]
            best_val = pivot_t.values[~pd.isna(pivot_t.values)].min()
            print(f"\n% change vs RF Baseline (MAE={rf_base:.2f}):")
            for model in pivot_t.index:
                for fs in fs_names:
                    val = pivot_t.loc[model, fs]
                    if pd.isna(val):
                        continue
                    pct = (val - rf_base) / rf_base * 100
                    tag = " [BEST]" if val == best_val else ""
                    print(f"  {model:15s}  {fs:15s}: {val:.2f}  ({pct:+.1f}%){tag}")

    print("\n" + "=" * 80)
    print("OVERALL CONCLUSIONS")
    print("=" * 80)
    overall = ablation_df.groupby(["model", "feature_set"])["mae"].mean()
    best    = overall.idxmin()
    rf_base = overall.get(("RF", "Baseline"), float("nan"))
    print(f"\nBest combination overall:  model={best[0]}, feature_set={best[1]}")
    print(f"  MAE = {overall[best]:.2f}  (RF Baseline: {rf_base:.2f})")
    print(f"  Improvement vs RF Baseline: {(overall[best] - rf_base) / rf_base * 100:+.1f}%")

    print("\nBest model per feature set:")
    for fs in fs_names:
        sub_fs = ablation_df[ablation_df["feature_set"] == fs]
        if sub_fs.empty:
            continue
        best_m = sub_fs.groupby("model")["mae"].mean().idxmin()
        best_v = sub_fs.groupby("model")["mae"].mean().min()
        print(f"  {fs:15s}: {best_m} (MAE={best_v:.2f})")

    print("\nMarginal MAE change per feature group (RF only):")
    rf_only = ablation_df[ablation_df["model"] == "RF"]
    for i in range(1, len(fs_names)):
        prev_mae = rf_only[rf_only["feature_set"] == fs_names[i - 1]]["mae"].mean()
        curr_mae = rf_only[rf_only["feature_set"] == fs_names[i]]["mae"].mean()
        delta    = curr_mae - prev_mae
        print(f"  {fs_names[i-1]:15s} -> {fs_names[i]:15s}: {delta:+.2f} MAE  ({delta/prev_mae*100:+.1f}%)")


def save_outputs(ablation_df: pd.DataFrame, feature_sets: dict) -> None:
    """Save heatmap PNG and raw results CSV to outputs/."""
    os.makedirs("outputs/figures", exist_ok=True)
    try:
        plot_ablation_heatmap(
            ablation_df, feature_sets, settings.targets,
            save_path="outputs/figures/ablation_heatmap.png",
        )
        print("Heatmap saved: outputs/figures/ablation_heatmap.png")
    except Exception as e:
        print(f"Could not save heatmap: {e}")

    out_csv = "outputs/ablation_results.csv"
    ablation_df.to_csv(out_csv, index=False)
    print(f"Raw results saved: {out_csv}")
