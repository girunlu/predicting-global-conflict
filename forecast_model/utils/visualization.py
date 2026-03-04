"""
utils/visualization.py
──────────────────────
Plotting utilities for the conflict forecasting pipeline.

Every function displays the figure via plt.show() then closes it with
plt.close() so that Jupyter does not render it a second time from the
implicit return value. Pass save_path to write the figure to disk before
it is shown.

Conventions used throughout:
  - "Baseline"  : model trained on original 31 predictors
  - "Enhanced"  : model trained on baseline + additional features
  - "Actual"    : held-out ground truth (black solid line)
  - Lower MAE is always better
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display as _ipy_display


# ── MAE comparison bar chart ───────────────────────────────────────────────

def plot_mae_comparison(results_df: pd.DataFrame, target: str,
                        save_path: str = None) -> None:
    """
    Horizontal bar chart comparing Baseline vs Enhanced MAE across regions.

    Regions are sorted by Baseline MAE ascending so the highest-conflict
    zones (where the model is under most pressure) appear at the top.
    A shorter Enhanced bar than Baseline bar means the new features helped.

    Parameters
    ----------
    results_df : output of run_comparison() — must have 'region', 'label', 'mae'.
    target     : target name used only for the chart title.
    save_path  : if provided, save figure to this path.
    """
    mae = results_df.pivot(index="region", columns="label", values="mae")
    mae = mae.sort_values("Baseline", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x, w = np.arange(len(mae)), 0.35
    ax.barh(x - w / 2, mae["Baseline"], w, label="Baseline",
            color="#4C72B0", alpha=0.9)
    ax.barh(x + w / 2, mae["Enhanced"], w, label="Enhanced (+risk indicators)",
            color="#DD8452", alpha=0.9)

    ax.set_yticks(x)
    ax.set_yticklabels(mae.index, fontsize=9)
    ax.set_xlabel("MAE (lower is better)")
    ax.set_title(f"Baseline vs Enhanced — MAE for '{target}' (top 10 regions)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    _ipy_display(fig)
    plt.close(fig)


# ── Forecast line plots ────────────────────────────────────────────────────

def plot_forecasts(results_df: pd.DataFrame, target: str,
                   save_path: str = None) -> None:
    """
    For each region, plot Actual, Baseline prediction, and Enhanced prediction
    in the same subplot so their accuracy can be directly compared.

    Layout: 2 columns, ceil(n_regions / 2) rows.

    Reading the chart:
      - Black solid line  = ground truth (what actually happened)
      - Blue dashed line  = Baseline model prediction
      - Orange dashed line = Enhanced model prediction (with risk indicators)
    MAE for each model is shown in the legend label.

    Parameters
    ----------
    results_df : output of run_comparison(), containing both 'Baseline' and
                 'Enhanced' labelled rows with y_test / y_pred arrays.
    target     : target name for the figure title.
    save_path  : optional path to save the figure.
    """
    regions = results_df["region"].unique()
    n       = len(regions)
    ncols   = 2
    nrows   = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, region in enumerate(regions):
        ax     = axes_flat[i]
        r_base = results_df[(results_df["region"] == region) & (results_df["label"] == "Baseline")]
        r_enh  = results_df[(results_df["region"] == region) & (results_df["label"] == "Enhanced")]

        # Both should have the same y_test (same holdout window)
        ref    = r_base.iloc[0] if not r_base.empty else r_enh.iloc[0]
        months = np.arange(len(ref["y_test"]))

        ax.plot(months, ref["y_test"], "o-", color="black", lw=2, ms=6,
                label="Actual")
        if not r_base.empty:
            r = r_base.iloc[0]
            ax.plot(months, r["y_pred"], "x--", color="#4C72B0", ms=6,
                    label=f"Baseline  (MAE={r['mae']})")
        if not r_enh.empty:
            r = r_enh.iloc[0]
            ax.plot(months, r["y_pred"], "s--", color="#DD8452", ms=6,
                    label=f"Enhanced  (MAE={r['mae']})")

        ax.set_title(region, fontsize=10)
        ax.set_xlabel("Test month (0 = earliest, 5 = most recent)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Actual vs Baseline vs Enhanced Predictions — '{target}'\n"
        f"Holdout: last 6 months per region",
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    _ipy_display(fig)
    plt.close(fig)


# ── Feature importance bar chart ───────────────────────────────────────────

def plot_risk_feature_importance(results_df: pd.DataFrame,
                                 risk_prefix: str = "risk_",
                                 save_path: str = None) -> None:
    """
    Average feature importance of risk-indicator columns across all regions
    and targets, drawn from Enhanced model results.

    Risk indicator importances tell us which early-warning signals the model
    actually uses. A high importance for 'risk_ethnic_tension' means that
    regions with recent ethnic tension signals tend to have systematically
    different violence levels, and the model exploits that pattern.

    Parameters
    ----------
    results_df  : combined results from multiple run_comparison() calls.
    risk_prefix : prefix identifying risk columns in feature_importances.
    save_path   : optional save path.
    """
    enhanced = results_df[results_df["label"] == "Enhanced"]

    fi_frames = []
    for _, row in enhanced.iterrows():
        fi = row["feature_importances"]
        risk_fi = fi[[c for c in fi.index if c.startswith(risk_prefix)]]
        if not risk_fi.empty:
            fi_frames.append(risk_fi)

    if not fi_frames:
        print("No risk feature importance data found.")
        return None

    avg_fi = pd.concat(fi_frames, axis=1).mean(axis=1).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    avg_fi.plot(kind="barh", ax=ax, color="#55A868", alpha=0.85)
    ax.set_xlabel("Mean Feature Importance (averaged across all regions & targets)")
    ax.set_title(
        "Risk Indicator Importance — Enhanced RF Model\n"
        "Higher = the model relies more heavily on this early-warning signal"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    _ipy_display(fig)
    plt.close(fig)


# ── Multi-model ablation heatmap ───────────────────────────────────────────

def plot_ablation_heatmap(ablation_df: pd.DataFrame, feature_sets: dict,
                          targets: list, save_path: str = None) -> None:
    """
    Heatmap of mean MAE for each model × feature-set combination, one panel
    per target variable.

    How to read this chart:
      - Each row is a model architecture (RF, LGBM-Tweedie, XGBoost, …).
      - Each column is a feature set (Baseline → +Risk → +Macro → …).
      - Green cells = low MAE (good). Red cells = high MAE (bad).
      - Reading left-to-right across a row shows how each additional feature
        group affects that model's accuracy.
      - Reading top-to-bottom in a column shows which model architecture works
        best for a given feature set.

    Parameters
    ----------
    ablation_df  : output of the ablation loop — columns: model, feature_set,
                   target, region, mae.
    feature_sets : ordered dict of {name: predictor_list} defining column order.
    targets      : list of target names for panel titles.
    save_path    : optional save path.
    """
    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        sub = ablation_df[ablation_df["target"] == target]
        mat = (
            sub.groupby(["model", "feature_set"])["mae"]
            .mean()
            .unstack("feature_set")
            .reindex(columns=list(feature_sets.keys()))
        )
        im   = ax.imshow(mat.values, cmap="RdYlGn_r", aspect="auto")
        vmax = np.nanmax(mat.values)

        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=9)

        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                val = mat.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if val > vmax * 0.7 else "black")

        ax.set_title(f"MAE: {target}", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        "Model x Feature Set Ablation — Mean MAE (lower = better)\n"
        "Evaluated on top-10 most active regions, 6-month hold-out",
        fontsize=12,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    _ipy_display(fig)
    plt.close(fig)
