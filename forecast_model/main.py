import argparse
from utils.preprocessing import (
    prepare_data_pipeline,
    prepare_enriched_pipeline,
    filter_admin1_data,
)
from models.simple_model import train_and_evaluate_model
from config import settings
from utils.ablation import build_feature_sets, run_ablation, print_results, save_outputs


def forecast_admin1_events(
    target_admin1: str,
    target_event: str,
    clean_data: bool = False,
    enrich: bool = False,
):
    """
    Full modeling pipeline for a given ADMIN1 region and target event type.

    enrich=True  → trains on the fully enriched dataset (risk + macro + holidays + engineered).
    enrich=False → trains on the baseline 30-feature dataset.
    """
    if enrich:
        model_data = prepare_enriched_pipeline(clean_data=clean_data)
        if "matched_admin1_id" in model_data.columns:
            model_data = model_data.set_index(["matched_admin1_id", "month_year"])
        # Full enriched predictor list — reuse ablation's feature-set definition
        predictors = build_feature_sets(model_data)["+Engineered"]
    else:
        model_data = prepare_data_pipeline(clean_data=clean_data)
        predictors = settings.predictors

    region_data = filter_admin1_data(model_data, target_admin1)
    train_and_evaluate_model(region_data, target_event, region_name=target_admin1,
                             predictors=predictors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast conflict events at the Admin1 level or build datasets."
    )
    parser.add_argument("--region",     type=str, help="Target ADMIN1 region name")
    parser.add_argument("--event",      type=str, help="Target event type (e.g. Battles)")
    parser.add_argument("--clean-data", action="store_true",
                        help="Rebuild datasets from raw data instead of loading from disk")
    parser.add_argument("--enrich",     action="store_true",
                        help="Build / use the fully enriched dataset (risk + macro + holidays + engineered)")
    parser.add_argument("--ablation",   action="store_true",
                        help="Run multi-model ablation benchmark (requires --enrich dataset)")
    parser.add_argument("--top-n",      type=int, default=10,
                        help="Top N regions for ablation evaluation (default: 10)")
    parser.add_argument("--holdout",    type=int, default=6,
                        help="Holdout months for ablation evaluation (default: 6)")

    args = parser.parse_args()

    # Ablation mode
    if args.ablation:
        df = prepare_enriched_pipeline(clean_data=args.clean_data)
        ablation_df  = run_ablation(df, top_n=args.top_n, holdout=args.holdout)
        feature_sets = build_feature_sets(df)
        print_results(ablation_df, feature_sets)
        save_outputs(ablation_df, feature_sets)

    # Dataset-only mode: just build and save, no modelling
    elif not args.region and not args.event:
        if args.enrich:
            prepare_enriched_pipeline(clean_data=args.clean_data)
        else:
            prepare_data_pipeline(clean_data=args.clean_data)

    else:
        if not args.region or not args.event:
            parser.error("--region and --event are both required for forecasting")

        forecast_admin1_events(
            target_admin1=args.region,
            target_event=args.event,
            clean_data=args.clean_data,
            enrich=args.enrich,
        )
