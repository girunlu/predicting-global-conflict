import argparse
from utils.preprocessing import (
    prepare_data_pipeline,
    prepare_enriched_pipeline,
    filter_admin1_data,
)
from models.simple_model import train_and_evaluate_model


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
    else:
        model_data = prepare_data_pipeline(clean_data=clean_data)

    region_data = filter_admin1_data(model_data, target_admin1)
    train_and_evaluate_model(region_data, target_event, region_name=target_admin1)


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

    args = parser.parse_args()

    # Dataset-only mode: just build and save, no modelling
    if not args.region and not args.event:
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
