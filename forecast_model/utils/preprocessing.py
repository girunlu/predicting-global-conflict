import os
import pandas as pd
import geopandas as gpd
from utils import data_cleaning, map_admin_regions
from config import settings

def prepare_data_pipeline(clean_data: bool = False):
    """
    Builds or loads the model-ready DataFrame.
    If clean_data=True, load from saved file. Otherwise, run the full pipeline.
    """
    output_path = "data/processed/model_data.csv"

    if not clean_data and os.path.exists(output_path):
        print("Loading cleaned data from disk...")
        df = pd.read_csv(output_path, index_col=[0, 1])
        return df

    print("Running full data preprocessing pipeline...")
    df = pd.read_csv("data/raw/1997-01-01-2025-07-03.csv")
    df = df[df['year'] >= 2018].copy()
    df['date'] = pd.to_datetime(df['event_date'], format='%d %B %Y')
    df['month_year'] = df['date'].dt.to_period('M').astype(str)

    gdf = gpd.read_file("data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp")
    df_neighbours = map_admin_regions.add_admin1_neighbors(df, gdf)

    neighbour_data = data_cleaning.summarise_neighbour_events(df_neighbours)
    event_data = data_cleaning.get_monthly_events(df_neighbours)
    subevent_data = data_cleaning.get_monthly_subevents(
        df_neighbours, ['Excessive force against protesters', 'Agreement']
    )

    combined = pd.concat([event_data, subevent_data], axis=1).join(neighbour_data, how='left')
    combined = data_cleaning.add_lagged_columns(combined)
    combined = data_cleaning.add_time_trend_features(combined)
    combined = data_cleaning.add_importance_weights(combined)

    model_data = combined[settings.predictors + settings.targets]

    # Save to disk for next time
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_data.to_csv(output_path)

    return model_data
    
def filter_admin1_data(df, admin1_region):
    return df.loc[admin1_region]