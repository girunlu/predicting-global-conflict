import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def get_monthly_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame showing monthly counts of specified event types 
    for each matched_admin1_id-region combination, including months with zero events.

    Parameters:
        df (pd.DataFrame): The input DataFrame with at least 'matched_admin1_id', 'month_year', 'event_type' columns.
        event_cols (list): List of event types (columns) to include in the result.

    Returns:
        pd.DataFrame: A pivoted DataFrame with MultiIndex (matched_admin1_id, month_year) 
                      and columns for each event_type in event_cols.
    """

    # Get all unique values
    admin1_vals = df['matched_admin1_id'].unique()
    month_year_vals = df['month_year'].unique()
    event_type_vals = df['event_type'].unique()

    # Create a complete multi-index of all combinations
    full_index = pd.MultiIndex.from_product(
        [admin1_vals, month_year_vals, event_type_vals],
        names=['matched_admin1_id', 'month_year', 'event_type']
    )

    # Group data
    grouped = df.groupby(['matched_admin1_id', 'month_year', 'event_type']).size()

    # Reindex with the full index and fill missing values with 0
    grouped_full = grouped.reindex(full_index, fill_value=0)

    #  Unstack to get event_type columns (filtered to event_cols if needed)
    result = grouped_full.unstack('event_type', fill_value=0)

    return result


def get_monthly_subevents(df: pd.DataFrame, subevent_cols: list) -> pd.DataFrame:
    """
    Returns a DataFrame showing monthly counts of specified sub-event types 
    for each matched_admin1_id-region combination, including months with zero events.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns: 'matched_admin1_id', 'month_year', 'sub_event_type'.
        subevent_cols (list): List of sub-event types (columns) to include in the result.

    Returns:
        pd.DataFrame: A pivoted DataFrame with MultiIndex (matched_admin1_id, month_year)
                      and columns for each sub_event_type in subevent_cols.
    """

    # Get all unique values
    admin1_vals = df['matched_admin1_id'].unique()
    month_year_vals = df['month_year'].unique()
    sub_event_type_vals = df['sub_event_type'].unique()

    # Create a complete multi-index of all combinations
    full_index = pd.MultiIndex.from_product(
        [admin1_vals, month_year_vals, sub_event_type_vals],
        names=['matched_admin1_id', 'month_year', 'sub_event_type']
    )

    # Group data
    grouped = df.groupby(['matched_admin1_id', 'month_year', 'sub_event_type']).size()

    # Reindex with the full index, fill missing with 0
    grouped_full = grouped.reindex(full_index, fill_value=0)

    # Unstack to get sub_event_type columns, filtered to subevent_cols if provided
    result = grouped_full.unstack('sub_event_type', fill_value=0)

    if subevent_cols:
        result = result[subevent_cols]

    return result
    

def add_lagged_columns(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Adds lagged (t - lag) columns for each column in the input DataFrame,
    grouped by the first index level (matched_admin1_id), and suffixes them with (t-1), (t-2), etc.

    Parameters:
        df (pd.DataFrame): MultiIndex DataFrame with (matched_admin1_id, month_year) as index.
        lag (int): The lag step. Default is 1 (previous month).

    Returns:
        pd.DataFrame: DataFrame with new lagged columns added.
    """
    # Prepare lagged columns
    lagged_df = df.groupby(level='matched_admin1_id').shift(lag)

    # Rename columns to include (t-1)
    lagged_df.columns = [f"{col} (t-{lag})" for col in lagged_df.columns]
    lagged_df = lagged_df.fillna(0)

    # Combine original and lagged
    combined = pd.concat([df, lagged_df], axis=1)

    return combined


def summarise_neighbour_events(df_neighbours):
    """
    For each possible (matched_admin1_id, month_year), compute the sum of event_type counts 
    across all its neighbours listed in 'admin1_neighbors', even if no events occurred.

    Returns:
        DataFrame: MultiIndexed by ['matched_admin1_id', 'month_year'] with one column 
                   per event_type, suffixed with '_neighbours'.
    """

    # Step 1: Aggregate counts by (admin1_id, month_year, event_type)
    grouped = (
        df_neighbours
        .groupby(['matched_admin1_id', 'month_year', 'event_type'])
        .size()
        .reset_index(name='count')
    )

    # Step 2: Convert to lookup dict
    event_dict = defaultdict(lambda: defaultdict(int))
    for _, row in tqdm(grouped.iterrows(), "Convert to dictionary", total=len(grouped)):
        key = (row['matched_admin1_id'], row['month_year'])
        event_dict[key][row['event_type']] += row['count']

    # Step 3: Create full admin-month grid
    unique_admins = df_neighbours['matched_admin1_id'].unique()
    unique_months = df_neighbours['month_year'].unique()
    full_index = pd.MultiIndex.from_product([unique_admins, unique_months], names=['matched_admin1_id', 'month_year'])

    # Step 4: Create neighbor lookup table: {admin_id: neighbors}
    neighbor_lookup = df_neighbours.drop_duplicates('matched_admin1_id').set_index('matched_admin1_id')['admin1_neighbors'].to_dict()

    # Step 5: Compute summary for each admin-month
    all_event_types = df_neighbours['event_type'].unique().tolist()
    rows = []

    for admin_id, month in tqdm(full_index, "Get neighbour counts"):
        neighbors = neighbor_lookup.get(admin_id, [])
        summary = {f"{etype}_neighbours": 0 for etype in all_event_types}

        if isinstance(neighbors, (list, tuple)) and neighbors:
            for neighbor_id in neighbors:
                neighbor_counts = event_dict.get((neighbor_id, month), {})
                for etype, count in neighbor_counts.items():
                    summary[f"{etype}_neighbours"] += count

        summary_row = {
            'matched_admin1_id': admin_id,
            'month_year': month,
            **summary
        }
        rows.append(summary_row)

    return pd.DataFrame(rows).set_index(['matched_admin1_id', 'month_year'])


def add_time_trend_features(df):
    """
    Adds temporal trend features (raw year, month and quarter dummies, and linear trend) 
    to a MultiIndexed DataFrame (matched_admin1_id, month_year).
    
    Parameters:
        df (pd.DataFrame): MultiIndexed with ['matched_admin1_id', 'month_year'].
                           'month_year' should be datetime-like.

    Returns:
        pd.DataFrame: DataFrame with added time trend features.
    """
    # Reset index to access 'month_year'
    df = df.reset_index()

    # Ensure datetime format
    df['month_year'] = pd.to_datetime(df['month_year'])

    # Linear monthly trend (starting Jan 2018 = 0)
    start_date = pd.Timestamp('2018-01-01')
    df['linear_month_trend'] = ((df['month_year'].dt.year - start_date.year) * 12 +
                                (df['month_year'].dt.month - start_date.month))

    # Extract date components
    df['year'] = df['month_year'].dt.year
    df['month'] = df['month_year'].dt.month
    df['quarter'] = df['month_year'].dt.quarter

    # Create dummies for month and quarter (drop first to avoid multicollinearity)
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    quarter_dummies = pd.get_dummies(df['quarter'], prefix='quarter', drop_first=True)

    # Concatenate to original df
    df = pd.concat([df, month_dummies, quarter_dummies], axis=1)

    # Drop month and quarter columns (keep year as integer)
    df.drop(columns=['month', 'quarter'], inplace=True)

    # Restore MultiIndex
    df.set_index(['matched_admin1_id', 'month_year'], inplace=True)

    return df


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 9 engineered features motivated by cross-variable relationship analysis:
      a) Lag-2 columns for each of the three main targets
      b) organized_violence (t-1)  — aggregate threat level
      c) is_active (t-1)           — binary zero-inflation splitter
      d) battles_x_remote (t-1)   — multiplicative co-escalation interaction
      e) 3-month rolling averages  — smoothed trend per target

    Parameters:
        df (pd.DataFrame): Must contain 'matched_admin1_id', 'month_year', and the
                           three (t-1) lag columns for Battles, Remote violence, VaC.

    Returns:
        pd.DataFrame: Original DataFrame with 9 new columns appended.
    """
    B1 = 'Battles (t-1)'
    E1 = 'Explosions/Remote violence (t-1)'
    V1 = 'Violence against civilians (t-1)'

    df = df.copy().sort_values(['matched_admin1_id', 'month_year']).reset_index(drop=True)
    grp = df.groupby('matched_admin1_id', sort=False)

    # a) Lag-2: shift existing t-1 columns by 1 more within each region
    for src, dst in [
        (B1, 'Battles (t-2)'),
        (E1, 'Explosions/Remote violence (t-2)'),
        (V1, 'Violence against civilians (t-2)'),
    ]:
        df[dst] = grp[src].shift(1)

    # b) Organized-violence aggregate (CAST's own combined target)
    df['organized_violence (t-1)'] = df[B1] + df[E1] + df[V1]

    # c) Binary activity flag — helps model split zero-inflation from low-activity
    df['is_active (t-1)'] = (df['organized_violence (t-1)'] > 0).astype(int)

    # d) Cross-type interaction — co-escalation: remote violence spikes 13x when Battles > 3
    df['battles_x_remote (t-1)'] = df[B1] * df[E1]

    # e) 3-month rolling averages of each t-1 lag (smoother trend signal)
    for col, name in [
        (B1, 'Battles_3mo_avg (t-1)'),
        (E1, 'Remote_3mo_avg (t-1)'),
        (V1, 'VaC_3mo_avg (t-1)'),
    ]:
        df[name] = grp[col].transform(lambda x: x.rolling(3, min_periods=1).mean())

    return df


def add_importance_weights(df, decay_rate=0.05):
    """
    Adds an 'importance_weight' column based on recency, with exponential decay.
    More recent observations get higher weights.

    Assumes df has a MultiIndex: ['matched_admin1_id', 'month_year'],
    where 'month_year' is datetime-like.
    """
    # Reset index to access 'month_year'
    df = df.reset_index()

    # Ensure datetime format
    df['month_year'] = pd.to_datetime(df['month_year'])

    # Compute months since most recent observation
    max_date = df['month_year'].max()
    months_since = (max_date.to_period('M') - df['month_year'].dt.to_period('M')).apply(lambda x: x.n)

    # Exponential decay weights
    df['importance_weight'] = np.exp(-decay_rate * months_since)

    # Restore original MultiIndex
    df.set_index(['matched_admin1_id', 'month_year'], inplace=True)

    return df