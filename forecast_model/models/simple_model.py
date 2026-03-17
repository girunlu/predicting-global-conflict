import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from config import settings

def sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")
    

def train_and_evaluate_model(region_data, target_event, region_name=None, predictors=None):
    # Ensure output directory exists
    output_dir = "outputs/figures"
    os.makedirs(output_dir, exist_ok=True)

    # Use provided predictor list or fall back to baseline predictors
    if predictors is None:
        predictors = settings.predictors
    # Only keep columns that actually exist in region_data
    predictors = [p for p in predictors if p in region_data.columns]

    # Split train/test
    train_data = region_data.iloc[:-6]
    test_data = region_data.iloc[-6:]

    X_train = train_data[predictors]
    X_test = test_data[predictors]
    y_train = train_data[target_event]
    y_test = test_data[target_event]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train, sample_weight=train_data["importance_weight"])
    y_pred = rf.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test[y_test != 0] - y_pred[y_test != 0]) / y_test[y_test != 0])) * 100

    print(f"\nForecast Results for {target_event} in {region_name}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # --- Plot 1: Predictions vs Actual ---
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title(f'{target_event} in {region_name}: Predictions vs Actual')
    plt.xlabel('Month Index')
    plt.ylabel('Event Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    forecast_path = os.path.join(output_dir, f"forecast_{sanitize_filename(region_name)}_{sanitize_filename(target_event)}.png")
    plt.savefig(forecast_path)
    plt.close()

    # --- Plot 2: Feature Importance ---
    importances = rf.feature_importances_
    importance_series = pd.Series(importances, index=X_train.columns).sort_values()

    plt.figure(figsize=(8, 6))
    importance_series.plot(kind='barh', title=f'Feature Importance: {target_event} in {region_name}')
    plt.tight_layout()

    importance_path = os.path.join(output_dir, f"feature_importance_{sanitize_filename(region_name)}_{sanitize_filename(target_event)}.png")
    plt.savefig(importance_path)
    plt.close()

    return mae, mape
