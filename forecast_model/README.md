# ACLED Conflict Forecasting

This project forecasts **conflict event counts** at the **Admin 1 level** using ACLED data, spatial and temporal features, and machine learning (Random Forests, XGBoost). It supports forecasting by region and event type, with evaluation metrics and visualizations. It also supports running the data cleaning pipeline independently.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/predicting-global-conflict.git
cd forecast_model
```

### 2. Install Dependencies

Create a virtual environment and install packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or in Google Colab:

```python
!pip install -r requirements.txt
```

---

## Running the Pipeline

### Data Cleaning Only Mode

You can run **only the data cleaning and preprocessing pipeline**, with no forecasting, using:

```bash
python main.py --clean-data
```

This mode:

* Runs the full preprocessing pipeline from raw data
* Saves cleaned and feature-engineered data to disk
* Does not require `--region` or `--event`
* Does not train or evaluate any model

This is useful for preparing data once before running multiple forecasts.

### Forecasting Mode

Run the full forecasting pipeline for a specific Admin1 region and event type:

```bash
python main.py --region "UKR - Donetsk" --event "Battles"
```

Optional flag:

* `--clean-data`: Run the full preprocessing pipeline from raw data before modeling. If omitted, the pipeline loads cleaned data from disk if available.

Example:

```bash
python main.py --region "UKR - Donetsk" --event "Battles" --clean-data
```

See the full list of valid regions in:

```
data/processed/valid_regions.txt
```

---

## Outputs

Running the forecasting pipeline will generate:

* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)
* Forecast plot (actual vs. predicted)
* Feature importance plot

Plots are saved to:

```
outputs/figures/
```

Filenames include the region and event type.

---

## Configuration

Edit `config/settings.py` to modify:

* Predictor feature columns
* Target event types
* Temporal or spatial features to include

---

## Data Requirements

Ensure the following data files are in place:

* ACLED event data:
  `data/raw/1997-01-01-2025-07-03.csv`

* Admin1 shapefiles:
  `data/raw/boundaries/ne_10m_admin_1_states_provinces/...`

These must be downloaded manually from the [Google Drive](https://drive.google.com/drive/folders/1qG9lFDUKTZW2kG6erbAqRSJhdm5l1255?usp=sharing) if not included in the repository.

---

## Modeling Details

* Forecasts are generated using a **Random Forest Regressor**
* Models include:

  * Lagged conflict event counts
  * Time trend features (monthly, quarterly, linear)
  * Neighbor region features
  * Importance weights prioritizing recent data
* Evaluation is done on a 6-month holdout set

---

## License

MIT License.
