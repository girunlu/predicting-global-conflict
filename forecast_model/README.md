# ACLED Conflict Forecasting

This project forecasts **conflict event counts** at the **Admin 1 level** using ACLED data, spatial and temporal features, and machine learning (Random Forests, XGBoost). It supports forecasting by region and event type, with evaluation metrics and visualizations.

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

## Running the Forecasting Pipeline

You can run the full forecasting pipeline from the command line using:

```bash
python main.py --region "UKR - Donetsk" --event "Battles"
```

Optional flag:

* `--clean-data`: Run the full preprocessing pipeline from raw data. Otherwise, the pipeline loads cleaned data from disk (if available).

See the full list of possible regions in '/data/processed/valid_regions.txt'.

### Example: Run full pipeline including data cleaning

```bash
python main.py --region "UKR - Donetsk" --event "Battles" --clean-data
```

---

## Outputs

Running the pipeline will generate:

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

MIT License — feel free to use, modify, and share with attribution.
