# Uber Ride Cancellation Prediction

A machine learning project to predict ride cancellations at booking time, enabling proactive intervention strategies to reduce the overall cancellation rate.

## Business Problem

Analysis of 2024 Uber ride data revealed a critical operational challenge:
- **32% of all bookings** end in cancellation (~37,500 rides)
- This represents approximately **$748,600 in lost revenue** annually
- Cost asymmetry: Missing a cancellation costs **4x more** than a false alarm

## Solution

Built a predictive model that identifies high-risk bookings at the time of booking, enabling:
- Proactive customer engagement (ETA updates, confirmations)
- Strategic driver assignment
- Targeted incentives for at-risk bookings

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| F2-Score | ≥ 0.68 | **0.73** |
| Recall | ≥ 70% | **95.6%** |
| Precision | ≥ 60% | 38.1% |
| Est. Annual Savings | $100K | **$174K** |

> **Note**: The model prioritizes recall over precision due to the 4:1 cost asymmetry (missing a cancellation costs $20, false alarm costs $5).

## Key Findings

### Strongest Predictor: VTAT (Vehicle Time to Arrival)
- **VTAT ≥ 15 minutes → 100% cancellation rate**
- Clear threshold effect - most actionable feature for intervention

### Other Insights
- **Location patterns**: Moderate predictive power (176 unique locations)
- **Temporal patterns**: Surprisingly flat - no "danger hours" identified
- **Vehicle type**: Weak predictor - all types show ~32% cancellation rate

## Project Structure

```
uber-analysis/
├── grafana/                   # Grafana monitoring dashboard
│   ├── docker-compose.yml    # Docker setup for Grafana
│   ├── export_data.py        # Exports analysis insights to SQLite
│   ├── generate_dashboard.py # Generates Grafana dashboard JSON
│   ├── start.sh              # One-command setup script
│   ├── data/                 # SQLite database (generated)
│   └── provisioning/         # Grafana auto-provisioning configs
├── data/
│   ├── raw/                  # Original dataset
│   ├── bronze/               # Cleaned data splits
│   └── silver/               # Feature-engineered data
├── models/                   # Trained model artifacts
├── notebooks/
│   ├── 01_ingest_data.ipynb
│   ├── 02_business_assumptions.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_univar_eda.ipynb
│   ├── 05_bivar_eda.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_baseline_logistic_regression.ipynb
│   ├── 08_random_forest.ipynb
│   ├── 09_xgboost.ipynb
│   └── 10_lightgbm.ipynb
├── DATASET_INFO.md
└── PROJECT_WALKTHROUGH.md    # Detailed methodology
```

## Models Compared

| Model | F2-Score | Recall | Precision | Status |
|-------|----------|--------|-----------|--------|
| Logistic Regression | 0.55 | 60% | 45% | Baseline |
| Random Forest | 0.68 | 85% | 42% | Improved |
| XGBoost | 0.73 | 96% | 38% | Best |
| **LightGBM** | **0.73** | **95.6%** | 38.1% | **Selected** |

## Features Engineered (17 total)

| Category | Features |
|----------|----------|
| VTAT | `avg_vtat_imputed`, `vtat_bucket`, `is_high_vtat` |
| Location | `pickup_encoded`, `drop_encoded` (target encoded, top 10 + Other) |
| Vehicle | `vehicle_type_encoded` |
| Temporal | `hour`, `dayofweek`, `month`, `is_weekend`, `is_peak_hour`, `is_late_night` |
| Cyclical | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos` |

## Run the Grafana Dashboard Locally

The Grafana dashboard provides interactive visualizations of the EDA insights and model monitoring metrics. It runs via Docker.

**Prerequisites**: Docker and Docker Compose installed.

```bash
# Navigate to the grafana directory
cd uber-analysis/grafana

# Option 1: One-command setup (recommended)
./start.sh

# Option 2: Step by step
python3 export_data.py           # Export insights to SQLite
python3 generate_dashboard.py    # Generate dashboard JSON
docker compose up -d             # Start Grafana
```

Open **http://localhost:3000** and log in with `admin` / `admin`.

The dashboard **"Uber Ride Cancellation — Analysis & Model Monitoring"** is set as the home dashboard and includes:

| Section | Panels |
|---------|--------|
| **KPIs** | Total bookings, cancellation rate, lost revenue, features engineered |
| **Univariate Analysis** | Target distribution, vehicle type, VTAT distribution, hourly/daily/monthly patterns |
| **Bivariate Analysis** | VTAT buckets vs cancellation, top pickup/drop locations, mutual information, correlations, Cramer's V |
| **Model Monitoring** | 4-model comparison table, LightGBM gauges (F2/Recall/Precision/PR-AUC/ROC-AUC), feature importance for all models |
| **Business Impact** | Net savings, ROI, daily interventions, confusion matrix breakdown, financial breakdown |

To stop Grafana:
```bash
cd uber-analysis/grafana
docker compose down
```

## Dataset

- **Source**: [Kaggle - Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
- **Size**: 150,000 bookings from 2024
- **License**: CC BY-SA 4.0

## Tech Stack

- **Analysis**: Python, Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Modeling**: Scikit-learn, XGBoost, LightGBM
- **Dashboard**: Grafana
- **Infrastructure**: Docker, Docker Compose
- **Data Storage**: Parquet, SQLite

## Future Improvements

- [ ] Implement A/B testing framework for interventions
- [ ] Explore deep learning approaches
- [ ] Add geographic clustering features
- [ ] Build API endpoint for production deployment
- [ ] Add data drift detection to Grafana monitoring

