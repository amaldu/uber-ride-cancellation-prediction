# Uber Ride Cancellation Prediction

A machine learning project to predict ride cancellations at booking time, enabling proactive intervention strategies to reduce the overall cancellation rate.

> **[Project Walkthrough](uber-analysis/PROJECT_WALKTROUGH.md)** Read this detailed document covering the full reasoning behind every decision in this project: problem framing (business objectives, cost matrix, metric selection), data acquisition and legal considerations, EDA insights (cleaning, leakage detection, univariate/bivariate findings), and future research directions. 

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/uber-ride-cancellation-prediction.git
cd uber-ride-cancellation-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order (01-09)
cd uber-analysis/notebooks
jupyter lab
```

## Make Predictions

```python
from uber_analysis.src.predict import predict_cancellation

# Single prediction
result = predict_cancellation(
    avg_vtat=8.5,
    pickup_location="Koramangala",
    drop_location="Whitefield",
    vtat_missing=0
)
print(f"Cancellation probability: {result['probability']:.2%}")
print(f"Prediction: {'Will Cancel' if result['prediction'] else 'Will Complete'}")
```

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
├── uber-analysis/
│   ├── data/
│   │   ├── raw/                  # Original dataset
│   │   ├── bronze/               # Cleaned data
│   │   └── silver/               # Feature-engineered data, pipelines
│   ├── models/                   # Trained model artifacts
│   ├── notebooks/
│   │   ├── 01_ingest_data.ipynb
│   │   ├── 02_business_assumptions.ipynb
│   │   ├── 03_data_cleaning.ipynb
│   │   ├── 04_univar_eda.ipynb
│   │   ├── 05_bivar_eda.ipynb
│   │   ├── 06_multivar_eda.ipynb
│   │   ├── 07_feature_engineering.ipynb
│   │   ├── 08_baseline_models.ipynb
│   │   └── 09_tuning_evaluation.ipynb
│   ├── src/
│   │   ├── feature_engineering.py  # Custom transformers
│   │   ├── predict.py              # Inference script
│   │   └── evaluation.py           # Evaluation utilities
│   ├── tests/                    # Unit tests
│   ├── grafana/                  # Monitoring dashboard
│   ├── DATASET_INFO.md
│   └── PROJECT_WALKTHROUGH.md
├── requirements.txt
└── README.md
```

## Notebook Pipeline

| Notebook | Purpose |
|----------|---------|
| 01-03 | Data ingestion, business context, cleaning |
| 04-06 | Univariate, bivariate, multivariate EDA |
| 07 | Feature engineering pipelines |
| 08 | True baselines + candidate model comparison |
| 09 | Hyperparameter tuning + final evaluation |

## Models Compared

| Model | ROC-AUC | Status |
|-------|---------|--------|
| True Baseline (vtat_missing) | ~0.65 | Floor |
| Logistic Regression | ~0.68 | Candidate |
| Random Forest | ~0.70 | Candidate |
| XGBoost | ~0.72 | Candidate |
| **LightGBM** | **~0.73** | **Selected** |

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `avg_vtat` | Numeric | Vehicle time to arrival (minutes) |
| `vtat_missing` | Binary | Whether VTAT is missing (strong signal) |
| `pickup_location` | Categorical | 176 unique locations |
| `drop_location` | Categorical | 176 unique locations |

## Run Tests

```bash
cd uber-analysis
python -m pytest tests/ -v
```

## Run the Grafana Dashboard

```bash
cd uber-analysis/grafana
./start.sh
# Open http://localhost:3000 (admin/admin)
```

## Dataset

- **Source**: [Kaggle - Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
- **Size**: 150,000 bookings from 2024
- **License**: CC BY-SA 4.0

## Tech Stack

- **Analysis**: Python, Pandas, NumPy
- **Modeling**: Scikit-learn, XGBoost, LightGBM, Optuna
- **Visualization**: Matplotlib, Seaborn
- **Dashboard**: Grafana, Docker
- **Testing**: pytest

## License

MIT License — see [LICENSE](LICENSE) for details.
