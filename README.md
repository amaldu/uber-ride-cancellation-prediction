# Uber Ride Cancellation Analysis

An automated data analysis pipeline that investigates ride cancellation patterns in Uber booking data, produces statistical findings, and generates a curated Markdown report with charts.

> **[Project Walkthrough](uber-analysis/PROJECT_WALKTROUGH.md)** — detailed reasoning behind every decision: problem framing, business objectives, data acquisition, EDA methodology, and key findings.

## Quick Start

```bash
git clone https://github.com/yourusername/uber-ride-cancellation-prediction.git
cd uber-ride-cancellation-prediction

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

./run_analysis.sh
```

The report is written to `uber-analysis/reports/analysis_<timestamp>.md` with a copy at `uber-analysis/reports/latest.md`.

## Key Findings

### 32% of bookings end in cancellation (~$960K estimated annual loss)

### Strongest predictor: VTAT (Vehicle Time to Arrival)
- **VTAT >= 15 min → 100% cancellation** (system auto-cancel)
- **VTAT missing → 100% cancellation** (early cancellations before vehicle assignment)
- Five behavioural zones identified with non-linear, non-monotonic relationship

### What does NOT predict cancellation
- **Time of day, day of week, month**: cancellation rate is flat (~32%) across all temporal dimensions
- **Vehicle type**: all 7 types show ~32% cancellation rate
- **Route (pickup × drop)**: high cardinality artifact; zero cross-validated signal

## What the Pipeline Does

1. **Clean** raw CSV → remove leakage columns, map target, cast types
2. **Univariate** analysis → distribution, cardinality, missing values, temporal patterns
3. **Bivariate** analysis → each feature vs cancellation with statistical tests
4. **Multivariate** analysis → redundancy checks, route validation, VIF, correlation
5. **Generate charts** → 6 publication-quality PNGs
6. **Build report** → curated Markdown with findings, tables, and recommendations

## Project Structure

```
├── run_analysis.sh              # Entry point — run this
├── requirements.txt
├── README.md
└── uber-analysis/
    ├── DATASET_INFO.md
    ├── PROJECT_WALKTHROUGH.md
    ├── data/
    │   ├── raw/                 # Original Kaggle CSV
    │   ├── bronze/              # Cleaned parquet
    │   └── silver/              # Enriched parquet (with derived features)
    ├── notebooks/               # Exploratory notebooks (historical reference)
    │   ├── 01_ingest_data.ipynb
    │   ├── 02_business_assumptions.ipynb
    │   ├── 03_data_cleaning.ipynb
    │   ├── 04_univar_eda.ipynb
    │   ├── 05_bivar_eda.ipynb
    │   └── 06_multivar_eda.ipynb
    ├── src/
    │   ├── eda_utils/           # Reusable statistical and plotting functions
    │   ├── analysis/            # Pipeline stages (cleaning, univar, bivar, multivar)
    │   └── reporting/           # Chart generation and Markdown report builder
    └── reports/
        ├── charts/              # Static PNGs
        ├── latest.md            # Most recent report
        └── analysis_*.md        # Timestamped reports
```

## Dataset

- **Source**: [Kaggle — Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
- **Size**: 150,000 bookings from 2024
- **License**: CC BY-SA 4.0

## Tech Stack

- **Analysis**: Python, Pandas, NumPy, SciPy, scikit-learn (statistical utilities only)
- **Visualization**: Matplotlib, Seaborn
- **Automation**: Bash

## License

MIT License — see [LICENSE](LICENSE) for details.
