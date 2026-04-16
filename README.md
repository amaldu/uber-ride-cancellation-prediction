# Uber Ride Cancellation Analysis

Automated data analysis pipeline that investigates ride cancellation patterns and presents the results in a Grafana dashboard.

> **[Project Walkthrough](uber-analysis/PROJECT_WALKTROUGH.md)** — full reasoning behind every decision: problem framing, business objectives, EDA methodology, and key findings.

## Usage

```bash
# 1. Run the analysis pipeline
python run_analysis.py

# 2. Start the dashboard
cd uber-analysis/grafana && docker compose up
```

Open **http://localhost:3000** (admin / admin).

To stop Grafana: `docker compose down` from the same directory.

## Key Findings

- **32% of bookings** end in cancellation (~$960K estimated annual loss)
- **VTAT >= 15 min → 100% cancellation** (system auto-cancel)
- **VTAT missing → 100% cancellation** (early cancellations before vehicle assignment)
- **Temporal features carry no signal** — rate is flat across hours, days, months
- **Vehicle type is not discriminative** — all 7 types show ~32%
- **Route feature fails cross-validation** — high Cramer's V was a cardinality artifact

## How It Works

```
Raw CSV → Clean → Enrich → Univariate → Bivariate → Multivariate
                                                          ↓
                                                     SQLite DB → Grafana
```

Every value in the dashboard is computed from data. Nothing is hardcoded.

## Project Structure

```
├── run_analysis.py              # Run this
├── requirements.txt
└── uber-analysis/
    ├── PROJECT_WALKTHROUGH.md
    ├── DATASET_INFO.md
    ├── data/
    │   ├── raw/                 # Kaggle CSV
    │   ├── bronze/              # Cleaned parquet
    │   └── silver/              # Enriched parquet
    ├── grafana/
    │   ├── docker-compose.yml   # Grafana container
    │   ├── data/                # SQLite DB (generated)
    │   └── provisioning/        # Auto-provisioned datasource + dashboard
    ├── notebooks/               # Exploratory (historical reference)
    └── src/
        ├── analysis/            # Pipeline: cleaning, univar, bivar, multivar
        ├── grafana/             # SQLite export + dashboard JSON generator
        └── eda_utils/           # Reusable stats/plotting functions
```

## Prerequisites

- Python 3.10+
- Docker

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

- **Source**: [Kaggle — Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
- **Size**: 150,000 bookings from 2024
- **License**: CC BY-SA 4.0

## License

MIT — see [LICENSE](LICENSE).
