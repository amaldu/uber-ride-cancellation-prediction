# Uber Ride Cancellation Analysis

An automated data analysis pipeline that investigates ride cancellation patterns in Uber booking data and presents the results in a Grafana dashboard.

> **[Project Walkthrough](uber-analysis/PROJECT_WALKTROUGH.md)** — detailed reasoning behind every decision: problem framing, business objectives, data acquisition, EDA methodology, and key findings.

## Quick Start

```bash
git clone https://github.com/yourusername/uber-ride-cancellation-prediction.git
cd uber-ride-cancellation-prediction

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run analysis and open dashboard
PYTHONPATH=uber-analysis/src python -m analysis run
```

Then open **http://localhost:3000** (login: admin / admin).

## Commands

All commands are run from the project root with `PYTHONPATH=uber-analysis/src`:

| Command | What it does |
|---------|-------------|
| `python -m analysis run` | Run full pipeline + start Grafana |
| `python -m analysis analyze` | Run analysis only (no Grafana) |
| `python -m analysis serve` | Start Grafana (analysis must have been run) |
| `python -m analysis stop` | Stop Grafana |
| `python -m analysis run --csv /path/to/data.csv` | Run on a different CSV |

## Key Findings

### 32% of bookings end in cancellation (~$960K estimated annual loss)

### Strongest predictor: VTAT (Vehicle Time to Arrival)
- **VTAT >= 15 min → 100% cancellation** (system auto-cancel)
- **VTAT missing → 100% cancellation** (early cancellations before vehicle assignment)
- Five behavioural zones with non-linear, non-monotonic relationship

### What does NOT predict cancellation
- **Time of day, day of week, month**: rate is flat (~32%) across all temporal dimensions
- **Vehicle type**: all 7 types show ~32% cancellation rate
- **Route (pickup × drop)**: high cardinality artifact; zero cross-validated signal

## How It Works

```
Raw CSV → Clean → Enrich → Univariate → Bivariate → Multivariate
                                                          ↓
                                                     SQLite DB ← Grafana reads via SQL
                                                          ↓
                                                   Dashboard JSON (auto-provisioned)
```

Every value in the dashboard is **computed from data** on each run. Nothing is hardcoded.

## Project Structure

```
├── requirements.txt
├── README.md
└── uber-analysis/
    ├── DATASET_INFO.md
    ├── PROJECT_WALKTHROUGH.md
    ├── data/
    │   ├── raw/                 # Original Kaggle CSV
    │   ├── bronze/              # Cleaned parquet
    │   └── silver/              # Enriched parquet
    ├── grafana/
    │   ├── docker-compose.yml   # Grafana container
    │   ├── data/                # SQLite DB (generated, gitignored)
    │   └── provisioning/        # Auto-provisioned datasource + dashboard
    ├── notebooks/               # Exploratory notebooks (historical reference)
    │   ├── 01_ingest_data.ipynb
    │   ├── 02_business_assumptions.ipynb
    │   ├── 03_data_cleaning.ipynb
    │   ├── 04_univar_eda.ipynb
    │   ├── 05_bivar_eda.ipynb
    │   └── 06_multivar_eda.ipynb
    └── src/
        ├── analysis/            # Pipeline stages
        │   ├── __main__.py      # CLI entry point
        │   ├── run.py           # Orchestrator (analyze / serve / stop)
        │   ├── cleaning.py      # Raw CSV → clean DataFrame
        │   ├── univariate.py    # Per-feature analysis
        │   ├── bivariate.py     # Feature vs target tests
        │   └── multivariate.py  # Interactions, redundancy, VIF
        ├── grafana/             # Dashboard generation
        │   ├── export_db.py     # Analysis dicts → SQLite
        │   └── dashboard.py     # Programmatic Grafana JSON
        └── eda_utils/           # Reusable stats/plotting functions
```

## Prerequisites

- Python 3.10+
- Docker (for Grafana)

## Dataset

- **Source**: [Kaggle — Uber Ride Analytics Dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-cancellation-dashboard)
- **Size**: 150,000 bookings from 2024
- **License**: CC BY-SA 4.0

## Tech Stack

- **Analysis**: Python, Pandas, NumPy, SciPy, scikit-learn (statistical utilities)
- **Database**: SQLite
- **Dashboard**: Grafana + frser-sqlite-datasource plugin
- **Infrastructure**: Docker Compose

## License

MIT License — see [LICENSE](LICENSE) for details.
