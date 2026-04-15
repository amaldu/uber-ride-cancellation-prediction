"""Main orchestrator: run the full analysis pipeline and export to Grafana.

Pipeline stages:
    1. Clean raw CSV → bronze parquet
    2. Enrich with derived columns
    3. Univariate analysis
    4. Bivariate analysis
    5. Multivariate analysis
    6. Export results to SQLite
    7. Generate Grafana dashboard JSON
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

UBER_ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent
GRAFANA_DIR = UBER_ANALYSIS_DIR / "grafana"

sys.path.insert(0, str(UBER_ANALYSIS_DIR / "src"))

from analysis.cleaning import clean
from analysis import univariate, bivariate, multivariate
from grafana.export_db import export as export_to_sqlite
from grafana.dashboard import build as build_dashboard

logger = logging.getLogger(__name__)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that bivariate and multivariate stages expect."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
    df["weekday"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_night"] = df["hour"].isin([*range(0, 6), *range(22, 24)]).astype(int)
    df["is_business"] = df["hour"].isin(range(9, 18)).astype(int)
    df["is_rush"] = df["hour"].isin([9, 10, 11, 15, 16, 17, 18, 19, 20, 21]).astype(int)
    df["vtat_missing"] = df["avg_vtat"].isna().astype(int)
    df["vtat_zone"] = pd.cut(
        df["avg_vtat"],
        bins=[0, 2.9, 5.0, 11.9, 15.0, 20.0],
        labels=["instant", "short", "normal", "dip", "timeout"],
        include_lowest=True,
    )
    return df


def analyze(raw_csv: str | None = None) -> None:
    """Run the full analysis pipeline and export to SQLite + Grafana JSON."""
    data_dir = UBER_ANALYSIS_DIR / "data"
    raw_csv = raw_csv or str(data_dir / "raw" / "ncr_ride_bookings.csv")

    logger.info("[1/7] Cleaning data...")
    df_clean = clean(raw_csv, output_dir=str(data_dir / "bronze"))

    logger.info("[2/7] Enriching features...")
    df = _enrich(df_clean)

    logger.info("[3/7] Running univariate analysis...")
    univar_results = univariate.run(df_clean)

    logger.info("[4/7] Running bivariate analysis...")
    bivar_results = bivariate.run(df)

    logger.info("[5/7] Running multivariate analysis...")
    multivar_results = multivariate.run(df)

    logger.info("[6/7] Exporting to SQLite...")
    db_path = export_to_sqlite(univar_results, bivar_results, multivar_results, df)

    logger.info("[7/7] Generating Grafana dashboard...")
    dash_path = build_dashboard()

    logger.info("Analysis complete.")
    logger.info("  Database: %s", db_path)
    logger.info("  Dashboard: %s", dash_path)


def serve() -> None:
    """Start Grafana via docker compose."""
    compose_file = GRAFANA_DIR / "docker-compose.yml"
    if not compose_file.exists():
        logger.error("docker-compose.yml not found at %s", compose_file)
        sys.exit(1)

    logger.info("Starting Grafana at http://localhost:3000 (admin/admin)...")
    subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=str(GRAFANA_DIR),
        check=True,
    )
    logger.info("Grafana is running. Open http://localhost:3000")


def stop() -> None:
    """Stop Grafana."""
    logger.info("Stopping Grafana...")
    subprocess.run(
        ["docker", "compose", "down"],
        cwd=str(GRAFANA_DIR),
        check=True,
    )
    logger.info("Grafana stopped.")
