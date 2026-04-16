"""Run the full analysis pipeline and export results to Grafana.

Usage: python run_analysis.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

UBER_ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(UBER_ANALYSIS_DIR / "src"))

from analysis.cleaning import clean
from analysis import univariate, bivariate, multivariate
from grafana.export_db import export as export_to_sqlite
from grafana.dashboard import build as build_dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
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


def main():
    data_dir = UBER_ANALYSIS_DIR / "data"
    raw_csv = str(data_dir / "raw" / "ncr_ride_bookings.csv")

    log.info("[1/7] Cleaning data...")
    df_clean = clean(raw_csv, output_dir=str(data_dir / "bronze"))

    log.info("[2/7] Enriching features...")
    df = _enrich(df_clean)

    log.info("[3/7] Univariate analysis...")
    univar = univariate.run(df_clean)

    log.info("[4/7] Bivariate analysis...")
    bivar = bivariate.run(df)

    log.info("[5/7] Multivariate analysis...")
    multivar = multivariate.run(df)

    log.info("[6/7] Exporting to SQLite...")
    db = export_to_sqlite(univar, bivar, multivar, df)

    log.info("[7/7] Generating dashboard...")
    dash = build_dashboard()

    log.info("Done. Database: %s", db)
    log.info("Now run: docker compose up  (in uber-analysis/grafana/)")


if __name__ == "__main__":
    main()
