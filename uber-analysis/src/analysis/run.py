"""Main orchestrator: run the full analysis pipeline and generate the report."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

UBER_ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(UBER_ANALYSIS_DIR / "src"))

from analysis.cleaning import clean
from analysis import univariate, bivariate, multivariate
from reporting.charts import generate_all as generate_charts
from reporting.report_builder import build as build_report


def _enrich_for_bivariate(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns that the bivariate and multivariate stages expect."""
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


def main(
    raw_csv: str | None = None,
    charts_dir: str | None = None,
    reports_dir: str | None = None,
    force_charts: bool = False,
) -> str:
    """Run the full pipeline. Returns the path to the generated report."""
    data_dir = UBER_ANALYSIS_DIR / "data"
    raw_csv = raw_csv or str(data_dir / "raw" / "ncr_ride_bookings.csv")
    charts_dir = charts_dir or str(UBER_ANALYSIS_DIR / "reports" / "charts")
    reports_dir = reports_dir or str(UBER_ANALYSIS_DIR / "reports")

    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Clean
    print("[1/6] Cleaning data...")
    df_clean = clean(raw_csv, output_dir=str(data_dir / "bronze"))

    # 2. Enrich
    print("[2/6] Enriching features...")
    df = _enrich_for_bivariate(df_clean)

    # 3. Univariate
    print("[3/6] Running univariate analysis...")
    univar_results = univariate.run(df_clean)

    # 4. Bivariate
    print("[4/6] Running bivariate analysis...")
    bivar_results = bivariate.run(df)

    # 5. Multivariate
    print("[5/6] Running multivariate analysis...")
    multivar_results = multivariate.run(df)

    # 6. Charts + Report
    print("[6/6] Generating charts and report...")
    chart_paths = generate_charts(df, univar_results, bivar_results, multivar_results, charts_dir)

    report_md = build_report(
        univar_results, bivar_results, multivar_results,
        chart_paths, charts_rel_dir="charts",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"analysis_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(report_md)

    latest_path = os.path.join(reports_dir, "latest.md")
    with open(latest_path, "w") as f:
        f.write(report_md)

    print(f"\nReport written to: {report_path}")
    print(f"Latest symlink:    {latest_path}")
    return report_path


if __name__ == "__main__":
    main()
