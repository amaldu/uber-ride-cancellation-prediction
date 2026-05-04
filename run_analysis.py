import logging
import re
import sqlite3
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

import pandas as pd

ROOT          = Path(__file__).resolve().parent
RAW_CSV       = ROOT / "data" / "raw" / "ncr_ride_bookings.csv"
BRONZE_DIR    = ROOT / "data" / "bronze"
SILVER_DIR    = ROOT / "data" / "silver"
SQLITE_DB     = SILVER_DIR / "rides.db"
DASHBOARD_ZIP = ROOT / "superset" / "dashboard.zip"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# transformation config

BOOKING_STATUS_MAP = {
    "Completed": 0,
    "Cancelled by Driver": 1,
    "No Driver Found": 1,
    "Cancelled by Customer": 1,
    "Incomplete": 0,
}

WHITELIST_COLS = [
    "date", 
    "time",
    "hour",
    "weekday",
    "vehicle_type",
    "pickup_location",
    "drop_location",
    "avg_vtat",
    "is_cancelled",
    "payment_method",
    "vtat_missing",
    "vtat_zone",
]

CATEGORICAL_COLUMNS = ["vehicle_type", "pickup_location", "drop_location"]
NUMERIC_COLUMNS = ["avg_vtat"]


# pipeline functions 

def to_snake_case(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    new_columns = []
    for col in df.columns:
        col = col.strip()
        col = re.sub(r"[\s\-]+", "_", col)
        col = re.sub(r"[^\w_]", "", col)
        col = col.lower()
        new_columns.append(col)
    df.columns = new_columns
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = to_snake_case(df, df.columns.tolist())

    df["is_cancelled"] = df["booking_status"].map(BOOKING_STATUS_MAP).astype("bool")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
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
    log.info("Loading raw data...")
    df_raw = pd.read_csv(RAW_CSV)

    log.info("Cleaning...")
    df_clean = clean(df_raw)
    log.info("Enriching features...")
    df = enrich(df_clean)
    df = df[[c for c in WHITELIST_COLS if c in df.columns]]

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(SILVER_DIR / "enriched_dataset.parquet"), index=False)

    log.info("Pipeline complete!")

    log.info("Writing SQLite database for Superset...")
    conn = sqlite3.connect(str(SQLITE_DB))
    df.to_sql("Uber Rides Dataset", conn, if_exists="replace", index=False)
    conn.close()
    log.info(f"Synced {len(df):,} rows")

    log.info("Starting Apache Superset...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d"],
        # captures stdout & stderr instead of print in terminal
        capture_output=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        log.error("Could not start Superset via Docker Compose.")
        #.decode() converts bytes to python string
        log.error(result.stderr.decode())
        return

    log.info("Waiting for Superset to be ready...it takes a bit of time to start up")
    # waiting 30s is necessary bc writing the csv into the db made it look like 
    # the browser wasn't loading 
    for _ in range(30):
        try:
            urllib.request.urlopen("http://localhost:8088/health", timeout=2)
            break
        except Exception:
            time.sleep(2)
    else:
        log.warning("Superset did not respond in time. Opening the browser anyway.")

    #the dashboard for the portfolio already exists inside superset/
    if DASHBOARD_ZIP.exists():
        log.info("Importing Superset dashboard...")
        subprocess.run(["docker", "cp", str(DASHBOARD_ZIP), "superset:/tmp/dashboard.zip"])
        subprocess.run(["docker", "exec", "superset", "superset", "import-dashboards", "-p", "/tmp/dashboard.zip", "-u", "admin"])

    log.info("Superset is running at http://localhost:8088")
    log.info("I hope you like it :)")
    webbrowser.open("http://localhost:8088")


if __name__ == "__main__":
    main()
