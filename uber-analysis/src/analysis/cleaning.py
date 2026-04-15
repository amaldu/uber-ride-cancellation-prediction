"""Data cleaning pipeline for raw Uber ride bookings CSV.

Reproduces the logic from 03_data_cleaning.ipynb as a callable function.
"""

import re
import os

import pandas as pd


def _to_snake_case(columns: list[str]) -> list[str]:
    result = []
    for col in columns:
        col = col.strip()
        col = re.sub(r"[\s\-]+", "_", col)
        col = re.sub(r"[^\w_]", "", col)
        col = col.lower()
        result.append(col)
    return result


LEAKAGE_COLUMNS = [
    "cancelled_rides_by_customer",
    "reason_for_cancelling_by_customer",
    "cancelled_rides_by_driver",
    "driver_cancellation_reason",
    "incomplete_rides",
    "incomplete_rides_reason",
    "driver_ratings",
    "customer_rating",
]

POST_OUTCOME_COLUMNS = ["avg_ctat", "booking_value", "ride_distance", "payment_method"]

BOOKING_STATUS_MAP = {
    "Completed": 0,
    "Cancelled by Driver": 1,
    "No Driver Found": 1,
    "Cancelled by Customer": 1,
    "Incomplete": 0,
}

CATEGORICAL_COLUMNS = ["vehicle_type", "pickup_location", "drop_location"]
NUMERIC_COLUMNS = ["avg_vtat"]
ID_COLUMNS = ["booking_id", "customer_id"]


def clean(raw_csv_path: str, output_dir: str | None = None) -> pd.DataFrame:
    """Read raw CSV, apply all cleaning steps, return clean DataFrame.

    Optionally writes the result to ``output_dir/clean_dataset.parquet``.
    """
    df = pd.read_csv(raw_csv_path)
    df.columns = _to_snake_case(df.columns.tolist())

    df["is_cancelled"] = df["booking_status"].map(BOOKING_STATUS_MAP)
    df = df.drop(columns=["booking_status"])

    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])
    df = df.drop(columns=[c for c in POST_OUTCOME_COLUMNS if c in df.columns])

    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].str.strip('"')

    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    df["is_cancelled"] = df["is_cancelled"].astype("float32")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(
            os.path.join(output_dir, "clean_dataset.parquet"),
            engine="pyarrow",
            index=False,
        )

    return df
