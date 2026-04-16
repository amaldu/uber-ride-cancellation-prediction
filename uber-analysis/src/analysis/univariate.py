"""Univariate EDA: analyse each feature independently.

Reproduces the curated highlights from 04_univar_eda.ipynb.
Returns a structured results dict consumed by the report builder.
"""

import numpy as np
import pandas as pd


def run(df: pd.DataFrame) -> dict:
    """Run all univariate analyses and return a results dict."""
    results: dict = {}

    results["shape"] = {"rows": len(df), "cols": len(df.columns)}
    results["columns"] = df.columns.tolist()
    results["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    # --- Target distribution ---
    target_counts = df["is_cancelled"].value_counts()
    results["target"] = {
        "completed": int(target_counts.get(0.0, 0)),
        "cancelled": int(target_counts.get(1.0, 0)),
        "cancellation_rate": float(df["is_cancelled"].mean()),
    }

    # --- Missing values ---
    missing = df.isna().sum()
    results["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing[col] / len(df) * 100)}
        for col in df.columns
        if missing[col] > 0
    }

    # --- Vehicle type distribution ---
    vtype = df["vehicle_type"].value_counts()
    results["vehicle_type"] = {
        "categories": vtype.index.tolist(),
        "counts": vtype.values.tolist(),
        "n_unique": int(df["vehicle_type"].nunique()),
    }

    # --- Location cardinality ---
    results["locations"] = {
        "pickup_unique": int(df["pickup_location"].nunique()),
        "drop_unique": int(df["drop_location"].nunique()),
        "pickup_top5": df["pickup_location"].value_counts().head(5).to_dict(),
        "drop_top5": df["drop_location"].value_counts().head(5).to_dict(),
    }

    # --- VTAT distribution ---
    vtat = df["avg_vtat"].dropna()
    results["avg_vtat"] = {
        "count": int(vtat.count()),
        "missing": int(df["avg_vtat"].isna().sum()),
        "missing_pct": float(df["avg_vtat"].isna().mean() * 100),
        "mean": float(vtat.mean()),
        "median": float(vtat.median()),
        "min": float(vtat.min()),
        "max": float(vtat.max()),
        "std": float(vtat.std()),
        "q25": float(vtat.quantile(0.25)),
        "q75": float(vtat.quantile(0.75)),
    }

    # --- Temporal flatness ---
    df_temp = df[["date", "time", "is_cancelled"]].copy()
    df_temp["date"] = pd.to_datetime(df_temp["date"])
    df_temp["hour"] = pd.to_datetime(df_temp["time"], format="%H:%M:%S").dt.hour
    df_temp["weekday"] = df_temp["date"].dt.dayofweek
    df_temp["month"] = df_temp["date"].dt.month

    temporal = {}
    for col, name in [("hour", "hourly"), ("weekday", "daily"), ("month", "monthly")]:
        rates = df_temp.groupby(col)["is_cancelled"].mean()
        temporal[name] = {
            "min_rate": float(rates.min()),
            "max_rate": float(rates.max()),
            "spread_pp": float((rates.max() - rates.min()) * 100),
        }
    results["temporal"] = temporal

    return results
