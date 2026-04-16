"""Univariate analysis and pipeline helpers

"""

import numpy as np
import pandas as pd

#############################################################################
# Notebooks
#############################################################################


def get_stats(data):
    stats = data.describe(include='all')
    stats.loc['dtype'] = data.dtypes
    stats.loc['rows_dataset'] = len(data)

    stats.loc['n_missing'] = data.isna().sum()
    stats.loc['% missing'] = round((data.isna().sum()/len(data)) * 100, 2)

    numeric_cols = data.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        stats.loc['skew', numeric_cols] = data[numeric_cols].skew()
        stats.loc['kurtosis', numeric_cols] = data[numeric_cols].kurtosis()
        stats.loc['range', numeric_cols] = data[numeric_cols].max() - data[numeric_cols].min()
        stats.loc['iqr', numeric_cols] = data[numeric_cols].quantile(0.75) - data[numeric_cols].quantile(0.25)
    return stats


def get_outliers(data, col):
    col_data = data[col].dropna()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = col_data[(col_data < lower) | (col_data > upper)]

    if len(outliers) > 0:
        print(f"\n{col} outliers")
        print(f"Range: [{outliers.min():.2f}, {outliers.max():.2f}]")
        print(f"Mean: {outliers.mean():.2f}")
        print(f"Median: {outliers.median():.2f}")
        print(f"Count: {len(outliers)} ({len(outliers)/len(col_data)*100:.2f}%)")
    else:
        print(f"\n {col}: no outliers detected")


def get_nans(data, col):
    cancelled_mask = data["is_cancelled"] == 1
    nan_mask = data[col].isna()

    exact_match = (cancelled_mask == nan_mask).all()
    print("All cancelled rows match NaNs in is_cancelled?", exact_match)

    print("Number of cancelled rows:", cancelled_mask.sum())
    print("Number of NaN avg_ctat rows:", nan_mask.sum())
    print("Number of rows where both are True:", (cancelled_mask & nan_mask).sum())

#############################################################################
# Pipeline functions
#############################################################################


def run_univariate(df: pd.DataFrame) -> dict:
    results: dict = {}

    results["shape"] = {"rows": len(df), "cols": len(df.columns)}
    results["columns"] = df.columns.tolist()
    results["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    target_counts = df["is_cancelled"].value_counts()
    results["target"] = {
        "completed": int(target_counts.get(0.0, 0)),
        "cancelled": int(target_counts.get(1.0, 0)),
        "cancellation_rate": float(df["is_cancelled"].mean()),
    }

    missing = df.isna().sum()
    results["missing"] = {
        col: {"count": int(missing[col]), "pct": float(missing[col] / len(df) * 100)}
        for col in df.columns
        if missing[col] > 0
    }

    vtype = df["vehicle_type"].value_counts()
    results["vehicle_type"] = {
        "categories": vtype.index.tolist(),
        "counts": vtype.values.tolist(),
        "n_unique": int(df["vehicle_type"].nunique()),
    }

    results["locations"] = {
        "pickup_unique": int(df["pickup_location"].nunique()),
        "drop_unique": int(df["drop_location"].nunique()),
        "pickup_top5": df["pickup_location"].value_counts().head(5).to_dict(),
        "drop_top5": df["drop_location"].value_counts().head(5).to_dict(),
    }

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
