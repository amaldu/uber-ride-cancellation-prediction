"""Bivariate EDA: analyse each feature against the target.

Reproduces the curated highlights from 05_bivar_eda.ipynb.
Returns a structured results dict consumed by the report builder.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, spearmanr


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _phi_coefficient(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.sum().sum()
    return float(np.sqrt(chi2 / n))


def _binary_vs_target(df: pd.DataFrame, col: str) -> dict:
    ct = pd.crosstab(df[col], df["is_cancelled"])
    chi2, p, _, _ = chi2_contingency(ct)
    phi = _phi_coefficient(df[col], df["is_cancelled"])

    if ct.shape == (2, 2):
        table = ct.values
        odds_ratio_val, fisher_p = fisher_exact(table)
    else:
        odds_ratio_val, fisher_p = float("nan"), float("nan")

    rates = df.groupby(col)["is_cancelled"].mean()
    return {
        "chi2": float(chi2),
        "p": float(p),
        "phi": phi,
        "odds_ratio": float(odds_ratio_val),
        "fisher_p": float(fisher_p),
        "rates": {str(k): float(v) for k, v in rates.items()},
    }


def run(df: pd.DataFrame) -> dict:
    """Run all bivariate analyses and return a results dict."""
    results: dict = {}

    # --- Temporal trend (Spearman on daily rate) ---
    daily = df[["date", "is_cancelled"]].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.groupby("date").agg(
        cancelled=("is_cancelled", "sum"),
        total=("is_cancelled", "count"),
    ).reset_index()
    daily["cancel_rate"] = daily["cancelled"] / daily["total"]
    daily["date_ordinal"] = daily["date"].map(pd.Timestamp.toordinal)

    rho, p = spearmanr(daily["date_ordinal"], daily["cancel_rate"])
    results["temporal_trend"] = {"spearman_rho": float(rho), "p_value": float(p)}

    # --- Temporal features chi-square ---
    df_t = df[["date", "time", "is_cancelled"]].copy()
    df_t["date"] = pd.to_datetime(df_t["date"])
    df_t["hour"] = pd.to_datetime(df_t["time"], format="%H:%M:%S").dt.hour
    df_t["weekday"] = df_t["date"].dt.dayofweek
    df_t["month"] = df_t["date"].dt.month

    temporal_tests = {}
    for col in ["hour", "weekday", "month"]:
        v = _cramers_v(df_t[col], df_t["is_cancelled"])
        chi2, p, _, _ = chi2_contingency(pd.crosstab(df_t[col], df_t["is_cancelled"]))
        temporal_tests[col] = {"cramers_v": v, "chi2": float(chi2), "p": float(p)}
    results["temporal_tests"] = temporal_tests

    # --- Vehicle type ---
    results["vehicle_type"] = {
        "cramers_v": _cramers_v(df["vehicle_type"], df["is_cancelled"]),
        "rates": {
            str(k): float(v)
            for k, v in df.groupby("vehicle_type", observed=True)["is_cancelled"]
            .mean()
            .items()
        },
    }

    # --- Locations ---
    results["pickup_location"] = {
        "cramers_v": _cramers_v(df["pickup_location"], df["is_cancelled"]),
    }
    results["drop_location"] = {
        "cramers_v": _cramers_v(df["drop_location"], df["is_cancelled"]),
    }

    pickup_rates = (
        df.groupby("pickup_location", observed=True)["is_cancelled"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    drop_rates = (
        df.groupby("drop_location", observed=True)["is_cancelled"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    results["pickup_location"]["top10"] = {
        str(k): {"rate": float(row["mean"]), "n": int(row["count"])}
        for k, row in pickup_rates.head(10).iterrows()
    }
    results["drop_location"]["top10"] = {
        str(k): {"rate": float(row["mean"]), "n": int(row["count"])}
        for k, row in drop_rates.head(10).iterrows()
    }

    # --- VTAT vs cancellation (the dominant predictor) ---
    vtat_complete = df.dropna(subset=["avg_vtat"])
    zones = [
        (2.0, 2.9, "instant"),
        (3.0, 5.0, "low"),
        (5.1, 11.9, "baseline"),
        (12.0, 15.0, "dip"),
        (15.1, 20.0, "timeout"),
    ]
    zone_results = {}
    for lo, hi, label in zones:
        mask = (vtat_complete["avg_vtat"] >= lo) & (vtat_complete["avg_vtat"] <= hi)
        zone = vtat_complete[mask]
        zone_results[label] = {
            "range": f"{lo:.1f}-{hi:.1f}",
            "n": int(len(zone)),
            "cancel_rate": float(zone["is_cancelled"].mean()) if len(zone) else 0.0,
        }
    results["vtat_zones"] = zone_results

    # Spearman on VTAT (continuous vs binary)
    rho_vtat, p_vtat = spearmanr(
        vtat_complete["avg_vtat"], vtat_complete["is_cancelled"]
    )
    results["vtat_spearman"] = {"rho": float(rho_vtat), "p": float(p_vtat)}

    # --- vtat_missing signal ---
    df_m = df.copy()
    df_m["vtat_missing"] = df_m["avg_vtat"].isna().astype(int)
    results["vtat_missing"] = _binary_vs_target(df_m, "vtat_missing")

    return results
