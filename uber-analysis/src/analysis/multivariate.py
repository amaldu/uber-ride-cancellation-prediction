"""Multivariate EDA: feature interactions, redundancy, and correlation structure.

Reproduces the curated highlights from 06_multivar_eda.ipynb.
Returns a structured results dict consumed by the report builder.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def _missingness_confounding(df: pd.DataFrame) -> dict:
    """Test whether non-VTAT features predict vtat missingness beyond is_cancelled."""
    non_vtat_cols = [
        "weekday", "month", "day_of_month", "week_of_year", "quarter",
        "hour", "is_night", "is_business", "is_rush",
        "vehicle_type", "pickup_location", "drop_location",
        "is_cancelled",
    ]
    available = [c for c in non_vtat_cols if c in df.columns]
    X = df[available].copy()
    for c in X.select_dtypes(include="category").columns:
        X[c] = LabelEncoder().fit_transform(X[c])

    y = df["vtat_missing"]

    lr_full = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr_full.fit(X, y)
    auc_full = roc_auc_score(y, lr_full.predict_proba(X)[:, 1])

    lr_base = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr_base.fit(df[["is_cancelled"]], y)
    auc_base = roc_auc_score(y, lr_base.predict_proba(df[["is_cancelled"]])[:, 1])

    return {
        "auc_baseline": float(auc_base),
        "auc_full": float(auc_full),
        "auc_lift": float(auc_full - auc_base),
    }


def _vtat_family_redundancy(df: pd.DataFrame) -> dict:
    """Measure overlap between avg_vtat and vtat_zone."""
    complete = df.dropna(subset=["avg_vtat"]).copy()
    groups = [
        g["avg_vtat"].values
        for _, g in complete.groupby("vtat_zone", observed=True)
    ]
    ss_bw = sum(len(g) * (g.mean() - complete["avg_vtat"].mean()) ** 2 for g in groups)
    ss_tot = ((complete["avg_vtat"] - complete["avg_vtat"].mean()) ** 2).sum()
    eta_sq = float(ss_bw / ss_tot)

    zone_codes = complete["vtat_zone"].cat.codes.values.astype(int)
    mi = mutual_info_classif(
        complete[["avg_vtat"]].values, zone_codes,
        discrete_features=[False], random_state=42, n_neighbors=5,
    )[0]

    cv = _cramers_v(df["vtat_zone"].astype(str), df["vtat_missing"].astype(str))

    return {
        "avg_vtat_vs_vtat_zone_eta_sq": eta_sq,
        "avg_vtat_vs_vtat_zone_mi": float(mi),
        "vtat_zone_vs_vtat_missing_cramers_v": cv,
    }


def _route_validation(df: pd.DataFrame) -> dict:
    """Cross-validated test of whether route adds signal beyond individual locations."""
    te_df = df[
        ["avg_vtat", "vtat_missing", "vehicle_type", "pickup_location",
         "drop_location", "is_cancelled"]
    ].copy()
    te_df["route"] = (
        te_df["pickup_location"].astype(str) + "_" + te_df["drop_location"].astype(str)
    )
    te_df["avg_vtat"] = te_df["avg_vtat"].fillna(te_df["avg_vtat"].median())

    for col in ["vehicle_type", "pickup_location", "drop_location"]:
        freq = te_df[col].value_counts()
        te_df[f"{col}_freq"] = te_df[col].map(freq).astype(float)

    base_cols = [
        "avg_vtat", "vtat_missing",
        "vehicle_type_freq", "pickup_location_freq", "drop_location_freq",
    ]
    X_base = te_df[base_cols].values
    y = te_df["is_cancelled"].values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    route_n_unique = te_df["route"].nunique()

    results_by_m: dict[str, list[float]] = {"baseline": []}
    for m in [5, 20, 50]:
        results_by_m[f"route_m{m}"] = []

    for train_idx, test_idx in kf.split(X_base, y):
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X_base[train_idx], y[train_idx])
        results_by_m["baseline"].append(
            roc_auc_score(y[test_idx], lr.predict_proba(X_base[test_idx])[:, 1])
        )

        for m in [5, 20, 50]:
            global_mean = te_df["is_cancelled"].iloc[train_idx].mean()
            agg = (
                te_df.iloc[train_idx]
                .groupby("route")["is_cancelled"]
                .agg(["mean", "count"])
            )
            agg["smoothed"] = (
                (agg["count"] * agg["mean"] + m * global_mean) / (agg["count"] + m)
            )
            enc_map = agg["smoothed"].to_dict()

            train_route = te_df["route"].iloc[train_idx].map(enc_map).fillna(global_mean).values
            test_route = te_df["route"].iloc[test_idx].map(enc_map).fillna(global_mean).values

            X_tr = np.column_stack([X_base[train_idx], train_route])
            X_te = np.column_stack([X_base[test_idx], test_route])

            lr_r = LogisticRegression(max_iter=1000, solver="lbfgs")
            lr_r.fit(X_tr, y[train_idx])
            results_by_m[f"route_m{m}"].append(
                roc_auc_score(y[test_idx], lr_r.predict_proba(X_te)[:, 1])
            )

    base_mean = np.mean(results_by_m["baseline"])
    summary = {}
    for key, aucs in results_by_m.items():
        summary[key] = {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "lift": float(np.mean(aucs) - base_mean),
        }
    summary["route_unique_values"] = route_n_unique
    return summary


def _correlation_and_vif(df: pd.DataFrame) -> dict:
    """Pearson correlation matrix + VIF for numeric features."""
    corr_df = df[
        ["avg_vtat", "vtat_missing", "vehicle_type", "pickup_location", "drop_location"]
    ].copy()

    for col in ["vehicle_type", "pickup_location", "drop_location"]:
        freq = corr_df[col].value_counts()
        corr_df[f"{col}_freq"] = corr_df[col].map(freq).astype(float)
    corr_df = corr_df.drop(columns=["vehicle_type", "pickup_location", "drop_location"])
    corr_df["avg_vtat"] = corr_df["avg_vtat"].fillna(corr_df["avg_vtat"].median())

    corr_matrix = corr_df.corr(method="pearson")

    vif_df = corr_df.copy()
    vif_df.insert(0, "const", 1.0)
    vif_results = {}
    for i in range(1, vif_df.shape[1]):
        vif_results[vif_df.columns[i]] = float(
            variance_inflation_factor(vif_df.values, i)
        )

    return {
        "correlation": {
            col: {col2: float(corr_matrix.loc[col, col2]) for col2 in corr_matrix.columns}
            for col in corr_matrix.index
        },
        "vif": vif_results,
    }


def run(df: pd.DataFrame) -> dict:
    """Run all multivariate analyses and return a results dict.

    Expects the enriched DataFrame from the bivariate stage (with vtat_zone,
    vtat_missing, temporal columns, etc.).
    """
    results: dict = {}

    results["missingness_confounding"] = _missingness_confounding(df)
    results["vtat_family"] = _vtat_family_redundancy(df)
    results["route_validation"] = _route_validation(df)
    results["correlation_vif"] = _correlation_and_vif(df)

    return results
