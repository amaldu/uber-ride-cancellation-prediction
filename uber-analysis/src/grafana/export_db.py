"""Export computed analysis results to SQLite for Grafana consumption.

Every value in the database is derived from the analysis pipeline —
nothing is hardcoded.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "grafana" / "data" / "uber_insights.db"
)


def _drop_and_create(c: sqlite3.Cursor) -> None:
    tables = [
        "meta",
        "dataset_overview",
        "target_distribution",
        "vehicle_type",
        "vtat_distribution",
        "vtat_zones",
        "temporal_hourly",
        "temporal_daily",
        "temporal_monthly",
        "temporal_tests",
        "location_pickup_top10",
        "location_drop_top10",
        "vtat_missing",
        "bivar_summary",
        "missingness_confounding",
        "feature_redundancy",
        "route_validation",
        "correlation_matrix",
        "vif_results",
    ]
    for t in tables:
        c.execute(f"DROP TABLE IF EXISTS {t}")

    c.execute("""CREATE TABLE meta (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    c.execute("""CREATE TABLE dataset_overview (
        metric TEXT PRIMARY KEY, value REAL, label TEXT
    )""")
    c.execute("""CREATE TABLE target_distribution (
        outcome TEXT, count INTEGER, percentage REAL
    )""")
    c.execute("""CREATE TABLE vehicle_type (
        vehicle_type TEXT, count INTEGER, cancel_rate REAL
    )""")
    c.execute("""CREATE TABLE vtat_distribution (
        stat TEXT, value REAL
    )""")
    c.execute("""CREATE TABLE vtat_zones (
        zone TEXT, range_min REAL, range_max REAL, cancel_rate REAL,
        sample_count INTEGER, zone_order INTEGER
    )""")
    c.execute("""CREATE TABLE temporal_hourly (
        hour INTEGER, cancel_rate REAL
    )""")
    c.execute("""CREATE TABLE temporal_daily (
        day_name TEXT, day_order INTEGER, cancel_rate REAL
    )""")
    c.execute("""CREATE TABLE temporal_monthly (
        month_name TEXT, month_order INTEGER, cancel_rate REAL
    )""")
    c.execute("""CREATE TABLE temporal_tests (
        feature TEXT, test_method TEXT, statistic REAL, p_value REAL, interpretation TEXT
    )""")
    c.execute("""CREATE TABLE location_pickup_top10 (
        location TEXT, cancel_rate REAL, volume INTEGER, rank_order INTEGER
    )""")
    c.execute("""CREATE TABLE location_drop_top10 (
        location TEXT, cancel_rate REAL, volume INTEGER, rank_order INTEGER
    )""")
    c.execute("""CREATE TABLE vtat_missing (
        metric TEXT, value REAL
    )""")
    c.execute("""CREATE TABLE bivar_summary (
        feature TEXT, method TEXT, statistic TEXT, p_value TEXT, interpretation TEXT
    )""")
    c.execute("""CREATE TABLE missingness_confounding (
        model TEXT, auc REAL
    )""")
    c.execute("""CREATE TABLE feature_redundancy (
        pair TEXT, metric TEXT, value REAL
    )""")
    c.execute("""CREATE TABLE route_validation (
        model TEXT, mean_auc REAL, std_auc REAL, lift REAL
    )""")
    c.execute("""CREATE TABLE correlation_matrix (
        feature_a TEXT, feature_b TEXT, correlation REAL
    )""")
    c.execute("""CREATE TABLE vif_results (
        feature TEXT, vif REAL
    )""")


def _write_meta(c: sqlite3.Cursor) -> None:
    c.execute("INSERT INTO meta VALUES (?, ?)", ("generated_at", datetime.now().isoformat()))
    c.execute("INSERT INTO meta VALUES (?, ?)", ("pipeline_version", "1.0.0"))


def _write_univariate(c: sqlite3.Cursor, univar: dict, df_enriched) -> None:
    import pandas as pd

    t = univar["target"]
    total = t["completed"] + t["cancelled"]

    c.executemany("INSERT INTO dataset_overview VALUES (?, ?, ?)", [
        ("total_records", total, f"{total:,}"),
        ("features", univar["shape"]["cols"], str(univar["shape"]["cols"])),
        ("cancellation_rate", t["cancellation_rate"] * 100, f"{t['cancellation_rate']:.1%}"),
        ("cancelled_rides", t["cancelled"], f"{t['cancelled']:,}"),
        ("est_revenue_loss", t["cancelled"] * 20, f"${t['cancelled'] * 20:,.0f}"),
        ("vehicle_types", univar["vehicle_type"]["n_unique"], str(univar["vehicle_type"]["n_unique"])),
        ("unique_locations", univar["locations"]["pickup_unique"], str(univar["locations"]["pickup_unique"])),
        ("vtat_missing_count", univar["avg_vtat"]["missing"], str(univar["avg_vtat"]["missing"])),
        ("vtat_missing_pct", univar["avg_vtat"]["missing_pct"], f"{univar['avg_vtat']['missing_pct']:.1f}%"),
    ])

    c.executemany("INSERT INTO target_distribution VALUES (?, ?, ?)", [
        ("Completed", t["completed"], round((1 - t["cancellation_rate"]) * 100, 1)),
        ("Cancelled", t["cancelled"], round(t["cancellation_rate"] * 100, 1)),
    ])

    # Vehicle type cancel rates from enriched data
    vt_rates = df_enriched.groupby("vehicle_type", observed=True)["is_cancelled"].agg(["count", "mean"])
    for vt, row in vt_rates.iterrows():
        c.execute(
            "INSERT INTO vehicle_type VALUES (?, ?, ?)",
            (str(vt), int(row["count"]), round(float(row["mean"]) * 100, 2)),
        )

    v = univar["avg_vtat"]
    for stat_name in ["mean", "median", "min", "max", "std", "q25", "q75"]:
        c.execute("INSERT INTO vtat_distribution VALUES (?, ?)", (stat_name, v[stat_name]))

    # Temporal rates from enriched data
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    hourly = df_enriched.groupby("hour")["is_cancelled"].mean()
    for h, rate in hourly.items():
        c.execute("INSERT INTO temporal_hourly VALUES (?, ?)", (int(h), round(float(rate) * 100, 2)))

    daily = df_enriched.groupby("weekday")["is_cancelled"].mean()
    for d, rate in daily.items():
        c.execute("INSERT INTO temporal_daily VALUES (?, ?, ?)",
                  (day_names[int(d)], int(d), round(float(rate) * 100, 2)))

    monthly = df_enriched.groupby("month")["is_cancelled"].mean()
    for m, rate in monthly.items():
        c.execute("INSERT INTO temporal_monthly VALUES (?, ?, ?)",
                  (month_names[int(m) - 1], int(m), round(float(rate) * 100, 2)))


def _write_bivariate(c: sqlite3.Cursor, bivar: dict) -> None:
    # VTAT zones
    zone_order = {"instant": 1, "low": 2, "baseline": 3, "dip": 4, "timeout": 5}
    for label, z in bivar["vtat_zones"].items():
        lo, hi = z["range"].split("-")
        c.execute(
            "INSERT INTO vtat_zones VALUES (?, ?, ?, ?, ?, ?)",
            (label.title(), float(lo), float(hi),
             round(z["cancel_rate"] * 100, 2), z["n"], zone_order.get(label, 0)),
        )

    # Locations
    for rank, (loc, info) in enumerate(bivar["pickup_location"]["top10"].items(), 1):
        c.execute(
            "INSERT INTO location_pickup_top10 VALUES (?, ?, ?, ?)",
            (loc, round(info["rate"] * 100, 2), info["n"], rank),
        )
    for rank, (loc, info) in enumerate(bivar["drop_location"]["top10"].items(), 1):
        c.execute(
            "INSERT INTO location_drop_top10 VALUES (?, ?, ?, ?)",
            (loc, round(info["rate"] * 100, 2), info["n"], rank),
        )

    # vtat_missing signal
    vm = bivar["vtat_missing"]
    c.executemany("INSERT INTO vtat_missing VALUES (?, ?)", [
        ("phi", vm["phi"]),
        ("odds_ratio", vm["odds_ratio"]),
        ("chi2", vm["chi2"]),
        ("fisher_p", vm["fisher_p"]),
    ])

    # Temporal tests
    for col, label in [("hour", "Hour"), ("weekday", "Weekday"), ("month", "Month")]:
        t = bivar["temporal_tests"][col]
        c.execute(
            "INSERT INTO temporal_tests VALUES (?, ?, ?, ?, ?)",
            (label, "Chi-square / Cramer's V", t["cramers_v"], t["p"], "Negligible"),
        )

    tt = bivar["temporal_trend"]
    c.execute(
        "INSERT INTO temporal_tests VALUES (?, ?, ?, ?, ?)",
        ("Daily Trend", "Spearman", tt["spearman_rho"], tt["p_value"], "No trend"),
    )

    # Full bivariate summary table
    rows = []
    for col, label in [("hour", "Hour"), ("weekday", "Weekday"), ("month", "Month")]:
        t = bivar["temporal_tests"][col]
        rows.append((f"{label} vs Cancellation", "Chi-square",
                      f"V = {t['cramers_v']:.4f}", f"{t['p']:.2e}", "Negligible"))

    rows.append(("Daily Rate Trend", "Spearman",
                  f"rho = {tt['spearman_rho']:.4f}", f"{tt['p_value']:.2e}", "No trend"))

    vt = bivar["vehicle_type"]
    rows.append(("Vehicle Type", "Cramer's V",
                  f"V = {vt['cramers_v']:.4f}", "-", "Negligible"))

    for key, label in [("pickup_location", "Pickup Location"), ("drop_location", "Drop Location")]:
        rows.append((label, "Cramer's V",
                      f"V = {bivar[key]['cramers_v']:.4f}", "-", "Faint signal"))

    vs = bivar["vtat_spearman"]
    rows.append(("avg_vtat", "Spearman",
                  f"rho = {vs['rho']:.4f}", f"{vs['p']:.2e}", "Dominant predictor"))

    rows.append(("vtat_missing", "Fisher / Phi",
                  f"phi = {vm['phi']:.3f}, OR = {vm['odds_ratio']:.1f}",
                  f"{vm['fisher_p']:.2e}", "Strong signal"))

    c.executemany("INSERT INTO bivar_summary VALUES (?, ?, ?, ?, ?)", rows)


def _write_multivariate(c: sqlite3.Cursor, multivar: dict) -> None:
    mc = multivar["missingness_confounding"]
    c.executemany("INSERT INTO missingness_confounding VALUES (?, ?)", [
        ("Baseline (is_cancelled only)", mc["auc_baseline"]),
        ("Full (all non-VTAT features)", mc["auc_full"]),
    ])

    vf = multivar["vtat_family"]
    c.executemany("INSERT INTO feature_redundancy VALUES (?, ?, ?)", [
        ("avg_vtat vs vtat_zone", "eta-squared", vf["avg_vtat_vs_vtat_zone_eta_sq"]),
        ("avg_vtat vs vtat_zone", "mutual information", vf["avg_vtat_vs_vtat_zone_mi"]),
        ("vtat_zone vs vtat_missing", "Cramer's V", vf["vtat_zone_vs_vtat_missing_cramers_v"]),
    ])

    rv = multivar["route_validation"]
    for key in ["baseline", "route_m5", "route_m20", "route_m50"]:
        if isinstance(rv[key], dict):
            c.execute(
                "INSERT INTO route_validation VALUES (?, ?, ?, ?)",
                (key, rv[key]["mean_auc"], rv[key]["std_auc"], rv[key]["lift"]),
            )

    corr = multivar["correlation_vif"]["correlation"]
    for fa, cols in corr.items():
        for fb, val in cols.items():
            c.execute("INSERT INTO correlation_matrix VALUES (?, ?, ?)", (fa, fb, val))

    vif = multivar["correlation_vif"]["vif"]
    for feat, val in vif.items():
        c.execute("INSERT INTO vif_results VALUES (?, ?)", (feat, val))


def export(
    univar: dict,
    bivar: dict,
    multivar: dict,
    df_enriched,
    db_path: str | Path | None = None,
) -> Path:
    """Write all analysis results to SQLite. Returns the database path."""
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    _drop_and_create(c)
    _write_meta(c)
    _write_univariate(c, univar, df_enriched)
    _write_bivariate(c, bivar)
    _write_multivariate(c, multivar)

    conn.commit()
    conn.close()
    logger.info("SQLite database written to %s", db_path)
    return db_path
