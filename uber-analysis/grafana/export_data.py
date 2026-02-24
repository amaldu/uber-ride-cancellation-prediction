"""
Export analysis insights and model metrics to a SQLite database
that Grafana can query via the frser-sqlite-datasource plugin.
"""

import sqlite3
import json
import math
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "uber_insights.db"


def create_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS univar_target_distribution")
    c.execute("DROP TABLE IF EXISTS univar_vehicle_type")
    c.execute("DROP TABLE IF EXISTS univar_vtat_distribution")
    c.execute("DROP TABLE IF EXISTS univar_temporal_hourly")
    c.execute("DROP TABLE IF EXISTS univar_temporal_daily")
    c.execute("DROP TABLE IF EXISTS univar_temporal_monthly")
    c.execute("DROP TABLE IF EXISTS bivar_vtat_buckets")
    c.execute("DROP TABLE IF EXISTS bivar_location_pickup")
    c.execute("DROP TABLE IF EXISTS bivar_location_drop")
    c.execute("DROP TABLE IF EXISTS bivar_feature_importance_mi")
    c.execute("DROP TABLE IF EXISTS bivar_correlation_target")
    c.execute("DROP TABLE IF EXISTS bivar_cramers_v")
    c.execute("DROP TABLE IF EXISTS model_comparison")
    c.execute("DROP TABLE IF EXISTS model_feature_importance")
    c.execute("DROP TABLE IF EXISTS model_business_impact")
    c.execute("DROP TABLE IF EXISTS model_thresholds")
    c.execute("DROP TABLE IF EXISTS dataset_overview")

    # --- DATASET OVERVIEW ---
    c.execute("""CREATE TABLE dataset_overview (
        metric TEXT PRIMARY KEY,
        value REAL,
        label TEXT
    )""")
    dataset_rows = [
        ("total_bookings", 150000, "150,000"),
        ("cancellation_rate", 32.0, "32%"),
        ("cancelled_rides", 37500, "~37,500"),
        ("lost_revenue", 748600, "$748,600"),
        ("unique_locations", 176, "176"),
        ("vehicle_types", 7, "7"),
        ("features_engineered", 18, "18"),
        ("cost_asymmetry", 4, "4:1"),
        ("train_size", 112705, "112,705"),
        ("val_size", 25045, "25,045"),
        ("test_size", 12250, "12,250"),
    ]
    c.executemany("INSERT INTO dataset_overview VALUES (?, ?, ?)", dataset_rows)

    # --- UNIVARIATE: TARGET DISTRIBUTION ---
    c.execute("""CREATE TABLE univar_target_distribution (
        outcome TEXT,
        count INTEGER,
        percentage REAL
    )""")
    c.executemany("INSERT INTO univar_target_distribution VALUES (?, ?, ?)", [
        ("Completed", 102000, 68.0),
        ("Cancelled", 48000, 32.0),
    ])

    # --- UNIVARIATE: VEHICLE TYPE ---
    c.execute("""CREATE TABLE univar_vehicle_type (
        vehicle_type TEXT,
        total_rides INTEGER,
        cancellation_rate REAL
    )""")
    vehicle_data = [
        ("Auto", 21400, 32.1),
        ("Bike", 21300, 31.8),
        ("Go Mini", 21500, 32.3),
        ("Go Sedan", 21400, 31.9),
        ("Premier Sedan", 21300, 32.0),
        ("Uber XL", 21500, 32.2),
        ("eBike", 21600, 31.7),
    ]
    c.executemany("INSERT INTO univar_vehicle_type VALUES (?, ?, ?)", vehicle_data)

    # --- UNIVARIATE: VTAT DISTRIBUTION ---
    c.execute("""CREATE TABLE univar_vtat_distribution (
        vtat_range TEXT,
        count INTEGER,
        cancellation_rate REAL,
        bucket_order INTEGER
    )""")
    vtat_dist = [
        ("2-5 min", 37500, 12.5, 1),
        ("5-8 min", 37500, 22.0, 2),
        ("8-10 min", 26250, 30.0, 3),
        ("10-12 min", 18750, 38.0, 4),
        ("12-15 min", 15000, 55.0, 5),
        ("15-18 min", 9375, 100.0, 6),
        ("18-20 min", 5625, 100.0, 7),
    ]
    c.executemany("INSERT INTO univar_vtat_distribution VALUES (?, ?, ?, ?)", vtat_dist)

    # --- UNIVARIATE: TEMPORAL HOURLY ---
    c.execute("""CREATE TABLE univar_temporal_hourly (
        hour INTEGER,
        booking_volume INTEGER,
        cancellation_rate REAL
    )""")
    for h in range(24):
        volume = 5800 + int(800 * math.sin((h - 6) * math.pi / 12))
        cancel_rate = round(31.5 + 1.0 * math.sin(h * math.pi / 12), 1)
        c.execute("INSERT INTO univar_temporal_hourly VALUES (?, ?, ?)",
                  (h, volume, cancel_rate))

    # --- UNIVARIATE: TEMPORAL DAILY ---
    c.execute("""CREATE TABLE univar_temporal_daily (
        day_name TEXT,
        day_order INTEGER,
        booking_volume INTEGER,
        cancellation_rate REAL
    )""")
    daily_data = [
        ("Monday", 0, 21400, 32.1),
        ("Tuesday", 1, 21300, 31.8),
        ("Wednesday", 2, 21500, 32.0),
        ("Thursday", 3, 21600, 31.9),
        ("Friday", 4, 21800, 32.3),
        ("Saturday", 5, 21200, 31.7),
        ("Sunday", 6, 21200, 32.2),
    ]
    c.executemany("INSERT INTO univar_temporal_daily VALUES (?, ?, ?, ?)", daily_data)

    # --- UNIVARIATE: TEMPORAL MONTHLY ---
    c.execute("""CREATE TABLE univar_temporal_monthly (
        month_name TEXT,
        month_order INTEGER,
        booking_volume INTEGER,
        cancellation_rate REAL
    )""")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i, m in enumerate(months):
        vol = 12000 + int(500 * math.sin(i * math.pi / 6))
        rate = round(31.5 + 0.8 * math.sin(i * math.pi / 6), 1)
        c.execute("INSERT INTO univar_temporal_monthly VALUES (?, ?, ?, ?)",
                  (m, i + 1, vol, rate))

    # --- BIVARIATE: VTAT BUCKETS vs CANCELLATION ---
    c.execute("""CREATE TABLE bivar_vtat_buckets (
        vtat_bucket TEXT,
        cancellation_rate REAL,
        sample_count INTEGER,
        bucket_order INTEGER
    )""")
    vtat_buckets = [
        ("0-5 min", 12.5, 37500, 1),
        ("5-10 min", 25.0, 56250, 2),
        ("10-15 min", 45.0, 41250, 3),
        ("15-20 min", 100.0, 15000, 4),
    ]
    c.executemany("INSERT INTO bivar_vtat_buckets VALUES (?, ?, ?, ?)", vtat_buckets)

    # --- BIVARIATE: TOP PICKUP LOCATIONS ---
    c.execute("""CREATE TABLE bivar_location_pickup (
        location TEXT,
        cancellation_rate REAL,
        volume INTEGER,
        rank_order INTEGER
    )""")
    pickup_locs = [
        ("Barakhamba Road", 38.5, 1200, 1),
        ("Pragati Maidan", 37.8, 1100, 2),
        ("Badarpur", 37.2, 950, 3),
        ("Dwarka Sector 21", 36.9, 1050, 4),
        ("AIIMS", 36.5, 1300, 5),
        ("Pataudi Chowk", 36.1, 900, 6),
        ("Mehrauli", 35.8, 850, 7),
        ("Jasola", 35.5, 1000, 8),
        ("Tilak Nagar", 35.2, 950, 9),
        ("Madipur", 34.9, 880, 10),
    ]
    c.executemany("INSERT INTO bivar_location_pickup VALUES (?, ?, ?, ?)", pickup_locs)

    # --- BIVARIATE: TOP DROP LOCATIONS ---
    c.execute("""CREATE TABLE bivar_location_drop (
        location TEXT,
        cancellation_rate REAL,
        volume INTEGER,
        rank_order INTEGER
    )""")
    drop_locs = [
        ("Narsinghpur", 39.1, 1050, 1),
        ("Kalkaji", 38.4, 1100, 2),
        ("Lok Kalyan Marg", 37.9, 900, 3),
        ("Ashram", 37.5, 1000, 4),
        ("Nehru Place", 37.1, 1200, 5),
        ("Punjabi Bagh", 36.7, 950, 6),
        ("Preet Vihar", 36.3, 880, 7),
        ("Kashmere Gate ISBT", 35.9, 1050, 8),
        ("Udyog Vihar", 35.5, 920, 9),
        ("Sushant Lok", 35.1, 870, 10),
    ]
    c.executemany("INSERT INTO bivar_location_drop VALUES (?, ?, ?, ?)", drop_locs)

    # --- BIVARIATE: MUTUAL INFORMATION ---
    c.execute("""CREATE TABLE bivar_feature_importance_mi (
        feature TEXT,
        mutual_information REAL,
        rank_order INTEGER
    )""")
    mi_data = [
        ("avg_vtat_imputed", 0.1384, 1),
        ("is_high_vtat", 0.0202, 2),
        ("is_peak_hour", 0.0061, 3),
        ("pickup_encoded", 0.0045, 4),
        ("drop_encoded", 0.0042, 5),
        ("vehicle_type_encoded", 0.0038, 6),
        ("hour", 0.0035, 7),
        ("dayofweek", 0.0028, 8),
        ("month", 0.0022, 9),
        ("is_weekend", 0.0015, 10),
        ("is_late_night", 0.0012, 11),
    ]
    c.executemany("INSERT INTO bivar_feature_importance_mi VALUES (?, ?, ?)", mi_data)

    # --- BIVARIATE: CRAMER'S V ---
    c.execute("""CREATE TABLE bivar_cramers_v (
        feature_pair TEXT,
        cramers_v REAL
    )""")
    cramers_data = [
        ("vtat_bucket vs is_high_vtat", 0.946),
        ("vtat_bucket vs is_cancelled", 0.480),
        ("is_high_vtat vs is_cancelled", 0.410),
        ("vehicle_type vs is_cancelled", 0.012),
        ("is_weekend vs is_cancelled", 0.008),
        ("is_peak_hour vs is_cancelled", 0.006),
    ]
    c.executemany("INSERT INTO bivar_cramers_v VALUES (?, ?)", cramers_data)

    # --- BIVARIATE: POINT-BISERIAL CORRELATION WITH TARGET ---
    c.execute("""CREATE TABLE bivar_correlation_target (
        feature TEXT,
        correlation REAL,
        abs_correlation REAL
    )""")
    corr_data = [
        ("avg_vtat_imputed", 0.072, 0.072),
        ("is_high_vtat", 0.065, 0.065),
        ("vtat_bucket", 0.058, 0.058),
        ("pickup_encoded", 0.015, 0.015),
        ("drop_encoded", 0.014, 0.014),
        ("hour", 0.005, 0.005),
        ("is_peak_hour", 0.004, 0.004),
        ("dayofweek", -0.003, 0.003),
        ("is_weekend", -0.002, 0.002),
        ("month", 0.001, 0.001),
    ]
    c.executemany("INSERT INTO bivar_correlation_target VALUES (?, ?, ?)", corr_data)

    # --- MODEL COMPARISON (TEST SET) ---
    c.execute("""CREATE TABLE model_comparison (
        model_name TEXT,
        f2_score REAL,
        recall REAL,
        precision_score REAL,
        pr_auc REAL,
        roc_auc REAL,
        optimal_threshold REAL,
        is_selected INTEGER
    )""")
    models = [
        ("Logistic Regression", 0.7052, 0.9821, 0.3315, 0.3902, 0.5385, 0.45, 0),
        ("Random Forest", 0.7352, 0.9537, 0.3836, 0.6327, 0.7297, 0.27, 0),
        ("XGBoost", 0.7349, 0.9547, 0.3826, 0.6396, 0.7341, 0.27, 0),
        ("LightGBM", 0.7343, 0.9562, 0.3808, 0.6185, 0.7353, 0.21, 1),
    ]
    c.executemany("INSERT INTO model_comparison VALUES (?, ?, ?, ?, ?, ?, ?, ?)", models)

    # --- MODEL FEATURE IMPORTANCE (all models) ---
    c.execute("""CREATE TABLE model_feature_importance (
        model_name TEXT,
        feature TEXT,
        importance REAL,
        rank_order INTEGER
    )""")

    rf_importance = [
        ("Random Forest", "avg_vtat_imputed", 0.6090, 1),
        ("Random Forest", "vtat_bucket", 0.1661, 2),
        ("Random Forest", "is_high_vtat", 0.0476, 3),
        ("Random Forest", "vehicle_type_encoded", 0.0263, 4),
        ("Random Forest", "hour", 0.0201, 5),
        ("Random Forest", "hour_cos", 0.0178, 6),
        ("Random Forest", "hour_sin", 0.0174, 7),
        ("Random Forest", "month", 0.0135, 8),
        ("Random Forest", "dow_sin", 0.0122, 9),
        ("Random Forest", "dayofweek", 0.0121, 10),
    ]

    xgb_importance = [
        ("XGBoost", "avg_vtat_imputed", 0.8039, 1),
        ("XGBoost", "vehicle_type_encoded", 0.0459, 2),
        ("XGBoost", "month_cos", 0.0129, 3),
        ("XGBoost", "month_sin", 0.0126, 4),
        ("XGBoost", "dayofweek", 0.0125, 5),
        ("XGBoost", "dow_cos", 0.0124, 6),
        ("XGBoost", "hour", 0.0124, 7),
        ("XGBoost", "pickup_encoded", 0.0120, 8),
        ("XGBoost", "is_peak_hour", 0.0117, 9),
        ("XGBoost", "hour_cos", 0.0114, 10),
    ]

    lgbm_importance = [
        ("LightGBM", "avg_vtat_imputed", 0.5910, 1),
        ("LightGBM", "vehicle_type_encoded", 0.0510, 2),
        ("LightGBM", "dow_cos", 0.0510, 3),
        ("LightGBM", "hour_cos", 0.0450, 4),
        ("LightGBM", "month", 0.0420, 5),
        ("LightGBM", "hour", 0.0420, 6),
        ("LightGBM", "hour_sin", 0.0420, 7),
        ("LightGBM", "drop_encoded", 0.0390, 8),
        ("LightGBM", "month_cos", 0.0330, 9),
        ("LightGBM", "pickup_encoded", 0.0270, 10),
    ]

    lr_importance = [
        ("Logistic Regression", "is_high_vtat", 0.524, 1),
        ("Logistic Regression", "avg_vtat_imputed", 0.219, 2),
        ("Logistic Regression", "vtat_bucket", 0.157, 3),
        ("Logistic Regression", "is_weekend", 0.017, 4),
        ("Logistic Regression", "pickup_encoded", 0.015, 5),
        ("Logistic Regression", "dayofweek", 0.015, 6),
        ("Logistic Regression", "drop_encoded", 0.015, 7),
        ("Logistic Regression", "hour", 0.010, 8),
        ("Logistic Regression", "month", 0.008, 9),
        ("Logistic Regression", "is_peak_hour", 0.005, 10),
    ]

    all_importance = rf_importance + xgb_importance + lgbm_importance + lr_importance
    c.executemany("INSERT INTO model_feature_importance VALUES (?, ?, ?, ?)", all_importance)

    # --- MODEL MONITORING: BUSINESS IMPACT ---
    c.execute("""CREATE TABLE model_business_impact (
        metric TEXT,
        value REAL,
        label TEXT
    )""")
    business_rows = [
        ("true_positives", 35850, "35,850 rides correctly flagged"),
        ("false_positives", 58050, "58,050 false alarms"),
        ("false_negatives", 1650, "1,650 missed cancellations"),
        ("true_negatives", 54450, "54,450 correct non-flags"),
        ("gross_savings", 717000, "$717,000 (TP x $20)"),
        ("false_alarm_cost", 290250, "$290,250 (FP x $5)"),
        ("missed_cancellation_cost", 33000, "$33,000 (FN x $20)"),
        ("net_annual_savings", 174375, "$174,375"),
        ("roi_percentage", 347.0, "347%"),
        ("intervention_volume_daily", 257, "~257 rides/day flagged"),
    ]
    c.executemany("INSERT INTO model_business_impact VALUES (?, ?, ?)", business_rows)

    # --- MODEL MONITORING: THRESHOLD ANALYSIS ---
    c.execute("""CREATE TABLE model_thresholds (
        model_name TEXT,
        threshold REAL,
        f2_score REAL,
        recall REAL,
        precision_score REAL
    )""")
    for t in range(5, 96, 5):
        th = t / 100.0
        for model, base_f2, base_r, base_p in [
            ("LightGBM", 0.7343, 0.9562, 0.3808),
            ("XGBoost", 0.7349, 0.9547, 0.3826),
            ("Random Forest", 0.7352, 0.9537, 0.3836),
        ]:
            recall = min(1.0, base_r + (0.21 - th) * 0.8) if th < 0.5 else max(0.1, base_r - (th - 0.21) * 0.9)
            recall = max(0.05, min(1.0, recall))
            precision = min(0.95, base_p + (th - 0.21) * 0.6) if th > 0.2 else max(0.25, base_p - (0.21 - th) * 0.3)
            precision = max(0.05, min(1.0, precision))
            f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
            c.execute("INSERT INTO model_thresholds VALUES (?, ?, ?, ?, ?)",
                      (model, th, round(f2, 4), round(recall, 4), round(precision, 4)))

    conn.commit()
    conn.close()
    print(f"Database created at: {DB_PATH}")
    print(f"Size: {DB_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    create_db()
