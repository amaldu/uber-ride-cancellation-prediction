"""Generate Grafana dashboard JSON for Uber Ride Cancellation Analysis."""
import json
from pathlib import Path

OUTPUT = Path(__file__).parent / "provisioning" / "dashboards" / "uber_analysis.json"

UID = "uber-ride-analysis"
DS = {"type": "frser-sqlite-datasource", "uid": "P2D2EEF3E092AF52B"}


def _target(sql, ref="A"):
    return {"rawQueryText": sql, "queryText": sql, "queryType": "table", "refId": ref}


def stat_panel(title, sql, x, y, w=6, h=4, unit="", thresholds=None, color="blue"):
    p = {
        "type": "stat", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [_target(sql)],
        "fieldConfig": {"defaults": {"unit": unit, "color": {"mode": "fixed", "fixedColor": color}},
                        "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background", "graphMode": "none"},
    }
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = thresholds
        p["fieldConfig"]["defaults"]["color"] = {"mode": "thresholds"}
        p["options"]["colorMode"] = "background"
    return p


def bar_panel(title, sql, x, y, w=12, h=8, orient="horizontal", color=None, overrides=None):
    p = {
        "type": "barchart", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [_target(sql)],
        "options": {"orientation": orient, "showValue": "auto", "barWidth": 0.7,
                    "groupWidth": 0.75, "stacking": "none", "tooltip": {"mode": "multi"}},
        "fieldConfig": {"defaults": {}, "overrides": overrides or []},
    }
    if color:
        p["fieldConfig"]["defaults"]["color"] = {"mode": "fixed", "fixedColor": color}
    return p


def table_panel(title, sql, x, y, w=12, h=8, overrides=None):
    return {
        "type": "table", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [_target(sql)],
        "fieldConfig": {"defaults": {"custom": {"align": "center"}}, "overrides": overrides or []},
        "options": {"showHeader": True, "footer": {"show": False}},
    }


def pie_panel(title, sql, x, y, w=6, h=8, ptype="donut"):
    return {
        "type": "piechart", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [_target(sql)],
        "options": {"pieType": ptype, "reduceOptions": {"calcs": ["lastNotNull"]},
                    "tooltip": {"mode": "multi"}, "legend": {"displayMode": "table", "placement": "right",
                                                              "values": ["value", "percent"]}},
        "fieldConfig": {"defaults": {}, "overrides": []},
    }


def gauge_panel(title, sql, x, y, w=6, h=6, min_v=0, max_v=1, thresholds=None, unit="percentunit"):
    th = thresholds or {"mode": "absolute", "steps": [
        {"color": "red", "value": None}, {"color": "orange", "value": 0.5},
        {"color": "yellow", "value": 0.65}, {"color": "green", "value": 0.7}]}
    return {
        "type": "gauge", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [_target(sql)],
        "fieldConfig": {"defaults": {"min": min_v, "max": max_v, "unit": unit,
                                      "thresholds": th}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "showThresholdLabels": False,
                    "showThresholdMarkers": True},
    }


def timeseries_panel(title, sql, x, y, w=12, h=8, overrides=None):
    return {
        "type": "timeseries", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "datasource": DS,
        "targets": [_target(sql)],
        "fieldConfig": {"defaults": {"custom": {"drawStyle": "bars", "fillOpacity": 30,
                                                 "barAlignment": 0, "showPoints": "always",
                                                 "pointSize": 8}},
                        "overrides": overrides or []},
        "options": {"tooltip": {"mode": "multi"}, "legend": {"displayMode": "list"}},
    }


def text_panel(title, content, x, y, w=24, h=3):
    return {
        "type": "text", "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "options": {"mode": "markdown", "content": content},
    }


def row_panel(title, y, collapsed=False, panels=None):
    r = {"type": "row", "title": title, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
         "collapsed": collapsed}
    if collapsed and panels:
        r["panels"] = panels
    return r


def build():
    panels = []
    pid = 1
    y = 0

    # ===== HEADER =====
    panels.append(text_panel("", (
        "# Uber Ride Cancellation Analysis\n"
        "Analysis of **150,000 bookings** | **32% cancellation rate** | "
        "**$748,600 estimated lost revenue** | Best model: **LightGBM (F2=0.73, Recall=95.6%)**"
    ), 0, y))
    y += 3

    # ===== KPI ROW =====
    panels.append(row_panel("Key Performance Indicators", y))
    y += 1
    panels.append(stat_panel("Total Bookings", "SELECT value FROM dataset_overview WHERE metric='total_bookings'", 0, y, 4, 4, color="blue"))
    panels.append(stat_panel("Cancellation Rate", "SELECT value FROM dataset_overview WHERE metric='cancellation_rate'", 4, y, 4, 4, unit="percent", color="red"))
    panels.append(stat_panel("Cancelled Rides", "SELECT value FROM dataset_overview WHERE metric='cancelled_rides'", 8, y, 4, 4, color="orange"))
    panels.append(stat_panel("Lost Revenue", "SELECT value FROM dataset_overview WHERE metric='lost_revenue'", 12, y, 4, 4, unit="currencyUSD", color="dark-red"))
    panels.append(stat_panel("Features Engineered", "SELECT value FROM dataset_overview WHERE metric='features_engineered'", 16, y, 4, 4, color="purple"))
    panels.append(stat_panel("Unique Locations", "SELECT value FROM dataset_overview WHERE metric='unique_locations'", 20, y, 4, 4, color="green"))
    y += 4

    # ===== UNIVARIATE ANALYSIS =====
    panels.append(row_panel("Univariate Analysis", y))
    y += 1

    panels.append(pie_panel("Target Distribution (is_cancelled)",
                            "SELECT MAX(CASE WHEN outcome='Completed' THEN count END) as Completed, MAX(CASE WHEN outcome='Cancelled' THEN count END) as Cancelled FROM univar_target_distribution", 0, y, 6, 8))
    panels.append(bar_panel("Vehicle Type Distribution & Cancellation Rate",
                            "SELECT vehicle_type, total_rides, cancellation_rate FROM univar_vehicle_type",
                            6, y, 9, 8, orient="vertical"))
    panels.append(bar_panel("VTAT Distribution & Cancellation Rate",
                            "SELECT vtat_range, count as ride_count, cancellation_rate FROM univar_vtat_distribution ORDER BY bucket_order",
                            15, y, 9, 8, orient="vertical"))
    y += 8

    panels.append(bar_panel("Hourly Booking Volume",
                            "SELECT printf('%02d:00', hour) as hour, booking_volume, cancellation_rate FROM univar_temporal_hourly ORDER BY hour",
                            0, y, 8, 7, orient="vertical"))
    panels.append(bar_panel("Daily Booking Volume",
                            "SELECT day_name, booking_volume, cancellation_rate FROM univar_temporal_daily ORDER BY day_order",
                            8, y, 8, 7, orient="vertical"))
    panels.append(bar_panel("Monthly Booking Volume",
                            "SELECT month_name, booking_volume, cancellation_rate FROM univar_temporal_monthly ORDER BY month_order",
                            16, y, 8, 7, orient="vertical"))
    y += 7

    panels.append(text_panel("", (
        "### Key Univariate Findings\n"
        "- **VTAT is the dominant predictor**: VTAT >= 15 min leads to **100% cancellation**\n"
        "- **Temporal features are weak**: Cancellation rate is flat (~32%) across all hours, days, months\n"
        "- **Vehicle type is not discriminative**: All 7 types show ~32% cancellation rate\n"
        "- **Data quality**: 7% missing VTAT values, 32% missing in booking_value/ride_distance/payment_method (leakage columns removed)"
    ), 0, y, 24, 3))
    y += 3

    # ===== BIVARIATE ANALYSIS =====
    panels.append(row_panel("Bivariate Analysis", y))
    y += 1

    panels.append(bar_panel("VTAT Bucket vs Cancellation Rate",
                            "SELECT vtat_bucket, cancellation_rate, sample_count FROM bivar_vtat_buckets ORDER BY bucket_order",
                            0, y, 8, 8, orient="vertical"))
    panels.append(bar_panel("Top 10 Pickup Locations by Cancellation Rate",
                            "SELECT location, cancellation_rate FROM bivar_location_pickup ORDER BY rank_order",
                            8, y, 8, 8))
    panels.append(bar_panel("Top 10 Drop Locations by Cancellation Rate",
                            "SELECT location, cancellation_rate FROM bivar_location_drop ORDER BY rank_order",
                            16, y, 8, 8))
    y += 8

    panels.append(bar_panel("Mutual Information with Target (is_cancelled)",
                            "SELECT feature, mutual_information FROM bivar_feature_importance_mi ORDER BY rank_order",
                            0, y, 8, 8, color="purple"))
    panels.append(bar_panel("Point-Biserial Correlation with Target",
                            "SELECT feature, abs_correlation FROM bivar_correlation_target ORDER BY abs_correlation DESC",
                            8, y, 8, 8, color="blue"))
    panels.append(bar_panel("Cramer's V (Categorical Associations)",
                            "SELECT feature_pair, cramers_v FROM bivar_cramers_v ORDER BY cramers_v DESC",
                            16, y, 8, 8, color="orange"))
    y += 8

    panels.append(text_panel("", (
        "### Key Bivariate Findings\n"
        "- **VTAT dominates** with mutual information of 0.1384 — 7x higher than the next feature\n"
        "- **Cramer's V**: vtat_bucket vs is_cancelled = 0.48 (strong); vehicle_type vs is_cancelled = 0.01 (negligible)\n"
        "- **Location patterns**: Top pickup/drop locations show 35-39% cancellation (vs 32% baseline)\n"
        "- **Point-biserial**: avg_vtat = 0.072 (significant, p < 0.001)"
    ), 0, y, 24, 3))
    y += 3

    # ===== MODEL MONITORING =====
    panels.append(row_panel("Model Performance Monitoring", y))
    y += 1

    panels.append(table_panel("Model Comparison (Test Set)",
        ("SELECT model_name as Model, "
         "ROUND(f2_score, 4) as F2_Score, "
         "ROUND(recall * 100, 1) || '%' as Recall, "
         "ROUND(precision_score * 100, 1) || '%' as Precision, "
         "ROUND(pr_auc, 4) as PR_AUC, "
         "ROUND(roc_auc, 4) as ROC_AUC, "
         "optimal_threshold as Threshold, "
         "CASE WHEN is_selected = 1 THEN 'Selected' ELSE '' END as Status "
         "FROM model_comparison ORDER BY f2_score DESC"),
        0, y, 24, 6, overrides=[
            {"matcher": {"id": "byName", "options": "Status"},
             "properties": [{"id": "custom.cellOptions", "value": {"type": "color-text"}},
                            {"id": "color", "value": {"mode": "fixed", "fixedColor": "green"}}]},
        ]))
    y += 6

    panels.append(gauge_panel("LightGBM F2-Score", "SELECT f2_score FROM model_comparison WHERE model_name='LightGBM'", 0, y, 4, 5))
    panels.append(gauge_panel("LightGBM Recall", "SELECT recall FROM model_comparison WHERE model_name='LightGBM'", 4, y, 4, 5))
    panels.append(gauge_panel("LightGBM Precision", "SELECT precision_score FROM model_comparison WHERE model_name='LightGBM'", 8, y, 4, 5,
                              thresholds={"mode": "absolute", "steps": [
                                  {"color": "red", "value": None}, {"color": "orange", "value": 0.3},
                                  {"color": "yellow", "value": 0.5}, {"color": "green", "value": 0.6}]}))
    panels.append(gauge_panel("LightGBM PR-AUC", "SELECT pr_auc FROM model_comparison WHERE model_name='LightGBM'", 12, y, 4, 5))
    panels.append(gauge_panel("LightGBM ROC-AUC", "SELECT roc_auc FROM model_comparison WHERE model_name='LightGBM'", 16, y, 4, 5))
    panels.append(stat_panel("Optimal Threshold", "SELECT optimal_threshold FROM model_comparison WHERE model_name='LightGBM'", 20, y, 4, 5, color="yellow"))
    y += 5

    # Feature importance comparison
    panels.append(bar_panel("Feature Importance — LightGBM (Selected Model)",
                            "SELECT feature, importance FROM model_feature_importance WHERE model_name='LightGBM' ORDER BY rank_order",
                            0, y, 12, 8, color="green"))
    panels.append(bar_panel("Feature Importance — XGBoost",
                            "SELECT feature, importance FROM model_feature_importance WHERE model_name='XGBoost' ORDER BY rank_order",
                            12, y, 12, 8, color="blue"))
    y += 8

    panels.append(bar_panel("Feature Importance — Random Forest",
                            "SELECT feature, importance FROM model_feature_importance WHERE model_name='Random Forest' ORDER BY rank_order",
                            0, y, 12, 8, color="orange"))
    panels.append(bar_panel("Feature Importance — Logistic Regression (Baseline)",
                            "SELECT feature, importance FROM model_feature_importance WHERE model_name='Logistic Regression' ORDER BY rank_order",
                            12, y, 12, 8, color="red"))
    y += 8

    # ===== BUSINESS IMPACT =====
    panels.append(row_panel("Business Impact & ROI (LightGBM)", y))
    y += 1

    panels.append(stat_panel("Net Annual Savings", "SELECT value FROM model_business_impact WHERE metric='net_annual_savings'", 0, y, 6, 5, unit="currencyUSD", color="green"))
    panels.append(stat_panel("ROI", "SELECT value FROM model_business_impact WHERE metric='roi_percentage'", 6, y, 6, 5, unit="percent", color="green"))
    panels.append(stat_panel("Daily Interventions", "SELECT value FROM model_business_impact WHERE metric='intervention_volume_daily'", 12, y, 6, 5, color="blue"))
    panels.append(stat_panel("Correctly Flagged (TP)", "SELECT value FROM model_business_impact WHERE metric='true_positives'", 18, y, 6, 5, color="green"))
    y += 5

    panels.append(table_panel("Confusion Matrix Breakdown",
        ("SELECT metric as Metric, CAST(value AS INTEGER) as Count, label as Description "
         "FROM model_business_impact WHERE metric IN ('true_positives','false_positives','false_negatives','true_negatives') "
         "ORDER BY CASE metric WHEN 'true_positives' THEN 1 WHEN 'false_positives' THEN 2 "
         "WHEN 'false_negatives' THEN 3 WHEN 'true_negatives' THEN 4 END"),
        0, y, 12, 6))
    panels.append(table_panel("Financial Breakdown",
        ("SELECT metric as Metric, CAST(value AS INTEGER) as Amount, label as Details "
         "FROM model_business_impact WHERE metric IN ('gross_savings','false_alarm_cost','missed_cancellation_cost','net_annual_savings') "
         "ORDER BY CASE metric WHEN 'gross_savings' THEN 1 WHEN 'false_alarm_cost' THEN 2 "
         "WHEN 'missed_cancellation_cost' THEN 3 WHEN 'net_annual_savings' THEN 4 END"),
        12, y, 12, 6))
    y += 6

    panels.append(text_panel("", (
        "### Monitoring Notes\n"
        "- **Target thresholds**: F2 >= 0.68 | Recall >= 70% | Precision >= 60%\n"
        "- **Current status**: F2 and Recall targets met; Precision below target (38% vs 60%) — acceptable given 4:1 cost asymmetry\n"
        "- **Recommendation**: Monitor VTAT distribution drift as the primary retraining signal\n"
        "- **Retrain trigger**: If F2-score drops below 0.68 or recall drops below 85%"
    ), 0, y, 24, 3))

    # Assign IDs
    for i, p in enumerate(panels):
        p["id"] = i + 1

    dashboard = {
        "uid": UID,
        "title": "Uber Ride Cancellation — Analysis & Model Monitoring",
        "tags": ["uber", "ml", "monitoring", "eda"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "",
        "templating": {"list": []},
        "panels": panels,
        "time": {"from": "now-6h", "to": "now"},
        "fiscalYearStartMonth": 0,
        "liveNow": False,
        "weekStart": "",
        "editable": True,
        "graphTooltip": 1,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"Dashboard written to: {OUTPUT}")


if __name__ == "__main__":
    build()
