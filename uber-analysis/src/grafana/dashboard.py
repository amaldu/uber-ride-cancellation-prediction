"""Generate a Grafana dashboard JSON that queries the computed SQLite database.

All panels use SQL queries against the tables populated by export_db.py.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

UID = "uber-eda-analysis"
DATASOURCE = {"type": "frser-sqlite-datasource", "uid": "uber_sqlite"}

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent.parent
    / "grafana" / "provisioning" / "dashboards" / "uber_analysis.json"
)

_panel_id = 0


def _next_id() -> int:
    global _panel_id
    _panel_id += 1
    return _panel_id


def _stat_panel(title, sql, x, y, w, h, unit="", color="green", decimals=None):
    p = {
        "id": _next_id(), "type": "stat",
        "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"rawSql": sql, "datasource": DATASOURCE}],
        "datasource": DATASOURCE,
        "fieldConfig": {"defaults": {
            "color": {"mode": "fixed", "fixedColor": color},
            "thresholds": {"mode": "absolute", "steps": [{"color": color, "value": None}]},
        }, "overrides": []},
        "options": {"graphMode": "none", "textMode": "value", "reduceOptions": {"calcs": ["lastNotNull"]}},
    }
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    if decimals is not None:
        p["fieldConfig"]["defaults"]["decimals"] = decimals
    return p


def _gauge_panel(title, sql, x, y, w, h, min_val=0, max_val=1, thresholds=None):
    if thresholds is None:
        thresholds = {"mode": "absolute", "steps": [
            {"color": "red", "value": None},
            {"color": "orange", "value": 0.3},
            {"color": "yellow", "value": 0.5},
            {"color": "green", "value": 0.7},
        ]}
    return {
        "id": _next_id(), "type": "gauge",
        "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"rawSql": sql, "datasource": DATASOURCE}],
        "datasource": DATASOURCE,
        "fieldConfig": {"defaults": {
            "min": min_val, "max": max_val,
            "thresholds": thresholds, "decimals": 4,
        }, "overrides": []},
    }


def _bar_panel(title, sql, x, y, w, h, orient="horizontal", color="green"):
    return {
        "id": _next_id(), "type": "barchart",
        "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"rawSql": sql, "datasource": DATASOURCE}],
        "datasource": DATASOURCE,
        "options": {"orientation": orient, "showValue": "auto", "barWidth": 0.7},
        "fieldConfig": {"defaults": {
            "color": {"mode": "fixed", "fixedColor": color},
        }, "overrides": []},
    }


def _table_panel(title, sql, x, y, w, h, overrides=None):
    p = {
        "id": _next_id(), "type": "table",
        "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"rawSql": sql, "datasource": DATASOURCE}],
        "datasource": DATASOURCE,
        "options": {"showHeader": True, "footer": {"show": False}},
        "fieldConfig": {"defaults": {}, "overrides": overrides or []},
    }
    return p


def _text_panel(title, content, x, y, w, h):
    return {
        "id": _next_id(), "type": "text",
        "title": title,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "options": {"mode": "markdown", "content": content},
    }


def _row_panel(title, y, collapsed=False):
    return {
        "id": _next_id(), "type": "row",
        "title": title,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        "collapsed": collapsed, "panels": [],
    }


def build(output_path: str | Path | None = None) -> Path:
    """Generate the dashboard JSON and write it to disk."""
    global _panel_id
    _panel_id = 0

    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT
    panels = []
    y = 0

    # ════════════════════════════════════════════════════════════
    # ROW: KPI Overview
    # ════════════════════════════════════════════════════════════
    panels.append(_row_panel("Dataset Overview & KPIs", y))
    y += 1

    panels.append(_stat_panel(
        "Total Bookings",
        "SELECT value FROM dataset_overview WHERE metric='total_records'",
        0, y, 4, 4, color="blue"))
    panels.append(_stat_panel(
        "Cancellation Rate",
        "SELECT value FROM dataset_overview WHERE metric='cancellation_rate'",
        4, y, 4, 4, unit="percent", color="red"))
    panels.append(_stat_panel(
        "Cancelled Rides",
        "SELECT value FROM dataset_overview WHERE metric='cancelled_rides'",
        8, y, 4, 4, color="orange"))
    panels.append(_stat_panel(
        "Est. Revenue Loss",
        "SELECT value FROM dataset_overview WHERE metric='est_revenue_loss'",
        12, y, 4, 4, unit="currencyUSD", color="red"))
    panels.append(_stat_panel(
        "Vehicle Types",
        "SELECT value FROM dataset_overview WHERE metric='vehicle_types'",
        16, y, 4, 4, color="purple"))
    panels.append(_stat_panel(
        "Unique Locations",
        "SELECT value FROM dataset_overview WHERE metric='unique_locations'",
        20, y, 4, 4, color="purple"))
    y += 4

    panels.append(_bar_panel(
        "Target Distribution",
        "SELECT outcome, count as rides, percentage as pct FROM target_distribution",
        0, y, 6, 6, orient="vertical", color="blue"))
    panels.append(_bar_panel(
        "Vehicle Type Cancellation Rate (%)",
        "SELECT vehicle_type, cancel_rate FROM vehicle_type ORDER BY cancel_rate DESC",
        6, y, 9, 6, orient="vertical", color="orange"))
    panels.append(_table_panel(
        "VTAT Statistics",
        "SELECT stat as Metric, ROUND(value, 2) as Value FROM vtat_distribution",
        15, y, 9, 6))
    y += 6

    # ════════════════════════════════════════════════════════════
    # ROW: Univariate — Temporal Flatness
    # ════════════════════════════════════════════════════════════
    panels.append(_row_panel("Temporal Analysis (No Signal Found)", y))
    y += 1

    panels.append(_bar_panel(
        "Hourly Cancellation Rate (%)",
        "SELECT printf('%02d:00', hour) as hour, cancel_rate FROM temporal_hourly ORDER BY hour",
        0, y, 8, 7, orient="vertical", color="blue"))
    panels.append(_bar_panel(
        "Daily Cancellation Rate (%)",
        "SELECT day_name, cancel_rate FROM temporal_daily ORDER BY day_order",
        8, y, 8, 7, orient="vertical", color="blue"))
    panels.append(_bar_panel(
        "Monthly Cancellation Rate (%)",
        "SELECT month_name, cancel_rate FROM temporal_monthly ORDER BY month_order",
        16, y, 8, 7, orient="vertical", color="blue"))
    y += 7

    panels.append(_text_panel("", (
        "### Finding: Temporal Features Carry No Signal\n"
        "Cancellation rate is flat (~32%) across all hours, days, and months. "
        "Chi-square tests return negligible Cramer's V (< 0.02) for every temporal dimension. "
        "Time-based interventions would not target the problem."
    ), 0, y, 24, 3))
    y += 3

    # ════════════════════════════════════════════════════════════
    # ROW: Bivariate — VTAT (The Dominant Predictor)
    # ════════════════════════════════════════════════════════════
    panels.append(_row_panel("VTAT Analysis (Dominant Predictor)", y))
    y += 1

    panels.append(_bar_panel(
        "Cancellation Rate by VTAT Zone (%)",
        "SELECT zone || ' (' || CAST(range_min AS TEXT) || '-' || CAST(range_max AS TEXT) || ' min)' as zone, "
        "cancel_rate, sample_count FROM vtat_zones ORDER BY zone_order",
        0, y, 12, 8, orient="vertical", color="red"))
    panels.append(_table_panel(
        "VTAT Zone Breakdown",
        "SELECT zone as Zone, "
        "CAST(range_min AS TEXT) || ' - ' || CAST(range_max AS TEXT) as 'Range (min)', "
        "cancel_rate || '%' as 'Cancel Rate', "
        "sample_count as 'Sample Size' "
        "FROM vtat_zones ORDER BY zone_order",
        12, y, 12, 8))
    y += 8

    panels.append(_text_panel("", (
        "### Finding: VTAT Zones Show Non-Linear, Non-Monotonic Pattern\n"
        "- **Instant (2-2.9 min)**: 0% cancellation — immediate pickup prevents cancellation\n"
        "- **Low (3-5 min)**: ~26% — below baseline\n"
        "- **Baseline (5.1-11.9 min)**: ~31% — matches overall rate\n"
        "- **Dip (12-15 min)**: ~9% — **sunk-cost effect** (riders who waited are committed)\n"
        "- **Timeout (15.1-20 min)**: 100% — **system auto-cancellation**"
    ), 0, y, 24, 3))
    y += 3

    # ════════════════════════════════════════════════════════════
    # ROW: Bivariate — vtat_missing & Locations
    # ════════════════════════════════════════════════════════════
    panels.append(_row_panel("Other Bivariate Findings", y))
    y += 1

    panels.append(_table_panel(
        "vtat_missing Signal",
        "SELECT metric as Metric, ROUND(value, 4) as Value FROM vtat_missing",
        0, y, 6, 6))
    panels.append(_bar_panel(
        "Top 10 Pickup Locations — Cancel Rate (%)",
        "SELECT location, cancel_rate FROM location_pickup_top10 ORDER BY rank_order",
        6, y, 9, 6, color="orange"))
    panels.append(_bar_panel(
        "Top 10 Drop Locations — Cancel Rate (%)",
        "SELECT location, cancel_rate FROM location_drop_top10 ORDER BY rank_order",
        15, y, 9, 6, color="orange"))
    y += 6

    panels.append(_text_panel("", (
        "### Finding: vtat_missing Is a Strong Binary Signal\n"
        "Every row with missing avg_vtat is cancelled (phi ≈ 0.40). "
        "These represent early cancellations before vehicle assignment — "
        "100% precision but only ~22% of all cancellations.\n\n"
        "### Finding: Location Patterns Are Faint\n"
        "Cramer's V ≈ 0.037 for both pickup and drop locations. "
        "Top locations show ~35-39% cancellation vs 32% baseline (~7pp swing)."
    ), 0, y, 24, 3))
    y += 3

    # ════════════════════════════════════════════════════════════
    # ROW: Statistical Tests Summary
    # ════════════════════════════════════════════════════════════
    panels.append(_row_panel("Statistical Tests Summary", y))
    y += 1

    panels.append(_table_panel(
        "All Bivariate Tests",
        "SELECT feature as Feature, method as Method, statistic as Statistic, "
        "p_value as 'p-value', interpretation as Interpretation "
        "FROM bivar_summary",
        0, y, 24, 8, overrides=[
            {"matcher": {"id": "byName", "options": "Interpretation"},
             "properties": [{"id": "custom.cellOptions", "value": {"type": "color-text"}},
                            {"id": "color", "value": {"mode": "fixed", "fixedColor": "yellow"}}]},
        ]))
    y += 8

    # ════════════════════════════════════════════════════════════
    # ROW: Multivariate
    # ════════════════════════════════════════════════════════════
    panels.append(_row_panel("Multivariate Analysis", y))
    y += 1

    panels.append(_table_panel(
        "Missingness Confounding Check",
        "SELECT model as Model, ROUND(auc, 4) as AUC FROM missingness_confounding",
        0, y, 8, 5))
    panels.append(_table_panel(
        "Feature Redundancy",
        "SELECT pair as 'Feature Pair', metric as Metric, ROUND(value, 4) as Value "
        "FROM feature_redundancy",
        8, y, 8, 5))
    panels.append(_table_panel(
        "Route Validation (CV AUC)",
        "SELECT model as Model, ROUND(mean_auc, 4) as 'Mean AUC', "
        "ROUND(std_auc, 4) as 'Std', ROUND(lift, 4) as Lift "
        "FROM route_validation",
        16, y, 8, 5))
    y += 5

    panels.append(_table_panel(
        "VIF (Variance Inflation Factor)",
        "SELECT feature as Feature, ROUND(vif, 2) as VIF FROM vif_results ORDER BY vif DESC",
        0, y, 12, 5))
    panels.append(_table_panel(
        "Correlation Matrix",
        "SELECT feature_a as Feature, "
        "ROUND(MAX(CASE WHEN feature_b='avg_vtat' THEN correlation END), 3) as avg_vtat, "
        "ROUND(MAX(CASE WHEN feature_b='vtat_missing' THEN correlation END), 3) as vtat_missing, "
        "ROUND(MAX(CASE WHEN feature_b='vehicle_type_freq' THEN correlation END), 3) as vehicle_type, "
        "ROUND(MAX(CASE WHEN feature_b='pickup_location_freq' THEN correlation END), 3) as pickup, "
        "ROUND(MAX(CASE WHEN feature_b='drop_location_freq' THEN correlation END), 3) as drop_loc "
        "FROM correlation_matrix GROUP BY feature_a",
        12, y, 12, 5))
    y += 5

    panels.append(_text_panel("", (
        "### Multivariate Findings\n"
        "- **Missingness confounding**: AUC lift of ~0.002 — vtat_missing is purely target-driven\n"
        "- **Route validation FAILED**: Cross-validated target encoding at m=5, 20, 50 shows zero lift. "
        "High Cramer's V was a cardinality artifact (30k+ unique routes)\n"
        "- **All VIFs ≈ 1**: No multicollinearity in the feature space\n"
        "- **Feature space is orthogonal**: No hidden non-linear dependencies"
    ), 0, y, 24, 3))
    y += 3

    # ════════════════════════════════════════════════════════════
    # ROW: Business Recommendations
    # ════════════════════════════════════════════════════════════
    panels.append(_row_panel("Business Recommendations", y))
    y += 1

    panels.append(_text_panel("", (
        "### Actionable Recommendations\n\n"
        "1. **VTAT is the actionable lever**: VTAT >= 15 min → 100% cancellation. "
        "Reducing vehicle assignment time directly reduces cancellations.\n\n"
        "2. **Early cancellation detection**: Missing VTAT perfectly predicts a subset. "
        "Proactive engagement at booking time could retain these riders.\n\n"
        "3. **Exploit the 12-15 min dip**: Riders who waited 12+ min show lower cancellation "
        "(sunk-cost effect). ETA updates and incentives in the 5-12 min window could push "
        "more riders into this retention zone.\n\n"
        "4. **Temporal/vehicle-type interventions won't help**: Rate is flat across all "
        "temporal dimensions and vehicle types.\n\n"
        "5. **Location strategies have limited upside**: ~7pp swing between best and worst "
        "locations — small effect size, secondary priority."
    ), 0, y, 24, 6))

    # ── Assemble dashboard ────────────────────────────────────
    dashboard = {
        "uid": UID,
        "title": "Uber Ride Cancellation — EDA Analysis Dashboard",
        "tags": ["uber", "eda", "analysis"],
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    logger.info("Dashboard JSON written to %s", output_path)
    return output_path
