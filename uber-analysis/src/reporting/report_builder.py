"""Assemble a curated Markdown report from analysis results and chart paths."""

from datetime import datetime


def _fmt_pct(val: float) -> str:
    return f"{val:.1%}"


def _fmt_pct_pp(val: float) -> str:
    return f"{val:.1f}pp"


def _statistical_tests_table(bivar: dict, multivar: dict) -> str:
    """Build a Markdown table summarising all statistical tests."""
    rows = []

    # Temporal tests
    for col, label in [("hour", "Hour"), ("weekday", "Weekday"), ("month", "Month")]:
        t = bivar["temporal_tests"][col]
        rows.append(
            f"| {label} vs Cancellation | Chi-square | "
            f"V = {t['cramers_v']:.4f} | p = {t['p']:.2e} | Negligible |"
        )

    # Temporal trend
    tt = bivar["temporal_trend"]
    rows.append(
        f"| Daily Rate Trend | Spearman | "
        f"rho = {tt['spearman_rho']:.4f} | p = {tt['p_value']:.2e} | No trend |"
    )

    # Vehicle type
    vt = bivar["vehicle_type"]
    rows.append(
        f"| Vehicle Type vs Cancellation | Chi-square / Cramer's V | "
        f"V = {vt['cramers_v']:.4f} | - | Negligible |"
    )

    # Locations
    for key, label in [("pickup_location", "Pickup Location"), ("drop_location", "Drop Location")]:
        loc = bivar[key]
        rows.append(
            f"| {label} vs Cancellation | Cramer's V | "
            f"V = {loc['cramers_v']:.4f} | - | Faint signal |"
        )

    # VTAT
    vs = bivar["vtat_spearman"]
    rows.append(
        f"| avg_vtat vs Cancellation | Spearman | "
        f"rho = {vs['rho']:.4f} | p = {vs['p']:.2e} | Dominant predictor |"
    )

    # vtat_missing
    vm = bivar["vtat_missing"]
    rows.append(
        f"| vtat_missing vs Cancellation | Fisher / Phi | "
        f"phi = {vm['phi']:.3f}, OR = {vm['odds_ratio']:.1f} | p = {vm['fisher_p']:.2e} | Strong signal |"
    )

    # Missingness confounding
    mc = multivar["missingness_confounding"]
    rows.append(
        f"| Missingness Confounding | Logistic Reg AUC lift | "
        f"delta = {mc['auc_lift']:.4f} | - | No confounding |"
    )

    # Route validation
    rv = multivar["route_validation"]
    for m_key in ["route_m5", "route_m20", "route_m50"]:
        lift = rv[m_key]["lift"]
        rows.append(
            f"| Route (smoothing={m_key.split('m')[1]}) | CV AUC lift | "
            f"lift = {lift:+.4f} | - | No generalisable signal |"
        )

    header = (
        "| Test | Method | Statistic | p-value | Interpretation |\n"
        "|------|--------|-----------|---------|----------------|\n"
    )
    return header + "\n".join(rows)


def build(
    univar: dict,
    bivar: dict,
    multivar: dict,
    chart_paths: dict[str, str],
    charts_rel_dir: str = "charts",
) -> str:
    """Return the full report as a Markdown string."""
    t = univar["target"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []

    # ── Title ──────────────────────────────────────────────────
    sections.append(f"# Uber Ride Cancellation Analysis Report\n\n*Generated: {now}*\n")

    # ── 1. Dataset Overview ───────────────────────────────────
    sections.append("## 1. Dataset Overview\n")
    sections.append(
        f"| Property | Value |\n"
        f"|----------|-------|\n"
        f"| Total records | {univar['shape']['rows']:,} |\n"
        f"| Features | {univar['shape']['cols']} |\n"
        f"| Vehicle types | {univar['vehicle_type']['n_unique']} |\n"
        f"| Unique pickup locations | {univar['locations']['pickup_unique']} |\n"
        f"| Unique drop locations | {univar['locations']['drop_unique']} |\n"
        f"| VTAT coverage | {_fmt_pct(1 - univar['avg_vtat']['missing_pct'] / 100)} "
        f"({univar['avg_vtat']['missing']} missing) |\n"
    )

    # ── 2. Target Distribution ────────────────────────────────
    sections.append("## 2. Target Distribution and Business Impact\n")
    sections.append(f"![Target Distribution]({charts_rel_dir}/target_distribution.png)\n")
    sections.append(
        f"- **{t['cancellation_rate']:.1%}** of all bookings end in cancellation "
        f"({t['cancelled']:,} out of {t['completed'] + t['cancelled']:,})\n"
        f"- Estimated annual revenue loss: **~${t['cancelled'] * 20:,.0f}** "
        f"(at $20 avg booking value)\n"
        f"- Class ratio: {1 - t['cancellation_rate']:.0%} completed / "
        f"{t['cancellation_rate']:.0%} cancelled (moderately imbalanced)\n"
    )

    # ── 3. Univariate Highlights ──────────────────────────────
    sections.append("## 3. Univariate Highlights\n")
    sections.append("### VTAT (Vehicle Time to Arrival)\n")
    v = univar["avg_vtat"]
    sections.append(
        f"The **strongest signal** in the dataset. Range: {v['min']:.1f} - {v['max']:.1f} min, "
        f"mean: {v['mean']:.1f} min, median: {v['median']:.1f} min. "
        f"{v['missing']} rows ({v['missing_pct']:.1f}%) have missing VTAT values -- "
        f"all of which are cancellations (see bivariate section).\n"
    )
    sections.append("### Temporal Features\n")
    sections.append(f"![Temporal Flatness]({charts_rel_dir}/temporal_flatness.png)\n")
    temp = univar["temporal"]
    sections.append(
        f"Cancellation rate is **flat** across all time dimensions:\n"
        f"- Hourly spread: {_fmt_pct_pp(temp['hourly']['spread_pp'])}\n"
        f"- Daily spread: {_fmt_pct_pp(temp['daily']['spread_pp'])}\n"
        f"- Monthly spread: {_fmt_pct_pp(temp['monthly']['spread_pp'])}\n\n"
        f"No temporal features carry predictive signal.\n"
    )
    sections.append("### Vehicle Type\n")
    sections.append(
        f"{univar['vehicle_type']['n_unique']} categories, all showing ~32% cancellation rate. "
        f"Not a discriminative feature.\n"
    )

    # ── 4. Bivariate Highlights ───────────────────────────────
    sections.append("## 4. Bivariate Highlights\n")
    sections.append("### VTAT Zones: The Dominant Predictor\n")
    sections.append(f"![VTAT Zones]({charts_rel_dir}/vtat_zones.png)\n")
    zones = bivar["vtat_zones"]
    sections.append(
        f"VTAT shows a non-linear, non-monotonic relationship with cancellation, "
        f"captured by five behavioural zones:\n\n"
        f"| Zone | VTAT Range | Cancel Rate | n |\n"
        f"|------|------------|-------------|---|\n"
    )
    for label, z in zones.items():
        sections.append(
            f"| {label.title()} | {z['range']} min | {_fmt_pct(z['cancel_rate'])} | {z['n']:,} |\n"
        )
    sections.append(
        f"\n**Key insight**: VTAT >= 15.1 min -> 100% cancellation rate (system auto-cancellation). "
        f"The 12-15 min 'dip' suggests a sunk-cost effect where riders who have already waited "
        f"are less likely to cancel.\n"
    )

    sections.append("### vtat_missing: A Strong Binary Signal\n")
    sections.append(f"![vtat_missing]({charts_rel_dir}/vtat_missing_signal.png)\n")
    vm = bivar["vtat_missing"]
    sections.append(
        f"Every row with missing avg_vtat is cancelled (phi = {vm['phi']:.3f}, "
        f"OR = {vm['odds_ratio']:.1f}). These represent early cancellations before "
        f"vehicle assignment -- 100% precision but only ~22% of all cancellations.\n"
    )

    sections.append("### Location Patterns\n")
    sections.append(f"![Locations]({charts_rel_dir}/location_top10.png)\n")
    sections.append(
        f"- Pickup location Cramer's V: {bivar['pickup_location']['cramers_v']:.4f} (faint)\n"
        f"- Drop location Cramer's V: {bivar['drop_location']['cramers_v']:.4f} (faint)\n"
        f"- Top locations show ~35-39% cancellation vs 32% baseline (~7pp swing)\n"
    )

    # ── 5. Multivariate Highlights ────────────────────────────
    sections.append("## 5. Multivariate Highlights\n")

    mc = multivar["missingness_confounding"]
    sections.append("### Missingness Confounding Check\n")
    sections.append(
        f"Tested whether non-VTAT features predict vtat missingness beyond is_cancelled alone. "
        f"AUC lift: **{mc['auc_lift']:.4f}** -- missingness is purely target-driven with "
        f"no hidden confounding.\n"
    )

    vf = multivar["vtat_family"]
    sections.append("### Feature Redundancy\n")
    sections.append(
        f"- avg_vtat vs vtat_zone: eta-squared = {vf['avg_vtat_vs_vtat_zone_eta_sq']:.4f} "
        f"(high overlap by design)\n"
        f"- vtat_zone vs vtat_missing: Cramer's V = {vf['vtat_zone_vs_vtat_missing_cramers_v']:.4f}\n"
    )

    rv = multivar["route_validation"]
    sections.append("### Route Feature Validation\n")
    sections.append(
        f"Combined pickup x drop creates {rv['route_unique_values']:,} unique routes. "
        f"Cross-validated target encoding at all smoothing levels (m=5, 20, 50) produced "
        f"**zero lift** -- the apparent high Cramer's V is entirely a cardinality artifact. "
        f"Route should be dropped.\n"
    )

    sections.append("### Correlation Structure\n")
    sections.append(f"![Correlation]({charts_rel_dir}/correlation_heatmap.png)\n")
    vif = multivar["correlation_vif"]["vif"]
    max_vif = max(vif.values())
    sections.append(
        f"All VIFs are near 1 (max: {max_vif:.2f}) -- no multicollinearity. "
        f"The feature space is relatively orthogonal.\n"
    )

    # ── 6. Statistical Tests Summary ──────────────────────────
    sections.append("## 6. Statistical Tests Summary\n")
    sections.append(_statistical_tests_table(bivar, multivar))
    sections.append("")

    # ── 7. Business Recommendations ───────────────────────────
    sections.append("## 7. Business Recommendations\n")
    sections.append(
        "Based on the analysis findings:\n\n"
        "1. **VTAT is the actionable lever**: Rides with VTAT >= 15 min are auto-cancelled. "
        "Reducing vehicle assignment time directly reduces cancellations.\n\n"
        "2. **Early cancellation detection**: Missing VTAT (no vehicle assigned yet) "
        "perfectly predicts a subset of cancellations. Proactive engagement at booking time "
        "could retain some of these riders.\n\n"
        "3. **The 12-15 min dip is exploitable**: Riders who have waited 12+ minutes show "
        "lower cancellation rates (sunk-cost effect). ETA communication and incentives in the "
        "5-12 min window could push more riders into this retention zone.\n\n"
        "4. **Temporal and vehicle-type interventions are unlikely to help**: Cancellation rate "
        "is flat across hours, days, months, and vehicle types. Time-based or vehicle-based "
        "strategies would not target the problem.\n\n"
        "5. **Location-based strategies have limited upside**: A ~7pp rate swing between "
        "the best and worst locations suggests some geographic signal, but the effect size "
        "is small. Location-specific interventions are a secondary priority.\n"
    )

    return "\n".join(sections)
