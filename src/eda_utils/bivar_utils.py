import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns 
from scipy import stats  
from scipy.stats import chi2_contingency, fisher_exact, kruskal  
import os
from scipy.stats import spearmanr


def plot_empirical_logit(df, feature, target, bins=10):
    df_temp = df[[feature, target]].dropna().copy()
    df_temp['bin'] = pd.qcut(df_temp[feature], bins, duplicates='drop')
    
    grouped = df_temp.groupby('bin')[target].agg(['mean', 'count'])
    grouped['empirical_logit'] = np.log(grouped['mean'] / (1 - grouped['mean']))
    grouped['bin_midpoint'] = [interval.mid for interval in grouped.index]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(grouped['bin_midpoint'], grouped['empirical_logit'], s=grouped['count']/10)
    ax.set_xlabel(feature)
    ax.set_ylabel('Empirical log-odds')
    ax.set_title(f'Linearity check: {feature} vs log-odds of {target}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    min_dim = min(confusion.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))


def spearman_trend(x, y, x_label="x", y_label="y"):

    rho, p_value = stats.spearmanr(x, y)
    print(f"Spearman correlation ({x_label} vs {y_label}): ρ = {rho:.4f}, p = {p_value:.4e}")

    if p_value < 0.05:
        direction = "increasing" if rho > 0 else "decreasing"
        strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.6 else "strong"
        print(f"  → Statistically significant {strength} {direction} trend (p < 0.05)")
    else:
        print("  → No statistically significant monotonic trend (p ≥ 0.05)")
    print()


def nominal_vs_binary(df, feature_col, target_col, labels,
                      feature_label=None, target_label=None,
                      target_pos_label="Cancelled",
                      target_neg_label="Not Cancelled",
                      rate_label="Cancellation Rate",
                      figsize=(12, 8),
                      bar_width=0.8, capsize=4, tick_step=1,
                      tick_fontsize=10, show_vol_labels=True):
    
    if feature_label is None:
        feature_label = feature_col
    if target_label is None:
        target_label = target_col

    overall_rate = df[target_col].mean()

    grp = (df.groupby(feature_col, observed=True)[target_col]
             .agg(["sum", "count"])
             .reset_index())
    grp.columns = [feature_col, "positive", "total"]
    grp["negative"] = grp["total"] - grp["positive"]
    grp["rate"] = grp["positive"] / grp["total"]

    z = 1.96
    n = grp["total"]
    p_hat = grp["rate"]
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
    grp["ci_lo"] = centre - margin
    grp["ci_hi"] = centre + margin

    grp["sig"] = np.where(
        grp["ci_hi"] < overall_rate, "below",
        np.where(grp["ci_lo"] > overall_rate, "above", "overlap"))

    color_map = {"above": "#c0392b", "below": "#27ae60", "overlap": "#7f8c8d"}

    fig, (ax_rate, ax_vol) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15},
        constrained_layout=True)

    x = np.arange(len(grp))
    ax_rate.bar(x, grp["rate"],
                color=[color_map[s] for s in grp["sig"]],
                edgecolor="white", alpha=0.85, width=bar_width)
    ax_rate.errorbar(x, grp["rate"],
                     yerr=[grp["rate"] - grp["ci_lo"],
                           grp["ci_hi"] - grp["rate"]],
                     fmt="none", ecolor="black", capsize=capsize, linewidth=1)
    ax_rate.axhline(overall_rate, color="black", ls="--", lw=1, alpha=0.7,
                    label=f"Overall mean ({overall_rate:.2%})")
    ax_rate.set_ylabel(rate_label)
    ax_rate.set_title(f"{rate_label} by {feature_label} (95% Wilson CI)")
    ax_rate.set_xticks(x[::tick_step])
    ax_rate.set_xticklabels([labels[i] for i in range(0, len(labels), tick_step)],
                            fontsize=tick_fontsize)
    ax_rate.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    legend_handles = [
        Patch(facecolor="#c0392b", alpha=0.85, label="Sig. above mean"),
        Patch(facecolor="#27ae60", alpha=0.85, label="Sig. below mean"),
        Patch(facecolor="#7f8c8d", alpha=0.85, label="Not significant"),
        plt.Line2D([], [], color="black", ls="--", lw=1,
                   label=f"Overall mean ({overall_rate:.2%})")
    ]
    ax_rate.legend(handles=legend_handles, loc="lower right", fontsize=9,
                   prop={"weight": "bold"})

    ax_vol.bar(x, grp["total"],
               color=[color_map[s] for s in grp["sig"]],
               edgecolor="white", alpha=0.65, width=bar_width)
    vol_mean = grp["total"].mean()
    ax_vol.axhline(vol_mean, color="black", ls="--", lw=1, alpha=0.7,
                   label=f"Mean volume ({vol_mean:,.0f})")
    ax_vol.set_ylabel("Ride Count")
    ax_vol.set_xlabel(feature_label)
    ax_vol.set_xticks(x[::tick_step])
    ax_vol.set_xticklabels([labels[i] for i in range(0, len(labels), tick_step)],
                           fontsize=tick_fontsize)
    ax_vol.legend(loc="lower right", fontsize=9, prop={"weight": "bold"})

    if show_vol_labels:
        for i_row, row in enumerate(grp.itertuples()):
            ax_vol.text(i_row, row.total + grp["total"].max() * 0.02,
                        f"{row.total:,}", ha="center", va="bottom", fontsize=9)

    plt.show()

    # stats tests 
    ct = pd.crosstab(df[feature_col], df[target_col])
    chi2, p, dof, expected = chi2_contingency(ct)
    v = cramers_v(df[feature_col], df[target_col])

    std_res = (ct.values - expected) / np.sqrt(expected)
    std_res_df = pd.DataFrame(std_res, index=labels,
                              columns=[target_neg_label, target_pos_label])
    std_res_cancelled = std_res_df[[target_pos_label]]

    print(f"\nChi-square: χ² = {chi2:.2f}, dof = {dof}, p = {p:.4e}")
    print(f"Cramér's V = {v:.4f}")

    noteworthy = std_res_cancelled[std_res_cancelled[target_pos_label].abs() > 2]
    if len(noteworthy):
        print(f"\nCategories with |standardized residual| > 2:")
        print(noteworthy.to_string())
    else:
        print("\nNo category stands out.")

    print("\nAll standardized residuals:")
    print(std_res_cancelled.to_string())

    return {
        "grouped": grp,
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "cramers_v": v,
        "std_residuals": std_res_df,
    }


def binary_vs_binary(df, feature_col, target_col,
                     labels=("0", "1"),
                     target_labels=("Not Cancelled", "Cancelled"),
                     feature_label=None, figsize=(10, 4)):

    if feature_label is None:
        feature_label = feature_col

    ct = pd.crosstab(df[feature_col], df[target_col])
    a, b = ct.iloc[0]          # feature=0 → (target=0, target=1)
    c, d = ct.iloc[1]          # feature=1 → (target=0, target=1)
    n = a + b + c + d

    #  rates + wilson CIs 
    overall_rate = df[target_col].mean()
    z = 1.96
    rates, ci_los, ci_his = [], [], []
    for pos, tot in [(b, a + b), (d, c + d)]:
        p_hat = pos / tot
        denom = 1 + z ** 2 / tot
        centre = (p_hat + z ** 2 / (2 * tot)) / denom
        margin = (z / denom) * np.sqrt(
            p_hat * (1 - p_hat) / tot + z ** 2 / (4 * tot ** 2))
        rates.append(p_hat)
        ci_los.append(centre - margin)
        ci_his.append(centre + margin)

    #  viz: rate bars + heatmap 
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.35},
        constrained_layout=True)

    x = np.arange(2)
    colors = ["#3498db", "#e67e22"]
    ax_bar.bar(x, rates, color=colors, edgecolor="white",
               alpha=0.85, width=0.55)
    ax_bar.errorbar(x, rates,
                    yerr=[[r - lo for r, lo in zip(rates, ci_los)],
                          [hi - r for r, hi in zip(rates, ci_his)]],
                    fmt="none", ecolor="black", capsize=6, linewidth=1.2)
    ax_bar.axhline(overall_rate, color="black", ls="--", lw=1, alpha=0.7,
                   label=f"Overall mean ({overall_rate:.2%})")
    for i, (r, lo, hi) in enumerate(zip(rates, ci_los, ci_his)):
        ax_bar.text(i, hi + 0.005, f"{r:.2%}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=11)
    ax_bar.set_ylabel("Cancellation Rate")
    ax_bar.set_title(f"Rate by {feature_label} (95% Wilson CI)")
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_bar.legend(fontsize=9)

    # Heatmap with counts + row percentages
    annot = np.empty_like(ct.values, dtype=object)
    for r_idx in range(ct.shape[0]):
        row_total = ct.iloc[r_idx].sum()
        for c_idx in range(ct.shape[1]):
            val = ct.iloc[r_idx, c_idx]
            pct = val / row_total * 100
            annot[r_idx, c_idx] = f"{val:,}\n({pct:.1f}%)"

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_orange", ["#d6eaf8", "#e67e22"])
    ax_heat.imshow(ct.values, cmap=cmap, aspect="auto")
    for r_idx in range(ct.shape[0]):
        for c_idx in range(ct.shape[1]):
            ax_heat.text(c_idx, r_idx, annot[r_idx, c_idx],
                         ha="center", va="center", fontsize=11,
                         fontweight="bold")
    ax_heat.set_xticks([0, 1])
    ax_heat.set_xticklabels(target_labels, fontsize=10)
    ax_heat.set_yticks([0, 1])
    ax_heat.set_yticklabels(labels, fontsize=10)
    ax_heat.set_title(f"2×2 Contingency Table")
    ax_heat.set_xlabel(target_col)
    ax_heat.set_ylabel(feature_label)

    plt.show()

    #  stats tests 
    odds_ratio, p_fisher = fisher_exact(ct.values)

    # Phi via chi-square (avoids large-int overflow)
    chi2 = chi2_contingency(ct, correction=False)[0]
    sign = 1 if (a * d - b * c) >= 0 else -1
    phi = sign * np.sqrt(chi2 / n)

    # odds-ratio 95% CI 
    log_or = np.log(odds_ratio) if odds_ratio > 0 else np.nan
    se = np.sqrt(1.0 / max(a, 1) + 1.0 / max(b, 1)
                 + 1.0 / max(c, 1) + 1.0 / max(d, 1))
    or_ci = (np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se))

    print(f"Fisher's exact test:  p = {p_fisher:.4e}")
    print(f"Phi coefficient:      φ = {phi:+.4f}")

    strength = ("negligible" if abs(phi) < 0.05
                else "weak" if abs(phi) < 0.15
                else "moderate" if abs(phi) < 0.25
                else "strong")
    direction = "positive" if phi > 0 else "negative"
    print(f"  → {strength} {direction} association")

    print(f"\nOdds ratio:           OR = {odds_ratio:.4f}  "
          f"95% CI [{or_ci[0]:.4f}, {or_ci[1]:.4f}]")
    if or_ci[0] <= 1 <= or_ci[1]:
        print("  → CI contains 1 — no significant difference in odds")
    else:
        print(f"  → {feature_label}=1 {'increases' if odds_ratio > 1 else 'decreases'}"
              f" the odds of cancellation by {abs(odds_ratio - 1):.1%}")

    return {
        "contingency": ct,
        "fisher_p": p_fisher,
        "odds_ratio": odds_ratio,
        "or_ci": or_ci,
        "phi": phi,
    }


def rolling_lineplot(dates, values, *,
                     rolling_windows=(7, 30),
                     ylabel="Value",
                     xlabel="Date",
                     title="",
                     daily_label="Daily",
                     rolling_label_fmt="{win}-day rolling avg",
                     mean_label_fmt="{mean:.2%}",
                     shade_weekends=True,
                     weekdays=None,
                     pct_axis=True,
                     xaxis_locator=None,
                     xaxis_formatter=None,
                     figsize=(16, 5)):
    
    rolling_colors = ["#c0392b", "#27ae60", "#2980b9", "#8e44ad"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, values, alpha=0.30, color="darkblue",
            linewidth=0.8, label=daily_label)

    for win, color in zip(rolling_windows, rolling_colors):
        ax.plot(dates, values.rolling(win, center=True).mean(),
                color=color, linewidth=2,
                label=rolling_label_fmt.format(win=win))

    overall = values.mean()
    label_text = mean_label_fmt.format(mean=overall)
    ax.axhline(overall, color="black", ls="--", lw=1, alpha=0.7,
               label=label_text)

    if shade_weekends:
        if weekdays is None:
            weekdays = pd.to_datetime(dates).dt.dayofweek
        for d, wd in zip(dates, weekdays):
            if wd >= 5:
                ax.axvspan(d - pd.Timedelta(hours=12),
                           d + pd.Timedelta(hours=12),
                           color="#f0e68c", alpha=0.25)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor="#f0e68c", alpha=0.4, label="Weekend"))
        ax.legend(handles=handles, loc="upper left", fontsize=9)
    else:
        ax.legend(loc="upper left", fontsize=9)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    if xaxis_locator is None:
        xaxis_locator = mdates.MonthLocator()
    if xaxis_formatter is None:
        xaxis_formatter = mdates.DateFormatter("%b")
    ax.xaxis.set_major_locator(xaxis_locator)
    ax.xaxis.set_major_formatter(xaxis_formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if pct_axis:
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.show()


def continuous_vs_binary(df, feature_col, target_col,
                         target_labels=("Not Cancelled", "Cancelled"),
                         feature_label=None, figsize=(14, 5),
                         n_bins=10):
    
    from scipy.stats import mannwhitneyu, pointbiserialr

    if feature_label is None:
        feature_label = feature_col

    sub = df[[feature_col, target_col]].dropna()
    g0 = sub.loc[sub[target_col] == 0, feature_col]
    g1 = sub.loc[sub[target_col] == 1, feature_col]

    colors = ["#27ae60", "#c0392b"]

    fig, (ax_kde, ax_box, ax_trend) = plt.subplots(
        1, 3, figsize=figsize, constrained_layout=True)

    #  KDE 
    for vals, label, color in [(g0, target_labels[0], colors[0]),
                                (g1, target_labels[1], colors[1])]:
        ax_kde.hist(vals, bins=50, density=True, alpha=0.25, color=color)
        vals.plot.kde(ax=ax_kde, color=color, linewidth=2, label=label)

    ax_kde.set_xlabel(feature_label)
    ax_kde.set_ylabel("Density")
    ax_kde.set_title(f"Distribution by {target_col}")
    ax_kde.legend(fontsize=9)

    #  boxplot 
    box_data = [g0.values, g1.values]
    bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.5,
                        tick_labels=target_labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax_box.set_ylabel(feature_label)
    ax_box.set_title("Box Plot")

    mean0, mean1 = g0.mean(), g1.mean()
    ax_box.scatter([1, 2], [mean0, mean1], color="black", zorder=5,
                   marker="D", s=40, label="Mean")
    ax_box.legend(fontsize=9)

    # binned 
    sub = sub.copy()
    sub["bin"] = pd.qcut(sub[feature_col], q=n_bins, duplicates="drop")
    binned = (sub.groupby("bin", observed=True)[target_col]
                 .agg(["mean", "count"])
                 .reset_index())
    binned.columns = ["bin", "cancel_rate", "n"]
    bin_labels = [f"{iv.mid:.1f}" for iv in binned["bin"]]

    ax_trend.plot(range(len(binned)), binned["cancel_rate"],
                  marker="o", color="#2c3e50", linewidth=2)
    overall = sub[target_col].mean()
    ax_trend.axhline(overall, color="black", ls="--", lw=1, alpha=0.7,
                     label=f"Overall mean ({overall:.2%})")
    ax_trend.set_xticks(range(len(binned)))
    ax_trend.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax_trend.set_xlabel(f"{feature_label} (bin midpoint)")
    ax_trend.set_ylabel("Cancellation Rate")
    ax_trend.set_title("Binned Cancel-Rate Trend")
    ax_trend.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_trend.legend(fontsize=9)

    plt.show()

    # stats tests
    U, p_mw = mannwhitneyu(g0, g1, alternative="two-sided")
    r_pb, p_pb = pointbiserialr(sub[target_col], sub[feature_col])

    pooled_std = np.sqrt(
        ((len(g0) - 1) * g0.std() ** 2 + (len(g1) - 1) * g1.std() ** 2)
        / (len(g0) + len(g1) - 2))
    d = (mean1 - mean0) / pooled_std

    print(f"Mann-Whitney U test:       U = {U:,.0f},  p = {p_mw:.4e}")
    if p_mw < 0.05:
        print("  → Distributions differ significantly (p < 0.05)")
    else:
        print("  → No significant distributional difference (p ≥ 0.05)")

    print(f"\nPoint-biserial correlation: r = {r_pb:+.4f},  p = {p_pb:.4e}")
    strength = ("negligible" if abs(r_pb) < 0.10
                else "weak" if abs(r_pb) < 0.30
                else "moderate" if abs(r_pb) < 0.50
                else "strong")
    direction = "positive" if r_pb > 0 else "negative"
    print(f"  → {strength} {direction} association")

    print(f"\nCohen's d:                 d = {d:+.4f}")
    magnitude = ("negligible" if abs(d) < 0.20
                 else "small" if abs(d) < 0.50
                 else "medium" if abs(d) < 0.80
                 else "large")
    print(f"  → {magnitude} effect size")
    print(f"\nGroup means:  {target_labels[0]} = {mean0:.3f},  "
          f"{target_labels[1]} = {mean1:.3f}  (Δ = {mean1 - mean0:+.3f})")

    return {
        "mann_whitney_U": U,
        "mann_whitney_p": p_mw,
        "point_biserial_r": r_pb,
        "point_biserial_p": p_pb,
        "cohens_d": d,
    }


def zone_breakdown(df, feature_col, target_col, zones,
                   bucket_size=0.5, feature_label=None,
                   figsize=(14, 7)):

    if feature_label is None:
        feature_label = feature_col

    sub = df[[feature_col, target_col]].dropna().copy()
    multiplier = round(1 / bucket_size)
    sub["bucket"] = (sub[feature_col] * multiplier).round(0) / multiplier
    detail = (sub.groupby("bucket")[target_col]
                 .agg(["mean", "count"]).reset_index())
    detail.columns = ["bucket", "cancel_rate", "n"]

    overall_rate = sub[target_col].mean()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15},
        constrained_layout=True)

    ax1.bar(range(len(detail)), detail["cancel_rate"],
            color="#2c3e50", alpha=0.7, width=0.8)
    ax1.axhline(overall_rate, color="red", ls="--", lw=1.5,
                label=f"Overall mean ({overall_rate:.2%})")
    ax1.set_xticks(range(len(detail)))
    ax1.set_xticklabels([f"{v:.1f}" for v in detail["bucket"]],
                         rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Cancellation Rate")
    ax1.set_title(f"Cancellation Rate by {feature_label} "
                  f"({bucket_size}-min buckets)")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.legend(fontsize=10)

    ax2.bar(range(len(detail)), detail["n"],
            color="#7f8c8d", alpha=0.6, width=0.8)
    ax2.set_xticks(range(len(detail)))
    ax2.set_xticklabels([f"{v:.1f}" for v in detail["bucket"]],
                         rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Ride Count")
    ax2.set_xlabel(f"{feature_label} (minutes)")

    plt.show()

    print("Zone summary:")
    for lo, hi, label in zones:
        mask = (sub[feature_col] >= lo) & (sub[feature_col] <= hi)
        zone = sub[mask]
        rate = zone[target_col].mean() if len(zone) else 0
        print(f"  {label:20s} [{lo:4.1f} – {hi:4.1f}]:  "
              f"n = {len(zone):6,}  rate = {rate:.4f}")


def continuous_by_category(df, feature_col, category_col,
                           feature_label=None, category_label=None,
                           figsize=(14, 12), cache_path=None):

    if feature_label is None:
        feature_label = feature_col
    if category_label is None:
        category_label = category_col

    sub = df.dropna(subset=[feature_col]).copy()
    print(f"Rows with {feature_label} present: {len(sub):,} / {len(df):,} "
          f"({len(df) - len(sub):,} dropped)")

    cat_order = sorted(sub[category_col].unique())
    groups = [grp[feature_col].values
              for _, grp in sub.groupby(category_col, observed=True)]

    H, p_kw = kruskal(*groups)

    ss_bw = sum(len(g) * (g.mean() - sub[feature_col].mean()) ** 2 for g in groups)
    ss_tot = ((sub[feature_col] - sub[feature_col].mean()) ** 2).sum()
    eta_sq = ss_bw / ss_tot

    if cache_path and os.path.exists(cache_path):
        img = plt.imread(cache_path)
        plt.figure(figsize=(figsize[0], figsize[1] // 2))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        fig, (ax_box, ax_violin) = plt.subplots(
            2, 1, figsize=figsize, constrained_layout=True)

        palette = plt.cm.Set2(np.linspace(0, 1, len(cat_order)))

        bp = ax_box.boxplot(groups, patch_artist=True, widths=0.5,
                            tick_labels=cat_order)
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        means = [g.mean() for g in groups]
        ax_box.scatter(range(1, len(means) + 1), means, color="black",
                       zorder=5, marker="D", s=40, label="Mean")
        ax_box.axhline(sub[feature_col].mean(), color="black", ls="--", lw=1,
                       alpha=0.7,
                       label=f"Overall mean ({sub[feature_col].mean():.2f})")
        ax_box.set_ylabel(feature_label)
        ax_box.set_title(f"{feature_label} by {category_label} — Box Plot")
        ax_box.legend(fontsize=9)
        plt.setp(ax_box.get_xticklabels(), rotation=30, ha="right")

        vp = ax_violin.violinplot(groups, showmeans=True, showmedians=True)
        for idx, body in enumerate(vp["bodies"]):
            body.set_facecolor(palette[idx])
            body.set_alpha(0.6)
        ax_violin.set_xticks(range(1, len(cat_order) + 1))
        ax_violin.set_xticklabels(cat_order, rotation=30, ha="right")
        ax_violin.set_ylabel(feature_label)
        ax_violin.set_title(f"{feature_label} by {category_label} — Violin Plot")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            plt.savefig(cache_path)
        plt.show()

    grp_stats = (sub.groupby(category_col, observed=True)[feature_col]
                    .agg(["count", "mean", "median", "std"])
                    .sort_values("mean", ascending=False))
    grp_stats.columns = ["n", "mean", "median", "std"]

    print(f"\nKruskal-Wallis H test: H = {H:.2f}, p = {p_kw:.4e}")
    if p_kw < 0.05:
        print(f"  → Significant difference in {feature_label} across "
              f"{category_label} (p < 0.05)")
    else:
        print(f"  → No significant difference across {category_label} (p ≥ 0.05)")

    print(f"\nEta-squared (η²): {eta_sq:.4f}")
    magnitude = ("negligible" if eta_sq < 0.01
                 else "small" if eta_sq < 0.06
                 else "medium" if eta_sq < 0.14
                 else "large")
    print(f"  → {magnitude} effect size")

    print(f"\nGroup statistics:")
    print(grp_stats.to_string())

    return {
        "kruskal_H": H,
        "kruskal_p": p_kw,
        "eta_squared": eta_sq,
        "group_stats": grp_stats,
    }


def categorical_vs_categorical(df, col_a, col_b,
                                col_a_label=None, col_b_label=None,
                                top_n=20, heatmap_figsize=(18, 6),
                                cache_prefix=None):

    if col_a_label is None:
        col_a_label = col_a
    if col_b_label is None:
        col_b_label = col_b

    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, expected = chi2_contingency(ct)
    v = cramers_v(df[col_a], df[col_b])

    print(f"Chi-square: χ² = {chi2:.2f}, dof = {dof}, p = {p:.4e}")
    print(f"Cramér\'s V = {v:.4f}")
    mag = ("negligible" if v < 0.10 else "small" if v < 0.30
           else "moderate" if v < 0.50 else "large")
    print(f"  → {mag} association")

    prop = ct.div(ct.sum(axis=1), axis=0)
    top_cols = ct.sum(axis=0).nlargest(top_n).index
    prop_top = prop[top_cols]

    std_res = (ct.values - expected) / np.sqrt(expected)
    std_res_df = pd.DataFrame(std_res, index=ct.index, columns=ct.columns)
    std_res_top = std_res_df[top_cols]

    # heatmap
    prop_cache = f"{cache_prefix}_proportions.png" if cache_prefix else None

    if prop_cache and os.path.exists(prop_cache):
        img = plt.imread(prop_cache)
        plt.figure(figsize=heatmap_figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=heatmap_figsize, constrained_layout=True)
        sns.heatmap(prop_top, annot=True, fmt=".2%", cmap="YlOrRd",
                    linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Row proportion"})
        ax.set_title(f"{col_a_label} × Top-{top_n} {col_b_label} "
                     f"(row-normalised proportions)")
        ax.set_xlabel(col_b_label)
        ax.set_ylabel(col_a_label)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if prop_cache:
            os.makedirs(os.path.dirname(prop_cache) or ".", exist_ok=True)
            plt.savefig(prop_cache)
        plt.show()

    #  res heatmap 
    res_cache = f"{cache_prefix}_residuals.png" if cache_prefix else None

    if res_cache and os.path.exists(res_cache):
        img = plt.imread(res_cache)
        plt.figure(figsize=heatmap_figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=heatmap_figsize, constrained_layout=True)
        sns.heatmap(std_res_top, center=0, cmap="RdBu_r", annot=True,
                    fmt=".1f", linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Std. residual"})
        ax.set_title(f"Standardised Residuals: {col_a_label} × Top-{top_n} "
                     f"{col_b_label} (|r| > 2 noteworthy)")
        ax.set_xlabel(col_b_label)
        ax.set_ylabel(col_a_label)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if res_cache:
            os.makedirs(os.path.dirname(res_cache) or ".", exist_ok=True)
            plt.savefig(res_cache)
        plt.show()

    noteworthy = (std_res_df.abs() > 2).sum().sum()
    print(f"\nCells with |std residual| > 2: {noteworthy} / {std_res_df.size}")

    return {
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "cramers_v": v,
        "std_residuals": std_res_df,
        "noteworthy_count": noteworthy,
    }

