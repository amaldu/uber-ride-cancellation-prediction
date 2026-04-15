"""Generate all report charts as static PNGs."""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def target_distribution(univar: dict, out: str) -> None:
    """Bar chart of completed vs cancelled rides."""
    t = univar["target"]
    labels = ["Completed", "Cancelled"]
    values = [t["completed"], t["cancelled"]]
    colors = ["#27ae60", "#c0392b"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{val:,}", ha="center", va="bottom", fontweight="bold",
        )
    ax.set_ylabel("Number of Rides")
    ax.set_title(f"Target Distribution (Cancellation Rate: {t['cancellation_rate']:.1%})")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    _save(fig, out)


def vtat_zones(bivar: dict, out: str) -> None:
    """Bar chart of cancellation rate by VTAT behavioural zone."""
    zones = bivar["vtat_zones"]
    labels = list(zones.keys())
    rates = [zones[z]["cancel_rate"] for z in labels]
    counts = [zones[z]["n"] for z in labels]
    range_labels = [zones[z]["range"] for z in labels]
    colors = ["#27ae60", "#2ecc71", "#7f8c8d", "#e67e22", "#c0392b"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [2, 1]})

    bars = ax1.bar(labels, rates, color=colors, edgecolor="white")
    for bar, rate, rl in zip(bars, rates, range_labels):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{rate:.0%}\n({rl} min)", ha="center", va="bottom", fontsize=9,
        )
    ax1.set_ylabel("Cancellation Rate")
    ax1.set_title("Cancellation Rate by VTAT Zone")
    ax1.set_ylim(0, 1.15)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(labels, counts, color=colors, edgecolor="white", alpha=0.6)
    ax2.set_ylabel("Ride Count")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:,.0f}"))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, out)


def temporal_flatness(df: pd.DataFrame, out: str) -> None:
    """Three-panel chart showing cancellation rate is flat across time dimensions."""
    df_t = df[["date", "time", "is_cancelled"]].copy()
    df_t["date"] = pd.to_datetime(df_t["date"])
    df_t["hour"] = pd.to_datetime(df_t["time"], format="%H:%M:%S").dt.hour
    df_t["weekday"] = df_t["date"].dt.dayofweek
    df_t["month"] = df_t["date"].dt.month

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    overall = df["is_cancelled"].mean()

    for ax, col, title, xlabels in [
        (axes[0], "hour", "Hourly", [f"{h:02d}" for h in range(24)]),
        (axes[1], "weekday", "Daily", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
        (axes[2], "month", "Monthly",
         ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
    ]:
        rates = df_t.groupby(col)["is_cancelled"].mean()
        x = rates.index
        ax.bar(range(len(x)), rates.values, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axhline(overall, color="red", linestyle="--", linewidth=1.5, label=f"Mean ({overall:.1%})")
        ax.set_title(f"{title} Cancellation Rate")
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(xlabels[:len(x)], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.25, 0.40)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Temporal Features Show No Predictive Signal", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out)


def vtat_missing_signal(bivar: dict, out: str) -> None:
    """Side-by-side bar chart of cancellation rate by vtat_missing status."""
    rates = bivar["vtat_missing"]["rates"]
    labels = ["VTAT Present", "VTAT Missing"]
    values = [rates.get("0", rates.get("0.0", 0)), rates.get("1", rates.get("1.0", 0))]
    colors = ["#27ae60", "#c0392b"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.4)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.0%}", ha="center", va="bottom", fontweight="bold",
        )
    phi = bivar["vtat_missing"]["phi"]
    ax.set_title(f"vtat_missing vs Cancellation (phi = {phi:.3f})")
    ax.set_ylabel("Cancellation Rate")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    _save(fig, out)


def location_top10(bivar: dict, out: str) -> None:
    """Horizontal bar chart of top-10 pickup and drop locations by cancellation rate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    overall_rate = 0.32

    for ax, key, title in [
        (ax1, "pickup_location", "Top 10 Pickup Locations"),
        (ax2, "drop_location", "Top 10 Drop Locations"),
    ]:
        top10 = bivar[key]["top10"]
        locs = list(top10.keys())
        rates = [top10[l]["rate"] for l in locs]

        colors = ["#c0392b" if r > overall_rate else "#7f8c8d" for r in rates]
        ax.barh(locs[::-1], rates[::-1], color=colors[::-1], edgecolor="white")
        ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Cancellation Rate")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.suptitle("Location Cancellation Rates (vs 32% baseline)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out)


def correlation_heatmap(multivar: dict, out: str) -> None:
    """Heatmap of Pearson correlations between numeric features."""
    corr = multivar["correlation_vif"]["correlation"]
    corr_df = pd.DataFrame(corr)

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson Correlation Matrix")
    plt.tight_layout()
    _save(fig, out)


def generate_all(
    df: pd.DataFrame,
    univar: dict,
    bivar: dict,
    multivar: dict,
    charts_dir: str,
) -> dict[str, str]:
    """Generate all charts and return a mapping of chart name to file path."""
    paths: dict[str, str] = {}

    mapping = {
        "target_distribution": (target_distribution, (univar,)),
        "vtat_zones": (vtat_zones, (bivar,)),
        "temporal_flatness": (temporal_flatness, (df,)),
        "vtat_missing_signal": (vtat_missing_signal, (bivar,)),
        "location_top10": (location_top10, (bivar,)),
        "correlation_heatmap": (correlation_heatmap, (multivar,)),
    }

    for name, (fn, args) in mapping.items():
        path = os.path.join(charts_dir, f"{name}.png")
        fn(*args, path)
        paths[name] = path

    return paths
