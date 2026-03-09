import numpy as np
import pandas as pd
from scipy import stats


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


def cramers_v(x, y):
    """Compute Cramer's V statistic for two categorical Series."""
    confusion = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    min_dim = min(confusion.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))


def get_outliers(data, col):
    col_data = data[col].dropna()
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = col_data[(col_data < lower) | (col_data > upper)]

    if len(outliers) > 0:
        print(f"\n--- {col} outliers ---")
        print(f"  Range: [{outliers.min():.2f}, {outliers.max():.2f}]")
        print(f"  Mean: {outliers.mean():.2f}")
        print(f"  Median: {outliers.median():.2f}")
        print(f"  Count: {len(outliers)} ({len(outliers)/len(col_data)*100:.2f}%)")
    else:
        print(f"\n {col}: no outliers detected")


def spearman_trend(x, y, x_label="x", y_label="y"):
    """Compute Spearman correlation and print an interpretation.

    Parameters
    ----------
    x, y : array-like
        The two variables to correlate.
    x_label, y_label : str
        Human-readable names used in the printed summary.

    Returns
    -------
    rho : float
    p_value : float
    """
    rho, p_value = stats.spearmanr(x, y)
    print(f"Spearman correlation ({x_label} vs {y_label}): ρ = {rho:.4f}, p = {p_value:.4e}")

    if p_value < 0.05:
        direction = "increasing" if rho > 0 else "decreasing"
        strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.6 else "strong"
        print(f"  → Statistically significant {strength} {direction} trend (p < 0.05)")
    else:
        print("  → No statistically significant monotonic trend (p ≥ 0.05)")
    print()



def get_nans(data, col):
    cancelled_mask = data["is_cancelled"] == 1
    nan_mask = data[col].isna()

    exact_match = (cancelled_mask == nan_mask).all()
    print("All cancelled rows match NaNs in is_cancelled?", exact_match)

    print("Number of cancelled rows:", cancelled_mask.sum())
    print("Number of NaN avg_ctat rows:", nan_mask.sum())
    print("Number of rows where both are True:", (cancelled_mask & nan_mask).sum())