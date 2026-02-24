"""
Survey Statistics Utilities
============================
Functions for calculating weighted survey statistics following Fraym standards.

Mirrors the R srvyr-based survey.R — every function has the same name,
parameters, and return shape so existing analysis code is easy to port.

Dependencies:  pandas, numpy, scipy
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Optional


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================

def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted mean (ignores NaN rows handled upstream)."""
    return float(np.average(values, weights=weights))


def _weighted_se(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted standard error of the mean.

    Uses the textbook formula for a weighted sample:
        se = sqrt(sum(w_i * (x_i - x_bar)^2) / (n_eff - 1)) / sqrt(n_eff)
    where n_eff = (sum w)^2 / sum(w^2)
    """
    w = weights / weights.sum()            # normalize
    x_bar = np.average(values, weights=w)
    variance = np.average((values - x_bar) ** 2, weights=w)
    n_eff = weights.sum() ** 2 / (weights ** 2).sum()
    se = np.sqrt(variance / n_eff)
    return float(se)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median via sorted cumulative weight."""
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_wts = weights[order]
    cumulative = np.cumsum(sorted_wts) / sorted_wts.sum()
    idx = np.searchsorted(cumulative, 0.5)
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


# ==============================================================================
# PUBLIC API
# ==============================================================================

def national_weighted_stats(
    df: pd.DataFrame,
    indicator_cols: list[str] | str,
    weight_col: str = "weight",
    strata_col: Optional[str] = None,     # reserved; not yet used in calculation
    cluster_col: Optional[str] = None,    # reserved; not yet used in calculation
    ci: bool = False,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Calculate national-level weighted statistics for one or more indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data frame.
    indicator_cols : list[str] or str
        Column name(s) to analyse.
    weight_col : str
        Weight column name (default: "weight").
    strata_col : str, optional
        Stratification column (reserved for future complex-survey support).
    cluster_col : str, optional
        Cluster column (reserved for future complex-survey support).
    ci : bool
        If True, add ci_lower / ci_upper columns (default: False).
    ci_level : float
        Confidence level for intervals (default: 0.95).

    Returns
    -------
    pd.DataFrame
        Columns: indicator, weighted_mean, se, weighted_median,
                 n, total_weight, [ci_lower, ci_upper].

    Examples
    --------
    >>> stats = national_weighted_stats(df, ["literacy_rate", "numeracy_rate"],
    ...                                  weight_col="pop_wgt_unclustered")
    >>> stats_ci = national_weighted_stats(df, "literacy_rate", ci=True)
    """
    if isinstance(indicator_cols, str):
        indicator_cols = [indicator_cols]

    rows = []
    for col in indicator_cols:
        subset = df[[col, weight_col]].dropna()
        if subset.empty:
            warnings.warn(f"All values missing for '{col}'. Skipping.")
            continue

        vals = subset[col].to_numpy(dtype=float)
        wts  = subset[weight_col].to_numpy(dtype=float)

        w_mean   = _weighted_mean(vals, wts)
        w_se     = _weighted_se(vals, wts)
        w_median = _weighted_median(vals, wts)

        row = {
            "indicator":       col,
            "weighted_mean":   w_mean,
            "se":              w_se,
            "weighted_median": w_median,
            "n":               len(subset),
            "total_weight":    float(wts.sum()),
        }

        if ci:
            z = scipy_stats.norm.ppf(0.5 + ci_level / 2)
            row["ci_lower"] = w_mean - z * w_se
            row["ci_upper"] = w_mean + z * w_se

        rows.append(row)

    return pd.DataFrame(rows)


def subnational_weighted_stats(
    df: pd.DataFrame,
    indicator_col: str,
    groupby_cols: list[str] | str,
    weight_col: str = "weight",
    strata_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
    min_n: int = 30,
) -> pd.DataFrame:
    """Calculate weighted statistics broken down by geography or demographics.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data frame.
    indicator_col : str
        Single indicator column to analyse.
    groupby_cols : list[str] or str
        Grouping column(s), e.g. "adm1_name" or ["adm1_name", "urban_rural"].
    weight_col : str
        Weight column name (default: "weight").
    min_n : int
        Groups below this sample size are flagged with small_sample=True (default: 30).

    Returns
    -------
    pd.DataFrame
        Columns: <groupby_cols>, weighted_mean, se, n, total_weight, small_sample.

    Examples
    --------
    >>> regional = subnational_weighted_stats(df, "literacy_rate", "adm1_name",
    ...                                        weight_col="pop_wgt_unclustered")
    >>> urban_rural = subnational_weighted_stats(df, "income",
    ...                                           ["adm1_name", "urban_rural"])
    """
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    cols_needed = groupby_cols + [indicator_col, weight_col]
    subset = df[cols_needed].dropna(subset=[indicator_col, weight_col])

    def _agg(group: pd.DataFrame) -> pd.Series:
        vals = group[indicator_col].to_numpy(dtype=float)
        wts  = group[weight_col].to_numpy(dtype=float)
        return pd.Series({
            "weighted_mean": _weighted_mean(vals, wts),
            "se":            _weighted_se(vals, wts),
            "n":             len(group),
            "total_weight":  float(wts.sum()),
        })

    result = (
        subset
        .groupby(groupby_cols, observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    result["small_sample"] = result["n"] < min_n

    return result


def weighted_crosstab(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    weight_col: str = "weight",
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """Weighted crosstabulation of two categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data frame.
    row_col : str
        Column for rows.
    col_col : str
        Column for columns.
    value_col : str
        Numeric column to aggregate (weighted mean within each cell).
    weight_col : str
        Weight column name (default: "weight").
    normalize : str or None
        None  — raw weighted values
        "all" — percentage of grand total
        "row" — percentage within each row
        "col" — percentage within each column

    Returns
    -------
    pd.DataFrame
        Long-format with columns: row_col, col_col, weighted_value.

    Examples
    --------
    >>> ct = weighted_crosstab(df, "education_level", "gender", "employed")
    >>> ct_pct = weighted_crosstab(df, "education", "gender", "employed",
    ...                             normalize="row")
    """
    subset = df[[row_col, col_col, value_col, weight_col]].dropna()

    def _wmean(grp: pd.DataFrame) -> float:
        vals = grp[value_col].to_numpy(dtype=float)
        wts  = grp[weight_col].to_numpy(dtype=float)
        return _weighted_mean(vals, wts)

    result = (
        subset
        .groupby([row_col, col_col], observed=True)
        .apply(_wmean, include_groups=False)
        .reset_index(name="weighted_value")
    )

    if normalize == "all":
        result["weighted_value"] /= result["weighted_value"].sum()
    elif normalize == "row":
        row_totals = result.groupby(row_col, observed=True)["weighted_value"].transform("sum")
        result["weighted_value"] /= row_totals
    elif normalize == "col":
        col_totals = result.groupby(col_col, observed=True)["weighted_value"].transform("sum")
        result["weighted_value"] /= col_totals
    elif normalize is not None:
        raise ValueError(f"normalize must be None, 'all', 'row', or 'col'. Got: {normalize!r}")

    return result


def time_series_stats(
    df: pd.DataFrame,
    indicator_col: str,
    time_col: str,
    weight_col: str = "weight",
    groupby_col: Optional[str] = None,
) -> pd.DataFrame:
    """Weighted statistics over time periods, optionally by subgroup.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data frame.
    indicator_col : str
        Indicator column to track over time.
    time_col : str
        Time column (e.g., "survey_year").
    weight_col : str
        Weight column name (default: "weight").
    groupby_col : str, optional
        Additional grouping column (e.g., "adm1_name").

    Returns
    -------
    pd.DataFrame
        Columns: time_col, [groupby_col], weighted_mean, se, n, total_weight.

    Examples
    --------
    >>> ts = time_series_stats(df, "unemployment_rate", "survey_year",
    ...                         weight_col="pop_wgt_unclustered")
    >>> regional_ts = time_series_stats(df, "income", "survey_year",
    ...                                  groupby_col="adm1_name")
    """
    group_vars = [time_col] if groupby_col is None else [time_col, groupby_col]
    cols_needed = group_vars + [indicator_col, weight_col]
    subset = df[cols_needed].dropna(subset=[indicator_col, weight_col])

    def _agg(grp: pd.DataFrame) -> pd.Series:
        vals = grp[indicator_col].to_numpy(dtype=float)
        wts  = grp[weight_col].to_numpy(dtype=float)
        return pd.Series({
            "weighted_mean": _weighted_mean(vals, wts),
            "se":            _weighted_se(vals, wts),
            "n":             len(grp),
            "total_weight":  float(wts.sum()),
        })

    return (
        subset
        .groupby(group_vars, observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )


def calculate_design_effect(
    df: pd.DataFrame,
    indicator_col: str,
    weight_col: str = "weight",
) -> float:
    """Calculate the design effect (DEFF) for a weighted survey indicator.

    DEFF > 1 indicates clustering effects; effective sample size = n / DEFF.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data frame.
    indicator_col : str
        Indicator column.
    weight_col : str
        Weight column name (default: "weight").

    Returns
    -------
    float
        Design effect value.

    Examples
    --------
    >>> deff = calculate_design_effect(df, "literacy_rate",
    ...                                 weight_col="pop_wgt_unclustered")
    >>> print(f"DEFF = {deff:.2f}, effective n = {len(df) / deff:.0f}")
    """
    subset = df[[indicator_col, weight_col]].dropna()
    wts = subset[weight_col].to_numpy(dtype=float)

    n               = len(wts)
    sum_w           = wts.sum()
    sum_w_sq        = (wts ** 2).sum()
    n_eff           = (sum_w ** 2) / sum_w_sq
    return float(n / n_eff)
