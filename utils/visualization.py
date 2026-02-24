"""
Visualization Utilities
========================
Functions for creating publication-ready charts and maps following Fraym standards.
Mirrors visualization.R — same function names, parameters, and output style.

Dependencies:  matplotlib, pandas, numpy, geopandas, rasterio, matplotlib-scalebar (optional)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from utils.fraym_palettes import (
    FRAYM_PRIMARY, FRAYM_NEUTRAL, FRAYM_EXTENDED, FRAYM_CHARTS,
    FRAYM_SEQUENTIAL, FRAYM_DIVERGENT,
    get_fraym_palette,
)

# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================

_BASE_THEME = {
    "figure.facecolor":        "white",
    "axes.facecolor":          "#f7f7f7",
    "axes.grid":               True,
    "grid.color":              "#e8e8e8",
    "grid.linewidth":          0.5,
    "axes.spines.top":         False,
    "axes.spines.right":       False,
    "font.family":             "sans-serif",
    "axes.titlesize":          14,
    "axes.titleweight":        "bold",
    "axes.titlepad":           10,
    "axes.labelsize":          11,
    "xtick.labelsize":         10,
    "ytick.labelsize":         10,
    "legend.fontsize":         10,
    "legend.frameon":          False,
}


def _apply_base_theme() -> None:
    plt.rcParams.update(_BASE_THEME)


def _ramp_to_cmap(colors: list[str], name: str = "fraym_ramp") -> mcolors.LinearSegmentedColormap:
    """Convert a Fraym ramp (list of hex strings) to a matplotlib colormap."""
    return mcolors.LinearSegmentedColormap.from_list(name, colors)


def _resolve_ramp(ramp_name: str | None, custom_ramp: list | None) -> list[str]:
    """Return a list of hex colors from ramp_name or custom_ramp."""
    if custom_ramp is not None:
        return custom_ramp
    ramp_name = ramp_name or "hello_darkness"
    all_ramps = {**FRAYM_SEQUENTIAL, **FRAYM_DIVERGENT}
    if ramp_name not in all_ramps:
        warnings.warn(f"Ramp '{ramp_name}' not found. Using 'hello_darkness'.")
        ramp_name = "hello_darkness"
    return all_ramps[ramp_name]


# ==============================================================================
# MAP FUNCTIONS
# ==============================================================================

def create_choropleth(
    sf_data,                               # geopandas GeoDataFrame
    value_col: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    legend_title: Optional[str] = None,
    ramp_name: str = "hello_darkness",
    custom_ramp: Optional[list] = None,
    boundary_color: str = "white",
    boundary_size: float = 0.3,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Create a choropleth map from a GeoDataFrame.

    Parameters
    ----------
    sf_data : geopandas.GeoDataFrame
        Spatial data with geometries and values.
    value_col : str
        Column name containing values to map.
    title : str, optional
        Map title.
    subtitle : str, optional
        Map subtitle / source note.
    legend_title : str, optional
        Legend label (defaults to value_col).
    ramp_name : str
        Fraym color ramp name (default: 'hello_darkness').
    custom_ramp : list, optional
        Custom list of hex colors (overrides ramp_name).
    boundary_color : str
        Boundary line color (default: 'white').
    boundary_size : float
        Boundary line width (default: 0.3).
    figsize : tuple
        Figure size in inches (default: (10, 8)).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.read_file("data/Iraq 2025/Zonal Statistics/iraq_adm1.gpkg")
    >>> gdf = gdf.merge(zonal_stats, on="adm1_pcode")
    >>> fig = create_choropleth(gdf, "literacy_rate",
    ...                         legend_title="Literacy Rate (%)",
    ...                         ramp_name="population_blues")
    """
    colors = _resolve_ramp(ramp_name, custom_ramp)
    cmap   = _ramp_to_cmap(colors)
    if legend_title is None:
        legend_title = value_col

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    sf_data.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        edgecolor=boundary_color,
        linewidth=boundary_size,
        legend=True,
        legend_kwds={
            "label":       legend_title,
            "orientation": "horizontal",
            "shrink":      0.6,
            "pad":         0.02,
        },
        missing_kwds={"color": "#cccccc", "label": "No data"},
    )

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=12)
    if subtitle:
        fig.text(0.12, 0.02, subtitle, fontsize=9, color="#555555")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def create_raster_map(
    raster_source,                         # file path string or numpy array tuple
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    legend_title: str = "Value",
    ramp_name: str = "hello_darkness",
    custom_ramp: Optional[list] = None,
    boundaries_gdf=None,                   # geopandas GeoDataFrame (optional overlay)
    boundary_color: str = "black",
    boundary_size: float = 0.5,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Create a map from a raster file with optional boundary overlay.

    Parameters
    ----------
    raster_source : str or (np.ndarray, dict)
        Path to a GeoTIFF file, OR a tuple of (array, transform_dict) from rasterio.
    title : str, optional
        Map title.
    subtitle : str, optional
        Map subtitle / source note.
    legend_title : str
        Legend label (default: 'Value').
    ramp_name : str
        Fraym color ramp name (default: 'hello_darkness').
    custom_ramp : list, optional
        Custom list of hex colors.
    boundaries_gdf : geopandas.GeoDataFrame, optional
        Boundary polygons to overlay on the raster.
    boundary_color : str
        Boundary line color (default: 'black').
    boundary_size : float
        Boundary line width (default: 0.5).
    figsize : tuple
        Figure size in inches (default: (10, 8)).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = create_raster_map(
    ...     "data/Iraq 2025/Rasters/irq_econ_now_good.tif",
    ...     legend_title="Economic Optimism (%)",
    ...     ramp_name="go_green",
    ...     boundaries_gdf=adm1_gdf,
    ... )
    """
    import rasterio
    from rasterio.plot import show as rasterio_show

    colors = _resolve_ramp(ramp_name, custom_ramp)
    cmap   = _ramp_to_cmap(colors)

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    # -- read and plot raster --
    if isinstance(raster_source, (str, Path)):
        with rasterio.open(raster_source) as src:
            arr = src.read(1).astype(float)
            arr[arr == src.nodata] = np.nan
            extent = [src.bounds.left, src.bounds.right,
                      src.bounds.bottom, src.bounds.top]
    else:
        arr, extent = raster_source  # user-supplied (array, extent)

    im = ax.imshow(
        arr,
        cmap=cmap,
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )

    # -- colorbar --
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        shrink=0.6, pad=0.02)
    cbar.set_label(legend_title, fontsize=10, fontweight="bold")

    # -- optional boundary overlay --
    if boundaries_gdf is not None:
        boundaries_gdf.plot(
            ax=ax, facecolor="none",
            edgecolor=boundary_color,
            linewidth=boundary_size,
        )

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=12)
    if subtitle:
        fig.text(0.12, 0.02, subtitle, fontsize=9, color="#555555")

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# ==============================================================================
# BAR CHART FUNCTIONS
# ==============================================================================

def create_bar_standard(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    sort_values: bool = True,
    show_values: bool = True,
    value_format: str = "{:.0f}%",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a standard vertical bar chart.

    Examples
    --------
    >>> fig = create_bar_standard(
    ...     regional_stats, x_col="adm1_name", y_col="literacy_pct",
    ...     title="Literacy Rates by Region",
    ...     subtitle="% of adults 18+ who are literate, 2025",
    ... )
    """
    df = data.copy()
    if sort_values:
        df = df.sort_values(y_col, ascending=False)

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)
    color = FRAYM_CHARTS["single_bar"]

    bars = ax.bar(
        range(len(df)), df[y_col],
        color=color, width=0.65,
    )
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df[x_col], rotation=0, ha="center")
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    if show_values:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ax.get_ylim()[1] * 0.01,
                value_format.format(h),
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )

    if title:    ax.set_title(title, loc="left")
    if subtitle: fig.text(0.12, -0.03, subtitle, fontsize=9, color="#555555")
    if xlabel:   ax.set_xlabel(xlabel)
    if ylabel:   ax.set_ylabel(ylabel)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def create_bar_horizontal(
    data: pd.DataFrame,
    x_col: str,         # values column
    y_col: str,         # categories column
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    sort_values: bool = True,
    show_values: bool = True,
    value_format: str = "{:.0f}%",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a horizontal bar chart (best for long category labels).

    Examples
    --------
    >>> fig = create_bar_horizontal(
    ...     data, x_col="percentage", y_col="policy_statement",
    ...     title="Agreement with Policy Statements",
    ... )
    """
    df = data.copy()
    if sort_values:
        df = df.sort_values(x_col, ascending=True)  # ascending so largest is on top

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)
    color = FRAYM_CHARTS["single_bar"]

    bars = ax.barh(range(len(df)), df[x_col], color=color, height=0.65)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df[y_col])
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    if show_values:
        max_val = df[x_col].max()
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                value_format.format(w),
                va="center", ha="left",
                fontsize=9, fontweight="bold",
                color=FRAYM_PRIMARY["teal"],
            )

    if title:    ax.set_title(title, loc="left")
    if subtitle: fig.text(0.12, -0.03, subtitle, fontsize=9, color="#555555")
    if xlabel:   ax.set_xlabel(xlabel)
    if ylabel:   ax.set_ylabel(ylabel)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def create_bar_comparison(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colors: Union[str, list] = "teal",
    show_values: bool = True,
    value_format: str = "{:.0f}%",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a grouped bar chart for comparing categories.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data with a group_col distinguishing series.
    colors : str or list
        'teal' (default), 'gray', or a list of hex strings.

    Examples
    --------
    >>> fig = create_bar_comparison(
    ...     data_long, x_col="region", y_col="literacy_pct",
    ...     group_col="urban_rural",
    ...     title="Urban-Rural Literacy Gap by Region",
    ... )
    """
    if colors == "teal":
        color_vals = FRAYM_CHARTS["comparison_teal"]
    elif colors == "gray":
        color_vals = FRAYM_CHARTS["comparison_gray"]
    else:
        color_vals = colors if isinstance(colors, list) else [colors]

    groups    = data[group_col].unique()
    categories = data[x_col].unique()
    n_groups   = len(groups)
    x          = np.arange(len(categories))
    width      = 0.7 / n_groups

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (grp, color) in enumerate(zip(groups, color_vals)):
        mask = data[group_col] == grp
        vals = data.loc[mask].set_index(x_col).reindex(categories)[y_col]
        offset = (i - (n_groups - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9, color=color, label=grp)

        if show_values:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + ax.get_ylim()[1] * 0.01,
                        value_format.format(h),
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    if title:    ax.set_title(title, loc="left")
    if subtitle: fig.text(0.12, -0.03, subtitle, fontsize=9, color="#555555")
    if xlabel:   ax.set_xlabel(xlabel)
    if ylabel:   ax.set_ylabel(ylabel)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def create_bar_stacked(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    fill_col: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    palette: Union[str, list] = "opinion_5",
    horizontal: bool = False,
    show_values: bool = True,
    value_threshold: float = 10,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a stacked bar chart.

    Parameters
    ----------
    palette : str or list
        One of 'opinion_5', 'intensity_5', 'rank_5', or a custom list of hex colors.

    Examples
    --------
    >>> fig = create_bar_stacked(
    ...     opinion_data, x_col="survey_year",
    ...     y_col="percentage", fill_col="opinion_level",
    ...     title="Trust in Government Over Time",
    ...     palette="opinion_5",
    ... )
    """
    if isinstance(palette, str):
        color_vals = FRAYM_CHARTS.get(palette, get_fraym_palette(palette))
    else:
        color_vals = palette

    categories = data[x_col].unique()
    fill_levels = data[fill_col].unique()

    # pivot to wide (categories × fill levels)
    wide = data.pivot_table(
        index=x_col, columns=fill_col, values=y_col, aggfunc="sum"
    ).reindex(fill_levels, axis=1)

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)

    bottoms = np.zeros(len(wide))
    for i, (level, color) in enumerate(zip(fill_levels, color_vals)):
        vals = wide[level].to_numpy(dtype=float)
        if horizontal:
            ax.barh(range(len(wide)), vals, left=bottoms,
                    color=color, height=0.75, label=level)
        else:
            ax.bar(range(len(wide)), vals, bottom=bottoms,
                   color=color, width=0.75, label=level)

        if show_values:
            for j, (v, b) in enumerate(zip(vals, bottoms)):
                if v >= value_threshold:
                    cx = b + v / 2
                    if horizontal:
                        ax.text(cx, j, value_format_val(v),
                                ha="center", va="center",
                                fontsize=8, color="white", fontweight="bold")
                    else:
                        ax.text(j, cx, value_format_val(v),
                                ha="center", va="center",
                                fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    tick_labels = wide.index.tolist()
    if horizontal:
        ax.set_yticks(range(len(wide)))
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_xticks(range(len(wide)))
        ax.set_xticklabels(tick_labels)

    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    if title:    ax.set_title(title, loc="left")
    if subtitle: fig.text(0.12, -0.03, subtitle, fontsize=9, color="#555555")
    if xlabel:   ax.set_xlabel(xlabel)
    if ylabel:   ax.set_ylabel(ylabel)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def value_format_val(v: float) -> str:
    return f"{v:.0f}"


# ==============================================================================
# LINE CHART FUNCTIONS
# ==============================================================================

def create_line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_cols: Union[str, list[str]],
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colors: Optional[list] = None,
    line_size: float = 2.0,
    show_points: bool = True,
    point_size: float = 6,
    legend_labels: Optional[list[str]] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a line chart for time-series or trend data.

    Examples
    --------
    >>> fig = create_line_chart(
    ...     ts_data, x_col="year",
    ...     y_cols=["urban_unemployment", "rural_unemployment"],
    ...     title="Urban vs Rural Unemployment",
    ...     legend_labels=["Urban", "Rural"],
    ... )
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    if colors is None:
        if len(y_cols) == 2:
            colors = FRAYM_CHARTS["line_2"]
        else:
            colors = [
                FRAYM_PRIMARY["aqua"],
                FRAYM_EXTENDED["red"],
                FRAYM_PRIMARY["bright_green"],
                FRAYM_EXTENDED["purple"],
                FRAYM_EXTENDED["orange"],
            ]

    labels = legend_labels or y_cols

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)

    for col, color, label in zip(y_cols, colors, labels):
        ax.plot(data[x_col], data[col],
                color=color, linewidth=line_size, label=label)
        if show_points:
            ax.scatter(data[x_col], data[col],
                       color=color, s=point_size ** 2, zorder=5)

    if len(y_cols) > 1:
        ax.legend(frameon=False)

    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    if title:    ax.set_title(title, loc="left")
    if subtitle: fig.text(0.12, -0.03, subtitle, fontsize=9, color="#555555")
    if xlabel:   ax.set_xlabel(xlabel)
    if ylabel:   ax.set_ylabel(ylabel)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# ==============================================================================
# SCATTER PLOT FUNCTIONS
# ==============================================================================

def create_scatter_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: Optional[str] = None,
    size: float = 50,
    alpha: float = 0.7,
    add_trendline: bool = False,
    trendline_color: Optional[str] = None,
    trendline_size: float = 1.5,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Create a scatter plot with optional linear trendline.

    Examples
    --------
    >>> fig = create_scatter_plot(
    ...     agg_data, x_col="literacy_rate", y_col="poverty_rate",
    ...     title="Literacy and Poverty by District",
    ...     add_trendline=True,
    ... )
    """
    if color is None:
        color = FRAYM_CHARTS["scatter_neutral"] if add_trendline else FRAYM_CHARTS["scatter_primary"]
    if trendline_color is None:
        trendline_color = FRAYM_EXTENDED["red"]

    _apply_base_theme()
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(data[x_col], data[y_col], color=color, s=size, alpha=alpha)

    if add_trendline:
        clean = data[[x_col, y_col]].dropna()
        m, b = np.polyfit(clean[x_col], clean[y_col], 1)
        x_line = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
        ax.plot(x_line, m * x_line + b,
                color=trendline_color, linewidth=trendline_size)

    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.5)
    ax.set_axisbelow(True)

    if title:    ax.set_title(title, loc="left")
    if subtitle: fig.text(0.12, -0.03, subtitle, fontsize=9, color="#555555")
    if xlabel:   ax.set_xlabel(xlabel)
    if ylabel:   ax.set_ylabel(ylabel)

    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def save_fraym_plot(
    fig: plt.Figure,
    filename: str,
    width: float = 10,
    height: float = 8,
    dpi: int = 300,
) -> None:
    """Save a matplotlib figure to disk following Fraym standards.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filename : str
        Output path (e.g., "work/figures/literacy_map_2026-02-24.png").
    width : float
        Width in inches (default: 10).
    height : float
        Height in inches (default: 8).
    dpi : int
        Resolution (default: 300).

    Examples
    --------
    >>> save_fraym_plot(fig, "work/figures/literacy_bars_2026-02-24.png")
    >>> save_fraym_plot(fig, "work/figures/map_2026-02-24.png", width=12, height=10)
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(width, height)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {filename}")


def list_color_ramps() -> None:
    """Print all available Fraym color ramps."""
    print("Sequential Ramps:")
    print("  hello_darkness, magma, go_green, off_grid")
    print("  candy_floss, candy_apple, population_blues")
    print("  grayscale (use only when color not available)\n")
    print("Divergent Ramps:")
    print("  sunshine, polar, peach_rings, hot_and_cold")
    print("  concord, colorblind_friendly\n")
    print("Chart Palettes:")
    print("  single_bar, comparison_teal, comparison_gray")
    print("  opinion_5, intensity_5, rank_5")
    print("  line_2, scatter_primary, scatter_neutral")
