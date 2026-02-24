# Fraym Utility Functions Reference (Python)

Complete documentation for all utility functions. Load everything at once with:

```python
import sys; sys.path.insert(0, "..")
from utils.import_all import *
```

---

## Survey Statistics  (`utils/survey.py`)

### national_weighted_stats()

Calculate national-level weighted statistics for one or more indicators.

**Signature:**
```python
national_weighted_stats(
    df: pd.DataFrame,
    indicator_cols: list[str] | str,
    weight_col: str = "weight",
    strata_col: str | None = None,
    cluster_col: str | None = None,
    ci: bool = False,
    ci_level: float = 0.95,
) -> pd.DataFrame
```

**Returns:** DataFrame with columns `indicator`, `weighted_mean`, `se`, `weighted_median`, `n`, `total_weight`, and optionally `ci_lower`/`ci_upper`.

**Examples:**
```python
# Simple usage
stats = national_weighted_stats(
    df, ["literacy_rate", "numeracy_rate"],
    weight_col="pop_wgt_unclustered",
)

# With 95% confidence intervals
stats_ci = national_weighted_stats(
    df, "literacy_rate",
    weight_col="pop_wgt_unclustered",
    ci=True,
)
```

---

### subnational_weighted_stats()

Weighted statistics broken down by geography or demographics.

**Signature:**
```python
subnational_weighted_stats(
    df: pd.DataFrame,
    indicator_col: str,
    groupby_cols: list[str] | str,
    weight_col: str = "weight",
    strata_col: str | None = None,
    cluster_col: str | None = None,
    min_n: int = 30,
) -> pd.DataFrame
```

**Returns:** DataFrame with grouping columns plus `weighted_mean`, `se`, `n`, `total_weight`, `small_sample`.

**Examples:**
```python
# By region
regional = subnational_weighted_stats(
    df, indicator_col="literacy_rate",
    groupby_cols="adm1_name",
    weight_col="pop_wgt_unclustered",
)

# Check flagged small samples
print(regional[regional["small_sample"]])

# Multiple groupings
urban_regional = subnational_weighted_stats(
    df, "literacy_rate",
    groupby_cols=["adm1_name", "urban_rural"],
    weight_col="pop_wgt_unclustered",
    min_n=50,
)
```

---

### weighted_crosstab()

Weighted crosstabulation of two categorical variables.

**Signature:**
```python
weighted_crosstab(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    weight_col: str = "weight",
    normalize: str | None = None,   # None | "all" | "row" | "col"
) -> pd.DataFrame
```

**Returns:** Long-format DataFrame with `row_col`, `col_col`, `weighted_value`.

**Examples:**
```python
# Basic crosstab
ct = weighted_crosstab(
    df, row_col="education_level", col_col="gender",
    value_col="employed", weight_col="pop_wgt_unclustered",
)

# Row-normalized (percentages within each education level)
ct_pct = weighted_crosstab(
    df, "education_level", "gender", "employed",
    weight_col="pop_wgt_unclustered",
    normalize="row",
)

# Convert to wide for tables
ct_wide = ct_pct.pivot(index="education_level",
                        columns="gender",
                        values="weighted_value")
```

---

### time_series_stats()

Weighted statistics over time, optionally by subgroup.

**Signature:**
```python
time_series_stats(
    df: pd.DataFrame,
    indicator_col: str,
    time_col: str,
    weight_col: str = "weight",
    groupby_col: str | None = None,
) -> pd.DataFrame
```

**Examples:**
```python
# National time series
ts = time_series_stats(
    df, "unemployment_rate", "survey_year",
    weight_col="pop_wgt_unclustered",
)

# Regional time series
ts_regional = time_series_stats(
    df, "poverty_rate", "survey_year",
    weight_col="pop_wgt_unclustered",
    groupby_col="adm1_name",
)
```

---

### calculate_design_effect()

Design effect (DEFF) for a weighted survey indicator.

**Signature:**
```python
calculate_design_effect(
    df: pd.DataFrame,
    indicator_col: str,
    weight_col: str = "weight",
) -> float
```

**Example:**
```python
deff = calculate_design_effect(df, "literacy_rate",
                                weight_col="pop_wgt_unclustered")
eff_n = len(df) / deff
print(f"DEFF = {deff:.2f}, effective n ≈ {eff_n:.0f}")
```

---

## Visualization Functions  (`utils/visualization.py`)

### Color Palettes

```python
from utils.fraym_palettes import FRAYM_PRIMARY, FRAYM_SEQUENTIAL, FRAYM_CHARTS

FRAYM_PRIMARY["teal"]              # "#196160"
FRAYM_SEQUENTIAL["hello_darkness"] # ["#f2f2f2", "#1dd3b0", "#196160"]
FRAYM_CHARTS["opinion_5"]          # list of 5 hex colors

# Get any color or palette by name
get_fraym_color("teal")           # → "#196160"
get_fraym_palette("colorblind_friendly")   # → list of hex strings
```

---

### create_choropleth()

Choropleth map from a GeoDataFrame.

**Signature:**
```python
create_choropleth(
    sf_data: gpd.GeoDataFrame,
    value_col: str,
    title: str | None = None,
    subtitle: str | None = None,
    legend_title: str | None = None,
    ramp_name: str = "hello_darkness",
    custom_ramp: list | None = None,
    boundary_color: str = "white",
    boundary_size: float = 0.3,
    figsize: tuple = (10, 8),
) -> plt.Figure
```

**Available ramps:** hello_darkness (default), magma, go_green, population_blues, candy_floss, candy_apple, off_grid, colorblind_friendly, sunshine, polar, hot_and_cold, concord, peach_rings

**Example:**
```python
import geopandas as gpd

gdf = gpd.read_file(DATA_PKG / "Zonal Statistics" / "iraq_adm1.gpkg")
gdf = gdf.merge(zonal_stats, on="adm1_pcode")

fig = create_choropleth(
    gdf, "literacy_rate",
    title="Literacy Rates by Governorate",
    subtitle="Adults aged 18+, Iraq 2025",
    legend_title="Literacy Rate (%)",
    ramp_name="population_blues",
)
save_fraym_plot(fig, FIGURES / f"literacy_map_{TODAY}.png")
```

---

### create_raster_map()

Map from a GeoTIFF raster with optional boundary overlay.

**Signature:**
```python
create_raster_map(
    raster_source: str | Path,
    title: str | None = None,
    subtitle: str | None = None,
    legend_title: str = "Value",
    ramp_name: str = "hello_darkness",
    custom_ramp: list | None = None,
    boundaries_gdf: gpd.GeoDataFrame | None = None,
    boundary_color: str = "black",
    boundary_size: float = 0.5,
    figsize: tuple = (10, 8),
) -> plt.Figure
```

**Example:**
```python
fig = create_raster_map(
    DATA_PKG / "Rasters" / "irq_econ_now_good.tif",
    title="Economic Optimism",
    subtitle="% who say current economic conditions are good, 2025",
    legend_title="Percentage (%)",
    ramp_name="go_green",
    boundaries_gdf=gdf,
)
save_fraym_plot(fig, FIGURES / f"econ_optimism_raster_{TODAY}.png",
                width=12, height=10)
```

---

### create_bar_standard()

Vertical bar chart for comparing categories.

**Signature:**
```python
create_bar_standard(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str | None = None,
    subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    sort_values: bool = True,
    show_values: bool = True,
    value_format: str = "{:.0f}%",
    figsize: tuple = (10, 6),
) -> plt.Figure
```

**Example:**
```python
regional = subnational_weighted_stats(df, "literacy_rate", "adm1_name",
                                       weight_col=WEIGHT_COL)
regional["literacy_pct"] = regional["weighted_mean"] * 100

fig = create_bar_standard(
    regional, x_col="adm1_name", y_col="literacy_pct",
    title="Literacy Rates by Governorate",
    subtitle="% adults 18+ who are literate, Iraq 2025",
    value_format="{:.1f}%",
)
save_fraym_plot(fig, FIGURES / f"literacy_bars_{TODAY}.png")
```

---

### create_bar_horizontal()

Horizontal bar chart (best for long category labels).

```python
fig = create_bar_horizontal(
    data, x_col="percentage", y_col="policy_statement",
    title="Agreement with Policy Statements",
    subtitle="% who agree or strongly agree, n=4,070",
)
```

---

### create_bar_comparison()

Grouped bars comparing two or more series.

```python
fig = create_bar_comparison(
    data_long, x_col="region", y_col="literacy_pct",
    group_col="urban_rural",
    title="Urban-Rural Literacy Gap by Governorate",
    subtitle="% literate among adults 18+, 2025",
    colors="teal",   # or "gray" or a custom list
)
```

---

### create_bar_stacked()

Stacked bars for showing composition.

```python
fig = create_bar_stacked(
    opinion_data, x_col="survey_year",
    y_col="percentage", fill_col="opinion_level",
    title="Trust in Government Over Time",
    subtitle="Distribution of responses, 2020-2025",
    palette="opinion_5",  # or "intensity_5", "rank_5", or custom list
)
```

---

### create_line_chart()

Line chart for time-series or trends.

```python
# Single series
fig = create_line_chart(
    ts_data, x_col="year", y_cols="unemployment_rate",
    title="National Unemployment Rate",
    subtitle="Annual average, 2015-2025",
)

# Multiple series
fig = create_line_chart(
    ts_data, x_col="year",
    y_cols=["urban_unemployment", "rural_unemployment"],
    title="Urban vs Rural Unemployment",
    legend_labels=["Urban", "Rural"],
)
```

---

### create_scatter_plot()

Scatter plot with optional linear trendline.

```python
fig = create_scatter_plot(
    agg_data, x_col="literacy_rate", y_col="poverty_rate",
    title="Literacy and Poverty by District",
    subtitle="Each point = one district",
    add_trendline=True,
)
```

---

### save_fraym_plot()

Save a figure to disk at Fraym standards (300 DPI, white background).

```python
save_fraym_plot(fig, "work/figures/my_chart_2026-02-24.png")
save_fraym_plot(fig, "work/figures/map_2026-02-24.png", width=12, height=10)
```

---

## Spatial Functions  (`utils/spatial.py`)

### fraym_login()

```python
# Uses INFRAYM_USER and INFRAYM_PASSWORD env vars
fraym_login()

# Or explicit
fraym_login("me@fraym.io", "mypassword")
```

### list_place_groups() / download_place_group()

```python
groups = list_place_groups("IRQ", admin_division_type="Governorate")
print(groups[["id", "description"]])

gdf = download_place_group(groups["id"].iloc[0])
```

### download_worldpop()

```python
arr = download_worldpop("IRQ", 2020,
                         save_to="work/irq_pop_2020.tif")

women = download_worldpop("IRQ", 2020,
                           age_lower=15, age_upper=49, gender="f",
                           save_to="work/irq_women_1549.tif")
```

### calc_zonal_stats()

```python
gdf = gpd.read_file(DATA_PKG / "Zonal Statistics" / "iraq_adm1.gpkg")
result = calc_zonal_stats(
    raster_path=str(DATA_PKG / "Rasters" / "irq_econ_now_good.tif"),
    zones=gdf,
    fun="mean",
    output_file="work/econ_zonal_adm1.csv",
)
```

---

## Complete Analysis Workflow

```python
"""Complete workflow example."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import geopandas as gpd
from utils.import_all import *

ROOT       = Path(__file__).parent.parent
DATA_PKG   = ROOT / "data" / "Iraq 2025"
FIGURES    = ROOT / "work" / "figures"
WEIGHT_COL = "pop_wgt_unclustered"
TODAY      = "2026-02-24"
FIGURES.mkdir(parents=True, exist_ok=True)

# 1. Load data
df      = pd.read_csv(DATA_PKG / "Training Data" / "training_data.csv")
zonal   = pd.read_csv(DATA_PKG / "Zonal Statistics" / "adm1_zonal_stats.csv")
gdf     = gpd.read_file(DATA_PKG / "Zonal Statistics" / "iraq_adm1.gpkg")

# 2. National statistics
national = national_weighted_stats(df, ["literacy_rate"], weight_col=WEIGHT_COL, ci=True)
print(national)

# 3. Subnational statistics
regional = subnational_weighted_stats(df, "literacy_rate", "adm1_name", weight_col=WEIGHT_COL)
regional["literacy_pct"] = regional["weighted_mean"] * 100

# 4. Bar chart
fig_bar = create_bar_standard(
    regional, x_col="adm1_name", y_col="literacy_pct",
    title="Literacy Rates by Governorate",
    subtitle="% adults 18+ who are literate, Iraq 2025",
)
save_fraym_plot(fig_bar, FIGURES / f"literacy_bars_{TODAY}.png")

# 5. Choropleth map
gdf_merged = gdf.merge(zonal, on="adm1_pcode")
fig_map = create_choropleth(
    gdf_merged, "literacy_rate",
    title="Literacy Rates by Governorate",
    legend_title="Literacy Rate (%)",
    ramp_name="population_blues",
)
save_fraym_plot(fig_map, FIGURES / f"literacy_map_{TODAY}.png", width=12, height=10)
```

---

## Tips

1. **Always use weights** — all survey functions require a weight column
2. **Check small_sample flag** — flag and caveat estimates where n < 30
3. **Use snake_case** for all column and variable names
4. **Prefer colorblind_friendly** divergent ramp for maximum accessibility
5. **Save consistently** — use `save_fraym_plot()` for Fraym-standard outputs
6. **Validate joins** — use `validate="1:1"` in `.merge()` to catch duplicates

For methodology details, see [FRAYM_METHODS.md](FRAYM_METHODS.md).
For visual design standards, see [VISUAL_STANDARDS.md](VISUAL_STANDARDS.md).
