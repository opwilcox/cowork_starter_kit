---
name: fraym-analyst-python
description: >
  Use when the user asks to "analyze survey data", "create a map",
  "run weighted statistics", "explore a data package", "make a chart",
  "generate a crosstab", "visualize indicators", "run survey analysis",
  or needs guidance on Fraym data structures, color palettes, utility
  functions, or geospatial analysis workflows. Python edition — uses
  pandas, geopandas, matplotlib, and rasterio instead of R.
version: 1.0.0
---

# Fraym Data Analysis (Python / Cowork)

Fraym provides high-resolution geospatial data and analytics focused on population characteristics, behaviors, and attitudes. This skill encodes the complete Fraym analyst workflow in **Python**: data exploration, survey statistics, visualization, and spatial analysis.

---

## Environment Setup

Always write Python code to `.py` files and run with `python`. Never use multiline `python -c` strings.

```python
# Standard boilerplate for every script in work/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))   # ← enables utils imports

import pandas as pd
import numpy as np
from utils.import_all import *

ROOT       = Path(__file__).parent.parent
DATA_PKG   = ROOT / "data" / "Iraq 2025"   # ← update from CLAUDE.md
WEIGHT_COL = "pop_wgt_unclustered"          # ← update from CLAUDE.md
FIGURES    = ROOT / "work" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
```

---

## Data Package Structure

| Asset | Location | Use |
|---|---|---|
| **Codebook** | `data/{pkg}/codebook.csv` | indicator_id, indicator_name, category |
| **Training Data** | `data/{pkg}/Training Data/` | Raw weighted survey microdata; use weight column |
| **Zonal Statistics** | `data/{pkg}/Zonal Statistics/` | CSV (tabular), GPKG (maps) |
| **Rasters** | `data/{pkg}/Rasters/` | GeoTIFF at 1km², named `irq-[INDICATOR_ID].tif` |

```python
codebook = pd.read_csv(DATA_PKG / "codebook.csv")
df       = pd.read_csv(DATA_PKG / "Training Data" / "training_data.csv")
zonal    = pd.read_csv(DATA_PKG / "Zonal Statistics" / "adm1_zonal_stats.csv")

import geopandas as gpd
gdf = gpd.read_file(DATA_PKG / "Zonal Statistics" / "iraq_adm1.gpkg")
```

---

## Survey Statistics (Always Use Weights)

Five core functions from `utils/survey.py`:

| Function | Purpose | Key Params |
|---|---|---|
| `national_weighted_stats()` | National means with optional CIs | `indicator_cols`, `weight_col`, `ci`, `ci_level` |
| `subnational_weighted_stats()` | Stats by geography/demographics | `indicator_col`, `groupby_cols`, `min_n=30` |
| `weighted_crosstab()` | Crosstabulation | `row_col`, `col_col`, `value_col`, `normalize` |
| `time_series_stats()` | Stats over time | `indicator_col`, `time_col`, `groupby_col` |
| `calculate_design_effect()` | Survey DEFF | `indicator_col`, `weight_col` |

```python
# National statistics
nat = national_weighted_stats(
    df, ["literacy_rate", "food_security"],
    weight_col=WEIGHT_COL, ci=True,
)

# Subnational by governorate
reg = subnational_weighted_stats(
    df, "literacy_rate", "adm1_name",
    weight_col=WEIGHT_COL,
)
reg["literacy_pct"] = reg["weighted_mean"] * 100

# Flag small samples
print(reg[reg["small_sample"]])
```

**Rules:**
- Always pass `weight_col` — never use unweighted means
- Flag groups where n < 30 (`small_sample` column)
- Include national reference alongside subnational results

---

## Visualization

All chart functions return a `matplotlib.figure.Figure`. Save with `save_fraym_plot()`.

**Chart functions** from `utils/visualization.py`:

| Function | When to Use |
|---|---|
| `create_choropleth()` | Geographic variation across admin units |
| `create_raster_map()` | High-resolution 1km spatial data |
| `create_bar_standard()` | Vertical bars comparing categories |
| `create_bar_horizontal()` | Bars with long category labels |
| `create_bar_comparison()` | Grouped bars (e.g., urban vs rural) |
| `create_bar_stacked()` | Composition/opinion scales |
| `create_line_chart()` | Time series trends |
| `create_scatter_plot()` | Bivariate relationships |
| `save_fraym_plot()` | Save at 300 DPI, white background |

**Color ramps** for maps — pass as `ramp_name=`:
- Sequential: `hello_darkness` (default), `magma`, `population_blues`, `go_green`, `off_grid`, `candy_floss`, `candy_apple`
- Divergent: `colorblind_friendly` (preferred), `sunshine`, `polar`, `hot_and_cold`, `concord`, `peach_rings`

**Chart palettes** — access via `FRAYM_CHARTS["palette_name"]`:
- `single_bar`, `comparison_teal`, `comparison_gray`
- `opinion_5`, `intensity_5`, `rank_5`, `line_2`

```python
# Choropleth example
gdf_merged = gdf.merge(zonal, on="adm1_pcode")
fig = create_choropleth(
    gdf_merged, "literacy_rate",
    title="Literacy Rates by Governorate",
    legend_title="Literacy Rate (%)",
    ramp_name="population_blues",
)
save_fraym_plot(fig, FIGURES / f"literacy_map_{TODAY}.png",
                width=12, height=10)

# Bar chart example
fig = create_bar_standard(
    reg, x_col="adm1_name", y_col="literacy_pct",
    title="Literacy Rates by Governorate",
    subtitle="% adults 18+, Iraq 2025",
)
save_fraym_plot(fig, FIGURES / f"literacy_bars_{TODAY}.png")
```

**Rules:** No vertical axis text. No error bars unless requested. Bar width ≤ 0.75. Use subtitle for context and units. Save all figures to `work/figures/`.

---

## Spatial Analysis (Fraym API)

Authenticate once per session:
```python
fraym_login()   # reads INFRAYM_USER and INFRAYM_PASSWORD env vars
```

Key spatial functions from `utils/spatial.py`:

```python
# List and download boundaries
groups = list_place_groups("IRQ", admin_division_type="Governorate")
gdf    = download_place_group(groups["id"].iloc[0])

# Country boundary
iraq = download_country("IRQ")

# WorldPop raster
arr = download_worldpop("IRQ", 2020, save_to="work/irq_pop_2020.tif")

# Zonal statistics
result = calc_zonal_stats(
    raster_path=str(DATA_PKG / "Rasters" / "irq_econ_now_good.tif"),
    zones=gdf,
    fun="mean",
    output_file="work/econ_zonal_adm1.csv",
)
```

---

## Python Style Standards

- `pathlib.Path` for all file paths
- f-strings for string formatting
- `groupby(..., observed=True)` to suppress warnings
- `merge(..., validate="1:1")` to catch join duplicates
- Type hints in function signatures
- `np.average(values, weights=weights)` for inline weighted means

---

## File Naming

- Scripts:  `work/{name}_{YYYY-MM-DD}.py`
- Figures:  `work/figures/{name}_{type}_{YYYY-MM-DD}.png`

---

## Quality Checklist

- [ ] Sampling weights used in all survey calculations
- [ ] Geographic joins validated (no orphaned records)
- [ ] Sample sizes adequate (n ≥ 30); small groups flagged
- [ ] Indicator values in valid ranges; NaN handled
- [ ] Fraym color palettes used exclusively
- [ ] Clear titles and subtitles with context and units
- [ ] Data sources and sample sizes noted

---

## Detailed References

- **`docs/UTILITY_FUNCTIONS.md`** — Complete Python function signatures and examples
- **`docs/VISUAL_STANDARDS.md`** — Full color hex codes, chart specs, accessibility guidelines
- **`docs/PYTHON_STYLE_GUIDE.md`** — Modern Python patterns, pathlib, pandas idioms
- **`docs/FRAYM_METHODS.md`** — Survey methodology, ML model, validation, covariates
