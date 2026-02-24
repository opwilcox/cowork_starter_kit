# Modern Python Development Guide

*Best practices for Fraym data analysis in Python. Emphasizes readable, idiomatic, and reproducible code.*

## Core Principles

1. **Prefer explicitness** — Clear variable names beat clever one-liners
2. **Use pathlib everywhere** — Portable, readable, no string concatenation
3. **Profile before optimizing** — Only optimize after measuring
4. **Write reproducible scripts** — Same inputs always produce same outputs

---

## File Paths with pathlib

Always use `pathlib.Path` — never `os.path.join` or string concatenation.

```python
# Good — pathlib
from pathlib import Path
ROOT     = Path(__file__).parent.parent
DATA_PKG = ROOT / "data" / "Iraq 2025"
fig_path = ROOT / "work" / "figures" / "literacy_map_2026-02-24.png"

# Check existence before reading
if not DATA_PKG.exists():
    raise FileNotFoundError(f"Data package not found: {DATA_PKG}")

# Create output directories
fig_path.parent.mkdir(parents=True, exist_ok=True)

# Avoid — string concatenation
data_path = "../data/Iraq 2025/codebook.csv"   # breaks on Windows
```

---

## Data Loading

```python
import pandas as pd

# Standard CSV load
df = pd.read_csv(DATA_PKG / "Training Data" / "survey.csv")

# Always inspect on load
print(df.shape)
print(df.dtypes)
print(df.isna().sum())

# Load spatial data
import geopandas as gpd
gdf = gpd.read_file(DATA_PKG / "Zonal Statistics" / "iraq_adm1.gpkg")
```

---

## Pandas Modern Patterns

### Method chaining

```python
# Good — readable chain
result = (
    df
    .dropna(subset=["literacy_rate", "pop_wgt_unclustered"])
    .query("adm1_name != 'Unknown'")
    .assign(literacy_pct=lambda x: x["literacy_rate"] * 100)
    .sort_values("literacy_pct", ascending=False)
)

# Avoid — repeated assignment
df2 = df.dropna(subset=["literacy_rate"])
df3 = df2.query("adm1_name != 'Unknown'")
df4 = df3.copy()
df4["literacy_pct"] = df4["literacy_rate"] * 100
```

### Groupby with observed=True (suppresses categorical warnings)

```python
# Good
regional = (
    df
    .groupby("adm1_name", observed=True)
    .agg(mean_literacy=("literacy_rate", "mean"), n=("literacy_rate", "count"))
    .reset_index()
)

# Use named aggregation — clearer than dict syntax
stats = df.groupby("region", observed=True).agg(
    weighted_mean=("value", lambda g: np.average(g, weights=df.loc[g.index, "weight"])),
    n=("value", "count"),
).reset_index()
```

### Merging / joining

```python
# Always specify how= and validate= to catch unexpected duplicates
merged = gdf.merge(
    zonal_stats,
    on="adm1_pcode",
    how="left",
    validate="1:1",       # raises if not 1-to-1
)
print(f"Unmatched rows: {merged['indicator_value'].isna().sum()}")
```

---

## Weighted Statistics

Use the utility functions from `utils/survey.py` rather than rolling your own:

```python
from utils.survey import national_weighted_stats, subnational_weighted_stats

# National
nat = national_weighted_stats(df, ["literacy_rate", "food_security"],
                               weight_col="pop_wgt_unclustered", ci=True)

# Subnational
reg = subnational_weighted_stats(df, "literacy_rate", "adm1_name",
                                  weight_col="pop_wgt_unclustered")
# Flag unreliable small-sample estimates
print(reg[reg["small_sample"]])
```

For manual weighted calculations:

```python
# Weighted mean
w_mean = np.average(df["indicator"], weights=df["pop_wgt_unclustered"])

# Weighted standard deviation
w = df["pop_wgt_unclustered"] / df["pop_wgt_unclustered"].sum()
variance = np.average((df["indicator"] - w_mean) ** 2, weights=w)
w_std = np.sqrt(variance)
```

---

## F-strings for Formatting

```python
# Good — f-strings
label = f"{region}: {pct:.1f}%"
filename = f"work/figures/literacy_bars_{date}.png"
print(f"n = {n:,}  |  weighted mean = {mean:.3f}")

# Avoid
label = "{}: {:.1f}%".format(region, pct)
label = "%s: %.1f%%" % (region, pct)
```

---

## Type Hints

Use type hints in function signatures — they serve as inline documentation:

```python
from typing import Optional, Union

def compute_gap(
    df: pd.DataFrame,
    group_col: str,
    indicator_col: str,
    weight_col: str = "pop_wgt_unclustered",
    min_n: int = 30,
) -> pd.DataFrame:
    """Return weighted mean gap between group pairs."""
    ...
```

---

## Reproducibility

```python
# Set random seed when sampling or using any stochastic operation
import numpy as np
np.random.seed(42)

# Record package versions in a comment block at the top of scripts
# pandas==2.1.4  numpy==1.26.2  geopandas==0.14.2  matplotlib==3.8.2
```

---

## Script Structure Template

```python
"""
Script description.
Author: <name>
Date: YYYY-MM-DD
"""
# ── stdlib ──────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# ── allow imports from project root ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ── Fraym utilities ─────────────────────────────────────────────────────────
from utils.import_all import *

# ── project constants ───────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PKG   = ROOT / "data" / "Iraq 2025"
WEIGHT_COL = "pop_wgt_unclustered"
FIGURES    = ROOT / "work" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
TODAY      = "2026-02-24"   # use: from datetime import date; TODAY = date.today().isoformat()

# ── load data ────────────────────────────────────────────────────────────────
codebook = pd.read_csv(DATA_PKG / "codebook.csv")
df = pd.read_csv(DATA_PKG / "Training Data" / "training_data.csv")

# ── analysis ─────────────────────────────────────────────────────────────────
nat_stats = national_weighted_stats(df, ["literacy_rate"], weight_col=WEIGHT_COL)
print(nat_stats)

# ── visualization ────────────────────────────────────────────────────────────
fig = create_bar_standard(
    nat_stats.rename(columns={"indicator": "Indicator", "weighted_mean": "Mean"}),
    x_col="Indicator", y_col="Mean",
    title="National Statistics",
)
save_fraym_plot(fig, FIGURES / f"national_stats_{TODAY}.png")
```

---

## Common Gotchas

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: utils` | Add `sys.path.insert(0, "..")` before imports |
| Pandas `FutureWarning` on groupby | Add `observed=True` to all `groupby()` calls |
| Matplotlib font warnings | Ignore or install `matplotlib-fonttools` |
| GeoDataFrame CRS mismatch | `gdf.to_crs(epsg=4326)` before merging/plotting |
| rasterio nodata as -9999 | `arr[arr == src.nodata] = np.nan` after reading |
| Empty figure from `plt.show()` in script | Use `fig.savefig(...)` instead of `plt.show()` |
