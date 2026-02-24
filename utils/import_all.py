"""
Import All Fraym Utilities
===========================
Master importer — loads all Fraym utility functions into the current namespace.
Equivalent to R's  source("../utils/source_all.R").

Usage (from a script in work/):
    import sys
    sys.path.insert(0, "..")
    from utils.import_all import *

Or import modules individually for cleaner namespacing:
    from utils.survey import national_weighted_stats, subnational_weighted_stats
    from utils.visualization import create_choropleth, save_fraym_plot
    from utils.fraym_palettes import FRAYM_PRIMARY, FRAYM_SEQUENTIAL
"""

# -- Color palettes --
from utils.fraym_palettes import (
    FRAYM_PRIMARY,
    FRAYM_NEUTRAL,
    FRAYM_EXTENDED,
    FRAYM_SEQUENTIAL,
    FRAYM_DIVERGENT,
    FRAYM_CHARTS,
    get_fraym_color,
    get_fraym_palette,
    list_fraym_palettes,
)

# -- Survey statistics --
from utils.survey import (
    national_weighted_stats,
    subnational_weighted_stats,
    weighted_crosstab,
    time_series_stats,
    calculate_design_effect,
)

# -- Visualization --
from utils.visualization import (
    create_choropleth,
    create_raster_map,
    create_bar_standard,
    create_bar_horizontal,
    create_bar_comparison,
    create_bar_stacked,
    create_line_chart,
    create_scatter_plot,
    save_fraym_plot,
    list_color_ramps,
)

# -- Spatial / API (auth required before use) --
from utils.spatial import (
    fraym_login,
    list_place_groups,
    download_place_group,
    download_default_place_group,
    download_country,
    download_worldpop,
    list_surveys,
    get_survey_url,
    calc_zonal_stats,
    fraym_spatial_help,
)

print("Loading Fraym utility functions...")
print("✓ Loaded: fraym_palettes")
print("✓ Loaded: survey")
print("✓ Loaded: visualization")
print("✓ Loaded: spatial")
print("\nAvailable functions:")

print("\nColor Palettes (fraym_palettes):")
print("  FRAYM_PRIMARY, FRAYM_NEUTRAL, FRAYM_EXTENDED (dicts)")
print("  FRAYM_SEQUENTIAL, FRAYM_DIVERGENT, FRAYM_CHARTS (dicts)")
print("  list_fraym_palettes()  get_fraym_color()  get_fraym_palette()")

print("\nSurvey Statistics (survey):")
print("  national_weighted_stats()     subnational_weighted_stats()")
print("  weighted_crosstab()           time_series_stats()")
print("  calculate_design_effect()")

print("\nVisualization (visualization):")
print("  create_choropleth()    create_raster_map()    create_bar_standard()")
print("  create_bar_horizontal() create_bar_comparison() create_bar_stacked()")
print("  create_line_chart()    create_scatter_plot()   save_fraym_plot()")

print("\nfraymr API (spatial) — requires fraym_login():")
print("  fraym_login()                  list_place_groups()")
print("  download_place_group()         download_default_place_group()")
print("  download_country()             download_worldpop()")
print("  list_surveys()                 get_survey_url()")
print("  calc_zonal_stats()             fraym_spatial_help()")

print("\nReady to analyze!")
