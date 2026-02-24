"""
Fraym Color Palettes
====================
Centralized color constants for consistent Fraym visualizations.

Load via:
    from utils.fraym_palettes import *
    # or
    import sys; sys.path.insert(0, '..'); from utils.fraym_palettes import *

See docs/VISUAL_STANDARDS.md for full specifications.
"""

# ==============================================================================
# PRIMARY PALETTE
# ==============================================================================

FRAYM_PRIMARY = {
    "dark_blue":     "#00162b",
    "electric_blue": "#202da5",
    "teal":          "#196160",
    "aqua":          "#1dd8b0",
    "bright_green":  "#94d931",
}

# ==============================================================================
# NEUTRAL PALETTE
# ==============================================================================

FRAYM_NEUTRAL = {
    "charcoal":   "#393e50",
    "dark_gray":  "#696b78",
    "gray":       "#d6d9dd",
    "pale_gray":  "#f2f2f2",
    "dark_sand":  "#8f9092",
    "sand":       "#d8d5ca",
    "pale_sand":  "#efeee8",
}

# ==============================================================================
# EXTENDED PALETTE
# ==============================================================================

FRAYM_EXTENDED = {
    "purple":     "#7152e2",
    "dark_red":   "#5b2036",
    "red":        "#d44244",
    "orange":     "#e8b934",
    "yellow":     "#efeb6a",
    "dark_green": "#237d07",
}

# ==============================================================================
# SEQUENTIAL COLOR RAMPS  (for maps and continuous data)
# ==============================================================================

FRAYM_SEQUENTIAL = {
    # General purpose - light to dark teal
    "hello_darkness":    ["#f2f2f2", "#1dd3b0", "#196160"],

    # High contrast - dark to light with multiple hues
    "magma":             ["#0b162b", "#7152e2", "#d44244", "#e8b934", "#efe6ba"],

    # Positive indicators - light to dark green
    "go_green":          ["#f2f2f2", "#94d931", "#257d07"],

    # Complex gradient - warm to cool
    "off_grid":          ["#efe6ba", "#94d931", "#1dd3b0", "#2024a5", "#196160", "#0b162b"],

    # Simple blue gradient
    "candy_floss":       ["#f2f2f2", "#2024a5"],

    # Grayscale (ONLY for B&W printing)
    "grayscale":         ["#ffffff", "#000000"],

    # Red gradient for intensity
    "candy_apple":       ["#efeee8", "#d44244", "#3b2036"],

    # Population data - blue tones
    "population_blues":  ["#f2f2f2", "#1dd3b0", "#196160", "#2024a5"],
}

# ==============================================================================
# DIVERGENT COLOR RAMPS  (for data with a meaningful midpoint)
# ==============================================================================

FRAYM_DIVERGENT = {
    # Teal to beige
    "sunshine":           ["#196160", "#f2f2f2", "#efe6ba"],

    # Aqua to purple
    "polar":              ["#1dd3b0", "#f2f2f2", "#7152e2"],

    # Warm gradient
    "peach_rings":        ["#efeee8", "#efe6ba", "#e8b934", "#d44244"],

    # Red to aqua
    "hot_and_cold":       ["#d44244", "#393e80", "#1dd3b0"],

    # Purple gradient
    "concord":            ["#f2f2f2", "#7152e2", "#3b2036"],

    # Colorblind-friendly (PREFERRED for accessibility)
    "colorblind_friendly": ["#2024a5", "#f2f2f2", "#d44244"],
}

# ==============================================================================
# CHART-SPECIFIC PALETTES
# ==============================================================================

FRAYM_CHARTS = {
    # Single color for simple bar charts
    "single_bar":       "#196160",   # Teal

    # Two-color comparisons
    "comparison_teal":  ["#196160", "#1dd8b0"],
    "comparison_gray":  ["#393e50", "#d6d9dd"],

    # Opinion scales (5 levels: positive → negative)
    "opinion_5":        ["#196160", "#1dd8b0", "#d6d9dd", "#e8b934", "#d44244"],

    # Intensity scales (5 levels: low → high)
    "intensity_5":      ["#efe6ba", "#e8b934", "#e8763a", "#d44244", "#5b2036"],

    # Ranking scales (5 levels)
    "rank_5":           ["#c8e6e5", "#7fcbc8", "#1dd3b0", "#196160", "#0b3938"],

    # Line charts (2 series)
    "line_2":           ["#1dd8b0", "#d44244"],

    # Scatter plots
    "scatter_primary":  "#196160",
    "scatter_neutral":  "#d6d9dd",
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_fraym_color(color_name: str) -> str:
    """Return the hex code for a named Fraym color.

    Examples
    --------
    >>> get_fraym_color("teal")
    '#196160'
    >>> get_fraym_color("purple")
    '#7152e2'
    """
    for palette in (FRAYM_PRIMARY, FRAYM_NEUTRAL, FRAYM_EXTENDED):
        if color_name in palette:
            return palette[color_name]
    raise ValueError(f"Color '{color_name}' not found in Fraym palettes.")


def get_fraym_palette(palette_name: str) -> list:
    """Return the color list for a named Fraym palette.

    Examples
    --------
    >>> get_fraym_palette("hello_darkness")
    ['#f2f2f2', '#1dd3b0', '#196160']
    """
    for lookup in (FRAYM_SEQUENTIAL, FRAYM_DIVERGENT, FRAYM_CHARTS):
        if palette_name in lookup:
            val = lookup[palette_name]
            return val if isinstance(val, list) else [val]
    raise ValueError(f"Palette '{palette_name}' not found in Fraym palettes.")


def list_fraym_palettes() -> None:
    """Print all available Fraym color palettes."""
    print("=== FRAYM COLOR PALETTES ===\n")

    print("PRIMARY COLORS:")
    for name, hex_val in FRAYM_PRIMARY.items():
        print(f"  {name}: {hex_val}")

    print("\nSEQUENTIAL RAMPS (for maps):")
    for name, colors in FRAYM_SEQUENTIAL.items():
        n = len(colors)
        print(f"  {name} ({n} colors)")

    print("\nDIVERGENT RAMPS (for maps with midpoint):")
    for name, colors in FRAYM_DIVERGENT.items():
        n = len(colors)
        print(f"  {name} ({n} colors)")

    print("\nCHART PALETTES:")
    for name, val in FRAYM_CHARTS.items():
        n = len(val) if isinstance(val, list) else 1
        print(f"  {name} ({n} color{'s' if n > 1 else ''})")

    print("\nFor detailed specifications, see: docs/VISUAL_STANDARDS.md")
