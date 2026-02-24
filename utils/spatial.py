"""
Fraym Spatial Utilities
========================
Python wrappers for the Fraym (InfraYm) REST API and common spatial operations.
Mirrors spatial.R — same function names and return types where possible.

geopandas replaces sf; rasterio replaces terra.

Dependencies:  requests, geopandas, rasterio, numpy, pandas

Credentials: Set environment variables before calling fraym_login():
    INFRAYM_USER=your_email
    INFRAYM_PASSWORD=your_password

Or pass them explicitly to fraym_login().
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
import geopandas as gpd


# ==============================================================================
# SESSION STATE
# ==============================================================================

_SESSION: requests.Session = requests.Session()
_BASE_URL: str = "https://api.fraym.io"   # update if the endpoint changes
_AUTHENTICATED: bool = False


# ==============================================================================
# AUTHENTICATION
# ==============================================================================

def fraym_login(
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Authenticate with the Fraym API.

    Reads credentials from INFRAYM_USER / INFRAYM_PASSWORD env vars by default.
    Call once per session before any api_* functions.

    Parameters
    ----------
    user : str, optional
        Email address (falls back to INFRAYM_USER env var).
    password : str, optional
        Password (falls back to INFRAYM_PASSWORD env var).

    Examples
    --------
    >>> fraym_login()                           # reads from env vars
    >>> fraym_login("me@example.com", "pass")   # explicit credentials
    """
    global _SESSION, _AUTHENTICATED

    user     = user     or os.environ.get("INFRAYM_USER")
    password = password or os.environ.get("INFRAYM_PASSWORD")

    if not user or not password:
        raise ValueError(
            "Credentials not found. Set INFRAYM_USER and INFRAYM_PASSWORD "
            "environment variables, or pass them to fraym_login()."
        )

    resp = _SESSION.post(
        f"{_BASE_URL}/auth/login",
        json={"email": user, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("token") or resp.json().get("access_token")
    if token:
        _SESSION.headers.update({"Authorization": f"Bearer {token}"})
    _AUTHENTICATED = True
    print("Fraym login successful.")


def _require_auth() -> None:
    if not _AUTHENTICATED:
        raise RuntimeError("Not authenticated. Call fraym_login() first.")


# ==============================================================================
# PLACE GROUPS
# ==============================================================================

def list_place_groups(
    iso3_code: str,
    place_type: Optional[str] = None,
    admin_division_type: Optional[str] = None,
    active_only: bool = True,
) -> pd.DataFrame:
    """List available place groups for a country.

    Parameters
    ----------
    iso3_code : str
        ISO-3 country code (e.g. "IRQ", "USA").
    place_type : str, optional
        Filter: "Country", "Urban", "Administrative Division", "City".
    admin_division_type : str, optional
        Filter sub-type, e.g. "Governorate", "State", "County".
    active_only : bool
        Return only active place groups (default: True).

    Returns
    -------
    pd.DataFrame
        Columns: id, description, placeType, adminDivisionType,
                 admLevel, source, numberFeatures, isDefault, isActive.

    Examples
    --------
    >>> groups = list_place_groups("IRQ")
    >>> admins = list_place_groups("IRQ", admin_division_type="Governorate")
    """
    _require_auth()
    params = {
        "iso3": iso3_code,
        "activeOnly": active_only,
    }
    if place_type:
        params["placeType"] = place_type
    if admin_division_type:
        params["adminDivisionType"] = admin_division_type

    resp = _SESSION.get(f"{_BASE_URL}/place-groups", params=params, timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())

    desired_cols = [
        "id", "description", "placeType", "adminDivisionType",
        "admLevel", "source", "numberFeatures", "isDefault", "isActive",
    ]
    existing = [c for c in desired_cols if c in df.columns]
    return df[existing].sort_values(["admLevel", "adminDivisionType"],
                                    ignore_index=True)


def download_place_group(id: int) -> gpd.GeoDataFrame:
    """Download a place group by its ID.

    Parameters
    ----------
    id : int
        Place group ID from list_place_groups()["id"].

    Returns
    -------
    geopandas.GeoDataFrame

    Examples
    --------
    >>> groups  = list_place_groups("IRQ", admin_division_type="Governorate")
    >>> gdf     = download_place_group(groups["id"].iloc[0])
    """
    _require_auth()
    resp = _SESSION.get(
        f"{_BASE_URL}/place-groups/{int(id)}/geojson", timeout=60
    )
    resp.raise_for_status()
    return gpd.GeoDataFrame.from_features(resp.json()["features"])


def download_default_place_group(
    iso3_code: str,
    place_type: str,
    admin_division_type: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Download the default (canonical) boundary for a country and type.

    Parameters
    ----------
    iso3_code : str
        ISO-3 country code.
    place_type : str
        "Country", "Urban", "Administrative Division", or "City".
    admin_division_type : str, optional
        Sub-type filter.

    Examples
    --------
    >>> country = download_default_place_group("IRQ", "Country")
    >>> govs    = download_default_place_group("IRQ", "Administrative Division",
    ...                                         admin_division_type="Governorate")
    """
    _require_auth()
    params = {"iso3": iso3_code, "placeType": place_type, "default": True}
    if admin_division_type:
        params["adminDivisionType"] = admin_division_type

    groups = list_place_groups(
        iso3_code,
        place_type=place_type,
        admin_division_type=admin_division_type,
    )
    default = groups[groups.get("isDefault", False) == True]
    if default.empty:
        default = groups
    if default.empty:
        raise ValueError(f"No place group found for {iso3_code} / {place_type}.")
    return download_place_group(int(default["id"].iloc[0]))


def download_country(iso3_code: str) -> gpd.GeoDataFrame:
    """Download the national boundary polygon.

    Examples
    --------
    >>> iraq = download_country("IRQ")
    >>> iraq.plot()
    """
    return download_default_place_group(iso3_code, "Country")


# ==============================================================================
# WORLDPOP
# ==============================================================================

def download_worldpop(
    iso3_code: str,
    year: int,
    age_lower: Optional[int] = None,
    age_upper: Optional[int] = None,
    gender: Optional[str] = None,
    mask_to_country: bool = False,
    partial_age_ranges: bool = True,
    save_to: Optional[str] = None,
) -> np.ndarray:
    """Download a WorldPop population raster.

    Parameters
    ----------
    iso3_code : str
        ISO-3 country code.
    year : int
        Year (WorldPop typically covers 2000-2020).
    age_lower : int, optional
        Lower age bound.
    age_upper : int, optional
        Upper age bound.
    gender : str, optional
        "m" (male), "f" (female), or None (total).
    mask_to_country : bool
        Clip to country boundary (default: False).
    partial_age_ranges : bool
        Include partial age-band overlaps (default: True).
    save_to : str, optional
        Path to save the GeoTIFF (e.g., "work/pop_iraq_2020.tif").

    Returns
    -------
    numpy.ndarray
        Raster array (call rasterio.open(save_to) for full metadata).

    Examples
    --------
    >>> arr = download_worldpop("IRQ", 2020, save_to="work/irq_pop_2020.tif")
    >>> women = download_worldpop("IRQ", 2020, age_lower=15, age_upper=49,
    ...                           gender="f", save_to="work/irq_women_1549.tif")
    """
    _require_auth()
    params = {
        "iso3": iso3_code, "year": int(year),
        "maskToCountry": mask_to_country,
        "partialAgeRanges": partial_age_ranges,
    }
    if age_lower is not None: params["ageLower"] = int(age_lower)
    if age_upper is not None: params["ageUpper"] = int(age_upper)
    if gender     is not None: params["gender"]   = gender

    resp = _SESSION.get(f"{_BASE_URL}/worldpop", params=params, timeout=120)
    resp.raise_for_status()

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, "wb") as f:
            f.write(resp.content)
        print(f"Saved WorldPop raster: {save_to}")
        import rasterio
        with rasterio.open(save_to) as src:
            return src.read(1)
    else:
        import io, rasterio
        with rasterio.open(io.BytesIO(resp.content)) as src:
            return src.read(1)


# ==============================================================================
# SURVEYS
# ==============================================================================

def list_surveys(
    iso3_code: str,
    start_year: int,
    raw_or_processed: str = "processed",
    limit: int = 10,
) -> pd.DataFrame:
    """List available surveys for a country.

    Examples
    --------
    >>> surveys = list_surveys("IRQ", start_year=2020)
    """
    _require_auth()
    params = {
        "iso3": iso3_code,
        "startYear": int(start_year),
        "type": raw_or_processed,
        "limit": int(limit),
    }
    resp = _SESSION.get(f"{_BASE_URL}/surveys", params=params, timeout=30)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def get_survey_url(survey_id: int, raw_or_processed: str = "processed") -> str:
    """Get the download URL for a specific survey.

    Examples
    --------
    >>> surveys = list_surveys("IRQ", 2020)
    >>> url = get_survey_url(surveys["id"].iloc[0])
    """
    _require_auth()
    params = {"type": raw_or_processed}
    resp = _SESSION.get(
        f"{_BASE_URL}/surveys/{int(survey_id)}/url",
        params=params, timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("url", "")


# ==============================================================================
# SPATIAL UTILITIES
# ==============================================================================

def calc_zonal_stats(
    raster_path: str,
    zones: gpd.GeoDataFrame,
    weight_raster_path: Optional[str] = None,
    fun: str = "mean",
    output_file: Optional[str] = None,
    nodata: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """Aggregate raster values within polygon zones (optionally population-weighted).

    Parameters
    ----------
    raster_path : str
        Path to the indicator GeoTIFF.
    zones : geopandas.GeoDataFrame
        Polygon zones for aggregation.
    weight_raster_path : str, optional
        Path to a weight raster (e.g., population) for weighted aggregation.
    fun : str
        Aggregation function: "mean", "sum", "min", "max" (default: "mean").
    output_file : str, optional
        Path to save output as CSV.
    nodata : float, optional
        Value to treat as nodata (if not encoded in raster metadata).

    Returns
    -------
    geopandas.GeoDataFrame
        Input zones with an added 'zonal_stat' column.

    Examples
    --------
    >>> gdf = gpd.read_file("data/Iraq 2025/Zonal Statistics/iraq_adm1.gpkg")
    >>> result = calc_zonal_stats(
    ...     "data/Iraq 2025/Rasters/irq_econ_now_good.tif",
    ...     zones=gdf,
    ...     fun="mean",
    ...     output_file="work/econ_zonal_adm1.csv",
    ... )
    """
    try:
        from rasterstats import zonal_stats as _zonal_stats
    except ImportError:
        raise ImportError(
            "rasterstats is required for calc_zonal_stats(). "
            "Install with: pip install rasterstats"
        )

    result = _zonal_stats(
        zones,
        raster_path,
        stats=[fun],
        nodata=nodata,
        geojson_out=False,
    )

    zones = zones.copy()
    zones["zonal_stat"] = [r.get(fun) for r in result]

    if weight_raster_path:
        # weighted mean: sum(value * pop) / sum(pop)
        val_stats = _zonal_stats(zones, raster_path,   stats=["sum"], nodata=nodata)
        wt_stats  = _zonal_stats(zones, weight_raster_path, stats=["sum"], nodata=nodata)
        zones["zonal_stat"] = [
            (v["sum"] / w["sum"]) if w["sum"] else np.nan
            for v, w in zip(val_stats, wt_stats)
        ]

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        zones.drop(columns="geometry").to_csv(output_file, index=False)
        print(f"Saved zonal stats: {output_file}")

    return zones


# ==============================================================================
# QUICK REFERENCE
# ==============================================================================

def fraym_spatial_help() -> None:
    """Print a quick-reference summary of available spatial functions."""
    print("=== Fraym Spatial Utilities Quick Reference ===\n")
    print("AUTHENTICATION:")
    print("  fraym_login()                         # authenticate (once per session)\n")
    print("PLACE GROUPS:")
    print("  list_place_groups(iso3, ...)           # list groups + IDs")
    print("  download_place_group(id)              # download by id → GeoDataFrame")
    print("  download_default_place_group(iso3, .) # download default boundary")
    print("  download_country(iso3)                # national boundary\n")
    print("WORLDPOP:")
    print("  download_worldpop(iso3, year, ...)    # population raster → np.ndarray\n")
    print("SURVEYS:")
    print("  list_surveys(iso3, start_year)        # list surveys + IDs")
    print("  get_survey_url(survey_id)             # get download URL\n")
    print("SPATIAL ANALYSIS:")
    print("  calc_zonal_stats(raster, zones, ...)  # aggregate raster by polygon\n")
    print("PLACE TYPE OPTIONS:")
    print("  'Country' | 'Urban' | 'Administrative Division' | 'City'")
