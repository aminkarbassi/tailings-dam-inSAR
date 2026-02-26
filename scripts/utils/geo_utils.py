"""
Geospatial Utilities
=====================
Helper functions for bounding box operations, WKT geometry generation,
DEM tile identification, and coordinate conversions used across the pipeline.
"""

import configparser
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(cfg_path: str | Path) -> configparser.ConfigParser:
    """Load project.cfg and return a ConfigParser object."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#",))
    cfg.read(str(cfg_path))
    return cfg


def get_aoi(cfg: configparser.ConfigParser, buffer: float = 0.0) -> Tuple[float, float, float, float]:
    """
    Return (lat_min, lat_max, lon_min, lon_max) from project.cfg [aoi] section.

    Args:
        cfg:    Loaded ConfigParser object.
        buffer: Optional extra degrees to add on all sides.

    Returns:
        Tuple of (south, north, west, east) floats.
    """
    south = cfg.getfloat("aoi", "lat_min") - buffer
    north = cfg.getfloat("aoi", "lat_max") + buffer
    west  = cfg.getfloat("aoi", "lon_min") - buffer
    east  = cfg.getfloat("aoi", "lon_max") + buffer
    return south, north, west, east


def bbox_to_wkt(south: float, north: float, west: float, east: float) -> str:
    """Convert a bounding box to a WKT POLYGON string for CDSE queries."""
    return (
        f"POLYGON(({west} {south}, {east} {south}, "
        f"{east} {north}, {west} {north}, {west} {south}))"
    )


def bbox_to_isce2_str(south: float, north: float, west: float, east: float) -> str:
    """Return bbox as the string format expected by ISCE2 / stackSentinel.py."""
    return f"{south} {north} {west} {east}"


# ---------------------------------------------------------------------------
# GLO-30 DEM tile helpers
# ---------------------------------------------------------------------------

def glo30_tile_names(south: float, north: float, west: float, east: float) -> list:
    """
    Return a list of GLO-30 DEM tile names covering the given bounding box.

    GLO-30 tiles are 1°x1° named as:
        Copernicus_DSM_COG_10_N20_00_W044_00_DEM/
        Copernicus_DSM_COG_10_S20_00_W044_00_DEM/  (for southern hemisphere)

    Args:
        south, north, west, east: Bounding box in decimal degrees.

    Returns:
        List of tile directory name strings.
    """
    import math
    tiles = []
    lat_min = int(math.floor(south))
    lat_max = int(math.ceil(north))
    lon_min = int(math.floor(west))
    lon_max = int(math.ceil(east))

    for lat in range(lat_min, lat_max):
        for lon in range(lon_min, lon_max):
            lat_hem = "N" if lat >= 0 else "S"
            lon_hem = "E" if lon >= 0 else "W"
            lat_abs = abs(lat)
            lon_abs = abs(lon)
            tile = (
                f"Copernicus_DSM_COG_10_{lat_hem}{lat_abs:02d}_00_{lon_hem}{lon_abs:03d}_00_DEM"
            )
            tiles.append(tile)

    logger.debug("GLO-30 tiles needed: %s", tiles)
    return tiles


def glo30_s3_url(tile_name: str) -> str:
    """Return the S3 URL for a GLO-30 DEM tile GeoTIFF (no authentication required)."""
    return (
        f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"
    )
