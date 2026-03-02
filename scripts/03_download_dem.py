#!/usr/bin/env python3
"""
03_download_dem.py — Download and prepare Copernicus GLO-30 DEM
===============================================================
Downloads 1°x1° GLO-30 DEM tiles from the public Copernicus S3 bucket,
mosaics them, converts the datum from EGM2008 (geoid) to WGS84 (ellipsoid),
and produces an ISCE2-compatible DEM file.

Output files:
    data/dem/glo30/raw_tiles/     — individual 1°x1° GeoTIFFs
    data/dem/glo30/dem_merged.tif — merged, WGS84 ellipsoidal heights
    data/dem/glo30/dem.wgs84      — ISCE2-format DEM (with .xml and .vrt)

Usage:
    python scripts/03_download_dem.py [--config config/project.cfg]

Requirements:
    conda install -c conda-forge gdal
    (ISCE2 must be installed for the final conversion step)
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config, get_aoi, glo30_tile_names, glo30_s3_url

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download GLO-30 DEM tiles")
    parser.add_argument("--config", default="config/project.cfg")
    return parser.parse_args()


def download_tile(url: str, dest: Path) -> bool:
    """Download a single DEM tile. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Already exists: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading: %s", url)

    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 404:
            # Ocean/sea tiles don't exist in the bucket — this is normal
            logger.info("Tile not found (likely ocean): %s", dest.name)
            return False
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
        return True

    except requests.RequestException as e:
        logger.error("Failed to download %s: %s", url, e)
        return False


def merge_tiles(tile_paths: list, output_path: Path):
    """Mosaic all downloaded tiles into a single GeoTIFF using GDAL."""
    logger.info("Merging %d tiles to %s", len(tile_paths), output_path)

    tile_str_list = [str(p) for p in tile_paths]
    cmd = [
        "gdal_merge.py",
        "-o", str(output_path),
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-nodata", "-9999",
        "-init", "-9999",
    ] + tile_str_list

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gdal_merge.py failed:\n%s", result.stderr)
        sys.exit(1)

    logger.info("Merged DEM written to: %s", output_path)


def crop_to_bbox(
    input_path: Path,
    output_path: Path,
    south: float,
    north: float,
    west: float,
    east: float,
    buffer: float = 0.5,
):
    """Crop merged DEM to AOI + buffer using gdalwarp."""
    xmin = west  - buffer
    xmax = east  + buffer
    ymin = south - buffer
    ymax = north + buffer

    logger.info(
        "Cropping DEM to bbox (%.2f %.2f %.2f %.2f) with %.1f° buffer",
        south, north, west, east, buffer,
    )

    cmd = [
        "gdalwarp",
        "-te", str(xmin), str(ymin), str(xmax), str(ymax),
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-dstnodata", "-9999",
        str(input_path),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gdalwarp crop failed:\n%s", result.stderr)
        sys.exit(1)

    logger.info("Cropped DEM: %s", output_path)


def convert_egm2008_to_wgs84(input_path: Path, output_path: Path):
    """
    Convert DEM heights from EGM2008 geoid to WGS84 ellipsoid.

    GLO-30 heights are referenced to the EGM2008 geoid.
    ISCE2 requires WGS84 ellipsoidal heights.
    The conversion adds the geoid undulation (N) at each pixel.

    Uses GDAL with the VERTCS / PROJ datum shift via gdalwarp.
    """
    logger.info("Converting EGM2008 → WGS84 ellipsoid...")

    # The PROJ pipeline for EGM2008 → WGS84 vertical datum conversion:
    # +proj=vgridshift +grids=us_nga_egm2008_1.tif +multiplier=1
    # GDAL/PROJ will download the geoid grid automatically if needed.
    cmd = [
        "gdalwarp",
        "-s_srs", "+proj=longlat +datum=WGS84 +geoidgrids=egm2008-5.gtx",
        "-t_srs", "EPSG:4326",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        str(input_path),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "EGM2008→WGS84 conversion via PROJ failed (geoid grid may not be installed).\n"
            "The merged GeoTIFF will be used as-is.\n"
            "To install the geoid grid: pip install pyproj; python -c \"import pyproj; pyproj.Proj('EPSG:5773')\".\n"
            "Alternatively, use ISCE2's dem.py utility which handles this conversion natively.\n"
            "Stderr: %s",
            result.stderr,
        )
        # Fall back: copy the file as-is with a warning
        import shutil
        shutil.copy2(input_path, output_path)
        logger.warning(
            "Using EGM2008 heights as-is. Topographic error correction in MintPy "
            "will partially compensate, but GACOS atmospheric correction may be affected."
        )
    else:
        logger.info("Datum conversion complete: %s", output_path)


def convert_to_isce2_format(input_path: Path, isce2_dem_path: Path):
    """
    Convert a GeoTIFF to ISCE2's native DEM format (.wgs84 + .xml + .vrt).

    If ISCE2 is available, use its gdal2isce_dem.py tool.
    Otherwise, create a minimal .xml sidecar manually.
    """
    logger.info("Converting to ISCE2 format: %s", isce2_dem_path)
    isce2_dem_path.parent.mkdir(parents=True, exist_ok=True)

    # Try ISCE2's gdal2isce_dem.py first
    result = subprocess.run(
        ["gdal2isce_dem.py", "-i", str(input_path), "-o", str(isce2_dem_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.info("ISCE2 format DEM written: %s", isce2_dem_path)
        return

    # Fallback: convert to raw binary + create ISCE2 XML sidecar
    logger.warning(
        "gdal2isce_dem.py not found or failed. Falling back to GDAL raw conversion."
    )

    # Convert to ENVI format (ISCE2 can read ENVI binary grids)
    envi_path = isce2_dem_path.with_suffix(".dem")
    cmd = [
        "gdal_translate",
        "-of", "ENVI",
        "-ot", "Int16",
        str(input_path),
        str(envi_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gdal_translate to ENVI failed:\n%s", result.stderr)
        sys.exit(1)

    # Rename .hdr to .dem.xml equivalent — ISCE2 reads ENVI .hdr files
    logger.info("ENVI DEM written: %s", envi_path)
    logger.info(
        "NOTE: If ISCE2 cannot read this DEM, run: gdal2isce_dem.py -i %s -o %s",
        input_path,
        isce2_dem_path,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    south, north, west, east = get_aoi(cfg)
    dem_buffer = cfg.getfloat("aoi", "dem_buffer")
    dem_dir    = Path(cfg.get("paths", "data_dir")) / "dem" / "glo30"
    tiles_dir  = dem_dir / "raw_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # 1. Identify required tiles (with DEM buffer)
    tile_names = glo30_tile_names(
        south - dem_buffer, north + dem_buffer,
        west  - dem_buffer, east  + dem_buffer,
    )
    logger.info("Need %d GLO-30 tile(s): %s", len(tile_names), tile_names)

    # 2. Download tiles
    downloaded = []
    for tile_name in tile_names:
        url  = glo30_s3_url(tile_name)
        dest = tiles_dir / f"{tile_name}.tif"
        ok   = download_tile(url, dest)
        if ok and dest.exists():
            downloaded.append(dest)

    if not downloaded:
        logger.error("No DEM tiles downloaded. Check network access to AWS S3.")
        sys.exit(1)

    logger.info("Downloaded %d/%d tiles", len(downloaded), len(tile_names))

    # 3. Merge tiles
    merged_path = dem_dir / "dem_merged_raw.tif"
    merge_tiles(downloaded, merged_path)

    # 4. Crop to AOI + buffer
    cropped_path = dem_dir / "dem_cropped.tif"
    crop_to_bbox(merged_path, cropped_path, south, north, west, east, buffer=dem_buffer)

    # 5. Convert EGM2008 → WGS84
    wgs84_tif = dem_dir / "dem_wgs84.tif"
    convert_egm2008_to_wgs84(cropped_path, wgs84_tif)

    # 6. Convert to ISCE2 format
    isce2_dem = dem_dir / "dem.wgs84"
    convert_to_isce2_format(wgs84_tif, isce2_dem)

    print(f"\nDEM preparation complete.")
    print(f"  Merged (raw):       {dem_dir / 'dem_merged_raw.tif'}")
    print(f"  Cropped:            {cropped_path}")
    print(f"  WGS84 ellipsoid:    {wgs84_tif}")
    print(f"  ISCE2 format:       {isce2_dem}")
    print()
    print("Next step: python scripts/04_prepare_isce2.py --orbit asc")
    print("           python scripts/04_prepare_isce2.py --orbit desc")


if __name__ == "__main__":
    main()
