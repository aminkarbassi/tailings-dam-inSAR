#!/usr/bin/env python3
"""
07_decompose_2d.py — Decompose LOS displacements into vertical + E-W components
================================================================================
Combines ascending and descending LOS displacement time-series to decompose
surface motion into its vertical (up-down) and east-west horizontal components.

Method:
    Two LOS observations (ascending and descending) are combined using the
    satellite geometry (incidence angle θ, heading angle α) to solve for
    vertical (dU) and east-west horizontal (dE) displacement, assuming the
    north-south component (dN) is negligible (InSAR is insensitive to N-S motion).

    LOS equations:
        d_asc  = -sin(θ_asc)·cos(α_asc)·dE + sin(θ_asc)·sin(α_asc)·dN + cos(θ_asc)·dU
        d_desc = -sin(θ_desc)·cos(α_desc)·dE + sin(θ_desc)·sin(α_desc)·dN + cos(θ_desc)·dU

    Setting dN = 0 yields a 2×2 system solved per pixel per epoch.

Inputs:
    processing/mintpy/asc/geo/geo_timeseries_ramp_demErr.h5
    processing/mintpy/desc/geo/geo_timeseries_ramp_demErr.h5
    processing/mintpy/asc/inputs/geometryGeo.h5  (incidence + heading angles)
    processing/mintpy/desc/inputs/geometryGeo.h5

Outputs:
    results/data/displacement_2d/vertical_YYYYMMDD.tif   — per epoch
    results/data/displacement_2d/ew_horizontal_YYYYMMDD.tif — per epoch
    results/data/displacement_2d/vertical_velocity.tif   — mean velocity
    results/data/displacement_2d/ew_velocity.tif

Usage:
    python scripts/07_decompose_2d.py [--config config/project.cfg]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose LOS time-series into vertical and E-W components"
    )
    parser.add_argument("--config", default="config/project.cfg")
    return parser.parse_args()


def load_hdf5_dataset(h5_path: Path, dataset: str) -> np.ndarray:
    """Read a dataset from an HDF5 file."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            available = list(f.keys())
            raise KeyError(
                f"Dataset '{dataset}' not found in {h5_path}.\nAvailable: {available}"
            )
        return f[dataset][...]


def load_hdf5_dates(h5_path: Path) -> list:
    """Return list of date strings from an HDF5 time-series file."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "date" in f:
            return [d.decode() if isinstance(d, bytes) else str(d) for d in f["date"][:]]
        elif "dates" in f:
            return [d.decode() if isinstance(d, bytes) else str(d) for d in f["dates"][:]]
    raise KeyError(f"No 'date' or 'dates' dataset found in {h5_path}")


def load_geometry(geom_path: Path):
    """
    Load incidence and heading angle arrays from MintPy geometryGeo.h5.

    Returns:
        incidence_angle (degrees, float32 array)
        heading_angle   (degrees, float32 array)
    """
    import h5py
    with h5py.File(geom_path, "r") as f:
        inc = f["incidenceAngle"][...]
        # Heading angle: MintPy uses "azimuthAngle" (clockwise from North)
        if "azimuthAngle" in f:
            head = f["azimuthAngle"][...]
        elif "headingAngle" in f:
            head = f["headingAngle"][...]
        else:
            # Default heading values for Sentinel-1
            # Ascending: ~-166° (i.e. heading ~194°)
            # Descending: ~-14° (i.e. heading ~346°)
            logger.warning(
                "Heading angle not found in %s. Using default Sentinel-1 values.", geom_path
            )
            head = np.full_like(inc, -166.0)

    return inc.astype(np.float32), head.astype(np.float32)


def build_los_unit_vectors(inc_deg: np.ndarray, head_deg: np.ndarray):
    """
    Compute the LOS unit vector components (E, N, U) from incidence and heading.

    For Sentinel-1 ascending/descending:
        heading is measured clockwise from North (degrees)
        incidence is measured from vertical (degrees)

    LOS unit vector pointing from ground to satellite:
        e_E = -sin(θ) · cos(α + π)  = sin(θ) · cos(α)   (roughly)
        e_N =  sin(θ) · sin(α + π)
        e_U =  cos(θ)

    MintPy convention: azimuthAngle is the heading of the satellite flight direction
    measured clockwise from North. The LOS vector is then:
        e_E = sin(θ) · sin(α - π)
        e_N = sin(θ) · cos(α - π)
        e_U = cos(θ)

    Positive LOS = ground moving toward satellite.
    """
    theta = np.deg2rad(inc_deg)
    alpha = np.deg2rad(head_deg - 90.0)  # convert heading to look direction

    e_E = np.sin(theta) * np.sin(alpha)
    e_N = np.sin(theta) * np.cos(alpha)
    e_U = np.cos(theta)

    return e_E, e_N, e_U


def decompose_epoch(
    d_asc:  np.ndarray,
    d_desc: np.ndarray,
    e_E_asc: np.ndarray, e_U_asc: np.ndarray,
    e_E_desc: np.ndarray, e_U_desc: np.ndarray,
    mask: np.ndarray,
) -> tuple:
    """
    Solve the 2×2 per-pixel system for dU (vertical) and dE (east-west).

    Assumes dN = 0 (north component negligible for InSAR).

    System:
        [ e_E_asc   e_U_asc  ] [ dE ]   [ d_asc  ]
        [ e_E_desc  e_U_desc ] [ dU ] = [ d_desc ]

    Args:
        d_asc, d_desc:   LOS displacement arrays (mm), same shape.
        e_E_asc, e_U_asc:  Unit vector components for ascending.
        e_E_desc, e_U_desc: Unit vector components for descending.
        mask:             Boolean array — True where both orbits are valid.

    Returns:
        (dE, dU) numpy arrays in mm, NaN where mask is False.
    """
    dE = np.full(d_asc.shape, np.nan, dtype=np.float32)
    dU = np.full(d_asc.shape, np.nan, dtype=np.float32)

    # Determinant of the design matrix (per pixel)
    det = e_E_asc * e_U_desc - e_U_asc * e_E_desc
    valid = mask & (np.abs(det) > 1e-6)

    dE[valid] = (d_asc[valid] * e_U_desc[valid] - d_desc[valid] * e_U_asc[valid]) / det[valid]
    dU[valid] = (d_desc[valid] * e_E_asc[valid] - d_asc[valid] * e_E_desc[valid]) / det[valid]

    return dE, dU


def save_geotiff(data: np.ndarray, ref_path: Path, out_path: Path, nodata: float = np.nan):
    """Save a 2D numpy array as a GeoTIFF, copying georeferencing from ref_path."""
    try:
        from osgeo import gdal, osr
    except ImportError:
        logger.error("GDAL Python bindings not found. Install with: conda install gdal")
        return

    ref_ds = gdal.Open(str(ref_path))
    if ref_ds is None:
        logger.error("Cannot open reference file: %s", ref_path)
        return

    driver = gdal.GetDriverByName("GTiff")
    rows, cols = data.shape
    out_ds = driver.Create(
        str(out_path), cols, rows, 1, gdal.GDT_Float32,
        ["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(data.astype(np.float32))
    band.SetNoDataValue(nodata)
    out_ds.FlushCache()
    ref_ds = None
    out_ds = None


def find_timeseries_file(mintpy_dir: Path) -> Path:
    """Find the best available MintPy time-series HDF5 file (geocoded, corrected)."""
    # Preference order: most corrections applied
    candidates = [
        mintpy_dir / "geo" / "geo_timeseries_ramp_demErr.h5",
        mintpy_dir / "geo" / "geo_timeseries_demErr.h5",
        mintpy_dir / "geo" / "geo_timeseries.h5",
        mintpy_dir / "timeseries_ramp_demErr.h5",
        mintpy_dir / "timeseries_demErr.h5",
        mintpy_dir / "timeseries.h5",
    ]
    for c in candidates:
        if c.exists():
            logger.info("Using time-series file: %s", c)
            return c

    logger.error("No MintPy time-series HDF5 found in %s", mintpy_dir)
    logger.error("Run 06_run_mintpy.py first (including the geocode step).")
    sys.exit(1)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    processing_dir = Path(cfg.get("paths", "processing_dir"))
    results_dir    = Path(cfg.get("paths", "results_dir"))
    out_dir        = results_dir / "data" / "displacement_2d"
    out_dir.mkdir(parents=True, exist_ok=True)

    asc_mintpy  = processing_dir / "mintpy" / "asc"
    desc_mintpy = processing_dir / "mintpy" / "desc"

    # Load time-series files
    asc_ts_path  = find_timeseries_file(asc_mintpy)
    desc_ts_path = find_timeseries_file(desc_mintpy)

    logger.info("Loading ascending time-series...")
    asc_ts     = load_hdf5_dataset(asc_ts_path, "timeseries")  # shape (T, H, W)
    asc_dates  = load_hdf5_dates(asc_ts_path)

    logger.info("Loading descending time-series...")
    desc_ts    = load_hdf5_dataset(desc_ts_path, "timeseries")
    desc_dates = load_hdf5_dates(desc_ts_path)

    # Convert to mm (MintPy outputs in metres)
    asc_ts_mm  = asc_ts  * 1000.0
    desc_ts_mm = desc_ts * 1000.0

    # Load geometry
    asc_geom_path  = asc_mintpy  / "inputs" / "geometryGeo.h5"
    desc_geom_path = desc_mintpy / "inputs" / "geometryGeo.h5"

    asc_inc,  asc_head  = load_geometry(asc_geom_path)
    desc_inc, desc_head = load_geometry(desc_geom_path)

    # Build LOS unit vectors
    e_E_asc,  e_N_asc,  e_U_asc  = build_los_unit_vectors(asc_inc,  asc_head)
    e_E_desc, e_N_desc, e_U_desc = build_los_unit_vectors(desc_inc, desc_head)

    # Find common dates
    common_dates = sorted(set(asc_dates) & set(desc_dates))
    logger.info(
        "Common dates: %d (ASC: %d, DESC: %d)",
        len(common_dates), len(asc_dates), len(desc_dates),
    )
    if len(common_dates) == 0:
        logger.error(
            "No common dates between ascending and descending time-series.\n"
            "Check that both orbits cover the same period and the same reference date."
        )
        sys.exit(1)

    # Build common mask: valid where both orbits have non-NaN, non-zero data
    asc_mean  = np.nanmean(np.abs(asc_ts_mm),  axis=0)
    desc_mean = np.nanmean(np.abs(desc_ts_mm), axis=0)
    mask      = np.isfinite(asc_mean) & np.isfinite(desc_mean) & (asc_mean > 0) & (desc_mean > 0)
    logger.info("Valid pixels (both orbits): %d (%.1f%%)", mask.sum(), 100*mask.mean())

    # Decompose per common date
    logger.info("Decomposing %d epochs...", len(common_dates))
    dU_all = []
    dE_all = []

    for date in common_dates:
        i_asc  = asc_dates.index(date)
        i_desc = desc_dates.index(date)

        d_asc  = asc_ts_mm[i_asc].astype(np.float32)
        d_desc = desc_ts_mm[i_desc].astype(np.float32)

        date_mask = mask & np.isfinite(d_asc) & np.isfinite(d_desc)

        dE, dU = decompose_epoch(
            d_asc, d_desc,
            e_E_asc, e_U_asc,
            e_E_desc, e_U_desc,
            date_mask,
        )
        dU_all.append(dU)
        dE_all.append(dE)

        # Save individual epoch GeoTIFFs
        ref_tif = out_dir / "reference.tif"
        if not ref_tif.exists():
            # Create a reference GeoTIFF from the HDF5 metadata on first epoch
            _create_reference_tif(asc_ts_path, ref_tif, d_asc)

        save_geotiff(dU, ref_tif, out_dir / f"vertical_{date}.tif")
        save_geotiff(dE, ref_tif, out_dir / f"ew_horizontal_{date}.tif")

    logger.info("Saved %d epoch GeoTIFFs to %s", len(common_dates), out_dir)

    # Compute and save mean velocity (linear trend) for both components
    dU_stack = np.stack(dU_all, axis=0)  # shape (T, H, W)
    dE_stack = np.stack(dE_all, axis=0)

    dates_years = _dates_to_years(common_dates)
    dU_vel = _fit_velocity(dU_stack, dates_years)
    dE_vel = _fit_velocity(dE_stack, dates_years)

    ref_tif = out_dir / "reference.tif"
    save_geotiff(dU_vel, ref_tif, out_dir / "vertical_velocity_mmyr.tif")
    save_geotiff(dE_vel, ref_tif, out_dir / "ew_velocity_mmyr.tif")
    logger.info("Saved velocity maps.")

    print(f"\n2D decomposition complete.")
    print(f"  Output directory: {out_dir}")
    print(f"  Per-epoch files:  vertical_YYYYMMDD.tif, ew_horizontal_YYYYMMDD.tif")
    print(f"  Velocity maps:    vertical_velocity_mmyr.tif, ew_velocity_mmyr.tif")
    print()
    print("Next step: python scripts/08_plot_maps.py")


def _dates_to_years(dates: list) -> np.ndarray:
    """Convert YYYYMMDD date strings to decimal years."""
    from datetime import datetime
    ref = datetime.strptime(dates[0], "%Y%m%d")
    years = []
    for d in dates:
        dt = datetime.strptime(d, "%Y%m%d")
        years.append((dt - ref).days / 365.25)
    return np.array(years)


def _fit_velocity(stack: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Fit linear velocity (mm/yr) to a 3D displacement stack (T, H, W)."""
    T, H, W = stack.shape
    data = stack.reshape(T, -1).astype(np.float64)
    # Least-squares fit: d = v * t + c
    A = np.vstack([years, np.ones(T)]).T  # (T, 2)
    result = np.full((H * W,), np.nan, dtype=np.float32)

    valid_pixels = np.sum(np.isfinite(data), axis=0) >= max(5, T // 2)
    for j in np.where(valid_pixels)[0]:
        col = data[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 5:
            continue
        v, _ = np.linalg.lstsq(A[finite], col[finite], rcond=None)[0]
        result[j] = v  # mm/yr

    return result.reshape(H, W)


def _create_reference_tif(h5_path: Path, out_path: Path, data_2d: np.ndarray):
    """
    Create a minimal GeoTIFF with correct geotransform from MintPy HDF5 metadata.
    Used as a template for subsequent save_geotiff calls.
    """
    import h5py
    try:
        from osgeo import gdal, osr
    except ImportError:
        return

    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)

    x0    = float(attrs.get("X_FIRST", attrs.get("LON_REF1", -44.35)))
    y0    = float(attrs.get("Y_FIRST", attrs.get("LAT_REF1", -19.92)))
    dx    = float(attrs.get("X_STEP",  0.0001))
    dy    = float(attrs.get("Y_STEP", -0.0001))
    rows, cols = data_2d.shape

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(out_path), cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([x0, dx, 0, y0, 0, dy])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(data_2d.astype(np.float32))
    ds.FlushCache()
    ds = None


if __name__ == "__main__":
    main()
