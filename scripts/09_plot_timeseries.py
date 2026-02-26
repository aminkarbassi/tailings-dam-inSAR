#!/usr/bin/env python3
"""
09_plot_timeseries.py — Plot displacement time-series at key points
====================================================================
Extracts cumulative displacement over time at user-defined points of interest
(dam crest, tailings surface, stable reference, etc.) and produces time-series
plots with the January 25, 2019 collapse date clearly annotated.

Features:
  - Extracts from both ascending and descending MintPy time-series HDF5 files
  - Averages over a 3×3 pixel window to reduce noise
  - Fits a linear velocity to the pre-collapse period
  - Marks the collapse date with a red dashed vertical line
  - Exports tabular data to CSV for further analysis

Points of interest are defined in config/poi.csv (created automatically if
it doesn't exist). Edit this file to add or rename locations.

Outputs:
    results/figures/timeseries/ts_{label}_asc.png
    results/figures/timeseries/ts_{label}_desc.png
    results/figures/timeseries/ts_{label}_combined.png
    results/data/timeseries_points.csv

Usage:
    python scripts/09_plot_timeseries.py [--config config/project.cfg]
                                          [--poi config/poi.csv]
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COLLAPSE_DATE = datetime(2019, 1, 25)
DAM_LON = -44.1231
DAM_LAT = -20.1113

# Both tracks are descending — "asc"/"desc" are just folder-name conventions
ORBIT_LABELS = {
    "asc":  "DESC Track 53 (inc. ~32°)",
    "desc": "DESC Track 155 (inc. ~45°)",
}

# Default points of interest (written to poi.csv if not present)
DEFAULT_POI = [
    {"label": "dam_crest",      "lat": -20.1113, "lon": -44.1231,
     "description": "Dam I crest centre (approx.)"},
    {"label": "tailings_upper", "lat": -20.1150, "lon": -44.1200,
     "description": "Upper tailings surface"},
    {"label": "tailings_lower", "lat": -20.1200, "lon": -44.1170,
     "description": "Lower tailings body"},
    {"label": "stable_ref",     "lat": -20.0600, "lon": -44.0800,
     "description": "Stable bedrock reference point"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot displacement time-series at POIs")
    parser.add_argument("--config",  default="config/project.cfg")
    parser.add_argument("--poi",     default="config/poi.csv",
                        help="Path to CSV with columns: label,lat,lon,description")
    parser.add_argument("--variant", default=None,
                        help="Processing variant suffix (e.g. 'isbas' → reads "
                             "processing/mintpy/asc_isbas/, saves to "
                             "results/figures/timeseries_isbas/)")
    return parser.parse_args()


def ensure_poi_csv(poi_path: Path):
    """Create default poi.csv if it doesn't exist."""
    if poi_path.exists():
        return
    logger.info("Creating default POI file: %s", poi_path)
    poi_path.parent.mkdir(parents=True, exist_ok=True)
    with open(poi_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "lat", "lon", "description"])
        writer.writeheader()
        writer.writerows(DEFAULT_POI)
    print(f"Default POI file created: {poi_path}")
    print("Edit this file to add your own points of interest before re-running.")


def load_poi(poi_path: Path) -> list:
    """Load points of interest from CSV."""
    pois = []
    with open(poi_path) as f:
        for row in csv.DictReader(f):
            pois.append({
                "label":       row["label"],
                "lat":         float(row["lat"]),
                "lon":         float(row["lon"]),
                "description": row.get("description", row["label"]),
            })
    return pois


def load_timeseries_h5(h5_path: Path):
    """Load MintPy geocoded time-series and coordinate arrays (WGS84 lat/lon)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        data_m = f["timeseries"][...]
        dates  = [
            d.decode() if isinstance(d, bytes) else str(d)
            for d in f["date"][:]
        ]
        attrs = dict(f.attrs)

    data_mm = data_m * 1000.0  # metres → mm

    x0 = float(attrs.get("X_FIRST", -44.35))
    y0 = float(attrs.get("Y_FIRST", -19.92))
    dx = float(attrs.get("X_STEP",   0.0001))
    dy = float(attrs.get("Y_STEP",  -0.0001))
    H, W = data_mm.shape[1], data_mm.shape[2]

    epsg = attrs.get("EPSG", None)
    if epsg and int(str(epsg)) != 4326:
        # Data is in a projected CRS (e.g. UTM) — convert corners to lat/lon
        from pyproj import Transformer
        t = Transformer.from_crs(f"EPSG:{int(str(epsg))}", "EPSG:4326", always_xy=True)
        lon0, lat0 = t.transform(x0,                  y0)                  # top-left
        lon1, lat1 = t.transform(x0 + (W - 1) * dx,  y0 + (H - 1) * dy)  # bottom-right
        lon_arr = np.linspace(lon0, lon1, W)
        lat_arr = np.linspace(lat0, lat1, H)
    else:
        lon_arr = x0 + np.arange(W) * dx
        lat_arr = y0 + np.arange(H) * dy

    return dates, data_mm, lat_arr, lon_arr


def find_timeseries_file(mintpy_dir: Path) -> Path:
    """Find best available geocoded time-series HDF5.

    HyP3 products are already geocoded (UTM), so MintPy stores the final
    corrected timeseries in the main working directory, not in geo/.
    """
    candidates = [
        # Main directory (HyP3 workflow — already geocoded in UTM)
        mintpy_dir / "timeseries_ERA5_ramp_demErr.h5",
        mintpy_dir / "timeseries_ramp_demErr.h5",
        mintpy_dir / "timeseries_demErr.h5",
        mintpy_dir / "timeseries_ramp.h5",
        mintpy_dir / "timeseries.h5",
        # geo/ subdirectory (ISCE2 workflow — reprojected to lat/lon)
        mintpy_dir / "geo" / "geo_timeseries_ramp_demErr.h5",
        mintpy_dir / "geo" / "geo_timeseries_demErr.h5",
        mintpy_dir / "geo" / "geo_timeseries.h5",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def latlon_to_rowcol(lat: float, lon: float, lat_arr: np.ndarray, lon_arr: np.ndarray):
    """Convert lat/lon to nearest row, col indices."""
    row = int(np.argmin(np.abs(lat_arr - lat)))
    col = int(np.argmin(np.abs(lon_arr - lon)))
    return row, col


def extract_timeseries(
    data_mm: np.ndarray,
    lat: float,
    lon: float,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    window: int = 3,
) -> np.ndarray:
    """
    Extract displacement time-series at (lat, lon) averaged over a window×window
    pixel neighbourhood to reduce noise.

    Returns:
        1D numpy array of cumulative displacement (mm), length T.
        NaN where all pixels in the window are masked.
    """
    row, col = latlon_to_rowcol(lat, lon, lat_arr, lon_arr)
    half = window // 2
    H, W = data_mm.shape[1], data_mm.shape[2]

    r0, r1 = max(0, row - half), min(H, row + half + 1)
    c0, c1 = max(0, col - half), min(W, col + half + 1)

    patch = data_mm[:, r0:r1, c0:c1]  # shape (T, h, w)
    patch[patch == 0] = np.nan

    with np.errstate(all="ignore"):
        ts = np.nanmean(patch.reshape(patch.shape[0], -1), axis=1)

    return ts


def dates_to_datetime(dates: list) -> list:
    return [datetime.strptime(d, "%Y%m%d") for d in dates]


def fit_linear_velocity(dts: list, ts_mm: np.ndarray):
    """
    Fit a linear trend to the time-series.

    Returns:
        velocity_mmyr:  mm/yr
        trend_mm:       fitted trend values (same length as ts_mm)
    """
    years = np.array([(dt - dts[0]).days / 365.25 for dt in dts])
    finite = np.isfinite(ts_mm)
    if finite.sum() < 3:
        return np.nan, np.full(len(ts_mm), np.nan)

    A = np.vstack([years[finite], np.ones(finite.sum())]).T
    result = np.linalg.lstsq(A, ts_mm[finite], rcond=None)
    v, c = result[0]
    trend = v * years + c
    return v, trend


def plot_single_orbit_ts(
    dts: list,
    ts_mm: np.ndarray,
    velocity: float,
    trend: np.ndarray,
    label: str,
    orbit: str,
    description: str,
    out_path: Path,
):
    """Plot displacement time-series for a single orbit and point."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Scatter plot of raw time-series
    ax.scatter(dts, ts_mm, s=20, color="steelblue", zorder=3,
               label=f"LOS displacement ({ORBIT_LABELS.get(orbit, orbit.upper())})")

    # Linear trend
    finite = np.isfinite(ts_mm)
    if finite.sum() > 3:
        ax.plot(dts, trend, "k--", linewidth=1.5, label=f"Linear trend: {velocity:.1f} mm/yr")

    # Collapse date vertical line
    ax.axvline(COLLAPSE_DATE, color="red", linewidth=2, linestyle="--", zorder=4,
               label="Collapse: 25 Jan 2019")
    ax.axvspan(
        datetime(2018, 10, 25), COLLAPSE_DATE,
        alpha=0.08, color="red", label="Final 3 months"
    )

    # Zero line
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative LOS displacement (mm)", fontsize=11)
    ax.set_title(
        f"Brumadinho Dam I — {description}\n"
        f"{ORBIT_LABELS.get(orbit, orbit.upper())} displacement time-series",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Annotation: days to collapse at last point
    last_finite = np.where(np.isfinite(ts_mm))[0]
    if len(last_finite):
        last_idx  = last_finite[-1]
        last_dt   = dts[last_idx]
        last_val  = ts_mm[last_idx]
        days_left = (COLLAPSE_DATE - last_dt).days
        if days_left > 0:
            ax.annotate(
                f"{days_left}d before\ncollapse",
                xy=(last_dt, last_val),
                xytext=(10, 15),
                textcoords="offset points",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"),
            )

    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_ts(
    asc_dts:  list,
    asc_ts:   np.ndarray,
    desc_dts: list,
    desc_ts:  np.ndarray,
    label: str,
    description: str,
    out_path: Path,
):
    """Plot ascending and descending on the same axes for comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(asc_dts,  asc_ts,  s=20, color="steelblue",  zorder=3, label=f"LOS ({ORBIT_LABELS['asc']})")
    ax.scatter(desc_dts, desc_ts, s=20, color="darkorange",  zorder=3, label=f"LOS ({ORBIT_LABELS['desc']})")

    ax.axvline(COLLAPSE_DATE, color="red", linewidth=2, linestyle="--",
               label="Collapse: 25 Jan 2019")
    ax.axvspan(datetime(2018, 10, 25), COLLAPSE_DATE, alpha=0.08, color="red")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative LOS displacement (mm)", fontsize=11)
    ax.set_title(
        f"Brumadinho Dam I — {description}\nTrack 53 + Track 155 LOS displacement comparison",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    processing_dir = Path(cfg.get("paths", "processing_dir"))
    results_dir    = Path(cfg.get("paths", "results_dir"))
    variant        = args.variant
    ts_suffix      = f"timeseries_{variant}" if variant else "timeseries"
    csv_name       = f"timeseries_points_{variant}.csv" if variant else "timeseries_points.csv"
    ts_out_dir     = results_dir / "figures" / ts_suffix
    data_out_dir   = results_dir / "data"
    ts_out_dir.mkdir(parents=True, exist_ok=True)
    data_out_dir.mkdir(parents=True, exist_ok=True)

    # Load points of interest
    poi_path = Path(args.poi)
    ensure_poi_csv(poi_path)
    pois = load_poi(poi_path)
    logger.info("Loaded %d points of interest from %s", len(pois), poi_path)

    # Load time-series for both orbits
    orbit_data = {}
    for orbit in ["asc", "desc"]:
        dir_tag    = f"{orbit}_{variant}" if variant else orbit
        mintpy_dir = processing_dir / "mintpy" / dir_tag
        ts_path    = find_timeseries_file(mintpy_dir)
        if ts_path is None:
            logger.warning("No geocoded time-series for %s — skipping.", ORBIT_LABELS.get(orbit, orbit.upper()))
            continue
        logger.info("Loading %s time-series: %s", ORBIT_LABELS.get(orbit, orbit.upper()), ts_path.name)
        dates, data_mm, lat_arr, lon_arr = load_timeseries_h5(ts_path)
        orbit_data[orbit] = {
            "dates": dates,
            "dts":   dates_to_datetime(dates),
            "data":  data_mm,
            "lat":   lat_arr,
            "lon":   lon_arr,
        }

    if not orbit_data:
        logger.error(
            "No time-series data found. Run 06_run_mintpy.py (with geocode step) first."
        )
        sys.exit(1)

    # Extract and plot time-series per POI
    all_rows = []

    for poi in pois:
        label       = poi["label"]
        lat, lon    = poi["lat"], poi["lon"]
        description = poi["description"]
        logger.info("Processing POI: %s (%.4f°N, %.4f°E)", label, lat, lon)

        orbit_ts = {}
        for orbit, od in orbit_data.items():
            ts = extract_timeseries(od["data"], lat, lon, od["lat"], od["lon"])
            orbit_ts[orbit] = ts

            velocity, trend = fit_linear_velocity(od["dts"], ts)
            logger.info(
                "  %s: velocity = %.1f mm/yr, %d valid epochs",
                ORBIT_LABELS.get(orbit, orbit.upper()), velocity if np.isfinite(velocity) else 0.0,
                np.sum(np.isfinite(ts)),
            )

            # Individual orbit plot
            plot_single_orbit_ts(
                dts=od["dts"],
                ts_mm=ts,
                velocity=velocity,
                trend=trend,
                label=label,
                orbit=orbit,
                description=description,
                out_path=ts_out_dir / f"ts_{label}_{orbit}.png",
            )

            # Collect rows for CSV
            for i, (dt, d) in enumerate(zip(od["dts"], ts)):
                all_rows.append({
                    "poi_label":   label,
                    "lat":         lat,
                    "lon":         lon,
                    "orbit":       orbit,
                    "date":        dt.strftime("%Y-%m-%d"),
                    "displacement_mm": round(float(d), 3) if np.isfinite(d) else "",
                    "velocity_mmyr":   round(float(velocity), 3) if np.isfinite(velocity) else "",
                })

        # Combined ASC + DESC plot (if both available)
        if "asc" in orbit_ts and "desc" in orbit_ts:
            plot_combined_ts(
                asc_dts  = orbit_data["asc"]["dts"],
                asc_ts   = orbit_ts["asc"],
                desc_dts = orbit_data["desc"]["dts"],
                desc_ts  = orbit_ts["desc"],
                label    = label,
                description = description,
                out_path = ts_out_dir / f"ts_{label}_combined.png",
            )

    # Write CSV
    csv_path = data_out_dir / csv_name
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        logger.info("Tabular data saved: %s", csv_path)

    print(f"\nTime-series plots saved to: {ts_out_dir}")
    print(f"Tabular data saved to:      {csv_path}")
    print()
    print("Analysis complete. Key files to review:")
    for poi in pois:
        label = poi["label"]
        print(f"  {ts_out_dir / f'ts_{label}_combined.png'}")
    print()
    print("Look for:")
    print("  1. Accelerating negative LOS displacement in the 3-6 months before collapse.")
    print("  2. The dam_crest point showing different signal from the stable_ref point.")
    print("  3. Any step-change or kink in the time-series indicating sudden movement.")


if __name__ == "__main__":
    main()
