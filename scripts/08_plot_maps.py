#!/usr/bin/env python3
"""
08_plot_maps.py — Plot cumulative displacement maps
===================================================
Generates publication-quality cumulative LOS displacement maps for each
acquisition date, plus 2D decomposed (vertical + E-W) maps if available.

All plots use:
  - RdBu_r diverging colormap centred on zero
  - Fixed colour scale across all epochs (for animation-compatible comparison)
  - Contextily basemap (OpenStreetMap or Stamen Terrain)
  - Dam location marked with a triangle
  - Scale bar, north arrow, and date title

Outputs:
    results/figures/displacement_maps/
        cumulative_asc_YYYYMMDD.png
        cumulative_desc_YYYYMMDD.png
    results/figures/decomposition/
        vertical_YYYYMMDD.png          (if 07_decompose_2d.py was run)
        ew_horizontal_YYYYMMDD.png     (if 07_decompose_2d.py was run)

Usage:
    python scripts/08_plot_maps.py [--config config/project.cfg]
                                    [--vmin -50] [--vmax 50]
                                    [--orbit {desc53,desc155,both}]
                                    [--no-decomp]
                                    [--every N]  # plot every Nth epoch
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DAM_LON = -44.1231
DAM_LAT = -20.1113
COLLAPSE_DATE = datetime(2019, 1, 25)

# Approximate outline of Dam I + tailings impoundment (Córrego do Feijão Mine).
# Derived from satellite imagery and published figures (Grebby et al. 2021).
# Polygon closes back to first point. Coordinates are WGS84 lon/lat.
DAM_OUTLINE_LON = [-44.132, -44.117, -44.113, -44.116, -44.126, -44.133, -44.132]
DAM_OUTLINE_LAT = [-20.097, -20.097, -20.105, -20.120, -20.123, -20.115, -20.097]

ORBIT_LABELS = {
    "desc53":  "DESC Track 53 (inc. ~32°)",
    "desc155": "DESC Track 155 (inc. ~45°)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cumulative displacement maps")
    parser.add_argument("--config",    default="config/project.cfg")
    parser.add_argument("--vmin",      type=float, default=None,
                        help="Colorscale minimum (mm). Auto-detected if not set.")
    parser.add_argument("--vmax",      type=float, default=None,
                        help="Colorscale maximum (mm). Auto-detected if not set.")
    parser.add_argument("--orbit",     choices=["desc53", "desc155", "both"], default="both")
    parser.add_argument("--variant",   default=None,
                        help="Processing variant suffix (e.g. 'isbas' → reads "
                             "processing/mintpy/desc53_isbas/, saves to "
                             "results/figures/displacement_maps_isbas/)")
    parser.add_argument("--no-decomp", action="store_true",
                        help="Skip 2D decomposition maps even if available")
    parser.add_argument("--every",     type=int, default=1,
                        help="Plot every Nth epoch (default: 1 = all)")
    parser.add_argument("--zoom-dam",  action="store_true",
                        help="Crop map to a tight ~6 km window centred on the dam")
    return parser.parse_args()


def load_timeseries_h5(h5_path: Path):
    """
    Load a MintPy geocoded time-series HDF5.

    Returns:
        dates:      list of 'YYYYMMDD' strings
        data_mm:    numpy array (T, H, W) in mm
        lat_arr:    1D array of latitudes (WGS84 degrees)
        lon_arr:    1D array of longitudes (WGS84 degrees)
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        ts = f["timeseries"][...] * 1000.0  # metres → mm
        dates = [
            d.decode() if isinstance(d, bytes) else str(d)
            for d in f["date"][:]
        ]
        attrs = dict(f.attrs)

    x0 = float(attrs.get("X_FIRST", -44.35))
    y0 = float(attrs.get("Y_FIRST", -19.92))
    dx = float(attrs.get("X_STEP",  0.0001))
    dy = float(attrs.get("Y_STEP", -0.0001))
    H, W = ts.shape[1], ts.shape[2]

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

    return dates, ts, lat_arr, lon_arr


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


def auto_vrange(data_mm: np.ndarray, percentile: float = 99.0) -> tuple:
    """Compute symmetric colour range from data percentile."""
    finite = data_mm[np.isfinite(data_mm)]
    if len(finite) == 0:
        return -50.0, 50.0
    vmax = np.percentile(np.abs(finite), percentile)
    return -vmax, vmax


def plot_displacement_map(
    data_2d: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    date_str: str,
    title_prefix: str,
    out_path: Path,
    vmin: float,
    vmax: float,
    cbar_label: str = "LOS displacement (mm)",
    zoom_extent: tuple = None,   # (lon_min, lon_max, lat_min, lat_max) or None
):
    """
    Render a single displacement map with basemap, dam marker, and colourbar.
    """
    try:
        import contextily as ctx
    except ImportError:
        ctx = None

    fig, ax = plt.subplots(figsize=(9, 8))

    # Extent for imshow: [left, right, bottom, top]
    extent = [lon_arr[0], lon_arr[-1], lat_arr[-1], lat_arr[0]]

    im = ax.imshow(
        data_2d,
        extent=extent,
        origin="upper",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        alpha=0.75,
        zorder=2,
    )

    # Add basemap if contextily is available
    if ctx is not None:
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom="auto",
                alpha=0.5,
                zorder=1,
            )
        except Exception as e:
            logger.debug("Basemap not available: %s", e)

    # Dam location marker
    ax.scatter(
        DAM_LON, DAM_LAT,
        marker="^", s=150, color="black", edgecolors="white",
        linewidths=1.5, zorder=5, label="Brumadinho Dam I",
    )

    # Date annotation
    dt = datetime.strptime(date_str, "%Y%m%d")
    days_to_collapse = (COLLAPSE_DATE - dt).days
    if days_to_collapse > 0:
        date_label = f"{dt.strftime('%d %b %Y')}  ({days_to_collapse}d before collapse)"
    elif days_to_collapse == 0:
        date_label = f"{dt.strftime('%d %b %Y')}  (COLLAPSE DATE)"
    else:
        date_label = f"{dt.strftime('%d %b %Y')}  ({-days_to_collapse}d after collapse)"

    ax.set_title(f"{title_prefix}\n{date_label}", fontsize=12, pad=10)
    ax.set_xlabel("Longitude (°)", fontsize=10)
    ax.set_ylabel("Latitude (°)",  fontsize=10)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label(cbar_label, fontsize=10)

    # Dam outline — approximate polygon of Dam I + tailings impoundment
    ax.plot(
        DAM_OUTLINE_LON, DAM_OUTLINE_LAT,
        color="white", linestyle="--", linewidth=1.5, zorder=6,
        label="Dam I extent (approx.)",
    )

    # Zoom to dam area if requested
    if zoom_extent is not None:
        ax.set_xlim(zoom_extent[0], zoom_extent[1])
        ax.set_ylim(zoom_extent[2], zoom_extent[3])

    # Legend
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_orbit_maps(
    mintpy_dir: Path,
    out_dir: Path,
    orbit: str,
    vmin: float,
    vmax: float,
    every: int,
    zoom_extent: tuple = None,
):
    """Plot all epoch maps for one orbit direction."""
    ts_path = find_timeseries_file(mintpy_dir)
    if ts_path is None:
        logger.warning(
            "No geocoded time-series found for %s in %s. "
            "Ensure 06_run_mintpy.py with geocode step has been run.",
            ORBIT_LABELS.get(orbit, orbit.upper()), mintpy_dir,
        )
        return

    dates, data_mm, lat_arr, lon_arr = load_timeseries_h5(ts_path)
    logger.info("%s: %d epochs", ORBIT_LABELS.get(orbit, orbit.upper()), len(dates))

    _vmin, _vmax = vmin, vmax
    if _vmin is None or _vmax is None:
        _vmin, _vmax = auto_vrange(data_mm)
        logger.info("Auto colour range: [%.1f, %.1f] mm", _vmin, _vmax)

    for i, date_str in enumerate(dates):
        if i % every != 0 and i != len(dates) - 1:
            continue  # always include last date
        frame = data_mm[i]
        frame[frame == 0] = np.nan  # mask zero-filled areas

        out_path = out_dir / f"cumulative_{orbit}_{date_str}.png"
        if out_path.exists():
            logger.debug("Already exists: %s", out_path.name)
            continue

        plot_displacement_map(
            data_2d=frame,
            lat_arr=lat_arr,
            lon_arr=lon_arr,
            date_str=date_str,
            title_prefix=f"Brumadinho InSAR — {ORBIT_LABELS.get(orbit, orbit.upper())} LOS Cumulative Displacement",
            out_path=out_path,
            vmin=_vmin,
            vmax=_vmax,
            zoom_extent=zoom_extent,
        )

    logger.info("Saved %s maps to %s", ORBIT_LABELS.get(orbit, orbit.upper()), out_dir)


def plot_2d_decomp_maps(decomp_dir: Path, out_dir: Path, vmin: float, vmax: float, every: int):
    """Plot vertical and E-W decomposed displacement maps."""
    vert_files = sorted(decomp_dir.glob("vertical_2019*.tif")) + \
                 sorted(decomp_dir.glob("vertical_2018*.tif")) + \
                 sorted(decomp_dir.glob("vertical_2017*.tif"))
    vert_files = sorted(set(vert_files))

    if not vert_files:
        logger.warning("No decomposed vertical GeoTIFFs found in %s", decomp_dir)
        return

    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio not installed — skipping decomp map plotting.")
        return

    for i, vert_path in enumerate(vert_files):
        if i % every != 0 and i != len(vert_files) - 1:
            continue

        date_str = vert_path.stem.replace("vertical_", "")
        ew_path  = decomp_dir / f"ew_horizontal_{date_str}.tif"

        for fpath, label, prefix, fname in [
            (vert_path, "Vertical displacement (mm)",   "Vertical",  f"vertical_{date_str}.png"),
            (ew_path,   "E-W displacement (+E / -W, mm)", "E-W horiz.", f"ew_{date_str}.png"),
        ]:
            if not fpath.exists():
                continue

            with rasterio.open(fpath) as src:
                data = src.read(1).astype(np.float32)
                data[data == src.nodata] = np.nan
                bounds = src.bounds
                lat_arr = np.linspace(bounds.top,    bounds.bottom, data.shape[0])
                lon_arr = np.linspace(bounds.left,   bounds.right,  data.shape[1])

            _vmin, _vmax = vmin, vmax
            if _vmin is None or _vmax is None:
                _vmin, _vmax = auto_vrange(data)

            plot_displacement_map(
                data_2d=data,
                lat_arr=lat_arr,
                lon_arr=lon_arr,
                date_str=date_str,
                title_prefix=f"Brumadinho InSAR — {prefix}",
                out_path=out_dir / fname,
                vmin=_vmin,
                vmax=_vmax,
                cbar_label=label,
            )

    logger.info("Saved decomposed displacement maps to %s", out_dir)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    processing_dir = Path(cfg.get("paths", "processing_dir"))
    results_dir    = Path(cfg.get("paths", "results_dir"))
    variant        = args.variant
    maps_suffix    = f"displacement_maps_{variant}" if variant else "displacement_maps"
    maps_dir       = results_dir / "figures" / maps_suffix
    decomp_dir_in  = results_dir / "data"   / "displacement_2d"
    decomp_dir_out = results_dir / "figures" / "decomposition"

    maps_dir.mkdir(parents=True, exist_ok=True)
    decomp_dir_out.mkdir(parents=True, exist_ok=True)

    orbits = ["desc53", "desc155"] if args.orbit == "both" else [args.orbit]

    # Tight zoom window: ±0.06° (~6.5 km) around dam centre
    zoom_extent = (
        DAM_LON - 0.06, DAM_LON + 0.06,   # lon_min, lon_max
        DAM_LAT - 0.06, DAM_LAT + 0.06,   # lat_min, lat_max
    ) if args.zoom_dam else None

    for orbit in orbits:
        dir_tag    = f"{orbit}_{variant}" if variant else orbit
        mintpy_dir = processing_dir / "mintpy" / dir_tag
        plot_orbit_maps(
            mintpy_dir=mintpy_dir,
            out_dir=maps_dir,
            orbit=orbit,
            vmin=args.vmin,
            vmax=args.vmax,
            every=args.every,
            zoom_extent=zoom_extent,
        )

    if not args.no_decomp and decomp_dir_in.exists():
        plot_2d_decomp_maps(
            decomp_dir=decomp_dir_in,
            out_dir=decomp_dir_out,
            vmin=args.vmin,
            vmax=args.vmax,
            every=args.every,
        )

    print(f"\nMaps saved to:")
    print(f"  {maps_dir}")
    if not args.no_decomp:
        print(f"  {decomp_dir_out}")
    print()
    print("Next step: python scripts/09_plot_timeseries.py")


if __name__ == "__main__":
    main()
