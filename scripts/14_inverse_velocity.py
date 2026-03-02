#!/usr/bin/env python3
"""
14_inverse_velocity.py — Fukuzono inverse velocity collapse prediction
=======================================================================
Applies the inverse velocity method (Fukuzono 1985) to the MintPy InSAR
time-series at the Brumadinho Dam I crest to retroactively predict the
collapse date, reproducing the core result of Grebby et al. 2021.

Method:
  1. Extract cumulative LOS displacement at the dam crest from each orbit's
     corrected time-series HDF5 (3×3 pixel neighbourhood average).
  2. Estimate instantaneous velocity at each epoch using a sliding-window
     linear fit (default window = 4 epochs ≈ 24 days for Sentinel-1).
  3. Compute 1/v at each epoch.
  4. Fit a linear regression to 1/v in the acceleration window
     (default: 2018-07-01 → last acquisition).
  5. Extrapolate to 1/v = 0 → predicted failure date.

A negative linear slope in 1/v vs time means the system is accelerating
toward failure. Extrapolating to zero gives the predicted collapse time.

Outputs:
  results/figures/inverse_velocity/
    inv_vel_{orbit}.png        — three-panel plot (displacement / velocity / 1/v)
  results/data/
    inverse_velocity_{orbit}.csv   — epoch-by-epoch table

References:
  Fukuzono T. (1985). A new method for predicting the failure time of a slope.
    Proc. 4th Int. Conf. Field Workshop Landslides, Tokyo, 145–150.
  Voight B. (1988). A method for prediction of volcanic eruptions. Nature 332, 125–130.
  Grebby S. et al. (2021). Mapping and monitoring pre-collapse surface deformation
    of the Brumadinho tailings dam. Nat. Commun. Earth Environ. 2, 80.

Usage:
  python scripts/14_inverse_velocity.py
  python scripts/14_inverse_velocity.py --orbit desc155
  python scripts/14_inverse_velocity.py --orbit desc155 --window 4 --accel-start 2018-07-01
  python scripts/14_inverse_velocity.py --orbit both --accel-start 2018-10-01
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
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
DAM_LAT = -20.1113
DAM_LON = -44.1231

ORBIT_LABELS = {
    "desc53":    "DESC Track 53 (inc. ~32°)",
    "desc155":   "DESC Track 155 (inc. ~45°)",
    "alos2_asc": "ALOS-2 L-band ASC (inc. ~30–36°)",
}
ORBIT_COLORS = {
    "desc53":    "steelblue",
    "desc155":   "darkorange",
    "alos2_asc": "forestgreen",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inverse velocity collapse time prediction for Brumadinho Dam I"
    )
    parser.add_argument("--config", default="config/project.cfg")
    parser.add_argument(
        "--orbit",
        default="both",
        choices=["desc53", "desc155", "alos2_asc", "both"],
        help="Orbit(s) to analyse. 'both' = desc53 + desc155 (default).",
    )
    parser.add_argument(
        "--poi-lat", type=float, default=DAM_LAT,
        help=f"Latitude of extraction point (default: {DAM_LAT})",
    )
    parser.add_argument(
        "--poi-lon", type=float, default=DAM_LON,
        help=f"Longitude of extraction point (default: {DAM_LON})",
    )
    parser.add_argument(
        "--window", type=int, default=4,
        help="Sliding window (epochs) for velocity estimation (default: 4)",
    )
    parser.add_argument(
        "--accel-start", default="2018-07-01",
        help="Start of acceleration window for 1/v regression (default: 2018-07-01)",
    )
    parser.add_argument(
        "--extract-window", type=int, default=3,
        help="Spatial pixel window for extraction average (default: 3×3)",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Processing variant suffix (e.g. 'isbas')",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (mirrors 09_plot_timeseries.py)
# ─────────────────────────────────────────────────────────────────────────────

def find_timeseries_file(mintpy_dir: Path) -> Path | None:
    """Find best available corrected time-series HDF5."""
    candidates = [
        mintpy_dir / "timeseries_ERA5_ramp_demErr.h5",
        mintpy_dir / "timeseries_ramp_demErr.h5",
        mintpy_dir / "timeseries_demErr.h5",
        mintpy_dir / "timeseries_ramp.h5",
        mintpy_dir / "timeseries.h5",
        mintpy_dir / "geo" / "geo_timeseries_ramp_demErr.h5",
        mintpy_dir / "geo" / "geo_timeseries_demErr.h5",
        mintpy_dir / "geo" / "geo_timeseries.h5",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_timeseries_h5(h5_path: Path):
    """Return dates (list[str]), data_mm (T,H,W), lat_arr, lon_arr."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        data_m = f["timeseries"][...]
        dates = [
            d.decode() if isinstance(d, bytes) else str(d)
            for d in f["date"][:]
        ]
        attrs = dict(f.attrs)

    data_mm = data_m * 1000.0

    x0 = float(attrs.get("X_FIRST", DAM_LON - 0.2))
    y0 = float(attrs.get("Y_FIRST", DAM_LAT - 0.2))
    dx = float(attrs.get("X_STEP",   0.0001))
    dy = float(attrs.get("Y_STEP",  -0.0001))
    H, W = data_mm.shape[1], data_mm.shape[2]

    epsg = attrs.get("EPSG", None)
    if epsg and int(str(epsg)) != 4326:
        from pyproj import Transformer
        t = Transformer.from_crs(f"EPSG:{int(str(epsg))}", "EPSG:4326", always_xy=True)
        lon0, lat0 = t.transform(x0, y0)
        lon1, lat1 = t.transform(x0 + (W - 1) * dx, y0 + (H - 1) * dy)
        lon_arr = np.linspace(lon0, lon1, W)
        lat_arr = np.linspace(lat0, lat1, H)
    else:
        lon_arr = x0 + np.arange(W) * dx
        lat_arr = y0 + np.arange(H) * dy

    return dates, data_mm, lat_arr, lon_arr


def extract_timeseries(data_mm, lat, lon, lat_arr, lon_arr, win=3):
    """
    Mean cumulative displacement (mm) over a win×win pixel neighbourhood.
    If fewer than 3 valid pixels are found, the window is doubled up to 4×
    to ensure sparse coherence near the dam crest is captured.
    """
    row = int(np.argmin(np.abs(lat_arr - lat)))
    col = int(np.argmin(np.abs(lon_arr - lon)))
    H, W = data_mm.shape[1], data_mm.shape[2]

    # Expand window until we get at least 3 valid pixels in the LAST epoch
    last = data_mm[-1]
    for trial_win in [win, win * 2, win * 3, win * 4]:
        half = trial_win // 2
        r0, r1 = max(0, row - half), min(H, row + half + 1)
        c0, c1 = max(0, col - half), min(W, col + half + 1)
        patch_last = last[r0:r1, c0:c1]
        n_valid = int(np.sum(patch_last != 0))
        if n_valid >= 3 or trial_win >= win * 4:
            break

    if trial_win != win:
        logger.info(
            "  Expanded extraction window from %d to %d px (%d valid pixels found)",
            win, trial_win, n_valid,
        )

    patch = data_mm[:, r0:r1, c0:c1].copy()
    # First epoch is always 0 by MintPy convention; masked pixels are also 0.
    # Replace exactly-zero values in non-first epochs with NaN (masked), but
    # keep first epoch as-is (all zero = valid reference).
    patch[1:][patch[1:] == 0] = np.nan
    with np.errstate(all="ignore"):
        ts = np.nanmean(patch.reshape(patch.shape[0], -1), axis=1)
    return ts


# ─────────────────────────────────────────────────────────────────────────────
# Inverse velocity maths
# ─────────────────────────────────────────────────────────────────────────────

def compute_sliding_velocity(dts: list, ts_mm: np.ndarray, win: int = 4) -> np.ndarray:
    """
    Instantaneous velocity at each epoch using a centred sliding-window
    linear fit over `win` epochs.

    Returns:
        vel_mmday: velocity in mm/day (NaN where data are insufficient)
    """
    T = len(dts)
    days = np.array([(dt - dts[0]).days for dt in dts], dtype=float)
    vel = np.full(T, np.nan)
    half = win // 2

    for i in range(T):
        i0 = max(0, i - half)
        i1 = min(T, i0 + win)          # keep window size fixed
        if i1 - i0 < 2:
            continue
        d_sl = days[i0:i1]
        v_sl = ts_mm[i0:i1]
        ok = np.isfinite(v_sl)
        if ok.sum() < 2:
            continue
        A = np.vstack([d_sl[ok], np.ones(ok.sum())]).T
        slope = np.linalg.lstsq(A, v_sl[ok], rcond=None)[0][0]
        vel[i] = slope

    return vel


def fit_inverse_velocity(dts: list, inv_vel: np.ndarray, accel_start: datetime):
    """
    Robust linear regression on 1/v in [accel_start, last epoch].

    Outliers are clipped at the 10th–90th percentile of the selected window
    before regression to prevent a single noisy epoch from dominating.

    Returns (slope, intercept, fit_dts, fit_inv_vel, predicted_failure_dt)
    or (None, ..., None) if the regression cannot be performed.
    """
    days0 = np.array([(dt - dts[0]).days for dt in dts], dtype=float)

    sel = np.array([dt >= accel_start for dt in dts]) & np.isfinite(inv_vel)

    if sel.sum() < 3:
        logger.warning(
            "Only %d valid points in acceleration window (need ≥3). "
            "Try an earlier --accel-start.", sel.sum()
        )
        return None, None, None, None, None

    x = days0[sel]
    y = inv_vel[sel]

    # Clip extreme outliers (outside 10–90th percentile) before regression
    lo, hi = np.nanpercentile(y, 10), np.nanpercentile(y, 90)
    keep = (y >= lo) & (y <= hi)
    if keep.sum() < 3:
        keep = np.ones(len(y), dtype=bool)   # fallback: use all points
    logger.info(
        "  1/v regression: %d/%d points in window after outlier clip [%.1f, %.1f]",
        keep.sum(), len(y), lo, hi,
    )

    A = np.vstack([x[keep], np.ones(keep.sum())]).T
    slope, intercept = np.linalg.lstsq(A, y[keep], rcond=None)[0]

    if slope >= 0:
        logger.warning(
            "1/v regression slope is positive (%.4f day/mm per day) — velocity is NOT "
            "accelerating in the selected window.\n"
            "  → Try a later --accel-start (e.g. 2018-10-01 or 2018-11-01).",
            slope,
        )
        return slope, intercept, None, None, None

    # t* where 1/v = 0
    t_star = -intercept / slope                          # days since dts[0]
    predicted = dts[0] + timedelta(days=float(t_star))

    # Build continuous fit line from accel_start to predicted failure
    x_fit = np.linspace(float(x[0]), t_star, 300)
    y_fit = slope * x_fit + intercept
    fit_dts = [dts[0] + timedelta(days=float(xi)) for xi in x_fit]

    return slope, intercept, fit_dts, y_fit, predicted


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_panels(
    dts, ts_mm, vel_mmday, inv_vel,
    accel_start, fit_dts, fit_inv_vel, predicted,
    orbit, poi_lat, poi_lon, out_path,
):
    """Three-panel figure: displacement / velocity / inverse velocity."""
    color = ORBIT_COLORS.get(orbit, "steelblue")
    label = ORBIT_LABELS.get(orbit, orbit.upper())

    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    fig.suptitle(
        f"Brumadinho Dam I — Inverse Velocity Analysis\n"
        f"{label}  ·  Extraction point: {poi_lat}°S, {abs(poi_lon)}°W",
        fontsize=13, fontweight="bold",
    )

    # shared formatting helper
    def _fmt_axis(ax):
        ax.axvline(COLLAPSE_DATE, color="red", lw=2, ls="--", zorder=5)
        ax.axvspan(accel_start, max(COLLAPSE_DATE, dts[-1]),
                   alpha=0.07, color="red", zorder=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.axhline(0, color="gray", lw=0.8, ls=":", zorder=1)

    # ── Panel 1: Cumulative displacement ─────────────────────────────────
    ax = axes[0]
    ax.scatter(dts, ts_mm, s=25, color=color, zorder=3,
               label="Cumulative LOS displacement")
    ax.axvline(COLLAPSE_DATE, color="red", lw=2, ls="--",
               label="Collapse: 25 Jan 2019", zorder=5)
    ax.axvspan(accel_start, max(COLLAPSE_DATE, dts[-1]),
               alpha=0.07, color="red", label="Acceleration window", zorder=0)
    ax.axhline(0, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_ylabel("Cumulative LOS displacement (mm)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Panel 2: Velocity ─────────────────────────────────────────────────
    ax = axes[1]
    ok = np.isfinite(vel_mmday)
    ax.plot(
        [dts[i] for i in range(len(dts)) if ok[i]],
        vel_mmday[ok],
        "o-", ms=5, color=color, lw=1.5, label="Velocity (mm/day)",
    )
    ax.set_ylabel("LOS velocity (mm/day)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    _fmt_axis(ax)

    # ── Panel 3: Inverse velocity ─────────────────────────────────────────
    ax = axes[2]
    ok_iv = np.isfinite(inv_vel)
    ax.scatter(
        [dts[i] for i in range(len(dts)) if ok_iv[i]],
        inv_vel[ok_iv],
        s=30, color=color, zorder=3, label="1/velocity (day/mm)",
    )

    if fit_dts is not None:
        ax.plot(fit_dts, fit_inv_vel, "k-", lw=2.5,
                label="Linear regression (acceleration window)")

    if predicted is not None:
        ax.axvline(predicted, color="darkred", lw=2, ls=":",
                   label=f"Predicted collapse: {predicted.strftime('%d %b %Y')}", zorder=6)
        error_days = (predicted - COLLAPSE_DATE).days
        sign = "+" if error_days >= 0 else ""
        ax.annotate(
            f"Prediction error: {sign}{error_days} days",
            xy=(predicted, 0),
            xytext=(0.55, 0.12),
            textcoords="axes fraction",
            fontsize=9, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
        )

    ax.axvline(COLLAPSE_DATE, color="red", lw=2, ls="--",
               label="Actual collapse: 25 Jan 2019", zorder=5)
    ax.axvspan(accel_start, max(COLLAPSE_DATE, dts[-1]),
               alpha=0.07, color="red", zorder=0)
    ax.axhline(0, color="gray", lw=1.5, ls="-", zorder=2)
    ax.set_ylabel("Inverse LOS velocity (day/mm)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-orbit runner
# ─────────────────────────────────────────────────────────────────────────────

def run_orbit(
    orbit, mintpy_dir, out_fig_dir, out_data_dir,
    poi_lat, poi_lon, win, accel_start, extract_win, variant,
):
    tag     = f"{orbit}_{variant}" if variant else orbit
    h5_path = find_timeseries_file(mintpy_dir)
    if h5_path is None:
        logger.warning("No time-series HDF5 in %s — skipping %s.", mintpy_dir, orbit)
        return

    logger.info("[%s] %s", orbit.upper(), h5_path.name)
    dates, data_mm, lat_arr, lon_arr = load_timeseries_h5(h5_path)
    dts = [datetime.strptime(d, "%Y%m%d") for d in dates]

    ts_mm = extract_timeseries(data_mm, poi_lat, poi_lon, lat_arr, lon_arr, win=extract_win)
    n_finite = int(np.isfinite(ts_mm).sum())
    logger.info("[%s] %d epochs, %d with valid data", orbit.upper(), len(dts), n_finite)

    vel = compute_sliding_velocity(dts, ts_mm, win=win)
    inv_vel = np.where(np.abs(vel) > 1e-6, 1.0 / vel, np.nan)

    slope, intercept, fit_dts, fit_inv_vel, predicted = fit_inverse_velocity(
        dts, inv_vel, accel_start
    )

    if predicted is not None:
        error = (predicted - COLLAPSE_DATE).days
        sign = "+" if error >= 0 else ""
        logger.info(
            "[%s] Predicted collapse: %s  (error: %s%d days vs actual 2019-01-25)",
            orbit.upper(), predicted.strftime("%Y-%m-%d"), sign, error,
        )

    # Plot
    plot_panels(
        dts, ts_mm, vel, inv_vel,
        accel_start, fit_dts, fit_inv_vel, predicted,
        orbit, poi_lat, poi_lon,
        out_path=out_fig_dir / f"inv_vel_{tag}.png",
    )

    # CSV
    out_data_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "date":                   [dt.strftime("%Y-%m-%d") for dt in dts],
        "displacement_mm":         np.round(ts_mm, 3),
        "velocity_mmday":          np.round(vel, 5),
        "inv_velocity_daymm":      np.round(inv_vel, 5),
    })
    csv_path = out_data_dir / f"inverse_velocity_{tag}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("[%s] CSV: %s", orbit.upper(), csv_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    processing_dir = Path(cfg.get("paths", "processing_dir"))
    results_dir    = Path(cfg.get("paths", "results_dir"))
    out_fig_dir    = results_dir / "figures" / "inverse_velocity"
    out_data_dir   = results_dir / "data"

    accel_start = datetime.strptime(args.accel_start, "%Y-%m-%d")
    orbits = ["desc53", "desc155"] if args.orbit == "both" else [args.orbit]

    logger.info("Inverse velocity analysis — window: %d epochs, accel start: %s",
                args.window, args.accel_start)

    for orbit in orbits:
        tag      = f"{orbit}_{args.variant}" if args.variant else orbit
        mintpy_d = processing_dir / "mintpy" / tag
        if not mintpy_d.exists():
            logger.warning("MintPy dir not found: %s — skipping.", mintpy_d)
            continue
        run_orbit(
            orbit       = orbit,
            mintpy_dir  = mintpy_d,
            out_fig_dir = out_fig_dir,
            out_data_dir= out_data_dir,
            poi_lat     = args.poi_lat,
            poi_lon     = args.poi_lon,
            win         = args.window,
            accel_start = accel_start,
            extract_win = args.extract_window,
            variant     = args.variant,
        )

    print(f"\nOutputs saved to:")
    print(f"  Figures: {out_fig_dir}/inv_vel_*.png")
    print(f"  Data:    {out_data_dir}/inverse_velocity_*.csv")
    print()
    print("Interpretation:")
    print("  • Negative slope in the 1/v panel = accelerating creep toward failure")
    print("  • Predicted failure where the regression line crosses 1/v = 0")
    print("  • Prediction within ±2 weeks is considered good for InSAR time-series")
    print("  • If prediction is poor, adjust --accel-start to a later onset date")
    print("    or increase --window to reduce noise in the velocity estimate")


if __name__ == "__main__":
    main()
