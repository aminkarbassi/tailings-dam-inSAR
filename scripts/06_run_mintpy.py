#!/usr/bin/env python3
"""
06_run_mintpy.py — Run MintPy SBAS time-series analysis
========================================================
Executes the full MintPy smallbaselineApp.py pipeline step by step for one
orbit direction. Each step is run individually using --dostep so that:
  - Failures are immediately obvious and traceable
  - Individual steps can be re-run after adjusting config
  - Logs are separated per step for easy debugging

Steps executed:
    load_data           → Read ISCE2 interferograms into MintPy HDF5
    modify_network      → Apply coherence-based network filtering
    reference_point     → Set stable reference pixel
    correct_unwrap_error → Fix integer-cycle phase unwrapping errors
    invert              → SBAS inversion → raw time-series
    correct_LOD         → Sentinel-1 local oscillator drift
    correct_troposphere → Atmospheric delay correction (ERA5 or GACOS)
    deramp              → Remove long-wavelength orbital ramps
    correct_topography  → DEM error correction
    residual_RMS        → Identify and flag noisy epochs
    reference_date      → Set first acquisition as displacement = 0
    velocity            → Fit linear velocity to time-series
    geocode             → Reproject to geographic coordinates (lat/lon)

Usage:
    python scripts/06_run_mintpy.py --orbit desc53   [--config config/project.cfg]
    python scripts/06_run_mintpy.py --orbit desc155
    python scripts/06_run_mintpy.py --orbit desc53  --from-step correct_troposphere
    python scripts/06_run_mintpy.py --orbit desc53  --only-step velocity

Requirements:
    conda install -c conda-forge mintpy
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# MintPy step order — must be executed in this sequence
MINTPY_STEPS = [
    "load_data",
    "modify_network",
    "reference_point",
    "correct_unwrap_error",
    "invert_network",
    "correct_LOD",
    "correct_troposphere",
    "deramp",
    "correct_topography",
    "residual_RMS",
    "reference_date",
    "velocity",
    "geocode",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MintPy SBAS time-series pipeline")
    parser.add_argument("--orbit",     required=True, choices=["desc53", "desc155"])
    parser.add_argument("--config",    default="config/project.cfg")
    parser.add_argument("--variant",   default=None,
                        help="Name suffix for config and work dir (e.g. 'isbas' → "
                             "mintpy_desc53_isbas.cfg / processing/mintpy/desc53_isbas/)")
    parser.add_argument("--from-step", default=None,
                        help="Resume from this step (e.g. correct_troposphere)")
    parser.add_argument("--only-step", default=None,
                        help="Run only this single step")
    return parser.parse_args()


def check_mintpy_available() -> bool:
    result = subprocess.run(["which", "smallbaselineApp.py"], capture_output=True, text=True)
    return result.returncode == 0


def run_mintpy_step(
    step: str,
    template_file: Path,
    work_dir: Path,
    log_path: Path,
    orbit_tag: str = "",
):
    """
    Run a single MintPy step using smallbaselineApp.py --dostep.

    Args:
        step:          MintPy step name.
        template_file: Path to the MintPy .cfg template file.
        work_dir:      MintPy working directory.
        log_path:      Path to append log output.
    """
    cmd = [
        "smallbaselineApp.py",
        str(template_file),
        "--dostep", step,
        "--dir", str(work_dir),
    ]

    logger.info("Running MintPy step: %s", step)

    with open(log_path, "a") as log_fh:
        log_fh.write(f"\n{'='*60}\n  STEP: {step}\n{'='*60}\n")
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        logger.error(
            "MintPy step FAILED: %s (exit %d)\nCheck log: %s",
            step,
            result.returncode,
            log_path,
        )
        logger.error("To re-run from this step:")
        logger.error(
            "  python scripts/06_run_mintpy.py --orbit %s --from-step %s",
            orbit_tag,
            step,
        )
        sys.exit(result.returncode)


def verify_load_data_inputs(template_file: Path):
    """
    Before running load_data, check that the ISCE2 output paths in the
    template file actually exist and contain interferograms.
    """
    import re
    with open(template_file) as f:
        content = f.read()

    unw_pattern = re.search(r"mintpy\.load\.unwFile\s*=\s*(.+)", content)
    if not unw_pattern:
        return

    glob_pattern = unw_pattern.group(1).strip()
    # Check the directory part exists
    base_dir = Path(glob_pattern.split("*")[0])
    if not base_dir.exists():
        logger.error(
            "ISCE2 interferogram directory not found: %s\n"
            "Run 05_run_isce2.py before MintPy.",
            base_dir,
        )
        sys.exit(1)

    import glob as glob_mod
    matches = glob_mod.glob(glob_pattern)
    if not matches:
        logger.error(
            "No unwrapped interferograms found matching: %s\n"
            "ISCE2 processing may be incomplete.",
            glob_pattern,
        )
        sys.exit(1)

    logger.info("Found %d interferograms for MintPy input.", len(matches))


def print_qc_checklist(orbit: str, work_dir: Path):
    """Print post-processing QC steps."""
    print(f"\nMintPy {orbit.upper()} orbit time-series complete.")
    print()
    print("QC checklist:")
    print(f"  1. Check temporal coherence map:")
    print(f"     {work_dir}/pic/temporalCoherence.png")
    print(f"     → Should be > 0.7 over bedrock, lower over vegetation/tailings.")
    print(f"  2. Check interferogram network:")
    print(f"     {work_dir}/pic/network.png")
    print(f"     → Should be well-connected; no isolated epochs.")
    print(f"  3. Check mean LOS velocity:")
    print(f"     {work_dir}/pic/velocity.png")
    print(f"     → Stable areas should be near-zero; dam should show clear signal.")
    print(f"  4. If coherence or velocity look wrong, adjust reference point in")
    print(f"     config/mintpy_{orbit}.cfg and re-run from reference_point step.")
    print()
    print(f"If both ASC and DESC are done:")
    print(f"  Next step: python scripts/07_decompose_2d.py")
    print(f"  Or skip decomp: python scripts/08_plot_maps.py")


def main():
    args  = parse_args()
    orbit = args.orbit
    cfg   = load_config(args.config)

    if not check_mintpy_available():
        logger.error(
            "smallbaselineApp.py not found. Install MintPy:\n"
            "  conda install -c conda-forge mintpy"
        )
        sys.exit(1)

    project_root   = Path(cfg.get("paths", "project_root"))
    processing_dir = Path(cfg.get("paths", "processing_dir"))
    logs_dir       = Path(cfg.get("paths", "logs_dir"))

    tag           = f"{orbit}_{args.variant}" if args.variant else orbit
    cfg_name      = f"mintpy_{tag}.cfg"
    template_file = project_root / "config" / cfg_name
    work_dir      = processing_dir / "mintpy" / tag
    log_path      = logs_dir / f"mintpy_{tag}.log"

    if not template_file.exists():
        logger.error("MintPy config not found: %s", template_file)
        sys.exit(1)

    work_dir.mkdir(parents=True, exist_ok=True)

    # Determine which steps to run
    if args.only_step:
        if args.only_step not in MINTPY_STEPS:
            logger.error("Unknown step: %s. Valid steps: %s", args.only_step, MINTPY_STEPS)
            sys.exit(1)
        steps = [args.only_step]
    elif args.from_step:
        if args.from_step not in MINTPY_STEPS:
            logger.error("Unknown step: %s. Valid steps: %s", args.from_step, MINTPY_STEPS)
            sys.exit(1)
        start_idx = MINTPY_STEPS.index(args.from_step)
        steps = MINTPY_STEPS[start_idx:]
    else:
        steps = MINTPY_STEPS

    logger.info("MintPy %s orbit pipeline: %d steps", orbit.upper(), len(steps))
    logger.info("Template: %s", template_file)
    logger.info("Work dir: %s", work_dir)
    logger.info("Log:      %s", log_path)

    # Verify ISCE2 outputs exist before starting load_data
    if "load_data" in steps:
        verify_load_data_inputs(template_file)

    start_time = time.time()

    for i, step in enumerate(steps, 1):
        logger.info("[%d/%d] %s", i, len(steps), step)
        step_start = time.time()

        run_mintpy_step(step, template_file, work_dir, log_path, orbit_tag=tag)

        elapsed = time.time() - step_start
        logger.info("Step done in %.1f min", elapsed / 60)

    total_elapsed = (time.time() - start_time) / 60
    logger.info("MintPy %s pipeline complete in %.1f minutes", orbit.upper(), total_elapsed)

    print_qc_checklist(orbit, work_dir)


if __name__ == "__main__":
    main()
