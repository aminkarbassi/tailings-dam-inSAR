#!/usr/bin/env python3
"""
04_prepare_isce2.py — Generate ISCE2 topsStack configuration and run files
==========================================================================
Calls stackSentinel.py (part of ISCE2) to set up the interferogram processing
network for one orbit direction. This generates:
  - Per-pair configuration XMLs
  - Numbered run_XX_... shell scripts to execute the full pipeline

Must be run separately for each orbit:
    python scripts/04_prepare_isce2.py --orbit asc
    python scripts/04_prepare_isce2.py --orbit desc

Then verify the output before running 05_run_isce2.py.

Usage:
    python scripts/04_prepare_isce2.py --orbit {asc,desc}
                                        [--config config/project.cfg]

Requirements:
    ISCE2 must be installed and on the PATH (stackSentinel.py available).
    See README.md for ISCE2 installation instructions.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config, get_aoi, bbox_to_isce2_str
from utils.isce2_utils import check_isce2_available, build_stack_sentinel_cmd, get_run_files

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ISCE2 topsStack run files")
    parser.add_argument("--orbit",  required=True, choices=["asc", "desc"],
                        help="Orbit direction to prepare")
    parser.add_argument("--config", default="config/project.cfg")
    return parser.parse_args()


def get_subswaths(cfg, orbit: str) -> list:
    """Parse subswath list from config, e.g. 'IW1 IW2' → ['IW1', 'IW2']."""
    key = f"{orbit}_subswaths"
    raw = cfg.get("sentinel1", key)
    return [s.strip() for s in raw.split() if s.strip()]


def verify_safe_dirs(slc_dir: Path, orbit: str) -> list:
    """
    Check that unzipped SAFE directories exist and look complete.

    Returns list of SAFE directory paths.
    """
    safe_dirs = sorted(slc_dir.glob("S1*.SAFE"))
    if not safe_dirs:
        logger.error(
            "No .SAFE directories found in %s.\n"
            "Run 02_download_data.py --orbit %s first.",
            slc_dir,
            orbit,
        )
        sys.exit(1)

    logger.info("Found %d SAFE directories in %s", len(safe_dirs), slc_dir)

    # Spot-check: verify consistent relative orbit numbers from directory names
    # S1A_IW_SLC__1SDV_20190122T085023_20190122T085050_025507_02D0CD_DD6B.SAFE
    orbits_seen = set()
    for safe in safe_dirs:
        parts = safe.name.split("_")
        if len(parts) >= 10:
            # The absolute orbit is in parts[7]; relative orbit not directly in filename.
            # Full check would require reading annotation XML — accept all here.
            pass
        orbits_seen.add(safe.name[:3])  # S1A or S1B

    logger.info("Satellite platforms found: %s", orbits_seen)
    if len(orbits_seen) > 2:
        logger.warning("More than 2 platforms detected — verify relative orbits are consistent.")

    return safe_dirs


def run_stack_sentinel(
    slc_dir: Path,
    dem_path: Path,
    orbit_dir: Path,
    work_dir: Path,
    bbox_str: str,
    subswaths: list,
    cfg,
    log_path: Path,
):
    """Run stackSentinel.py to generate the interferogram network."""
    num_connections  = cfg.getint("isce2", "num_connections")
    coregistration  = cfg.get("isce2", "coregistration")
    workflow         = cfg.get("isce2", "workflow")

    cmd = build_stack_sentinel_cmd(
        slc_dir=slc_dir,
        dem_file=dem_path,
        orbit_dir=orbit_dir,
        work_dir=work_dir,
        bbox_str=bbox_str,
        num_connections=num_connections,
        coregistration=coregistration,
        workflow=workflow,
        subswaths=subswaths,
    )

    # Remove --useGPU if not applicable (will silently fall back to CPU in ISCE2)
    # but keeping it is harmless.

    logger.info("Running: %s", " ".join(cmd))
    logger.info("Working directory: %s", work_dir)
    logger.info("Log: %s", log_path)

    work_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        logger.error(
            "stackSentinel.py failed (exit %d). Check log: %s",
            result.returncode,
            log_path,
        )
        sys.exit(result.returncode)

    logger.info("stackSentinel.py completed successfully.")


def print_next_steps(run_files: list, orbit: str, work_dir: Path):
    """Print a summary of the generated network and next steps."""
    print(f"\n{'='*60}")
    print(f"  ISCE2 network prepared for {orbit.upper()} orbit")
    print(f"{'='*60}")
    print(f"  Run files generated: {len(run_files)}")
    print(f"  Working directory  : {work_dir}")
    print()
    print("ACTION REQUIRED before running 05_run_isce2.py:")
    print("  1. Check the log for any warnings about missing/problematic scenes.")
    print("  2. Open 2-3 run files to verify the burst selection looks correct.")
    print(f"     ls {work_dir}/run_files/")
    print("  3. Verify the interferogram network in the configs/ directory.")
    print("     A good network has ~3 connections per node and no isolated epochs.")
    print()
    print(f"Next step: python scripts/05_run_isce2.py --orbit {orbit}")


def main():
    args  = parse_args()
    orbit = args.orbit
    cfg   = load_config(args.config)

    if not check_isce2_available():
        logger.error(
            "stackSentinel.py not found on PATH.\n"
            "ISCE2 must be installed before running this script.\n"
            "See README.md for installation instructions."
        )
        sys.exit(1)

    data_dir       = Path(cfg.get("paths", "data_dir"))
    processing_dir = Path(cfg.get("paths", "processing_dir"))
    logs_dir       = Path(cfg.get("paths", "logs_dir"))

    slc_dir    = data_dir / "raw" / f"slc_{orbit}"
    dem_path   = data_dir / "dem" / "glo30" / "dem.wgs84"
    orbit_dir  = data_dir / "aux" / "orbits"
    work_dir   = processing_dir / "isce2" / orbit
    log_path   = logs_dir / f"isce2_{orbit}_setup.log"

    # Validate inputs
    if not dem_path.exists():
        logger.error(
            "DEM not found: %s\nRun 03_download_dem.py first.", dem_path
        )
        sys.exit(1)

    safe_dirs  = verify_safe_dirs(slc_dir, orbit)
    subswaths  = get_subswaths(cfg, orbit)
    south, north, west, east = get_aoi(cfg)
    bbox_str   = bbox_to_isce2_str(south, north, west, east)

    logger.info("Orbit:      %s", orbit.upper())
    logger.info("Scenes:     %d SAFE directories", len(safe_dirs))
    logger.info("Subswaths:  %s", subswaths)
    logger.info("Bbox:       %s", bbox_str)
    logger.info("Connections: %s", cfg.get("isce2", "num_connections"))

    run_stack_sentinel(
        slc_dir=slc_dir,
        dem_path=dem_path,
        orbit_dir=orbit_dir,
        work_dir=work_dir,
        bbox_str=bbox_str,
        subswaths=subswaths,
        cfg=cfg,
        log_path=log_path,
    )

    run_files = get_run_files(work_dir / "run_files")
    print_next_steps(run_files, orbit, work_dir)


if __name__ == "__main__":
    main()
