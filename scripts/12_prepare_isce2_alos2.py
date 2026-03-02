#!/usr/bin/env python3
"""
12_prepare_isce2_alos2.py — Generate ISCE2 stripmapStack configuration for ALOS-2
===================================================================================
Calls stripmapStack.py (ISCE2 contrib/stack/stripmapStack/) to set up the
interferogram processing network for the ALOS-2 ascending pass data.

This generates:
  - Per-pair configuration XMLs in processing/isce2/alos2_asc/configs/
  - Numbered run_XX_* shell scripts in processing/isce2/alos2_asc/run_files/

The run files are executed by 13_run_isce2_alos2.py.

ALOS-2 processing pipeline (ISCE2 stripmapStack):
    run_01  — Focus SLCs (if raw Level 1.0 data; skip for Level 1.1 pre-focused)
    run_02  — Compute baselines
    run_03  — Estimate dense offsets (coregistration)
    run_04  — Resample secondary SLCs
    run_05  — Generate interferograms + multi-look
    run_06  — Filter + unwrap (SNAPHU)
    run_07  — Geocode

Requirements:
    - ISCE2 installed with stripmapStack in PATH
      (typically at $ISCE_HOME/contrib/stack/stripmapStack/stripmapStack.py)
    - ALOS-2 Level 1.1 SLC data in data/raw/alos2/asc/
      (each scene as a directory containing LED-*, IMG-HH-*, etc.)
    - GLO-30 DEM in ISCE2 format at data/dem/glo30/dem.wgs84
      (run scripts/03_download_dem.py if not present)

Usage:
    python scripts/12_prepare_isce2_alos2.py
    python scripts/12_prepare_isce2_alos2.py --config config/project.cfg
    python scripts/12_prepare_isce2_alos2.py --dry-run   # print command, don't execute
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config, get_aoi, bbox_to_isce2_str

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ISCE2 stripmapStack for ALOS-2 ascending pass"
    )
    parser.add_argument("--config",  default="config/project.cfg")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stripmapStack.py command without executing it",
    )
    return parser.parse_args()


def check_stripmapstack() -> bool:
    """Check that stripmapStack.py is on the PATH."""
    result = subprocess.run(
        ["which", "stripmapStack.py"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def find_slc_dirs(alos2_dir: Path) -> list:
    """
    Find ALOS-2 SLC scene directories.

    ALOS-2 scenes downloaded from ASF are ZIP archives that extract to directories
    named like:  ALOS2_PALSAR2_<date>_<orbit>_<beam>/
    Each directory must contain a LED-* (leader) file for ISCE2 to recognise it.

    Returns sorted list of directories containing LED files.
    """
    # Pattern: any directory directly under alos2_dir that contains a LED-* file
    slc_dirs = []
    for d in sorted(alos2_dir.iterdir()):
        if d.is_dir() and list(d.glob("LED-*")):
            slc_dirs.append(d)

    return slc_dirs


def build_stripmapstack_cmd(
    slc_dir:         Path,
    dem_file:        Path,
    work_dir:        Path,
    bbox_str:        str,
    num_connections: int,
    az_looks:        int,
    rg_looks:        int,
    max_temp_days:   int,
    max_perp_m:      int,
) -> list:
    """
    Build the stripmapStack.py command for ALOS-2 FBD mode.

    Key ALOS-2-specific flags:
        -s alos2        : sensor type
        --nofocus       : data is already focused (Level 1.1 SLC from JAXA/ASF)
        --useGPU        : GPU acceleration if CUDA available (remove if not)

    Multi-look factors:
        ALOS-2 FBD native resolution: ~7 m range × ~3 m azimuth.
        az_looks=10, rg_looks=4 → ~30 m × 28 m (approximately square pixel).
        Adjust in config/project.cfg [alos2] az_looks / rg_looks.
    """
    cmd = [
        "stripmapStack.py",
        "-s",    "alos2",
        "-slc",  str(slc_dir),
        "-dem",  str(dem_file),
        "-w",    str(work_dir),
        "-b",    bbox_str,
        "--nofocus",                          # Level 1.1 is already focused
        "--numConnections",  str(num_connections),
        "--azimuthLooks",    str(az_looks),
        "--rangeLooks",      str(rg_looks),
        "--maxTemporal",     str(max_temp_days),
        "--maxSpatial",      str(max_perp_m),
        "--unwMethod",       "snaphu",
        "--filterStrength",  "0.5",
    ]
    return cmd


def print_next_steps(run_files: list, work_dir: Path):
    """Print a summary of the generated network and next steps."""
    print(f"\n{'='*65}")
    print(f"  ISCE2 ALOS-2 network prepared")
    print(f"{'='*65}")
    print(f"  Run files generated: {len(run_files)}")
    print(f"  Working directory  : {work_dir}")
    print()
    print("ACTION REQUIRED before running 13_run_isce2_alos2.py:")
    print("  1. Inspect the log for warnings about missing/skipped scenes.")
    print("  2. Check configs/ for burst/scene selections:")
    print(f"     ls {work_dir}/configs/")
    print("  3. Verify run_files/ — a complete network has run_01 through run_07:")
    print(f"     ls {work_dir}/run_files/")
    print()
    print("Next step: python scripts/13_run_isce2_alos2.py")


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if not args.dry_run and not check_stripmapstack():
        logger.error(
            "stripmapStack.py not found on PATH.\n"
            "ISCE2 must be installed with the stripmapStack contrib package.\n"
            "Typical location: $ISCE_HOME/contrib/stack/stripmapStack/\n"
            "Ensure it is added to your PATH and PYTHONPATH."
        )
        sys.exit(1)

    data_dir       = Path(cfg.get("paths", "data_dir"))
    processing_dir = Path(cfg.get("paths", "processing_dir"))
    logs_dir       = Path(cfg.get("paths", "logs_dir"))

    slc_dir   = data_dir       / "raw"   / "alos2" / "asc"
    dem_path  = data_dir       / "dem"   / "glo30"  / "dem.wgs84"
    work_dir  = processing_dir / "isce2" / "alos2_asc"
    log_path  = logs_dir       / "isce2_alos2_asc_setup.log"

    # Validate inputs
    if not dem_path.exists():
        logger.error(
            "DEM not found: %s\n"
            "Run scripts/03_download_dem.py first.", dem_path
        )
        sys.exit(1)

    if not slc_dir.exists():
        logger.error(
            "ALOS-2 data directory not found: %s\n"
            "Run scripts/11_download_alos2.py first.", slc_dir
        )
        sys.exit(1)

    slc_dirs = find_slc_dirs(slc_dir)
    if not slc_dirs:
        logger.error(
            "No ALOS-2 SLC scene directories found in %s.\n"
            "Each scene must be extracted to its own sub-directory\n"
            "containing a LED-* leader file.\n"
            "Check that 11_download_alos2.py completed extraction successfully.",
            slc_dir,
        )
        sys.exit(1)

    logger.info("Found %d ALOS-2 scene directories in %s", len(slc_dirs), slc_dir)
    for d in slc_dirs:
        logger.info("  %s", d.name)

    # Read parameters from config
    num_connections = cfg.getint("alos2", "num_connections")
    max_temp        = cfg.getint("alos2", "max_temp_baseline")
    max_perp        = cfg.getint("alos2", "max_perp_baseline")
    az_looks        = cfg.getint("alos2", "az_looks")
    rg_looks        = cfg.getint("alos2", "rg_looks")

    south, north, west, east = get_aoi(cfg)
    bbox_str = bbox_to_isce2_str(south, north, west, east)

    logger.info("Bounding box   : %s", bbox_str)
    logger.info("Connections    : %d", num_connections)
    logger.info("Temp baseline  : %d days", max_temp)
    logger.info("Perp baseline  : %d m", max_perp)
    logger.info("Multi-look     : %d az × %d rg", az_looks, rg_looks)

    cmd = build_stripmapstack_cmd(
        slc_dir=slc_dir,
        dem_file=dem_path,
        work_dir=work_dir,
        bbox_str=bbox_str,
        num_connections=num_connections,
        az_looks=az_looks,
        rg_looks=rg_looks,
        max_temp_days=max_temp,
        max_perp_m=max_perp,
    )

    if args.dry_run:
        print("\nDRY RUN — command that would be executed:")
        print("  " + " \\\n    ".join(cmd))
        print(f"\n  Working directory: {work_dir}")
        return

    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running stripmapStack.py ...")
    logger.info("Log: %s", log_path)

    logs_dir.mkdir(parents=True, exist_ok=True)
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
            "stripmapStack.py failed (exit %d). Check log: %s",
            result.returncode,
            log_path,
        )
        sys.exit(result.returncode)

    logger.info("stripmapStack.py completed successfully.")

    # List generated run files
    run_files = sorted((work_dir / "run_files").glob("run_[0-9]*"))
    print_next_steps(run_files, work_dir)


if __name__ == "__main__":
    main()
