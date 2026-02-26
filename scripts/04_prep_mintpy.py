#!/usr/bin/env python3
"""
04_prep_mintpy.py — Prepare MintPy inputs from HyP3 interferogram products
===========================================================================
1. Extracts all HyP3 zip files into per-pair subdirectories.
2. Runs prep_hyp3.py on the extracted TIF files to generate .rsc sidecar files
   (required by MintPy's data loader).
3. Runs smallbaselineApp.py --dostep load_data to create:
       processing/mintpy/{orbit}/inputs/ifgramStack.h5
       processing/mintpy/{orbit}/inputs/geometryGeo.h5

Usage:
    python scripts/04_prep_mintpy.py --orbit asc  [--config config/project.cfg]
    python scripts/04_prep_mintpy.py --orbit desc
"""

import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MintPy inputs from HyP3 products")
    parser.add_argument("--orbit",  required=True, choices=["asc", "desc"])
    parser.add_argument("--config", default="config/project.cfg")
    return parser.parse_args()


def check_command(name):
    r = subprocess.run(["which", name], capture_output=True, text=True)
    return r.returncode == 0


def unzip_products(hyp3_dir: Path):
    """Extract all zip files in hyp3_dir.

    HyP3 zips already contain a top-level directory named the same as the zip
    stem, so extracting to hyp3_dir directly gives the correct structure:
        hyp3_dir/S1BB_.../S1BB_..._unw_phase.tif
    which matches the glob pattern */*_unw_phase.tif in the MintPy config.
    """
    zips = sorted(hyp3_dir.glob("*.zip"))
    if not zips:
        logger.error("No zip files found in %s", hyp3_dir)
        sys.exit(1)

    logger.info("Extracting %d zip files in %s ...", len(zips), hyp3_dir)
    for i, zf in enumerate(zips, 1):
        dest = hyp3_dir / zf.stem   # the directory the zip will create
        if dest.exists() and any(dest.iterdir()):
            # already extracted correctly
            continue
        with zipfile.ZipFile(zf) as z:
            z.extractall(hyp3_dir)  # zip creates its own subdir inside hyp3_dir
        if i % 20 == 0 or i == len(zips):
            logger.info("  Extracted %d/%d", i, len(zips))

    logger.info("Extraction complete.")


def run_prep_hyp3(hyp3_dir: Path, log_path: Path):
    """Run prep_hyp3.py on all unw_phase TIF files to generate .rsc sidecar files."""
    unw_files = sorted(hyp3_dir.rglob("*_unw_phase.tif"))
    if not unw_files:
        logger.error("No *_unw_phase.tif files found after extraction.")
        sys.exit(1)
    logger.info("Running prep_hyp3.py on %d unwrapped interferograms...", len(unw_files))

    cmd = ["prep_hyp3.py"] + [str(f) for f in unw_files]
    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        logger.error("prep_hyp3.py failed. Check log: %s", log_path)
        sys.exit(1)
    logger.info("prep_hyp3.py complete — .rsc sidecar files written.")


def run_load_data(mintpy_dir: Path, template: Path, log_path: Path):
    """Run smallbaselineApp.py --dostep load_data to build ifgramStack.h5."""
    logger.info("Running smallbaselineApp.py --dostep load_data ...")

    cmd = [
        "smallbaselineApp.py",
        str(template),
        "--dostep", "load_data",
    ]
    with open(log_path, "a") as log_fh:
        result = subprocess.run(
            cmd,
            cwd=str(mintpy_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        logger.error("smallbaselineApp.py load_data failed. Check log: %s", log_path)
        sys.exit(1)
    logger.info("load_data complete.")


def main():
    args = parse_args()
    orbit = args.orbit
    cfg   = load_config(args.config)

    data_dir       = Path(cfg.get("paths", "data_dir"))
    processing_dir = Path(cfg.get("paths", "processing_dir"))
    project_root   = Path(cfg.get("paths", "project_root"))
    logs_dir       = Path(cfg.get("paths", "logs_dir"))

    hyp3_dir   = data_dir / "hyp3" / orbit
    mintpy_dir = processing_dir / "mintpy" / orbit
    inputs_dir = mintpy_dir / "inputs"
    template   = project_root / "config" / f"mintpy_{orbit}.cfg"
    log_path   = logs_dir / f"prep_mintpy_{orbit}.log"

    mintpy_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # --- Prerequisite checks ---
    if not hyp3_dir.exists() or not any(hyp3_dir.iterdir()):
        logger.error(
            "HyP3 products not found in %s.\n"
            "Run 02_download_data.py --download --orbit %s first.",
            hyp3_dir, orbit,
        )
        sys.exit(1)

    if not check_command("prep_hyp3.py"):
        logger.error("prep_hyp3.py not found. Install MintPy:\n  conda install -c conda-forge mintpy")
        sys.exit(1)

    if not check_command("smallbaselineApp.py"):
        logger.error("smallbaselineApp.py not found. Install MintPy:\n  conda install -c conda-forge mintpy")
        sys.exit(1)

    # --- Step 1: Extract zip files ---
    unzip_products(hyp3_dir)

    # --- Step 2: Generate .rsc sidecar files ---
    run_prep_hyp3(hyp3_dir, log_path)

    # --- Step 3: Load data into MintPy HDF5 format ---
    run_load_data(mintpy_dir, template, log_path)

    # --- Verify outputs ---
    stack_h5 = inputs_dir / "ifgramStack.h5"
    geom_h5  = inputs_dir / "geometryGeo.h5"

    if stack_h5.exists() and geom_h5.exists():
        logger.info("MintPy inputs ready:")
        logger.info("  %s (%.1f MB)", stack_h5, stack_h5.stat().st_size / 1e6)
        logger.info("  %s (%.1f MB)", geom_h5,  geom_h5.stat().st_size  / 1e6)
    else:
        logger.warning(
            "Expected HDF5 files not found in %s.\nCheck log: %s", inputs_dir, log_path
        )

    print(f"\nMintPy inputs prepared for {orbit.upper()} orbit.")
    print(f"  Stack:    {stack_h5}")
    print(f"  Geometry: {geom_h5}")
    print(f"\nNext step: python scripts/06_run_mintpy.py --orbit {orbit}")


if __name__ == "__main__":
    main()
