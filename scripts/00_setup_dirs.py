#!/usr/bin/env python3
"""
00_setup_dirs.py — Initialize project directory structure
==========================================================
Creates all working directories required by the Brumadinho InSAR pipeline.
Run this once before executing any other script.

Usage:
    python scripts/00_setup_dirs.py [--config config/project.cfg]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow imports from scripts/utils/
sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Create project directory tree")
    parser.add_argument(
        "--config",
        default="config/project.cfg",
        help="Path to project.cfg (default: config/project.cfg)",
    )
    return parser.parse_args()


def build_directory_tree(cfg) -> dict:
    """Return a dict of {label: Path} for all directories to create."""
    data_dir       = Path(cfg.get("paths", "data_dir"))
    processing_dir = Path(cfg.get("paths", "processing_dir"))
    results_dir    = Path(cfg.get("paths", "results_dir"))
    logs_dir       = Path(cfg.get("paths", "logs_dir"))

    dirs = {
        # --- Raw scene catalogues (no SLC downloads needed with HyP3) ---
        "raw":                data_dir / "raw",
        # --- HyP3 processed interferogram products ---
        "hyp3_asc":           data_dir / "hyp3" / "asc",
        "hyp3_desc":          data_dir / "hyp3" / "desc",
        # --- GACOS atmospheric corrections (optional, for final results) ---
        "gacos_asc":          data_dir / "gacos" / "asc",
        "gacos_desc":         data_dir / "gacos" / "desc",
        # --- MintPy (ascending) ---
        "mintpy_asc":         processing_dir / "mintpy" / "asc",
        "mintpy_asc_inputs":  processing_dir / "mintpy" / "asc" / "inputs",
        "mintpy_asc_pic":     processing_dir / "mintpy" / "asc" / "pic",
        # --- MintPy (descending) ---
        "mintpy_desc":        processing_dir / "mintpy" / "desc",
        "mintpy_desc_inputs": processing_dir / "mintpy" / "desc" / "inputs",
        "mintpy_desc_pic":    processing_dir / "mintpy" / "desc" / "pic",
        # --- Results ---
        "figures_maps":       results_dir / "figures" / "displacement_maps",
        "figures_ts":         results_dir / "figures" / "timeseries",
        "figures_decomp":     results_dir / "figures" / "decomposition",
        "results_data":       results_dir / "data",
        "results_2d":         results_dir / "data" / "displacement_2d",
        # --- Logs ---
        "logs":               logs_dir,
    }
    return dirs


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    cfg = load_config(cfg_path)
    dirs = build_directory_tree(cfg)

    created, skipped = 0, 0
    for label, path in dirs.items():
        if path.exists():
            logger.debug("Already exists: %s", path)
            skipped += 1
        else:
            path.mkdir(parents=True, exist_ok=True)
            logger.info("Created: %s", path)
            created += 1

    # Write a manifest so other scripts can discover paths programmatically
    manifest = {label: str(path) for label, path in dirs.items()}
    manifest_path = Path(cfg.get("paths", "project_root")) / "workspace_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to: %s", manifest_path)

    print(f"\nDone. Created {created} directories, {skipped} already existed.")
    print(f"Manifest: {manifest_path}")
    print("\nNext step: python scripts/01_query_data.py")


if __name__ == "__main__":
    main()
