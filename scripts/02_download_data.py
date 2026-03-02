#!/usr/bin/env python3
"""
02_download_data.py — Submit HyP3 InSAR jobs and download results
==================================================================
Submits Sentinel-1 interferogram pairs (from 01_query_data.py) to the
ASF HyP3 cloud processing service, waits for completion, then downloads
the processed interferogram products.

HyP3 returns per-pair directories containing:
    *_unw_phase.tif     — unwrapped interferogram (radians)
    *_corr.tif          — coherence
    *_dem.tif           — DEM used in processing
    *_lv_theta.tif      — incidence angle
    *_lv_phi.tif        — azimuth angle
    *_water_mask.tif    — water body mask

Total download: ~100-300 MB per pair (vs 4 GB raw SLC).

Usage:
    python scripts/02_download_data.py --submit [--config config/project.cfg]
    python scripts/02_download_data.py --download           # download completed jobs
    python scripts/02_download_data.py --status             # check job status

Requirements:
    pip install hyp3_sdk
    NASA Earthdata credentials in env vars: EARTHDATA_LOGIN, EARTHDATA_PASSWORD
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JOB_NAMES = {
    "asc":  "brumadinho_asc",
    "desc": "brumadinho_desc",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Submit HyP3 jobs and download results")
    parser.add_argument("--config",   default="config/project.cfg")
    parser.add_argument("--orbit",    choices=["asc", "desc", "both"], default="both")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit",   action="store_true", help="Submit new jobs to HyP3")
    group.add_argument("--download", action="store_true", help="Download completed jobs")
    group.add_argument("--status",   action="store_true", help="Show current job status")
    return parser.parse_args()


def get_hyp3():
    """Authenticate with HyP3 using NASA Earthdata credentials.

    Credential lookup order:
      1. EARTHDATA_LOGIN / EARTHDATA_PASSWORD environment variables
      2. ~/.netrc entry for urs.earthdata.nasa.gov (hyp3_sdk uses this automatically)
    """
    try:
        import hyp3_sdk
    except ImportError:
        logger.error("hyp3_sdk not installed. Run: pip install hyp3_sdk")
        sys.exit(1)

    user = os.environ.get("EARTHDATA_LOGIN")
    pwd  = os.environ.get("EARTHDATA_PASSWORD")

    if user and pwd:
        hyp3 = hyp3_sdk.HyP3(username=user, password=pwd)
        logger.info("Authenticated with HyP3 via env vars as: %s", user)
    else:
        # hyp3_sdk reads ~/.netrc for urs.earthdata.nasa.gov automatically
        try:
            hyp3 = hyp3_sdk.HyP3()
            logger.info("Authenticated with HyP3 via ~/.netrc")
        except Exception as e:
            logger.error(
                "NASA Earthdata credentials not found.\n"
                "Run:  python scripts/setup_credentials.py\n"
                "  or: export EARTHDATA_LOGIN=... EARTHDATA_PASSWORD=...\n"
                "Register at: https://urs.earthdata.nasa.gov/\n"
                "Error: %s", e
            )
            sys.exit(1)

    return hyp3


def submit_orbit(hyp3, pairs_path: Path, job_name: str, orbit: str):
    """Submit all pairs for one orbit as HyP3 INSAR_GAMMA jobs."""
    if not pairs_path.exists():
        logger.error("Pairs file not found: %s\nRun 01_query_data.py first.", pairs_path)
        sys.exit(1)

    with open(pairs_path) as f:
        pairs = json.load(f)

    logger.info("Preparing %d HyP3 InSAR jobs for %s orbit...", len(pairs), orbit.upper())

    prepared = []
    for ref_name, sec_name in pairs:
        prepared.append(hyp3.prepare_insar_job(
            granule1             = ref_name,
            granule2             = sec_name,
            name                 = job_name,
            looks                = "20x4",   # 80 m resolution, standard for SBAS
            include_dem          = True,
            include_look_vectors = True,
            apply_water_mask     = True,
        ))

    # Submit in batches of 200 (HyP3 API limit per request)
    batch_size = 200
    all_jobs   = []
    for i in range(0, len(prepared), batch_size):
        chunk = prepared[i:i + batch_size]
        logger.info("Submitting batch %d/%d (%d jobs)...",
                    i // batch_size + 1, (len(prepared) - 1) // batch_size + 1, len(chunk))
        batch = hyp3.submit_prepared_jobs(chunk)
        all_jobs.extend(batch)
        time.sleep(2)  # brief pause between batches

    logger.info("Submitted %d jobs for %s orbit. Job name: %s", len(all_jobs), orbit.upper(), job_name)
    return all_jobs


def download_orbit(hyp3, job_name: str, out_dir: Path, orbit: str):
    """Wait for all jobs with job_name to finish, then download."""
    logger.info("Fetching jobs with name '%s'...", job_name)
    batch = hyp3.find_jobs(name=job_name)
    jobs  = list(batch)

    if not jobs:
        logger.warning("No jobs found with name '%s'. Submit first.", job_name)
        return

    pending  = [j for j in jobs if j.status_code in ("PENDING", "RUNNING")]
    complete = [j for j in jobs if j.status_code == "SUCCEEDED"]
    failed   = [j for j in jobs if j.status_code == "FAILED"]

    logger.info(
        "%s orbit: %d total | %d complete | %d pending/running | %d failed",
        orbit.upper(), len(jobs), len(complete), len(pending), len(failed),
    )

    if pending:
        logger.info("Waiting for %d jobs to complete (this may take hours)...", len(pending))
        batch = hyp3.watch(batch)

    # Download all succeeded jobs
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading to: %s", out_dir)
    succeeded_batch = batch.filter_jobs(succeeded=True, pending=False, running=False)
    downloaded = succeeded_batch.download_files(location=out_dir)
    logger.info("Download complete. %d files in %s", len(downloaded), out_dir)


def show_status(hyp3, job_names: list):
    for name in job_names:
        jobs = list(hyp3.find_jobs(name=name))
        if not jobs:
            print(f"  {name}: no jobs found")
            continue
        counts = {}
        for j in jobs:
            counts[j.status_code] = counts.get(j.status_code, 0) + 1
        status_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(f"  {name}: {len(jobs)} total — {status_str}")


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    data_dir = Path(cfg.get("paths", "data_dir"))
    raw_dir  = data_dir / "raw"

    orbits = ["asc", "desc"] if args.orbit == "both" else [args.orbit]

    hyp3 = get_hyp3()

    if args.status:
        print("\nHyP3 job status:")
        show_status(hyp3, [JOB_NAMES[o] for o in orbits])
        return

    if args.submit:
        for orbit in orbits:
            pairs_path = raw_dir / f"pairs_{orbit}.json"
            submit_orbit(hyp3, pairs_path, JOB_NAMES[orbit], orbit)
        print("\nJobs submitted. To check status:")
        print("  python scripts/02_download_data.py --status")
        print("\nOnce all jobs show SUCCEEDED:")
        print("  python scripts/02_download_data.py --download")

    elif args.download:
        for orbit in orbits:
            out_dir = data_dir / "hyp3" / orbit
            download_orbit(hyp3, JOB_NAMES[orbit], out_dir, orbit)

        print("\nDownload complete.")
        print("Next step: python scripts/04_prep_mintpy.py --orbit asc")
        print("           python scripts/04_prep_mintpy.py --orbit desc")


if __name__ == "__main__":
    main()
