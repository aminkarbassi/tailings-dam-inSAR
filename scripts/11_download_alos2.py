#!/usr/bin/env python3
"""
11_download_alos2.py — Download ALOS-2 PALSAR-2 data from ASF Vertex
======================================================================
Downloads ALOS-2 Level 1.1 (SLC) products from the Alaska SAR Facility (ASF)
Vertex data archive. Authentication uses your NASA Earthdata credentials from
~/.netrc (same account used for ASF HyP3).

DATA ACCESS REQUIREMENT:
    ALOS-2 Level 1.1 (SLC) products require JAXA Research Announcement (RA)
    approval in addition to an Earthdata account. Apply at:
        https://www.eorc.jaxa.jp/ALOS/en/alos-2/a2_proposal.htm
    Without RA approval, downloads will fail with a 403 Forbidden error.
    Run 10_query_alos2.py first to find available scenes.

ALOS-2 data format from ASF:
    Each scene downloads as a ZIP archive containing:
        - IMG-HH-ALPSRP*  : HH polarization SLC image
        - IMG-HV-ALPSRP*  : HV polarization SLC image (if FBD mode)
        - LED-ALPSRP*     : Leader file (orbit + metadata)
        - TRL-ALPSRP*     : Trailer file
        - VOL-ALPSRP*     : Volume directory
    ISCE2 reads the LED file to identify the dataset.

Usage:
    # Download all ascending scenes from the catalog:
    python scripts/11_download_alos2.py

    # Download only a specific track:
    python scripts/11_download_alos2.py --track 127

    # Download descending too (for QC comparison):
    python scripts/11_download_alos2.py --flight-dir BOTH

    # Use a specific catalog file:
    python scripts/11_download_alos2.py --catalog data/catalogs/alos2_scenes.json

Requirements:
    pip install asf-search
    ~/.netrc must contain NASA Earthdata credentials:
        machine urs.earthdata.nasa.gov login <user> password <pass>
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ALOS-2 PALSAR-2 SLC data from ASF Vertex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        default="data/catalogs/alos2_scenes.json",
        help="GeoJSON scene catalog from 10_query_alos2.py "
             "(default: data/catalogs/alos2_scenes.json)",
    )
    parser.add_argument(
        "--flight-dir",
        default="ASCENDING",
        choices=["ASCENDING", "DESCENDING", "BOTH"],
        help="Flight direction to download (default: ASCENDING)",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help="Download only scenes from this relative orbit / path number",
    )
    parser.add_argument(
        "--beam-mode",
        default=None,
        help="Filter by beam mode (e.g. FBD, FBS). Default: all modes.",
    )
    parser.add_argument("--config", default="config/project.cfg")
    parser.add_argument(
        "--nproc",
        type=int,
        default=2,
        help="Number of parallel downloads (default: 2)",
    )
    return parser.parse_args()


def check_asf_search():
    try:
        import asf_search  # noqa: F401
        return True
    except ImportError:
        logger.error("asf_search not installed.  Run: pip install asf-search")
        return False


def check_netrc() -> bool:
    """Verify that ~/.netrc contains Earthdata credentials."""
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        logger.error(
            "~/.netrc not found.\n"
            "Create it with:\n"
            "  echo 'machine urs.earthdata.nasa.gov login <user> password <pass>' >> ~/.netrc\n"
            "  chmod 600 ~/.netrc"
        )
        return False
    content = netrc_path.read_text()
    if "urs.earthdata.nasa.gov" not in content:
        logger.error(
            "NASA Earthdata credentials not found in ~/.netrc.\n"
            "Add: machine urs.earthdata.nasa.gov login <user> password <pass>"
        )
        return False
    return True


def load_catalog(catalog_path: Path) -> list:
    """Load scene GeoJSON from 10_query_alos2.py output."""
    if not catalog_path.exists():
        logger.error(
            "Catalog not found: %s\n"
            "Run 10_query_alos2.py first to search for available scenes.",
            catalog_path,
        )
        sys.exit(1)

    with open(catalog_path) as f:
        fc = json.load(f)

    return fc.get("features", [])


def filter_scenes(features: list, flight_dir: str, track: int = None, beam_mode: str = None) -> list:
    """Filter catalog features by flight direction, track, and beam mode."""
    out = []
    for feat in features:
        props = feat.get("properties", {})

        fd = props.get("flightDirection", "").upper()
        if flight_dir != "BOTH" and fd != flight_dir.upper():
            continue

        if track is not None:
            if str(props.get("pathNumber", "")) != str(track):
                continue

        if beam_mode is not None:
            if props.get("beamMode", "").upper() != beam_mode.upper():
                continue

        out.append(feat)

    return out


def reconstruct_asf_results(features: list):
    """
    Reconstruct asf_search SearchResult objects from the stored GeoJSON features.
    asf_search can deserialise its own GeoJSON format.
    """
    import asf_search as asf

    # asf_search 6.x+ can load results from GeoJSON
    try:
        results = asf.ASFSearchResults(
            [asf.ASFProduct(f) for f in features]
        )
    except Exception:
        # Older API fallback: use the raw GeoJSON URLs
        results = features

    return results


def download_scenes(features: list, out_dir: Path, n_proc: int = 2):
    """Download ALOS-2 scenes to out_dir using asf_search's batch download."""
    import asf_search as asf

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build authenticated session from ~/.netrc
    try:
        session = asf.ASFSession().auth_with_creds(
            username=None,   # reads ~/.netrc automatically
            password=None,
        )
    except Exception as e:
        logger.error(
            "Authentication failed: %s\n"
            "Ensure ~/.netrc has valid Earthdata credentials.", e
        )
        sys.exit(1)

    # Reconstruct asf_search result objects
    try:
        results = asf.ASFSearchResults(
            [asf.ASFProduct(f) for f in features]
        )
    except Exception as e:
        logger.error("Failed to reconstruct search results from catalog: %s", e)
        sys.exit(1)

    # Filter already-downloaded scenes (check for extracted directory or zip)
    to_download = []
    for r in results:
        fname  = r.properties.get("fileName", "")
        zpath  = out_dir / fname
        # Check both zip and extracted directory
        stem   = fname.replace(".zip", "")
        expath = out_dir / stem
        if zpath.exists() or expath.exists():
            logger.info("Already downloaded: %s", fname)
        else:
            to_download.append(r)

    if not to_download:
        logger.info("All %d scenes already downloaded.", len(results))
        return

    logger.info("Downloading %d scenes to %s ...", len(to_download), out_dir)
    logger.info("(JAXA RA approval required — will fail with 403 if not approved)")

    try:
        asf.ASFSearchResults(to_download).download(
            path=str(out_dir),
            session=session,
            processes=n_proc,
        )
        logger.info("Download complete.")
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            logger.error(
                "HTTP 403 Forbidden — ALOS-2 SLC data requires JAXA RA approval.\n"
                "Apply at: https://www.eorc.jaxa.jp/ALOS/en/alos-2/a2_proposal.htm"
            )
        else:
            logger.error("Download failed: %s", e)
        sys.exit(1)


def extract_zips(out_dir: Path):
    """Unzip downloaded ALOS-2 archives in-place."""
    import zipfile

    zips = sorted(out_dir.glob("*.zip"))
    if not zips:
        return

    logger.info("Extracting %d ZIP archives ...", len(zips))
    for zpath in zips:
        stem = zpath.stem
        if (out_dir / stem).exists():
            logger.info("Already extracted: %s", stem)
            continue

        logger.info("Extracting: %s", zpath.name)
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(out_dir)

        # Verify extraction produced something
        if not (out_dir / stem).exists():
            logger.warning(
                "Expected directory %s not found after extraction. "
                "Check the ZIP contents.", stem
            )

    logger.info("Extraction complete.")


def print_summary(features: list, out_dir: Path):
    """Print a summary of what was downloaded."""
    from collections import defaultdict

    by_group = defaultdict(list)
    for f in features:
        p = f.get("properties", {})
        key = (p.get("flightDirection", "?"), str(p.get("pathNumber", "?")), p.get("beamMode", "?"))
        by_group[key].append(p.get("startTime", "")[:10])

    print(f"\n{'='*60}")
    print(f"  ALOS-2 download summary — {len(features)} scenes")
    print(f"{'='*60}")
    for (fd, track, beam), dates in sorted(by_group.items()):
        print(f"  {fd:<15} track {track:<6} {beam:<6} — {len(dates)} scenes")
    print()
    print(f"  Output directory: {out_dir}")
    print()
    print("Next step:")
    print("  Verify the extracted scene directories contain LED-* files.")
    print("  Then: python scripts/12_prepare_isce2_alos2.py")


def main():
    args = parse_args()

    if not check_asf_search():
        sys.exit(1)

    if not check_netrc():
        sys.exit(1)

    cfg        = load_config(args.config)
    data_dir   = Path(cfg.get("paths", "data_dir"))
    out_dir    = data_dir / "raw" / "alos2" / args.flight_dir.lower().replace("both", "asc")

    catalog_path = Path(args.catalog)
    features     = load_catalog(catalog_path)

    if not features:
        logger.error("Catalog is empty. Re-run 10_query_alos2.py.")
        sys.exit(1)

    # Filter scenes
    filtered = filter_scenes(
        features,
        flight_dir=args.flight_dir,
        track=args.track,
        beam_mode=args.beam_mode,
    )

    if not filtered:
        logger.error(
            "No scenes match the filters (flight_dir=%s, track=%s, beam_mode=%s).",
            args.flight_dir, args.track, args.beam_mode,
        )
        logger.error(
            "Check the catalog or re-run 10_query_alos2.py with --flight-dir BOTH."
        )
        sys.exit(1)

    logger.info(
        "Selected %d / %d scenes (flight_dir=%s, track=%s, beam=%s).",
        len(filtered), len(features),
        args.flight_dir, args.track or "all", args.beam_mode or "all",
    )

    download_scenes(filtered, out_dir, n_proc=args.nproc)
    extract_zips(out_dir)
    print_summary(filtered, out_dir)


if __name__ == "__main__":
    main()
