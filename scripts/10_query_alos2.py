#!/usr/bin/env python3
"""
10_query_alos2.py — Search ASF Vertex for ALOS-2 PALSAR-2 scenes over Brumadinho
==================================================================================
Queries the ASF Vertex catalog for available ALOS-2 PALSAR-2 SLC products over
the study area and writes a GeoJSON scene catalog.

Key output: a summary table showing which ascending/descending tracks are available
and how many scenes each has. Use this to choose the best ascending track for 2D
decomposition with the existing Sentinel-1 descending data.

ALOS-2 data access requirements:
    Access to ALOS-2 SLC (Level 1.1) products requires a JAXA Research Announcement
    (RA) approval IN ADDITION to a NASA Earthdata account.
    RA registration: https://www.eorc.jaxa.jp/ALOS/en/alos-2/a2_proposal.htm
    ASF Earthdata:   https://urs.earthdata.nasa.gov/

    Searching is free and requires no approval; only downloading requires RA access.

Usage:
    # Search ascending passes (for 2D decomposition with S1 descending):
    python scripts/10_query_alos2.py --flight-dir ASCENDING

    # Search all passes (to see full availability):
    python scripts/10_query_alos2.py --flight-dir BOTH

    # Filter by beam mode (FBD = Fine Beam Double, recommended for InSAR):
    python scripts/10_query_alos2.py --flight-dir ASCENDING --beam-mode FBD

Requirements:
    pip install asf-search
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config, get_aoi, bbox_to_wkt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search ASF Vertex for ALOS-2 PALSAR-2 scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ascending passes (for 2D decomp with S1 descending):
  python scripts/10_query_alos2.py --flight-dir ASCENDING

  # All passes:
  python scripts/10_query_alos2.py --flight-dir BOTH

  # Filter to fine-beam mode only:
  python scripts/10_query_alos2.py --flight-dir ASCENDING --beam-mode FBD
""",
    )
    parser.add_argument(
        "--flight-dir",
        default="ASCENDING",
        choices=["ASCENDING", "DESCENDING", "BOTH"],
        help="Flight direction filter (default: ASCENDING — for 2D decomp with S1 desc)",
    )
    parser.add_argument(
        "--beam-mode",
        default=None,
        help="Beam mode filter: FBD (70 km, dual-pol), FBS (70 km, single-pol), "
             "WBD (350 km). Default: all modes.",
    )
    parser.add_argument("--config",  default="config/project.cfg")
    parser.add_argument(
        "--output",
        default="data/catalogs/alos2_scenes.json",
        help="Output GeoJSON catalog path (default: data/catalogs/alos2_scenes.json)",
    )
    return parser.parse_args()


def check_asf_search():
    """Verify asf_search is installed."""
    try:
        import asf_search  # noqa: F401
        return True
    except ImportError:
        logger.error(
            "asf_search not installed. Install it with:\n"
            "  pip install asf-search\n"
            "or:\n"
            "  conda install -c conda-forge asf_search"
        )
        return False


def search_alos2(
    wkt: str,
    start_date: str,
    end_date: str,
    flight_directions: list,
    beam_mode: str = None,
) -> list:
    """
    Search ASF Vertex for ALOS-2 PALSAR-2 SLC scenes.

    Args:
        wkt:              WKT POLYGON string for the AOI.
        start_date:       ISO date string (YYYY-MM-DD).
        end_date:         ISO date string (YYYY-MM-DD).
        flight_directions: List of asf_search.FLIGHT_DIRECTION constants.
        beam_mode:        Optional beam mode filter string.

    Returns:
        List of asf_search SearchResult objects.
    """
    import asf_search as asf

    all_results = []

    for fdir in flight_directions:
        logger.info(
            "Searching ALOS-2 %s scenes from %s to %s ...",
            fdir.value if hasattr(fdir, "value") else fdir,
            start_date,
            end_date,
        )

        kwargs = dict(
            dataset=[asf.DATASET.ALOS_2],
            processingLevel=[asf.PRODUCT_TYPE.L1_1],   # SLC equivalent
            intersectsWith=wkt,
            start=start_date,
            end=end_date,
            flightDirection=fdir,
        )

        if beam_mode:
            kwargs["beamMode"] = [beam_mode]

        try:
            results = asf.search(**kwargs)
            logger.info(
                "  Found %d scenes (%s).",
                len(results),
                fdir.value if hasattr(fdir, "value") else fdir,
            )
            all_results.extend(results)
        except Exception as e:
            logger.error("Search failed: %s", e)
            logger.error(
                "If this is an authentication error, ensure ~/.netrc has your "
                "NASA Earthdata credentials."
            )
            raise

    return all_results


def print_summary(results: list):
    """Print a formatted table of scenes grouped by track, direction, and beam mode."""
    if not results:
        return

    by_group = defaultdict(list)
    for r in results:
        p = r.properties
        key = (
            p.get("flightDirection", "?"),
            str(p.get("pathNumber", "?")),
            p.get("beamMode", "?"),
        )
        by_group[key].append(p.get("startTime", "")[:10])

    print(f"\n{'='*72}")
    print(f"  ALOS-2 PALSAR-2 scene catalog  —  {len(results)} total scenes")
    print(f"{'='*72}")
    print(f"  {'Direction':<15}  {'Track':>6}  {'Beam':<8}  {'Scenes':>7}  Date range")
    print(f"  {'-'*65}")

    for (fdir, track, beam), dates in sorted(by_group.items(), key=lambda x: (x[0][0], int(x[0][1]) if x[0][1].isdigit() else 0)):
        dates_sorted = sorted(set(dates))
        print(
            f"  {fdir:<15}  {track:>6}  {beam:<8}  {len(dates_sorted):>7}  "
            f"{dates_sorted[0]} → {dates_sorted[-1]}"
        )

    print(f"{'='*72}")
    print()
    print("ALOS-2 data access note:")
    print("  Searching is free. Downloading Level 1.1 (SLC) data requires")
    print("  JAXA Research Announcement (RA) approval:")
    print("  https://www.eorc.jaxa.jp/ALOS/en/alos-2/a2_proposal.htm")
    print()
    print("Recommended ascending track selection criteria:")
    print("  1. Choose the ascending track with the most scenes covering 2017–2019.")
    print("  2. Prefer FBD (Fine Beam Double) mode — dual-pol improves InSAR quality.")
    print("  3. At least ~10 scenes needed for a useful time-series.")
    print()


def print_next_steps(output_path: Path, results: list):
    """Print instructions for after the catalog has been saved."""
    # Find the best ascending candidate
    from collections import defaultdict
    asc_groups = defaultdict(list)
    for r in results:
        p = r.properties
        if p.get("flightDirection", "").upper() == "ASCENDING":
            key = (str(p.get("pathNumber", "?")), p.get("beamMode", "?"))
            asc_groups[key].append(p.get("startTime", "")[:10])

    print("Next steps:")
    if asc_groups:
        best = max(asc_groups.items(), key=lambda x: len(x[1]))
        best_track, best_beam = best[0]
        print(f"  1. Update config/project.cfg → [alos2] asc_relative_orbit = {best_track}")
        print(f"     (track {best_track}, beam {best_beam} has the most scenes: {len(best[1])})")
    else:
        print("  1. No ascending scenes found — check dates and AOI, or try --flight-dir BOTH.")
    print(f"  2. Download: python scripts/11_download_alos2.py --flight-dir ASCENDING")
    print(f"  3. Catalog saved: {output_path}")


def main():
    args = parse_args()

    if not check_asf_search():
        sys.exit(1)

    import asf_search as asf

    cfg = load_config(args.config)
    south, north, west, east = get_aoi(cfg)
    start_date = cfg.get("dates", "start_date")
    end_date   = cfg.get("dates", "end_date")
    wkt = bbox_to_wkt(south, north, west, east)

    # Resolve flight direction(s)
    if args.flight_dir == "BOTH":
        flight_dirs = [asf.FLIGHT_DIRECTION.ASCENDING, asf.FLIGHT_DIRECTION.DESCENDING]
    else:
        flight_dirs = [getattr(asf.FLIGHT_DIRECTION, args.flight_dir)]

    results = search_alos2(
        wkt=wkt,
        start_date=start_date,
        end_date=end_date,
        flight_directions=flight_dirs,
        beam_mode=args.beam_mode,
    )

    if not results:
        logger.error(
            "No ALOS-2 scenes found for this area (%s to %s).", start_date, end_date
        )
        logger.error("Try --flight-dir BOTH or remove --beam-mode filter.")
        sys.exit(1)

    print_summary(results)

    # Save GeoJSON catalog
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = [r.geojson() for r in results]
    catalog  = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(catalog, f, indent=2)
    logger.info("Catalog saved: %s  (%d scenes)", out_path, len(results))

    print_next_steps(out_path, results)


if __name__ == "__main__":
    main()
