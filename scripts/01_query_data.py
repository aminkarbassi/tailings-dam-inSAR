#!/usr/bin/env python3
"""
01_query_data.py — Search ASF for Sentinel-1 scenes and define interferogram pairs
===================================================================================
Uses NASA's asf_search library to find all Sentinel-1 IW SLC scenes over
the Brumadinho AOI for both ascending and descending orbits.

Then builds a sequential-3 SBAS interferogram network (each scene paired with
its 3 nearest temporal neighbours). These pairs are submitted to ASF HyP3
in the next step — no raw SLC download required.

No login needed for scene search. HyP3 submission requires a NASA Earthdata account.

Outputs:
    data/raw/scene_catalog_asc.json  — ascending scene metadata
    data/raw/scene_catalog_desc.json — descending scene metadata
    data/raw/pairs_asc.json          — (ref, sec) pair list (ascending)
    data/raw/pairs_desc.json         — (ref, sec) pair list (descending)

Usage:
    python scripts/01_query_data.py [--config config/project.cfg]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.geo_utils import load_config, get_aoi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Search ASF for Sentinel-1 scenes")
    parser.add_argument("--config", default="config/project.cfg")
    return parser.parse_args()


def search_scenes(start, end, south, north, west, east, orbit_direction, relative_orbit):
    import asf_search as asf
    logger.info("Searching ASF: %s orbit, track %d, %s to %s",
                orbit_direction, relative_orbit, start, end)
    aoi_wkt = (f"POLYGON(({west} {south},{east} {south},"
               f"{east} {north},{west} {north},{west} {south}))")
    results = asf.search(
        platform        = asf.PLATFORM.SENTINEL1,
        processingLevel = asf.PRODUCT_TYPE.SLC,
        beamMode        = asf.BEAMMODE.IW,
        flightDirection = orbit_direction,
        relativeOrbit   = relative_orbit,
        start           = start,
        end             = end,
        intersectsWith  = aoi_wkt,
    )
    results.sort(key=lambda r: r.properties["startTime"])
    logger.info("Found %d scenes", len(results))
    return results


def build_sbas_pairs(scenes, num_connections=3):
    n = len(scenes)
    pairs = []
    seen  = set()
    for i in range(n):
        for j in range(i + 1, min(i + num_connections + 1, n)):
            ref = scenes[i].properties["sceneName"]
            sec = scenes[j].properties["sceneName"]
            key = (ref, sec)
            if key not in seen:
                pairs.append(list(key))
                seen.add(key)
    logger.info("SBAS network: %d scenes → %d pairs", n, len(pairs))
    return pairs


def check_date_gaps(scenes, max_gap_days=24):
    dates = [datetime.fromisoformat(s.properties["startTime"][:10]) for s in scenes]
    gaps = []
    for i in range(1, len(dates)):
        delta = (dates[i] - dates[i-1]).days
        if delta > max_gap_days:
            gaps.append((dates[i-1].date(), dates[i].date(), delta))
    return gaps


def scenes_to_dict(scenes):
    return [{"sceneName":  s.properties["sceneName"],
             "startTime":  s.properties["startTime"],
             "pathNumber": s.properties.get("pathNumber"),
             "platform":   s.properties.get("platform"),
             "url":        s.properties.get("url", "")} for s in scenes]


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    south, north, west, east = get_aoi(cfg)
    start   = cfg.get("dates", "start_date")
    end     = cfg.get("dates", "end_date")
    asc_t    = cfg.getint("sentinel1", "asc_relative_orbit")
    desc_t   = cfg.getint("sentinel1", "desc_relative_orbit")
    asc_dir  = cfg.get("sentinel1", "asc_flight_direction",  fallback="ASCENDING").strip().upper()
    desc_dir = cfg.get("sentinel1", "desc_flight_direction", fallback="DESCENDING").strip().upper()
    n_conn   = cfg.getint("isce2", "num_connections")

    raw_dir = Path(cfg.get("paths", "data_dir")) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    for direction, rel_orbit, tag in [(asc_dir, asc_t, "asc"),
                                       (desc_dir, desc_t, "desc")]:
        scenes = search_scenes(start, end, south, north, west, east,
                               direction, rel_orbit)
        pairs  = build_sbas_pairs(scenes, num_connections=n_conn)
        gaps   = check_date_gaps(scenes)
        total_pairs += len(pairs)

        # Print summary
        dates = [s.properties["startTime"][:10] for s in scenes]
        print(f"\n{'='*60}")
        print(f"  {direction} (track {rel_orbit})")
        print(f"{'='*60}")
        print(f"  Scenes : {len(scenes)}  ({dates[0] if dates else 'none'} → {dates[-1] if dates else 'none'})")
        print(f"  Pairs  : {len(pairs)}")
        if gaps:
            print(f"  GAPS   : {len(gaps)} gap(s) > 24 days detected")
            for b, a, d in gaps:
                print(f"           {b} → {a}  ({d} days)")

        with open(raw_dir / f"scene_catalog_{tag}.json", "w") as f:
            json.dump(scenes_to_dict(scenes), f, indent=2)
        with open(raw_dir / f"pairs_{tag}.json", "w") as f:
            json.dump(pairs, f, indent=2)

    print(f"\nTotal HyP3 jobs to submit: {total_pairs}")
    print(f"Pair files written to: {raw_dir}")
    print("\nNext step: python scripts/02_download_data.py --submit")


if __name__ == "__main__":
    main()
