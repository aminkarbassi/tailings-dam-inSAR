# Brumadinho Tailings Dam — InSAR Pre-Collapse Deformation Analysis

Sentinel-1 InSAR time-series analysis of the Córrego do Feijão Mine Dam I (Brumadinho,
Brazil), replicating the methodology of Grebby et al. (2021, *Communications Earth & Environment*).

The pipeline detects and quantifies ground surface deformation in the 18 months prior to
the dam's catastrophic collapse on **25 January 2019**.

| | |
|---|---|
| **Study period** | July 2017 – January 25, 2019 |
| **Dam coordinates** | 20.1113°S, 44.1231°W |
| **SAR data** | Sentinel-1 IW (two descending tracks) + ALOS-2 PALSAR-2 ascending (optional) |
| **S1 Tracks** | Track 53 (inc. ~32°) + Track 155 (inc. ~45°) |
| **ALOS-2** | L-band ascending — enables 2D decomposition (vertical + E-W) |
| **Method** | SBAS time-series via ASF HyP3 + MintPy 1.6.2 (S1); ISCE2 stripmapStack + MintPy (ALOS-2) |
| **Key result** | ~−9 mm/yr LOS on tailings; −35 mm cumulative by Jan 2019 |

---

## Overview

This is a **cloud-based InSAR pipeline** — interferogram generation is handled by the
[ASF HyP3](https://hyp3-docs.asf.alaska.edu/) cloud service (free for registered users),
avoiding the need to install ISCE2 or process raw SLCs locally. MintPy then performs the
SBAS time-series inversion on your local machine.

> **Note on orbit coverage:** No ascending Sentinel-1 coverage exists over Brumadinho.
> The Grebby et al. (2021) paper also used two descending tracks (Track 53 + Track 155).
> Folders and scripts use `desc53` and `desc155` to make this explicit.

---

## Project Structure

```
EarthMovements/
├── environment.yml              # Conda environment (brumadinho_insar)
├── config/
│   ├── project.cfg              # Master config — paths, AOI, ALOS-2 parameters
│   ├── aoi.geojson              # Area of interest polygon
│   ├── mintpy_desc53.cfg         # MintPy config — S1 Track 53 (standard SBAS)
│   ├── mintpy_desc155.cfg        # MintPy config — S1 Track 155 (standard SBAS)
│   ├── mintpy_desc53_isbas.cfg   # MintPy config — Track 53 (ISBAS-like variant)
│   ├── mintpy_desc155_isbas.cfg  # MintPy config — Track 155 (ISBAS-like variant)
│   ├── mintpy_alos2_asc.cfg      # MintPy config — ALOS-2 ascending (ISCE2 inputs)
│   └── poi.csv                  # Points of interest (auto-created on first run)
├── scripts/
│   ├── 04_prep_mintpy.py        # Submit HyP3 jobs, download, load into MintPy (S1)
│   ├── 06_run_mintpy.py         # Run MintPy SBAS pipeline (S1 + ALOS-2)
│   ├── 07_decompose_2d.py       # 2D decomposition: ALOS-2 ASC + S1 DESC → vertical + E-W
│   ├── 08_plot_maps.py          # Cumulative displacement maps per epoch
│   ├── 09_plot_timeseries.py    # Time-series plots at points of interest
│   ├── 10_query_alos2.py        # Search ASF Vertex for ALOS-2 PALSAR-2 scenes
│   ├── 11_download_alos2.py     # Download ALOS-2 SLC data from ASF
│   ├── 12_prepare_isce2_alos2.py # Set up ISCE2 stripmapStack for ALOS-2
│   ├── 13_run_isce2_alos2.py    # Execute ISCE2 ALOS-2 processing pipeline
│   └── utils/
│       ├── geo_utils.py         # Config loading, coordinate utilities
│       └── isce2_utils.py       # ISCE2 run-file helpers (shared by S1 + ALOS-2)
├── data/
│   ├── hyp3/
│   │   ├── desc53/              # HyP3 products — S1 Track 53 (zip + extracted)
│   │   └── desc155/             # HyP3 products — S1 Track 155 (zip + extracted)
│   ├── raw/
│   │   └── alos2/asc/           # ALOS-2 SLC scene directories (LED-* + IMG-* files)
│   ├── dem/glo30/               # GLO-30 DEM tiles + merged dem.wgs84 (for ISCE2)
│   └── catalogs/
│       └── alos2_scenes.json    # ALOS-2 scene catalog from 10_query_alos2.py
├── processing/
│   ├── isce2/
│   │   └── alos2_asc/           # ISCE2 working dir — run_files/, configs/, merged/
│   └── mintpy/
│       ├── desc53/              # MintPy working dir — S1 Track 53, standard SBAS
│       ├── desc155/             # MintPy working dir — S1 Track 155, standard SBAS
│       ├── desc53_isbas/        # MintPy working dir — Track 53, ISBAS-like
│       ├── desc155_isbas/       # MintPy working dir — Track 155, ISBAS-like
│       └── alos2_asc/           # MintPy working dir — ALOS-2 ascending
├── results/
│   ├── figures/
│   │   ├── displacement_maps/        # Per-epoch LOS displacement maps (standard SBAS)
│   │   ├── displacement_maps_isbas/  # Per-epoch maps (ISBAS-like variant)
│   │   ├── decomposition/            # 2D decomposed vertical + E-W maps (if ALOS-2 run)
│   │   ├── timeseries/               # Time-series plots at POIs (standard SBAS)
│   │   └── timeseries_isbas/         # Time-series plots (ISBAS-like variant)
│   └── data/
│       ├── timeseries_points.csv     # Tabular displacement at POIs (standard SBAS)
│       ├── timeseries_points_isbas.csv
│       └── displacement_2d/          # Per-epoch vertical + E-W GeoTIFFs (if ALOS-2 run)
└── logs/                        # Processing logs
```

---

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate brumadinho_insar
```

### 2. ASF HyP3 / Earthdata credentials

Register a free account at https://urs.earthdata.nasa.gov/ (NASA Earthdata), then
configure `~/.netrc`:

```
machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
```

Or set environment variables:
```bash
export HYP3_USERNAME="your_earthdata_username"
export HYP3_PASSWORD="your_earthdata_password"
```

### 3. ERA5 credentials (atmospheric correction — Track 155 only)

Register at https://cds.climate.copernicus.eu/ and create `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

### 4. Edit project paths

Open [config/project.cfg](config/project.cfg) and update the `[paths]` section
to match your filesystem.

---

## Running the Pipeline

### Step 1 — Submit HyP3 jobs and prepare MintPy inputs

Searches for Sentinel-1 SLC pairs, submits INSAR_GAMMA jobs to ASF HyP3, waits for
completion, downloads the products (~60 GB), and loads them into MintPy's HDF5 format.

```bash
python scripts/04_prep_mintpy.py --orbit desc53   # Track 53 only
python scripts/04_prep_mintpy.py --orbit desc155  # Track 155 only
```

HyP3 cloud processing takes ~30–60 min. No local SAR processing required.

### Step 2 — Run MintPy SBAS time-series analysis

```bash
# Run both tracks in parallel (each takes ~1–2 h)
python scripts/06_run_mintpy.py --orbit desc53  > logs/mintpy_desc53.log 2>&1 &
python scripts/06_run_mintpy.py --orbit desc155 > logs/mintpy_desc155.log 2>&1 &

# Resume from a failed step
python scripts/06_run_mintpy.py --orbit desc155 --from-step correct_troposphere
```

ERA5 atmospheric correction is applied to Track 155 automatically (~40 GRIB files,
cached in `processing/mintpy/desc155/ERA5/`). Track 53 ERA5 is disabled due to a geometry
dimension mismatch in the HyP3 products for that track.

### Step 3 — Generate displacement maps

```bash
python scripts/08_plot_maps.py --orbit both --every 3 --no-decomp
```

`--every 3` plots every 3rd epoch to reduce output volume.

### Step 4 — Generate time-series plots

```bash
python scripts/09_plot_timeseries.py
```

Edit [config/poi.csv](config/poi.csv) to define your own points of interest (lat, lon, label),
then re-run.

---

## ISBAS-like Variant

Standard SBAS masks out pixels with insufficient coherence across the interferogram network,
which often excludes the dam face and tailings margins. The Grebby et al. (2021) paper used
**ISBAS** (Intermittent SBAS), which processes intermittently coherent pixels.

This pipeline includes an ISBAS-like approximation using MintPy's variance-weighted
inversion with looser coherence thresholds:

```bash
# Run ISBAS-like pipeline (results saved separately — does not overwrite standard SBAS)
python scripts/06_run_mintpy.py --orbit desc53  --variant isbas > logs/mintpy_desc53_isbas.log 2>&1 &
python scripts/06_run_mintpy.py --orbit desc155 --variant isbas > logs/mintpy_desc155_isbas.log 2>&1 &

# Plot results
python scripts/08_plot_maps.py   --variant isbas --orbit both --every 3 --no-decomp
python scripts/09_plot_timeseries.py --variant isbas
```

| Parameter | Standard SBAS | ISBAS-like |
|---|---|---|
| `network.minCoherence` | 0.7 | 0.5 |
| `networkInversion.weightFunc` | `no` (uniform) | `var` (1/variance) |
| `networkInversion.maskThreshold` | 0.4 | 0.2 |

> **Note:** The true ISBAS algorithm (Cigna & Sowter 2017) is proprietary. This variant
> is an open-source approximation using MintPy's built-in weighted inversion.

---

## ALOS-2 PALSAR-2 Extension (Optional)

Adding ALOS-2 ascending data enables **2D displacement decomposition** (vertical + east-west),
since Sentinel-1 has no ascending coverage over Brumadinho.
ALOS-2 L-band (23.6 cm) also offers better coherence on soil and tailings than C-band.

> **Data access:** ALOS-2 Level 1.1 (SLC) products require a [JAXA Research Announcement](https://www.eorc.jaxa.jp/ALOS/en/alos-2/a2_proposal.htm)
> approval in addition to a NASA Earthdata account. Searching the catalog is free.

### Step 1 — Find available ALOS-2 scenes

```bash
pip install asf-search   # one-time
python scripts/10_query_alos2.py --flight-dir ASCENDING
```

Review the output table. Note the ascending track number with the most scenes.
Update `config/project.cfg` → `[alos2] asc_relative_orbit = <track>`.

### Step 2 — Download ALOS-2 data

```bash
python scripts/11_download_alos2.py --flight-dir ASCENDING --track <track>
```

Each scene downloads as a ZIP archive and is automatically extracted.
Data lands in `data/raw/alos2/asc/`.

### Step 3 — Process with ISCE2 stripmapStack

ISCE2 must be installed with `stripmapStack.py` on the PATH, and the GLO-30 DEM
must be available in ISCE2 format (run `scripts/03_download_dem.py` if not).

```bash
# Set up ISCE2 network and run files:
python scripts/12_prepare_isce2_alos2.py

# Execute the full pipeline (~12–48 h depending on scene count and hardware):
python scripts/13_run_isce2_alos2.py
```

Resume after interruption with `--from-step <run_XX_name>`.

### Step 4 — Run MintPy on ALOS-2 interferograms

```bash
python scripts/06_run_mintpy.py --orbit alos2_asc > logs/mintpy_alos2_asc.log 2>&1
```

ERA5 atmospheric correction is applied automatically (same as S1 Track 155).

### Step 5 — 2D Decomposition

```bash
# ALOS-2 ascending + S1 Track 155 descending → vertical + E-W displacement:
python scripts/07_decompose_2d.py

# Plot decomposed maps:
python scripts/08_plot_maps.py --orbit alos2_asc --every 3   # ALOS-2 LOS maps
python scripts/08_plot_maps.py --no-decomp False             # shows 2D decomp maps too
```

### ALOS-2 vs Sentinel-1 key parameters

| Parameter | Sentinel-1 (C-band) | ALOS-2 PALSAR-2 (L-band) |
|---|---|---|
| Wavelength | 5.6 cm | 23.6 cm |
| Repeat cycle | 6–12 days (S1A+B) | 14 days |
| Coherence on tailings | Moderate | Higher |
| Max temp baseline | 180 days | 336 days (24 cycles) |
| Max perp baseline | 200 m | 1500 m |
| Preprocessor | ASF HyP3 (cloud) | ISCE2 stripmapStack (local) |

---

## Storage Requirements

| Component | Size (approx.) |
|---|---|
| HyP3 products — Track 53 (zip + extracted) | ~25 GB |
| HyP3 products — Track 155 (zip + extracted) | ~35 GB |
| MintPy working dirs (standard SBAS, both tracks) | ~10 GB |
| MintPy working dirs (ISBAS-like, both tracks) | ~10 GB |
| ALOS-2 SLC data (ascending, ~18 months) | ~30–50 GB |
| ISCE2 ALOS-2 working dir | ~20 GB |
| MintPy ALOS-2 working dir | ~5 GB |
| Final results (figures + CSV) | < 1 GB |
| **Total (S1 only)** | **~80–90 GB** |
| **Total (S1 + ALOS-2)** | **~150–170 GB** |

---

## Key Configuration Notes

### Reference point
Set to stable bedrock NE of the dam (`−20.06°S, −44.08°W`) in both MintPy configs.
This is defined in `config/mintpy_desc53.cfg` and `config/mintpy_desc155.cfg` under
`mintpy.reference.lalo`.

### Two descending Sentinel-1 tracks
Both Track 53 and Track 155 are descending with the same satellite heading (~−104°).
With S1 data alone, 2D decomposition into vertical + east-west is **not** applicable;
the two LOS time-series are analysed independently, consistent with Grebby et al. (2021).

2D decomposition becomes possible with the **optional ALOS-2 extension** (see above),
which provides an ascending geometry. See `config/mintpy_alos2_asc.cfg` and
`scripts/07_decompose_2d.py`.

### Orbit data gap
Track 53 has a gap in the Sentinel-1 archive around December 2017 – January 2018.
This is a known characteristic of the archive and does not affect the analysis.

---

## Results

After running the full pipeline, key outputs are:

- `results/figures/displacement_maps/` — one PNG per epoch per track showing cumulative LOS displacement
- `results/figures/timeseries/` — time-series plots at the dam crest, tailings surface, and stable reference
- `results/data/timeseries_points.csv` — tabular data for quantitative comparison

Expected signals consistent with Grebby et al. (2021):

| Location | LOS velocity | Cumulative (Jul 2017 – Jan 2019) |
|---|---|---|
| Upper tailings (Track 155, ERA5-corrected) | ~−9 mm/yr | ~−35 mm |
| Lower tailings (Track 155) | ~−9 mm/yr | ~−33 mm |
| Dam crest | masked (low coherence) | — |

Acceleration visible in November 2018 – January 2019, ~8 weeks before collapse.

---

## Reference

Grebby, S., Sowter, A., Gluyas, J., Toll, D., Gee, D., Athab, A. & Girindran, R. (2021).
Advanced analysis of satellite data reveals ground deformation precursors to the Brumadinho
Tailings Dam collapse. *Communications Earth & Environment*, 2(1).
https://doi.org/10.1038/s43247-020-00079-2

> A 2025 "Matters Arising" response questions whether the deformation pattern was a precursor
> to the liquefaction failure: https://doi.org/10.1038/s43247-025-02067-w
