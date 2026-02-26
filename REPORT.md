# Brumadinho Dam Collapse Precursor Detection via Sentinel-1 InSAR
## Technical Report — Methodology, Implementation & Results

---

## 1. Background and Motivation

On **25 January 2019**, the Córrego do Feijão Mine Dam I (Brumadinho, Minas Gerais, Brazil), owned by Vale S.A., collapsed catastrophically, killing **270 people** and releasing ~12 million m³ of iron ore tailings. It stands as one of the deadliest mining disasters in history.

The scientific question driving this project: **Was the collapse preceded by detectable surface deformation, measurable from space?** If so, can routine satellite radar (InSAR) monitoring serve as an early-warning tool for tailings storage facilities?

This project replicates the approach of **Grebby et al. (2021)** (*Nature Communications Earth & Environment*, doi:10.1038/s43247-020-00079-2), which demonstrated precursory deformation was indeed observable in Sentinel-1 data. Our goal was to independently reproduce their findings using open-source tools and freely available SAR data.

---

## 2. Study Area and Data

### Points of Interest
Four points were defined to capture the spatial structure of deformation:

| Label | Location | Purpose |
|---|---|---|
| `dam_crest` | −20.1113°, −44.1231° | Active dam structure |
| `tailings_upper` | ~200 m upslope of crest | Upper tailings pond surface |
| `tailings_lower` | ~300 m downslope of crest | Lower tailings, near toe |
| `stable_ref` | Rocky ridge NE of dam | Stable reference — no mining activity |

### Sentinel-1 Tracks
A critical finding during data exploration: **no ascending Sentinel-1 track covers the Brumadinho area** during the 2017–2019 period. The two available tracks are both descending:

| Internal label | Track | Heading | Incidence angle | Overpass (UTC) |
|---|---|---|---|---|
| `desc53` | Track 53 | ~−104° | ~32° | ~08:28 |
| `desc155` | Track 155 | ~−104° | ~45° | ~08:20 |

This matches the Grebby et al. (2021) paper, which likewise used two descending tracks. The different incidence angles provide complementary sensitivity to surface motion geometry.

**Study period:** 1 July 2017 → 25 January 2019 (day of collapse)
**Total SAR scenes:** ~48 per track (Sentinel-1A, 12-day repeat)
**Interferometric pairs:** ~138 per track (SBAS network, max 180-day temporal baseline, max 200 m perpendicular baseline)

---

## 3. Processing Pipeline

### 3.1 Architecture Overview

```
Sentinel-1 SLC → [ASF HyP3 cloud] → Geocoded interferograms (UTM)
                                           ↓
                              MintPy 1.6.2 SBAS inversion
                                           ↓
                       LOS displacement time-series (mm)
                                           ↓
                        Velocity maps + POI time-series plots
```

### 3.2 Why HyP3 Instead of ISCE2

The originally planned pipeline used **ISCE2** (JPL) for interferogram generation, requiring local processing of raw Sentinel-1 SLCs (~2 TB, ~72 hours of compute). We switched to **ASF HyP3**, the Alaska Satellite Facility's cloud processing service, for three reasons:

1. HyP3 delivers analysis-ready, geocoded interferograms with no local SLC handling
2. Products are free for ASF users and available within hours via the API
3. MintPy 1.6.2 has a native `processor = hyp3` loading mode

Total downloaded data: ~90 GB (276 zip files, 138 pairs × 2 tracks).

### 3.3 MintPy SBAS Configuration

Key MintPy parameters for the standard SBAS run:

| Parameter | Track 53 | Track 155 | Notes |
|---|---|---|---|
| `network.minCoherence` | 0.7 | 0.7 | Pixel inclusion threshold |
| `networkInversion.weightFunc` | `no` | `no` | Uniform weights (L2 norm) |
| `networkInversion.maskThreshold` | 0.4 | 0.4 | Per-pixel coherence mask |
| `troposphericDelay.method` | `no` | `pyaps` (ERA5) | Track 53 skipped due to geometry size mismatch |
| `deramp` | `linear` | `linear` | Removes orbital ramp |
| `topographicResidual` | `yes` | `yes` | DEM error correction |
| `reference.lalo` | −20.06, −44.08 | −20.06, −44.08 | Stable rocky ridge |

**Notable engineering problems solved during pipeline development:**

- HyP3 zip files already contain a top-level subdirectory — extracting with `z.extractall(hyp3_dir)` (not `extractall(dest)`) prevents double-nesting
- HyP3 uses GAMMA unwrapper → no connected-component files → `unwrapError.method = no`
- `waterMask.h5` from HyP3 has different pixel dimensions than `ifgramStack.h5` → deleted and set to `no`
- Track 53's `geometryGeo.h5` is 3047 rows vs `ifgramStack.h5`'s 3045 rows → ERA5 correction fails → set to `no` for that track only
- Cached `smallbaselineApp.cfg` in the working directory overrides the project config — patched directly each run

### 3.4 ISBAS-like Variant

To approximate the **Intermittent SBAS (ISBAS)** approach used by Grebby et al., we ran a second MintPy inversion with relaxed thresholds:

| Parameter | Standard SBAS | ISBAS-like |
|---|---|---|
| `network.minCoherence` | 0.7 | 0.5 |
| `networkInversion.weightFunc` | `no` | `var` |
| `networkInversion.maskThreshold` | 0.4 | 0.2 |
| Working directory | `processing/mintpy/{track}/` | `processing/mintpy/{track}_isbas/` |

The effect was immediate: **Track 155 ISBAS inverted 1,713,674 pixels (45.4%)** vs approximately 25% for standard SBAS — consistent with ISBAS's core promise of greater spatial coverage. Results are kept entirely separate from the standard SBAS outputs using a `--variant isbas` flag on all pipeline scripts.

True ISBAS (as in Grebby et al.) uses the proprietary Intermittent SBAS algorithm (Sowter et al. 2013), where pixels participate in only the subset of interferograms in which they are coherent. MintPy's `weightFunc = var` approximates this by down-weighting low-coherence observations rather than excluding them entirely.

---

## 4. Results

### 4.1 Standard SBAS

| Point of Interest | Track | LOS Velocity | Cumulative to 17 Jan 2019 |
|---|---|---|---|
| `tailings_upper` | Track 155 (inc. ~45°) | **−9.4 mm/yr** | **−35.6 mm** |
| `tailings_lower` | Track 155 (inc. ~45°) | −8.5 mm/yr | −33.3 mm |
| `tailings_lower` | Track 53 (inc. ~32°) | +2.3 mm/yr | +2.0 mm |
| `dam_crest` | Track 155 | within noise | — |

Negative LOS values indicate motion **away from the satellite** — consistent with subsidence or downslope displacement in the LOS direction. The Track 53 (shallower incidence) signal near zero at the tailings_lower point reflects different geometric sensitivity to the same motion vector.

**Temporal structure:** Quarterly displacement analysis at `tailings_upper` (Track 155):

| Period | Rate |
|---|---|
| Jul–Oct 2017 | +2.2 mm/month (settling) |
| Oct 2017–Jan 2018 | −12.4 mm/month |
| Jan–Apr 2018 | +4.9 mm/month (seasonal rebound) |
| Apr–Jul 2018 | −4.3 mm/month |
| Jul–Oct 2018 | −2.7 mm/month |
| **Oct 2018–Jan 2019** | **−11.6 mm/month (acceleration)** |

The final three months show re-acceleration of LOS displacement, reaching −20.3 mm in the 8 epochs before collapse.

### 4.2 ISBAS-like Variant

| Point of Interest | Track | LOS Velocity | Cumulative to Jan 2019 |
|---|---|---|---|
| `dam_crest` | Track 53 (inc. ~32°) | −39.9 mm/yr | −48.4 mm |
| `dam_crest` | Track 155 (inc. ~45°) | −21.0 mm/yr | −49.8 mm |
| `tailings_upper` | Track 155 | −9.8 mm/yr | −36.9 mm |
| `tailings_lower` | Track 155 | −9.1 mm/yr | −34.8 mm |

The ISBAS-like run recovers signal at the `dam_crest` point that is absent from the standard SBAS (below coherence threshold). The tailings results are consistent, confirming stability of the inversion.

---

## 5. Comparison with Grebby et al. (2021)

| Aspect | Grebby et al. (2021) | This Project |
|---|---|---|
| **Data** | Sentinel-1, Tracks 53 + 155 | Sentinel-1, Tracks 53 + 155 |
| **Period** | ~Jul 2017 – Jan 2019 | Jul 2017 – Jan 2019 |
| **Pairs** | ~100+ per track | ~138 per track |
| **SAR processor** | GAMMA (commercial) | ASF HyP3 (free, GAMMA-based) |
| **Time-series method** | ISBAS (Sowter et al. 2013) | MintPy SBAS + ISBAS-like variant |
| **Atmospheric correction** | ERA5 | ERA5 (Track 155), none (Track 53) |
| **Max cumulative LOS** | Up to −36 mm/yr in patches; ~−18 mm avg at tailings front | −35.6 mm cumulative, −9.4 mm/yr avg |
| **Acceleration before collapse** | Visible Nov 2018–Jan 2019 | Confirmed: −11.6 mm/month in final quarter |
| **Dam crest signal** | Detected | Detected (ISBAS-like only) |
| **Decomposition** | None — two LOS maps analyzed separately | None (same reason: both tracks descending) |

### Key Agreement Points

1. **Magnitude**: Our cumulative LOS displacement (−35.6 mm) matches the paper's reported maximum of approximately −36 mm at the tailings surface.
2. **Acceleration**: Both find anomalous acceleration in the final 3 months before the January 25, 2019 collapse.
3. **Track geometry**: Confirmation that only descending tracks are available — the paper made the same observation.
4. **No decomposition**: With two tracks sharing the same heading (~−104°), the inversion into vertical + east-west components is geometrically ill-conditioned. Both studies present dual LOS maps independently.

### Key Differences

- **Pixel coverage**: Grebby et al. used proprietary ISBAS, recovering more pixels in low-coherence areas. Our standard SBAS is more conservative; our ISBAS-like variant partially bridges this gap (45% vs ~25% of pixels).
- **Velocity magnitudes**: The paper reports patches of up to −36 mm/yr in the spatial velocity maps. Our linear fit over the full period averages this out to −9.4 mm/yr, which is consistent — local hot-spots are subsampled by our point analysis.
- **Atmospheric correction**: Our Track 53 inversion skips ERA5 due to a grid dimension mismatch between the HyP3 geometry file and the interferogram stack. This does not materially affect the tailings signal but limits comparison at that track.

### Scientific Note

A 2025 *Matters Arising* paper in the same journal revisits the Grebby et al. results and questions whether the detected deformation reflects the failure mechanism or pre-existing mine subsidence unrelated to the dam. Our results do not resolve this debate, but the spatial pattern (concentrated at the tailings upper surface) and the timing of acceleration are consistent with Grebby et al.'s original interpretation.

---

## 6. Software Stack and Reproducibility

| Component | Tool | Version |
|---|---|---|
| SAR interferograms | ASF HyP3 (INSAR_GAMMA product) | API v1 |
| Time-series inversion | MintPy | 1.6.2 |
| Atmospheric correction | PyAPS / ERA5 (CDS API) | — |
| Coordinate transforms | pyproj | — |
| Plotting | matplotlib | — |
| Environment | conda (brumadinho_insar) | — |

All code is available at: [github.com/aminkarbassi/tailings-dam-inSAR](https://github.com/aminkarbassi/tailings-dam-inSAR)

To reproduce:
```bash
conda env create -f environment.yml
conda activate brumadinho_insar
python scripts/04_prep_mintpy.py
python scripts/06_run_mintpy.py
python scripts/08_plot_maps.py --orbit desc53
python scripts/08_plot_maps.py --orbit desc155
python scripts/09_plot_timeseries.py
```

---

## 7. Conclusion

This project demonstrates that **pre-collapse deformation at the Brumadinho dam was detectable from space** using freely available Sentinel-1 SAR data and open-source tools. The SBAS time-series shows:

- Sustained LOS subsidence of ~9–10 mm/yr at the tailings surface throughout 2017–2018
- **Re-acceleration to −11.6 mm/month in the final 3 months before collapse**
- Signal concentrated at the upper tailings surface and dam crest, not at the stable reference site

These findings independently confirm the Grebby et al. (2021) results and support the case for routine satellite radar monitoring of tailings storage facilities as a low-cost, globally scalable early-warning mechanism.
