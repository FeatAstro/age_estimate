# Revealing the Star Formation History of the Local Milky Way Through Its Young Stellar Groups

**M1 Internship — LIRA, Observatoire de Paris / CNRS**  
**Supervisor: Alexis Quintana**  
**Student: Gael**

---

## Overview

This pipeline identifies OB association members in Gaia DR3, clusters them kinematically
using HDBSCAN, and fits PARSEC isochrones to derive ages. It is designed to be general
for any young stellar complex within 1 kpc, and has been applied to Sco-Cen (~140 pc)
the Cep-Her complex (~350 pc) and Orion complex (~xxx pc) as primary targets.

---

## Repository structure

```
.
├── data/
│   ├── processed/          # Gaia catalogs after Script 1
│   └── crossmatch/         # Reference catalogs 
├── grids/
│   └── parsec_solar_gaia.dat   # PARSEC isochrone grid (see below)
├── outputs/
│   ├── hdbscan/            # HDBSCAN cluster catalogs
│   └── ages/               # Isochrone fitting results
├── images/
│   ├── hdbscan/            # Diagnostic plots from Script 2b
│   ├── ages/               # CMD + posterior plots from Script 5
│   ├── agemap/             # Age maps from Script 4
│   └── sfh/                # Star formation history plots from Script 6
├── Script1_Download_Gaia.py        # Gaia ADQL query
├── Script2_HDBSCAN.py              # HDBSCAN clustering
├── script2_inspect.py              # Visual inspection of clusters
├── fit_ages.py                     # Bayesian isochrone age fitting
├── script4_agemap.py               # Spatial age map
├── script5_cmd_posteriors.py       # CMD + posterior diagnostics
├── script6_sfh.py                  # Star formation history plots
├── download_parsec.py              # Automated PARSEC grid downloader (utility)
├── load_parsec.py                  # PARSEC grid loader (utility)
└── fit_isochrone.py                # Isochrone fitting core (utility)
```

---

## Installation

### Python version
Python 3.10 or 3.11 recommended.

### Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
```

### Install dependencies
```bash
pip install numpy scipy matplotlib astropy scikit-learn dynesty astroquery
```

Full dependency list:
| Package | Version | Purpose |
|---|---|---|
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.10 | Interpolation, statistics |
| matplotlib | ≥3.7 | All plotting |
| astropy | ≥5.3 | FITS I/O, coordinates, units |
| scikit-learn | ≥1.3 | HDBSCAN clustering |
| dynesty | ≥2.1 | Nested sampling for age fitting |
| astroquery | ≥0.4 | VizieR catalog download |

---

## PARSEC isochrone grid

The grid file `grids/parsec_solar_gaia.dat` is required by `load_parsec.py` and `fit_ages.py`.

### Automated download — `download_parsec.py`

The recommended way to obtain the grid is to run the dedicated download script:

```bash
python3 download_parsec.py
```

This script queries the PARSEC CMD web interface at
[http://stev.oapd.inaf.it/cgi-bin/cmd](http://stev.oapd.inaf.it/cgi-bin/cmd)
with the correct parameters and saves the result to `grids/parsec_solar_gaia.dat`.

Default parameters used:
- **Evolutionary tracks**: PARSEC version 1.2S
- **Photometric system**: Gaia EDR3 (Weiler 2018)
- **Initial mass function**: Kroupa (2001)
- **Metallicity**: Z = 0.0152 (solar)
- **Age range**: log(age) = 5.0 to 7.95, step 0.05

### Manual download

If the automated download fails (the CMD web interface occasionally changes):

1. Go to [http://stev.oapd.inaf.it/cgi-bin/cmd](http://stev.oapd.inaf.it/cgi-bin/cmd)
2. Set the parameters listed above
3. Download the `.dat` file and place it in `grids/parsec_solar_gaia.dat`

### Grid utilities — `load_parsec.py`

`load_parsec.py` reads the grid and exposes two functions used throughout the pipeline:
- `load_parsec(path, label_max=3)` — loads the full grid, filtering to MS+PMS phases (label ≤ 3)
- `get_isochrone(grid, log_age)` — interpolates the grid at a requested log age

---

## Scripts

### Script 1 — `Script1_Download_Gaia.py`
Downloads Gaia DR3 data for a given sky region using an ADQL query through the Gaia archive. Applies parallax zero-point correction (Lindegren+2021), cross-matches with BailerJones distances, 2MASS, PanSTARRS DR1, and Rybizki+2022 fidelity scores.

**Key parameters to set:**
```python
name_complex = 'Sco_Cen'       # for e.g.
sky_tag      = 'ra100_300_dec-90_0'
ra_min, ra_max, dec_min, dec_max = 100, 300, -90, 0
plx_min, plx_max = 5.0, 14.0   # parallax cut in mas
```

**Output:** `data/processed/catalog_complete_<name_complex>_<sky_tag>.fits`

---

### Script 2 — `Script2_HDBSCAN.py`
Applies selection cuts to the Gaia catalog and runs HDBSCAN clustering in 5D phase space (X, Y, Z, CV×V_l, CV×V_b). 
Proper motions are converted to Galactic frame and corrected for the LSR.
Selection cuts applied: astrometric quality (RUWE/fidelity), parallax range, M_G faint limit (automatic, distance-dependent), optional CMD pre-selection.

**Key parameters:**
```python
MIN_CLUSTER_SIZE = 20    # minimum stars per cluster
MIN_SAMPLES      = 70    # density threshold (higher = fewer, larger clusters)
CV               = 8     # velocity scaling (higher = more kinematic weight)
```

**Output:** `outputs/hdbscan/hdbscan_clusters_<run_tag>.fits`

---

### Script 2b — `script2_inspect.py`
Visual inspection of HDBSCAN clusters. Produces one 4-panel diagnostic figure per cluster: CMD with 20 Myr PARSEC isochrone overlay, proper motion diagram, parallax histogram, and RA/Dec sky map colored by membership probability.

Also outputs a summary FITS table with quality flags (σ_ϖ, σ_μ, N) and a list of quality-OK clusters ready for age fitting.

**Usage:**
```bash
python3 script2_inspect.py                   # all clusters
python3 script2_inspect.py --clusters 7 11   # specific clusters
```

**Output:** `outputs/<run_tag>/inspect_hdbscan/cluster_<ID>_N<N>.png`

---

### Script 3 — `fit_ages.py`
Bayesian isochrone age fitting for each HDBSCAN cluster using the skewed Cauchy likelihood of Ratzenböck+2023 (Paper II). Sampling is done with dynesty nested sampling, which handles the age-extinction degeneracy well.

Fits 4 parameters: log_age, Av (extinction), s (scatter scale), a (skewness).
Distance is fixed per-star using BailerJones geometric distances, no single cluster distance is assumed, which is importan for extended associations.

**Usage:**
```bash
python3 fit_ages.py                         # all clusters
python3 fit_ages.py --clusters 7 11 13      # specific clusters
python3 fit_ages.py --resume                # skip already-fitted clusters
python3 fit_ages.py --nlive 500             # more live points (slower, better)
python3 fit_ages.py --cmd GRP               # use G-RP instead of BP-RP
```

**Output:**
- `outputs/<run_tag>/ages/ages_BPRP.fits` — age summary table
- `outputs/<run_tag>/ages/posteriors/cluster_<ID>_BPRP.npz` — posterior samples

---

### Script 4 — `script4_agemap.py`
Produces the main science figure: a spatial age map in Galactic coordinates
with clusters shown as bubbles colored by age and sized by their number of stars, and also shown with the individual stars. Includes optional crossmatch with reference catalogs (Ratzenböck+2023 for Sco-Cen, Kerr+2024 for Cep-Her, Sanchez-Sanjuan+2023 for Orion) for direct comparison.

Also produces a summary table (CSV) with cluster positions, ages, and reference catalog matches.

**Usage:**
```bash
python3 script4_agemap.py --logz-min 10
python3 script4_agemap.py --logz-age          # age-dependent logZ threshold
python3 script4_agemap.py --age-max 25        # Sco-Cen only
python3 script4_agemap.py --logz-age \
  --cat data/crossmatch/ScoCenGroups_Ratzenbock2023_crossmatch.fits \
  --name-col SigMA                            # with crossmatch
```

**Output:**
- `images/agemap/<run_tag>/agemap_BPRP.png`
- `outputs/agemap/<run_tag>/summary_BPRP.csv`

---

### Script 5 — `script5_cmd_posteriors.py`
Diagnostic figures for each fitted cluster: CMD with best-fit isochrone overlay (MAP age + 68% HDI bounds), age posterior histogram, and Av posterior histogram. Essential for validating age fits before presenting results.

Uses an age-dependent logZ threshold to avoid discarding young clusters that have intrinsically wider CMD scatter (PMS stars, disk excess).

**Usage:**
```bash
python3 script5_cmd_posteriors.py --logz-min 10
python3 script5_cmd_posteriors.py --logz-age           # age dependent
python3 script5_cmd_posteriors.py --clusters 7 11 13
python3 script5_cmd_posteriors.py --cmd GRP
```

**Output:** `images/<run_tag>/ages/cmd_cluster_<ID>_BPRP.png`

---

### Script 6 — `script6_sfh.py`
Star formation history plots:
- **Age vs Galactic longitude** — spatial age gradient across the complex
- **Age vs distance** — radial age structure
- **Timeline (Gantt chart)** — clusters ordered by distance, horizontal
  HDI bars showing age uncertainty per group

All figures include optional catalog reference comparison.

**Usage:**
```bash
python3 script6_sfh.py --logz-age
python3 script6_sfh.py --logz-age --age-max 25
python3 script6_sfh.py --logz-age \
  --cat data/crossmatch/ScoCenGroups_Ratzenbock2023_crossmatch.fits \
  --name-col SigMA
```

**Output:**
- `images/<run_tag>/sfh/sfh_age_vs_l_BPRP.png`
- `images/<run_tag>/sfh/sfh_age_vs_dist_BPRP.png`
- `images/<run_tag>/sfh/sfh_timeline_BPRP.png`

---

## Age-dependent logZ threshold

The `--logz-age` flag applies a physically motivated logZ filter:

| Age range | logZ threshold | Reason |
|---|---|---|
| < 10 Myr | -20 | PMS scatter, disk excess — very permissive |
| 10–20 Myr | 0 | UCL/LCC-age — moderate |
| 20–35 Myr | 5 | Her OB1-age — some CMD scatter remains |
| > 35 Myr | 10 | Background populations — clean MS, strict |

Young clusters have intrinsically worse isochrone fits due to pre-main sequence scatter, differential extinction, and disk excess. A fixed logZ threshold would incorrectly discard real young detections.

---

## Typical workflow

```bash
# 0. Download PARSEC isochrone grid (once only)
python3 download_parsec.py

# 1. Download Gaia data
python3 Script1_Download_Gaia.py

# 2. Run HDBSCAN clustering
python3 Script2_HDBSCAN.py

# 3. Inspect clusters visually
python3 script2_inspect.py

# 4. Fit ages
python3 fit_ages.py --resume

# 5. Validate fits
python3 script5_cmd_posteriors.py --logz-age

# 6. Age map
python3 script4_agemap.py --logz-age --cat <crossmatch.fits> --name-col <col>

# 7. Star formation history
python3 script6_sfh.py --logz-age
```

---

## Reference catalogs

Reference catalogs are used for crossmatch validation and comparison only, they are not required to run the pipeline. Any complex within 1 kpc can be targeted by adjusting the sky box, parallax cut, and HDBSCAN parameters.

| Complex | Distance | Reference | File | Key column |
|---|---|---|---|---|
| Sco-Cen | ~140 pc | Ratzenböck+2023 | `ScoCenGroups_Ratzenbock2023_crossmatch.fits` | `SigMA` |
| Cep-Her | ~350 pc | Kerr+2024 | `CepHerGroups_SPYGLASS.fits` | `SUBGROUP` |

The `SUBGROUP` column in the Kerr+2024 catalog is a composite key built from `AS` (association: ORPH/CINR/CUPV/ROS6) and `SG` (subgroup ID), e.g. `ORPH_28`. Stars with `SG = -1` (unassigned) are excluded from the crossmatch.

For a new complex with no reference catalog, omit `--cat` from Scripts 4 and 6.
Clusters will be labelled by ID only and the age map will show NEW for all groups.

---

## Key references

- Ratzenböck et al. (2022, 2023) — SigMA algorithm, Sco-Cen age fitting
- Kerr et al. (2023) — SPYGLASS IV, Cep-Her base catalog
- Kerr et al. (2024) — SPYGLASS V, Cep-Her substructure (Orpheus, Cinyras, Cupavo)
- Lindegren et al. (2021) — Gaia EDR3 parallax zero-point correction
- Rybizki et al. (2022) — Gaia fidelity scores
- Bailer-Jones et al. (2021) — Geometric distances from Gaia parallaxes
- Choi et al. (2016) / Dotter (2016) — MIST stellar models
- Speagle (2020) — dynesty nested sampling