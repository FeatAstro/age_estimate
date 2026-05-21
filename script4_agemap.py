#!/usr/bin/env python3
"""
script4_results.py — Summary table and age map for fitted clusters
==================================================================
Reads the age fitting results from fit_ages.py and produces:

  1. A formatted summary table printed to terminal and saved as CSV.

  2. An age map in Galactic coordinates (l/b) with fluid bubbles colored
     and sized by age/N.

  3. An age vs Galactic longitude plot showing the star formation history.

Usage
-----
    python3 script4_agemap.py
    python3 script4_agemap.py --cmd-compare
    python3 script4_agemap.py --cat data/crossmatch/ScoCenGroups_Ratzenbock2023_crossmatch.fits \
                               --name-col SigMA
    python3 script4_agemap.py --age-max 25   # Sco-Cen only
    python3 script4_agemap.py --logz-min 10   
	python3 script4_agemap.py --logz-age	 # Age-dependent threshold for logz

Input
-----
    outputs/<run_tag>/ages/ages_BPRP.fits
    outputs/<run_tag>/ages/ages_GRP.fits   (optional)

Output
------
    outputs/<run_tag>/agemap/summary_<cmd>.csv
    images/<run_tag>/agemap/agemap_<cmd>.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from astropy.table import Table, join

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
name_complex = 'Sco_Cen'  # choose
sky_tag      = 'ra100_300_dec-90_0' # choose
ms_tag       = '80'
mc_tag       = '20'
cv_tag       = '10' 

catalog_name = 'Ratzenbock+2023'  # choose

run_tag      = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_ages    = f'outputs/{run_tag}/ages/'
path_hdbscan = 'outputs/hdbscan/'
path_results = f'outputs/{run_tag}/agemap/{catalog_name}/'
path_summary = 'summary/'

os.makedirs(path_results, exist_ok=True)

# Orion OB1 (Sanchez-Sanjuan et al. 2024)
# CAT_AGES = {
# 1:  (4.7, 2.3, 11.0),   # lambda Ori
# 2:  (13.36, 8.73, 17.55),   # Ori-North
# 3:  (9.0, 5.6, 13.0),   # Briceno-1A
# 4:  (9.0, 5.6, 13.0),   # Briceno-1B
# 5:  (10.0,  8.0, 12.0),   # Ori-East
# 6:  (9.0,  7.0, 11.0),   # OBP-Far
# 7:  (2.5, 2.2, 2.8),   # sigma Ori
# 8:  (17.9, 11.4, 24.0),   # OBP-b
# 9:  (6.4, 2.1, 12.8),   # OBP-d
# 10: (6.8, 3.9, 10.4),   # OBP-Near
# 11: (2.0,  1.0,  3.0),   # ONC
# 12: (3.1, 1.0, 5.7),   # Ori-South
# 13: (18.2, 12.9, 28.0),   # Orion Y
# }

# # Cluster
# CAT_NAMES = {
# 1:  'lambda Ori',   2:  'Ori-North',
# 3:  'Briceno-1A',   4:  'Briceno-1B',
# 5:  'Ori-East',     6:  'OBP-Far',
# 7:  'sigma Ori',    8:  'OBP-b',
# 9:  'OBP-d',        10: 'OBP-Near',
# 11: 'ONC',          12: 'Ori-South',
# 13: 'Orion Y',
# }

# ------------------------------------------------------------------------------#

# Her OB1 (Kerr et al. 2024)
# CAT_AGES = {
#     'ORPH_1' : (27.9, 25.5, 30.3),   'ORPH_2' : (26.5, 23.7, 29.3),
#     'ORPH_3' : (39.0, 37.2, 40.8),   'ORPH_4' : (34.5, 32.0, 37.0),
#     'ORPH_5' : (30.0, 28.1, 31.9),   'ORPH_6' : (33.7, 29.9, 37.5),
#     'ORPH_7' : (39.8, 36.5, 43.1),   'ORPH_8' : (27.1, 25.8, 28.4),
#     'ORPH_9' : (32.8, 30.0, 35.6),   'ORPH_10': (24.9, 23.0, 26.8),
#     'ORPH_11': (32.4, 30.4, 34.4),   'ORPH_12': (27.7, 26.1, 29.3),
#     'ORPH_13': (26.5, 22.7, 30.3),   'ORPH_14': (26.0, 23.7, 28.3),
#     'ORPH_15': (26.5, 24.4, 28.6),   'ORPH_16': (34.5, 32.9, 36.1),
#     'ORPH_17': (27.9, 25.1, 30.7),   'ORPH_18': (30.0, 28.6, 31.4),
#     'ORPH_19': (27.3, 26.0, 28.6),   'ORPH_20': (32.6, 30.6, 34.6),
#     'ORPH_21': (31.4, 29.5, 33.3),   'ORPH_22': (31.7, 29.0, 34.4),
#     'ORPH_23': (30.1, 28.5, 31.7),   'ORPH_24': (27.4, 25.5, 29.3),
#     'ORPH_25': (34.9, 32.3, 37.5),   'ORPH_26': (28.0, 25.3, 30.7),
#     'ORPH_27': (28.7, 26.9, 30.5),   'ORPH_28': (32.1, 30.9, 33.3),
#     'CINR_1' : (27.9, 26.0, 29.8),   'CINR_2' : (37.1, 32.6, 41.6),
#     'CINR_3' : (29.3, 27.4, 31.2),   'CINR_4' : (38.5, 33.7, 43.3),
#     'CINR_5' : (31.4, 30.0, 32.8),   'CINR_6' : (29.9, 28.3, 31.5),
#     'CINR_7' : (32.9, 31.7, 34.1),   'CINR_8' : (39.3, 35.1, 43.5),
#     'CINR_9' : (43.1, 39.1, 47.1),   'CINR_10': (35.7, 34.2, 37.2),
#     'CUPV_1' : (68.2, 59.0, 77.4),   'CUPV_2' : (62.6, 50.8, 74.4),
#     'CUPV_3' : (55.3, 53.0, 57.6),   'CUPV_4' : (68.4, 59.4, 77.4),
#     'CUPV_5' : (71.0, 64.5, 77.5),   'CUPV_6' : (68.1, 61.4, 74.8),
#     'CUPV_7' : (80.0, 80.0, 99.9),   'CUPV_8' : (55.5, 51.4, 59.6),
#     'CUPV_9' : (53.8, 49.1, 58.5),   'CUPV_10': (58.4, 55.4, 61.4),
#     'ROS6_1' : (80.0, 80.0, 99.9),   'ROS6_2' : (71.2, 64.8, 77.6),
# }

# # SUBGROUP
# CAT_NAMES = {
#     'ORPH_1' : 'Orpheus 1',    'ORPH_2' : 'Orpheus 2',
#     'ORPH_3' : 'Orpheus 3',    'ORPH_4' : 'Orpheus 4',
#     'ORPH_5' : 'Orpheus 5',    'ORPH_6' : 'Orpheus 6',
#     'ORPH_7' : 'Orpheus 7',    'ORPH_8' : 'Orpheus 8',
#     'ORPH_9' : 'Orpheus 9',    'ORPH_10': 'Orpheus 10',
#     'ORPH_11': 'Orpheus 11',   'ORPH_12': 'Orpheus 12',
#     'ORPH_13': 'Orpheus 13',   'ORPH_14': 'Orpheus 14',
#     'ORPH_15': 'Orpheus 15',   'ORPH_16': 'Orpheus 16',
#     'ORPH_17': 'Orpheus 17',   'ORPH_18': 'Orpheus 18',
#     'ORPH_19': 'Orpheus 19',   'ORPH_20': 'Orpheus 20',
#     'ORPH_21': 'Orpheus 21',   'ORPH_22': 'Orpheus 22',
#     'ORPH_23': 'Orpheus 23',   'ORPH_24': 'Orpheus 24',
#     'ORPH_25': 'Orpheus 25',   'ORPH_26': 'Orpheus 26',
#     'ORPH_27': 'Orpheus 27',   'ORPH_28': 'Orpheus 28',
#     'CINR_1' : 'Cinyras 1',    'CINR_2' : 'Cinyras 2',
#     'CINR_3' : 'Cinyras 3',    'CINR_4' : 'Cinyras 4',
#     'CINR_5' : 'Cinyras 5',    'CINR_6' : 'Cinyras 6',
#     'CINR_7' : 'Cinyras 7',    'CINR_8' : 'Cinyras 8',
#     'CINR_9' : 'Cinyras 9',    'CINR_10': 'Cinyras 10',
#     'CUPV_1' : 'Cupavo 1',     'CUPV_2' : 'Cupavo 2',
#     'CUPV_3' : 'Cupavo 3',     'CUPV_4' : 'Cupavo 4',
#     'CUPV_5' : 'Cupavo 5',     'CUPV_6' : 'Cupavo 6',
#     'CUPV_7' : 'Cupavo 7',     'CUPV_8' : 'Cupavo 8',
#     'CUPV_9' : 'Cupavo 9',     'CUPV_10': 'Cupavo 10',
#     'ROS6_1' : 'Roslund 6-1',  'ROS6_2' : 'Roslund 6-2',
# }

# ------------------------------------------------------------------------------#

# Sco-Cen (Ratzenböck et al. 2023)
CAT_AGES = {
  #  ID   age   lo    hi
     1: ( 3.8,  3.4,  4.2),    2: ( 5.8,  5.3,  7.6),    3: ( 9.8,  8.4, 11.0),
     4: ( 7.6,  6.9,  8.4),    5: (10.0,  9.5, 11.0),    6: (12.7, 11.0, 13.1),
     7: (13.7, 13.1, 15.0),    8: (14.7, 14.0, 15.5),    9: (19.1, 17.8, 21.5),
    10: (15.0, 13.6, 15.9),   11: (17.2, 14.8, 18.1),   12: (20.0, 17.8, 22.5),
    13: ( 6.0,  5.1,  6.6),   14: (15.3, 15.0, 15.9),   15: (16.9, 16.3, 17.8),
    16: (42.1, 33.2, 51.1),   17: (20.9, 20.1, 21.6),   18: (13.4, 12.7, 14.8),
    19: (14.4, 13.5, 14.8),   20: (15.7, 14.8, 16.0),   21: (15.5, 15.0, 16.1),
    22: (11.2, 10.2, 12.2),   23: (10.2,  9.5, 11.2),   24: ( 8.8,  8.4,  9.4),
    25: ( 9.4,  8.5, 10.8),   26: ( 3.4,  2.5,  6.5),   27: (15.9, 13.8, 17.5),
    28: (15.4, 13.5, 16.2),   29: ( 8.5,  6.1, 10.5),   30: (11.6, 10.8, 12.1),
    31: (14.5, 13.9, 15.1),   32: ( 8.5,  7.2,  9.6),   33: ( 3.8,  2.9,  5.7),
    34: ( 2.8,  1.7,  3.5),   35: ( 9.6,  7.4, 11.3),   36: ( 9.2,  7.5, 12.5),
    37: (19.1, 14.5, 25.7),
}

# SigMA
CAT_NAMES = {
    1:'rho Oph/L1688', 2:'nu Sco',         3:'delta Sco',      4:'beta Sco',
    5:'sigma Sco',     6:'Antares',        7:'rho Sco',        8:'Scorpio-Body',
    9:'US-foreground', 10:'V1062-Sco',     11:'mu Sco',        12:'Libra-South',
    13:'Lupus-1-4',    14:'eta Lup',       15:'phi Lup',       16:'Norma-North',
    17:'e Lup',        18:'UPK606',        19:'rho Lup',       20:'nu Cen',
    21:'sig Cen',      22:'Acrux',         23:'Musca-fgd',     24:'eps Cham',
    25:'eta Cham',     26:'B59',           27:'Pipe-North',    28:'tet Oph',
    29:'CrA-Main',     30:'CrA-North',     31:'Scorpio-Sting', 32:'Centaurus-Far',
    33:'Chamaeleon-1', 34:'Chamaeleon-2',  35:'L134/L183',     36:'Oph SE',
    37:'Oph NorthFar',
}


# ------------------------------------------------------------------------------#

# Sco-Cen (Kerr et al. 2023)
# CAT_AGES = {
# }

# EOM & Leaf
# CAT_NAMES = {
# }

# Cluster name mapping — built automatically from crossmatch if --cat is given,
# otherwise stays empty (cluster IDs are used as labels).
CLUSTER_NAMES = {}

# ---------------------------------------------------------------------------
# Manual label-direction overrides
# ---------------------------------------------------------------------------
# Per-plot dicts: {cluster_id: direction}
# Valid directions: 'NE', 'NW', 'SE', 'SW', 'E', 'W', 'N', 'S'
# Leave empty to rely entirely on the automatic placement algorithm.
LABEL_DIR_STARS_LB      = {7: 'SW', 4: 'S', 9: 'S'}   
LABEL_DIR_BUBBLES_LB    = {4: 'S', 9: 'S'}  
LABEL_DIR_STARS_RADEC   = {}   
LABEL_DIR_BUBBLES_RADEC = {}

# Sco-Cen
# LABEL_DIR_STARS_LB      = {24: 'SW', 25: 'SE'}   # galactic coords, individual stars
# LABEL_DIR_BUBBLES_LB    = {17: 'NW', 32: 'SW', 33: 'NW', 31: 'NE'}   # galactic coords, bubbles
# LABEL_DIR_STARS_RADEC   = {24: 'SW', 25: 'SE', 28: 'NE', 26: 'S', 30: 'S', 17: 'S'}   # RA/Dec, individual stars
# LABEL_DIR_BUBBLES_RADEC = {17: 'NW', 25: 'S', 23: 'E', 30: 'S'}   # RA/Dec, bubbles

def _resolve_cat_key(raw):
    """Convert a raw catalog value to the key type used in CAT_AGES/CAT_NAMES."""
    s = raw.decode().strip() if isinstance(raw, bytes) else str(raw).strip()
    try:
        return int(s), s
    except (ValueError, TypeError):
        return s, s


def _valid_labels(col):
    """Return non-empty, non-masked string labels from a (possibly masked) column slice."""
    import numpy.ma as ma
    arr = np.asarray(col)
    mask = ma.getmaskarray(col)
    out = []
    for i, v in enumerate(arr):
        if mask[i]:
            continue
        s = v.decode().strip() if isinstance(v, bytes) else str(v).strip()
        if s and s not in ('--', 'nan'):
            out.append(s)
    return np.array(out)


def build_cluster_names(cat, name_col, hdb, min_frac=0.30, min_count=3):
    cat = cat.copy(); hdb = hdb.copy()
    cat['source_id'] = np.array(cat['source_id']).astype(str)
    hdb['source_id'] = np.array(hdb['source_id']).astype(str)
    _, keep = np.unique(np.array(cat['source_id']), return_index=True)
    cat_dedup = cat[keep]

    try:
        matched = join(hdb, cat_dedup[['source_id', name_col]],
                       keys='source_id', join_type='inner')
    except Exception as e:
        print(f"  Warning: crossmatch join failed ({e})")
        return {}

    names = {}
    for cid in np.unique(np.array(hdb['cluster_id'])):
        n_total    = (np.array(hdb['cluster_id']) == cid).sum()
        match_mask = np.array(matched['cluster_id']) == cid
        if match_mask.sum() < min_count:
            continue
        raw = _valid_labels(matched[name_col][match_mask])
        if len(raw) == 0:
            continue
        vals, counts = np.unique(raw, return_counts=True)
        best_raw     = vals[np.argmax(counts)]
        best_key, best_str = _resolve_cat_key(best_raw)
        if counts.max() / n_total >= min_frac:
            names[int(cid)] = CAT_NAMES.get(best_key, best_str)
    return names																																																																		

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--cat', default=None,
                   help='Catalog FITS for cross-match labels')
    p.add_argument('--name-col', default='AS',
                   help='Column in catalog with cluster IDs')
    p.add_argument('--cmd-compare', action='store_true',
                   help='Load both BPRP and GRP and compare')
    p.add_argument('--age-max', type=float, default=None,
                   help='Maximum age in Myr to include')
    p.add_argument('--logz-min', type=float, default=10.0,
                   help='Minimum log_evidence to include')
    p.add_argument('--logz-age', action='store_true',
                   help='Use age-dependent logZ threshold (permissive for young clusters)')
    p.add_argument('--l-min', type=float, default=None)
    p.add_argument('--l-max', type=float, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_results(cmd='BPRP'):
    path = path_ages + f'ages_{cmd}.fits'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    return Table.read(path)


def apply_filters(results, args):
    mask = np.ones(len(results), dtype=bool)
    if getattr(args, 'logz_age', False):
        logz_mask = np.array([
            float(row['log_evidence']) > logz_threshold(float(row['age_map_myr']))
            for row in results
        ])
        mask &= logz_mask
    else:
        mask &= np.array(results['log_evidence']) > args.logz_min
    if args.age_max is not None:
        mask &= np.array(results['age_lo_myr']) <= args.age_max
    if args.l_min is not None:
        mask &= np.array(results['l_mean']) >= args.l_min
    if args.l_max is not None:
        mask &= np.array(results['l_mean']) <= args.l_max
    return results[mask]


# ---------------------------------------------------------------------------
# Cross-match
# ---------------------------------------------------------------------------
def crossmatch_labels(results, cat, name_col, min_count=3):
    hdb_file = path_hdbscan + f'hdbscan_clusters_{run_tag}.fits'
    hdb = Table.read(hdb_file)
    cat = cat.copy(); hdb = hdb.copy()
    cat['source_id'] = np.array(cat['source_id']).astype(str)
    hdb['source_id'] = np.array(hdb['source_id']).astype(str)
    _, keep = np.unique(np.array(cat['source_id']), return_index=True)
    cat_dedup = cat[keep]

    matched = join(hdb, cat_dedup[['source_id', name_col]],
                   keys='source_id', join_type='inner')

    ref_labels, ref_ages, ref_ages_lo, ref_ages_hi = [], [], [], []
    for row in results:
        k     = int(row['cluster_id'])
        n_tot = int(row['n_members'])
        mask  = np.array(matched['cluster_id']) == k

        if mask.sum() < min_count:
            ref_labels.append('NEW')
            ref_ages.append(np.nan); ref_ages_lo.append(np.nan); ref_ages_hi.append(np.nan)
            continue

        raw = _valid_labels(matched[name_col][mask])
        if len(raw) == 0:
            ref_labels.append('NEW')
            ref_ages.append(np.nan); ref_ages_lo.append(np.nan); ref_ages_hi.append(np.nan)
            continue

        vals, counts = np.unique(raw, return_counts=True)
        best_raw     = vals[np.argmax(counts)]
        best_key, best_str = _resolve_cat_key(best_raw)
        best_count   = counts.max()

        cat_mask         = np.array(cat_dedup[name_col]).astype(str) == best_str
        n_subgroup_total = cat_mask.sum()
        frac_hdbscan     = best_count / n_tot * 100
        frac_catalog     = best_count / n_subgroup_total * 100 if n_subgroup_total > 0 else np.nan

        ref_labels.append(
            f'{CAT_NAMES.get(best_key, best_str)} '
            f'({frac_hdbscan:.0f}% of HDBSCAN | {frac_catalog:.0f}% of catalog)'
        )
        age_tuple = CAT_AGES.get(best_key, (np.nan, np.nan, np.nan))
        ref_ages.append(age_tuple[0])
        ref_ages_lo.append(age_tuple[1])
        ref_ages_hi.append(age_tuple[2])

    results['ref_label']   = ref_labels
    results['ref_age_myr'] = np.array(ref_ages,    dtype=float)
    results['ref_age_lo']  = np.array(ref_ages_lo, dtype=float)
    results['ref_age_hi']  = np.array(ref_ages_hi, dtype=float)
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(results, cmd='BPRP', compare=None):
    has_ref = 'ref_label' in results.colnames
    print(f"\n{'='*140}")
    print(f"  {name_complex}  run={run_tag}  CMD={cmd}  "
          f"{len(results)} clusters")
    print(f"{'='*140}")
    print(f"{'ID':>4}  {'N':>5}  {'l':>6}  {'b':>5}  {'dist':>5}  "
          f"{'Age [lo, hi] Myr':>20}  {'Av':>5}  {'logZ':>7}"
          + (f"  {'Ref':>45} {'RefAge':>8} {'Δ':>6}" if has_ref else ''))
    print('-' * 140)

    csv_rows = []
    for row in results:
        k       = int(row['cluster_id'])
        age     = float(row['age_map_myr'])
        age_lo  = float(row['age_lo_myr'])
        age_hi  = float(row['age_hi_myr'])
        age_str = f"{age:.1f} [{age_lo:.1f}, {age_hi:.1f}]"
        av      = float(row['Av_map'])
        logz    = float(row['log_evidence'])
        l       = float(row['l_mean'])
        b       = float(row['b_mean'])
        dist    = float(row['dist_mean'])
        n       = int(row['n_members'])

        line = (f"{k:>4}  {n:>5}  {l:>6.1f}  {b:>5.1f}  {dist:>5.0f}  "
                f"{age_str:>20}  {av:>5.2f}  {logz:>7.1f}")

        if has_ref:
            ref     = str(row['ref_label'])
            ref_age = float(row['ref_age_myr'])
            offset  = age - ref_age if not np.isnan(ref_age) else np.nan
            ref_str = f"{ref_age:.1f}" if not np.isnan(ref_age) else '—'
            off_str = f"{offset:+.1f}" if not np.isnan(offset) else '—'
            line   += f"  {ref:>45}  {ref_str:>8}  {off_str:>6}"
            csv_rows.append([k, n, l, b, dist, age, age_lo, age_hi,
                             av, logz, ref, ref_age, offset])
        else:
            csv_rows.append([k, n, l, b, dist, age, age_lo, age_hi, av, logz])

        print(line)

    print('=' * 140)

    # save CSV
    csv_path = path_results + f'summary_{catalog_name}_{cmd}.csv'
    csv_path2 = path_summary + f'summary_{catalog_name}_{cmd}_{run_tag}.csv'
    os.makedirs(path_summary, exist_ok=True)
    hdr = 'cluster_id,n_members,l_mean,b_mean,dist_mean,age_map_myr,age_lo_myr,age_hi_myr,Av_map,log_evidence'
    if has_ref:
        hdr += ',ref_label,ref_age_myr,age_offset_myr'
    with open(csv_path, 'w') as f:
        f.write(hdr + '\n')
        for r in csv_rows:
            f.write(','.join(str(x) for x in r) + '\n')
    with open(csv_path2, 'w') as f:
        f.write(hdr + '\n')
        for r in csv_rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print(f"\nCSV -> {csv_path} and {csv_path2}")

# ---------------------------------------------------------------------------
# Age map — white background, deep colormap, clean bubbles
# ---------------------------------------------------------------------------
def _compute_axes_limits(res, members=None, pad_deg=3.0):
    """Shared axis limits for both age map plots.

    If members is provided, limits are computed from the selected member
    star positions so the padding reflects the star distribution rather than
    the cluster bubble/center positions.
    """
    if members is not None and len(members) > 0:
        cluster_ids = np.unique(np.array(res['cluster_id'], dtype=int))
        star_mask = np.isin(np.array(members['cluster_id'], dtype=int), cluster_ids)
        if star_mask.sum() > 0:
            l = np.array(members['l'])[star_mask]
            b = np.array(members['b'])[star_mask]
        else:
            l = np.array(res['l_mean'])
            b = np.array(res['b_mean'])
    else:
        l = np.array(res['l_mean'])
        b = np.array(res['b_mean'])

    return ((l.max() + pad_deg, l.min() - pad_deg),   # xlim: inverted
            (b.min() - pad_deg, b.max() + pad_deg))   # ylim


def _choose_figsize(xlim, ylim, max_dim=24, min_dim=10):
    """Choose a rectangular figure size from axis extents."""
    dx = abs(xlim[0] - xlim[1])
    dy = abs(ylim[1] - ylim[0])
    if dy <= 0 or dx <= 0:
        return (18, 14)

    aspect = dx / dy
    if aspect >= 1.0:
        width = min(max_dim, max(min_dim, 16 * aspect))
        height = min(max_dim, max(min_dim, 12))
    else:
        width = min(max_dim, max(min_dim, 14))
        height = min(max_dim, max(min_dim, 14 / aspect))
    return (width, height)


def _wrap_longitudes_to_center(l_values, center):
    """Normalize longitudes to the branch around a reference center value."""
    l_values = np.array(l_values, dtype=float)
    return center + ((l_values - center + 180.0) % 360.0) - 180.0


def _wrap_longitude_array(l_values):
    """Wrap longitude values to a consistent branch for plotting."""
    l_values = np.array(l_values, dtype=float)
    center = (np.degrees(np.arctan2(
        np.mean(np.sin(np.radians(l_values))),
        np.mean(np.cos(np.radians(l_values)))
    )) % 360.0)
    return _wrap_longitudes_to_center(l_values, center)


def _compute_axes_limits_xy(x, y, pad_deg=3.0):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return ((x.max() + pad_deg, x.min() - pad_deg),
            (y.min() - pad_deg, y.max() + pad_deg))


_DIR_TO_OFFSET = {
    'NE': ( 2,  1), 'NW': (-4,  1), 'SE': ( 2, -1), 'SW': (-4, -1),
    'E':  ( 2,  0), 'W':  (-4,  0), 'N':  ( 0,  2), 'S':  ( 0, -2),
}


def _resolve_label_positions(l, b, ids, logz, names, offset_pts=20, min_sep_pts=15,
                             xlim=None, ylim=None, figsize=None, overrides=None):
    """
    Return per-cluster (dx, dy) offsets in points that avoid collisions and
    keep labels inside the axes bounds.

    overrides: dict {cluster_id: direction_str} where direction_str is one of
               'NE', 'NW', 'SE', 'SW', 'E', 'W', 'N', 'S'.  Overridden
               clusters skip the algorithm and always use the given direction.

    Converts point offsets to data units using figsize and axis limits so that
    the collision radius and OOB margin are physically meaningful.
    Scores each candidate: collision costs 2, out-of-bounds costs 1.
    Picks the lowest-scoring candidate; list order breaks ties so diagonals win.
    """
    candidates = [
        ( offset_pts,  offset_pts),
        (-offset_pts,  offset_pts),
        ( offset_pts, -offset_pts),
        (-offset_pts, -offset_pts),
        ( offset_pts*2,  0),
        (-offset_pts*2,  0),
        ( 0,  offset_pts*2),
        ( 0, -offset_pts*2),
    ]

    # Fraction of figure width/height actually occupied by the axes.
    axes_frac = 0.78

    if xlim is not None:
        xmin, xmax = min(xlim), max(xlim)
        ymin, ymax = min(ylim), max(ylim)
        x_range = xmax - xmin
        y_range = ymax - ymin
        if figsize is not None:
            scale_x = x_range / (figsize[0] * 72 * axes_frac)
            scale_y = y_range / (figsize[1] * 72 * axes_frac)
        else:
            ref = min(x_range, y_range) / 900.0
            scale_x = scale_y = ref
        # OOB margin: 1.5× offset to account for text extending past its anchor.
        margin_x = offset_pts * scale_x * 1.5
        margin_y = offset_pts * scale_y * 1.5
        xmin_eff = xmin + margin_x
        xmax_eff = xmax - margin_x
        ymin_eff = ymin + margin_y
        ymax_eff = ymax - margin_y
    else:
        scale_x = scale_y = 0.1
        xmin_eff = xmax_eff = ymin_eff = ymax_eff = None

    placed = []
    offsets = []

    for i in range(len(l)):
        cid = int(ids[i])
        if overrides and cid in overrides:
            sx, sy = _DIR_TO_OFFSET[overrides[cid]]
            best = (sx * offset_pts, sy * offset_pts)
            placed.append((l[i] + best[0] * scale_x, b[i] + best[1] * scale_y))
            offsets.append(best)
            continue

        best, best_score = candidates[0], float('inf')
        for dx, dy in candidates:
            cx = l[i] + dx * scale_x
            cy = b[i] + dy * scale_y
            collision = any(
                abs(cx - px) < min_sep_pts * scale_x and abs(cy - py) < min_sep_pts * scale_y
                for px, py in placed
            )
            oob = (xlim is not None and
                   (cx < xmin_eff or cx > xmax_eff or cy < ymin_eff or cy > ymax_eff))
            score = int(collision) * 2 + int(oob)
            if score < best_score:
                best, best_score = (dx, dy), score
                if score == 0:
                    break
        placed.append((l[i] + best[0] * scale_x, b[i] + best[1] * scale_y))
        offsets.append(best)
    return offsets


def plot_age_map_stars(members, results, args, cmd='BPRP'):
    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    age_lookup = {int(row['cluster_id']): float(row['age_map_myr']) for row in res}
    l_mean = np.array(res['l_mean'])
    b_mean = np.array(res['b_mean'])
    ids    = np.array(res['cluster_id'], dtype=int)
    logz   = np.array(res['log_evidence'])
    cluster_centers = {int(row['cluster_id']): float(row['l_mean']) for row in res}

    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, min(age_lookup.values()) - 2),
                             vmax=max(age_lookup.values()) + 2)

    star_coords = []
    all_l = []
    all_b = []
    for cid, age in age_lookup.items():
        mask = np.array(members['cluster_id']) == cid
        if mask.sum() == 0:
            continue
        lvals = np.array(members['l'])[mask]
        bvals = np.array(members['b'])[mask]
        lvals = _wrap_longitudes_to_center(lvals, cluster_centers[cid])
        star_coords.append((lvals, bvals, age))
        all_l.append(lvals)
        all_b.append(bvals)

    if len(all_l) > 0:
        all_l = np.concatenate(all_l)
        all_b = np.concatenate(all_b)
        plot_l = np.concatenate([all_l, l_mean])
        plot_b = np.concatenate([all_b, b_mean])
    else:
        plot_l = l_mean
        plot_b = b_mean

    xlim = (plot_l.max() + 3.0, plot_l.min() - 3.0)
    ylim = (plot_b.min() - 3.0, plot_b.max() + 3.0)
    figsize = _choose_figsize(xlim, ylim)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for lvals, bvals, age in star_coords:
        ax.scatter(lvals, bvals,
                   color=cmap(norm(age)),
                   s=3, alpha=0.5, zorder=3, rasterized=True)

    names  = [CLUSTER_NAMES.get(cid, '') for cid in ids]
    offsets = _resolve_label_positions(l_mean, b_mean, ids, logz, names,
                                       xlim=xlim, ylim=ylim, figsize=figsize,
                                       overrides=LABEL_DIR_STARS_LB)

    for i in range(len(res)):
        cid  = ids[i]
        name = names[i]
        dx, dy = offsets[i]
        ax.plot(l_mean[i], b_mean[i], '+',
                color='black', markersize=5, markeredgewidth=1.0, zorder=6)

        label = f'{cid}  logZ={logz[i]:.0f}'
        ax.annotate(label,
                    xy=(l_mean[i], b_mean[i]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'),
                    arrowprops=dict(arrowstyle='-', color='#888888',
                                    lw=0.6, shrinkA=0, shrinkB=2))
        if name:
            ax.annotate(name,
                        xy=(l_mean[i], b_mean[i]),
                        xytext=(dx, dy - 11), textcoords='offset points',
                        fontsize=8.5, fontstyle='italic', color='#1a6644', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Age (Myr)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('Galactic longitude $l$ (deg)', fontsize=12)
    ax.set_ylabel('Galactic latitude $b$ (deg)', fontsize=12)
    ax.set_title(rf'{name_complex} — Age map (stars)  (HDBSCAN $m_s$={ms_tag}, PARSEC-{cmd})',
                 fontsize=12, pad=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, linestyle='--', color='#aaaaaa')

    plt.tight_layout()
    out = path_results + f'agemap_stars_{catalog_name}{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Age map (stars) -> {out}")


def plot_age_map_bubbles(members, results, args, cmd='BPRP'):
    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    ages  = np.array(res['age_map_myr'])
    l_raw = np.array(res['l_mean'])
    b     = np.array(res['b_mean'])
    ids   = np.array(res['cluster_id'], dtype=int)
    logz  = np.array(res['log_evidence'])

    ref_l = (np.degrees(np.arctan2(
                np.mean(np.sin(np.radians(l_raw))),
                np.mean(np.cos(np.radians(l_raw))))) % 360.0)
    l = _wrap_longitudes_to_center(l_raw, ref_l)
    res_wrapped = res.copy()
    res_wrapped['l_mean'] = l
    xlim, ylim = _compute_axes_limits(res_wrapped, members=None)

    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, float(ages.min()) - 2),
                             vmax=float(ages.max()) + 2)
    colors = cmap(norm(ages))

    sqrt_n = np.sqrt(np.array(res['n_members'], dtype=float))
    denom  = sqrt_n.max()
    s_min, s_max = 50, 1000
    sizes  = (s_min + (s_max - s_min) * (sqrt_n / denom)
              if denom > 0 else np.full(len(sqrt_n), (s_min + s_max) / 2.0))
    ms = np.sqrt(sizes)

    order  = np.argsort(sizes)[::-1]
    names  = [CLUSTER_NAMES.get(cid, '') for cid in ids]
    figsize = _choose_figsize(xlim, ylim)
    offsets = _resolve_label_positions(l, b, ids, logz, names, offset_pts=20,
                                       xlim=xlim, ylim=ylim, figsize=figsize,
                                       overrides=LABEL_DIR_BUBBLES_LB)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i in order:
        ax.plot(l[i], b[i], 'o',
                color=colors[i], markersize=ms[i],
                alpha=0.75,
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=3)

    for i in range(len(res)):
        cid  = ids[i]
        name = names[i]
        dx, dy = offsets[i]

        label = f'{cid}  logZ={logz[i]:.0f}'
        ax.annotate(label,
                    xy=(l[i], b[i]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'),
                    arrowprops=dict(arrowstyle='-', color='#888888',
                                    lw=0.6, shrinkA=0, shrinkB=2))
        if name:
            ax.annotate(name,
                        xy=(l[i], b[i]),
                        xytext=(dx, dy - 11), textcoords='offset points',
                        fontsize=8.5, fontstyle='italic', color='#1a6644', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Age (Myr)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    legend_handles = []
    for ref_n, lbl in [(100, 'N=100'), (500, 'N=500')]:
        ref_s  = s_min + (s_max - s_min) * (np.sqrt(float(ref_n)) / denom)
        ref_ms = np.sqrt(np.clip(ref_s, s_min, s_max))
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#777777',
                   markersize=ref_ms, label=lbl))
    ax.legend(handles=legend_handles, fontsize=9, loc='best',
              framealpha=0.9, edgecolor='#aaaaaa',
              labelspacing=1.8, borderpad=1.2, handletextpad=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('Galactic longitude $l$ (deg)', fontsize=12)
    ax.set_ylabel('Galactic latitude $b$ (deg)', fontsize=12)
    ax.set_title(rf'{name_complex} — Age map (bubbles)  (HDBSCAN $m_s$={ms_tag}, PARSEC-{cmd})',
                 fontsize=12, pad=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, linestyle='--', color='#aaaaaa')

    plt.tight_layout()
    out = path_results + f'agemap_bubbles_{catalog_name}{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Age map (bubbles) -> {out}")


def plot_age_map_stars_radec(members, results, args, cmd='BPRP'):
    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    age_lookup = {int(row['cluster_id']): float(row['age_map_myr']) for row in res}
    ra_raw = np.array(res['ra_mean'])
    dec_mean = np.array(res['dec_mean'])
    ids = np.array(res['cluster_id'], dtype=int)
    logz = np.array(res['log_evidence'])
    ra_mean = _wrap_longitude_array(ra_raw)

    star_coords = []
    all_ra, all_dec = [], []
    for i, cid in enumerate(ids):
        mask = np.array(members['cluster_id'], dtype=int) == cid
        if mask.sum() == 0:
            continue
        ra_vals = _wrap_longitudes_to_center(np.array(members['ra'])[mask], ra_mean[i])
        dec_vals = np.array(members['dec'])[mask]
        star_coords.append((ra_vals, dec_vals, age_lookup[cid]))
        all_ra.append(ra_vals)
        all_dec.append(dec_vals)

    if len(all_ra) > 0:
        all_ra = np.concatenate(all_ra)
        all_dec = np.concatenate(all_dec)
        plot_ra = np.concatenate([all_ra, ra_mean])
        plot_dec = np.concatenate([all_dec, dec_mean])
    else:
        plot_ra = ra_mean
        plot_dec = dec_mean

    xlim, ylim = _compute_axes_limits_xy(plot_ra, plot_dec)
    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, min(age_lookup.values()) - 2),
                             vmax=max(age_lookup.values()) + 2)

    figsize = _choose_figsize(xlim, ylim)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for ra_vals, dec_vals, age in star_coords:
        ax.scatter(ra_vals, dec_vals,
                   color=cmap(norm(age)),
                   s=3, alpha=0.5, zorder=3, rasterized=True)

    names = [CLUSTER_NAMES.get(cid, '') for cid in ids]
    offsets = _resolve_label_positions(ra_mean, dec_mean, ids, logz, names,
                                       xlim=xlim, ylim=ylim, figsize=figsize,
                                       overrides=LABEL_DIR_STARS_RADEC)

    for i in range(len(res)):
        cid = ids[i]
        name = names[i]
        dx, dy = offsets[i]
        ax.plot(ra_mean[i], dec_mean[i], '+',
                color='black', markersize=5, markeredgewidth=1.0, zorder=6)
        label = f'{cid}  logZ={logz[i]:.0f}'
        ax.annotate(label,
                    xy=(ra_mean[i], dec_mean[i]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'),
                    arrowprops=dict(arrowstyle='-', color='#888888',
                                    lw=0.6, shrinkA=0, shrinkB=2))
        if name:
            ax.annotate(name,
                        xy=(ra_mean[i], dec_mean[i]),
                        xytext=(dx, dy - 11), textcoords='offset points',
                        fontsize=8.5, fontstyle='italic', color='#1a6644', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Age (Myr)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Right ascension $\alpha$ (deg)', fontsize=12)
    ax.set_ylabel(r'Declination $\delta$ (deg)', fontsize=12)
    ax.set_title(rf'{name_complex} — Age map (stars, RA/Dec)  (HDBSCAN $m_s$={ms_tag}, PARSEC-{cmd})',
                 fontsize=12, pad=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, linestyle='--', color='#aaaaaa')

    plt.tight_layout()
    out = path_results + f'agemap_stars_radec_{catalog_name}{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Age map (stars, RA/Dec) -> {out}")


def plot_age_map_bubbles_radec(results, args, cmd='BPRP'):
    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    ages = np.array(res['age_map_myr'])
    ra_raw = np.array(res['ra_mean'])
    dec = np.array(res['dec_mean'])
    ids = np.array(res['cluster_id'], dtype=int)
    logz = np.array(res['log_evidence'])
    ra = _wrap_longitude_array(ra_raw)

    xlim, ylim = _compute_axes_limits_xy(ra, dec)
    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, float(ages.min()) - 2),
                             vmax=float(ages.max()) + 2)
    colors = cmap(norm(ages))

    sqrt_n = np.sqrt(np.array(res['n_members'], dtype=float))
    denom = sqrt_n.max()
    s_min, s_max = 50, 1000
    sizes = (s_min + (s_max - s_min) * (sqrt_n / denom)
             if denom > 0 else np.full(len(sqrt_n), (s_min + s_max) / 2.0))
    ms = np.sqrt(sizes)

    order = np.argsort(sizes)[::-1]
    names = [CLUSTER_NAMES.get(cid, '') for cid in ids]
    figsize = _choose_figsize(xlim, ylim)
    offsets = _resolve_label_positions(ra, dec, ids, logz, names, offset_pts=20,
                                       xlim=xlim, ylim=ylim, figsize=figsize,
                                       overrides=LABEL_DIR_BUBBLES_RADEC)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i in order:
        ax.plot(ra[i], dec[i], 'o',
                color=colors[i], markersize=ms[i],
                alpha=0.75,
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=3)

    for i in range(len(res)):
        cid = ids[i]
        name = names[i]
        dx, dy = offsets[i]
        label = f'{cid}  logZ={logz[i]:.0f}'
        ax.annotate(label,
                    xy=(ra[i], dec[i]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'),
                    arrowprops=dict(arrowstyle='-', color='#888888',
                                    lw=0.6, shrinkA=0, shrinkB=2))
        if name:
            ax.annotate(name,
                        xy=(ra[i], dec[i]),
                        xytext=(dx, dy - 11), textcoords='offset points',
                        fontsize=8.5, fontstyle='italic', color='#1a6644', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=1, ec='none'))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Age (Myr)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    legend_handles = []
    for ref_n, lbl in [(100, 'N=100'), (500, 'N=500')]:
        ref_s = s_min + (s_max - s_min) * (np.sqrt(float(ref_n)) / denom)
        ref_ms = np.sqrt(np.clip(ref_s, s_min, s_max))
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='#777777',
                   markersize=ref_ms, label=lbl))
    ax.legend(handles=legend_handles, fontsize=9, loc='best',
              framealpha=0.9, edgecolor='#aaaaaa',
              labelspacing=1.8, borderpad=1.2, handletextpad=0.8)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'Right ascension $\alpha$ (deg)', fontsize=12)
    ax.set_ylabel(r'Declination $\delta$ (deg)', fontsize=12)
    ax.set_title(rf'{name_complex} — Age map (bubbles, RA/Dec)  (HDBSCAN $m_s$={ms_tag}, PARSEC-{cmd})',
                 fontsize=12, pad=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, linestyle='--', color='#aaaaaa')

    plt.tight_layout()
    out = path_results + f'agemap_bubbles_radec_{catalog_name}{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Age map (bubbles, RA/Dec) -> {out}")


def plot_age_map_stars_xyz(members, results, args, cmd='BPRP'):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping interactive XYZ plot (pip install plotly)")
        return

    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    age_lookup = {int(row['cluster_id']): float(row['age_map_myr']) for row in res}
    ids  = np.array(res['cluster_id'], dtype=int)
    logz = np.array(res['log_evidence'], dtype=float)

    star_coords = []
    cx_mean, cy_mean, cz_mean, cx_ids, cx_logz = [], [], [], [], []
    for i, cid in enumerate(ids):
        mask = np.array(members['cluster_id'], dtype=int) == cid
        if mask.sum() == 0:
            continue
        xs = np.array(members['X'], dtype=float)[mask]
        ys = np.array(members['Y'], dtype=float)[mask]
        zs = np.array(members['Z'], dtype=float)[mask]
        age = age_lookup[cid]
        star_coords.append((xs, ys, zs, age, cid))
        cx_mean.append(np.mean(xs))
        cy_mean.append(np.mean(ys))
        cz_mean.append(np.mean(zs))
        cx_ids.append(cid)
        cx_logz.append(float(logz[i]))

    age_min = max(0, min(age_lookup.values()) - 2)
    age_max = max(age_lookup.values()) + 2

    # --- stars trace (all clusters, one array, colored by age) ---
    flat_x, flat_y, flat_z, flat_age, flat_hover = [], [], [], [], []
    for xs, ys, zs, age, cid in star_coords:
        name = CLUSTER_NAMES.get(cid, '')
        hover = f'Cluster {cid}' + (f' — {name}' if name else '') + f'<br>Age: {age:.1f} Myr'
        flat_x.extend(xs); flat_y.extend(ys); flat_z.extend(zs)
        flat_age.extend([age] * len(xs))
        flat_hover.extend([hover] * len(xs))

    stars_trace = go.Scatter3d(
        x=np.array(flat_x, dtype=float), y=np.array(flat_y, dtype=float), z=np.array(flat_z, dtype=float),
        mode='markers',
        marker=dict(size=2, color=np.array(flat_age, dtype=float), colorscale='Plasma', reversescale=True,
                    cmin=age_min, cmax=age_max, opacity=0.5,
                    colorbar=dict(title='Age (Myr)', thickness=15)),
        text=flat_hover,
        hovertemplate='%{text}<extra></extra>',
        name='Stars',
    )

    # --- cluster center crosses with labels ---
    center_hover = []
    center_text  = []
    for i, cid in enumerate(cx_ids):
        name = CLUSTER_NAMES.get(cid, '')
        center_hover.append(
            f'<b>Cluster {cid}</b>' + (f' — {name}' if name else '') +
            f'<br>Age: {age_lookup[cid]:.1f} Myr<br>logZ: {cx_logz[i]:.0f}')
        center_text.append(str(cid) + (f'<br><i>{name}</i>' if name else ''))

    centers_trace = go.Scatter3d(
        x=np.array(cx_mean, dtype=float), y=np.array(cy_mean, dtype=float), z=np.array(cz_mean, dtype=float),
        mode='markers+text',
        marker=dict(size=6, symbol='cross', color='black'),
        text=center_text,
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=center_hover,
        name='Centers',
        showlegend=False,
    )

    fig = go.Figure(data=[stars_trace, centers_trace])
    fig.update_layout(
        title=f'{name_complex} — Age map (stars, XYZ)  (HDBSCAN ms={ms_tag}, PARSEC-{cmd})',
        scene=dict(
            xaxis_title='X (pc)', yaxis_title='Y (pc)', zaxis_title='Z (pc)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor='white',
        width=1400, height=900,
    )

    out_html = path_results + f'agemap_stars_xyz_{catalog_name}{cmd}.html'
    fig.write_html(out_html)
    print(f"Age map (stars, XYZ) -> {out_html}")

    try:
        out_png = path_results + f'agemap_stars_xyz_{catalog_name}{cmd}.png'
        fig.write_image(out_png, width=1800, height=1200)
        print(f"Age map (stars, XYZ) -> {out_png}")
    except Exception:
        print("  (kaleido not installed — PNG skipped; pip install kaleido)")


def plot_age_map_bubbles_xyz(members, results, args, cmd='BPRP'):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping interactive XYZ plot (pip install plotly)")
        return

    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    ages = np.array(res['age_map_myr'], dtype=float)
    ids  = np.array(res['cluster_id'], dtype=int)
    logz = np.array(res['log_evidence'], dtype=float)
    n_members = np.array(res['n_members'], dtype=float)

    x = np.array([np.mean(np.array(members['X'], dtype=float)[np.array(members['cluster_id'], dtype=int) == cid])
                  for cid in ids], dtype=float)
    y = np.array([np.mean(np.array(members['Y'], dtype=float)[np.array(members['cluster_id'], dtype=int) == cid])
                  for cid in ids], dtype=float)
    z = np.array([np.mean(np.array(members['Z'], dtype=float)[np.array(members['cluster_id'], dtype=int) == cid])
                  for cid in ids], dtype=float)

    age_min = max(0, float(ages.min()) - 2)
    age_max = float(ages.max()) + 2

    sqrt_n = np.sqrt(n_members)
    denom  = sqrt_n.max()
    ms_min, ms_max = 8, 40
    marker_sizes = (ms_min + (ms_max - ms_min) * (sqrt_n / denom)
                    if denom > 0 else np.full(len(sqrt_n), (ms_min + ms_max) / 2.0))

    hover_texts = []
    label_texts = []
    for i, cid in enumerate(ids):
        name = CLUSTER_NAMES.get(cid, '')
        hover_texts.append(
            f'<b>Cluster {cid}</b>' + (f' — {name}' if name else '') +
            f'<br>Age: {ages[i]:.1f} Myr<br>N: {int(n_members[i])}<br>logZ: {logz[i]:.0f}')
        label_texts.append(str(cid) + (f'<br><i>{name}</i>' if name else ''))

    bubbles_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        marker=dict(
            size=marker_sizes,
            color=ages, colorscale='Plasma', reversescale=True,
            cmin=age_min, cmax=age_max, opacity=0.75,
            line=dict(color='white', width=1),
            colorbar=dict(title='Age (Myr)', thickness=15),
        ),
        text=label_texts,
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
        name='Clusters',
    )

    fig = go.Figure(data=[bubbles_trace])
    fig.update_layout(
        title=f'{name_complex} — Age map (bubbles, XYZ)  (HDBSCAN ms={ms_tag}, PARSEC-{cmd})',
        scene=dict(
            xaxis_title='X (pc)', yaxis_title='Y (pc)', zaxis_title='Z (pc)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor='white',
        width=1400, height=900,
    )

    out_html = path_results + f'agemap_bubbles_xyz_{catalog_name}{cmd}.html'
    fig.write_html(out_html)
    print(f"Age map (bubbles, XYZ) -> {out_html}")

    try:
        out_png = path_results + f'agemap_bubbles_xyz_{catalog_name}{cmd}.png'
        fig.write_image(out_png, width=1800, height=1200)
        print(f"Age map (bubbles, XYZ) -> {out_png}")
    except Exception:
        print("  (kaleido not installed — PNG skipped; pip install kaleido)")


# ---------------------------------------------------------------------------
# Age vs Galactic longitude
# ---------------------------------------------------------------------------

def get_label(cid):
    """Return physical name if available, else cluster ID string."""
    name = CLUSTER_NAMES.get(int(cid), '')
    return name if name else f'{int(cid)}'


def plot_age_vs_l(res, args, cmd='BPRP'):
    ages  = np.array(res['age_map_myr'])
    lo    = np.array(res['age_lo_myr'])
    hi    = np.array(res['age_hi_myr'])
    l     = np.array(res['l_mean'])
    ids   = np.array(res['cluster_id'], dtype=int)
    dist  = np.array(res['dist_mean'])
    yerr  = np.vstack([np.maximum(0, ages - lo), np.maximum(0, hi - ages)])

    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, float(ages.min()) - 2),
                             vmax=float(ages.max()) + 2)
    colors = cmap(norm(ages))

    sqrt_n = np.sqrt(np.array(res['n_members'], dtype=float))
    denom  = sqrt_n.max()
    s_min, s_max = 50, 1000
    sizes  = (s_min + (s_max - s_min) * (sqrt_n / denom)
              if denom > 0 else np.full(len(sqrt_n), (s_min + s_max) / 2.0))
    ms = np.sqrt(sizes)

    # largest bubbles painted first, smallest on top
    order = np.argsort(sizes)[::-1]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i in order:
        ax.errorbar(l[i], ages[i],
                    yerr=[[yerr[0, i]], [yerr[1, i]]],
                    fmt='none',
                    capsize=3, capthick=1.2, elinewidth=1.0,
                    ecolor='#555555', zorder=4)
        ax.plot(l[i], ages[i], 'o',
                color=colors[i], markersize=ms[i],
                alpha=0.75,
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=3)
        label = get_label(ids[i])
        ax.annotate(label,
                    xy=(l[i], ages[i]),
                    xytext=(-14, 0), textcoords='offset points',
                    fontsize=5, ha='center', va='bottom',
                    color='green', zorder=5, fontweight='bold',
                    rotation=90, rotation_mode='anchor')

    has_cat = False
    if 'ref_age_myr' in res.colnames:
        ref_mask = np.isfinite(np.array(res['ref_age_myr']))
        if ref_mask.sum() > 0:
            has_cat = True
            r_ages = np.array(res['ref_age_myr'])[ref_mask]
            r_lo   = np.array(res['ref_age_lo'])[ref_mask]
            r_hi   = np.array(res['ref_age_hi'])[ref_mask]
            r_l    = l[ref_mask]
            r_ye   = np.vstack([np.maximum(0, r_ages - r_lo),
                                 np.maximum(0, r_hi - r_ages)])
            ax.errorbar(r_l, r_ages, yerr=r_ye,
                        fmt='^', color='red',
                        capsize=3, capthick=1, markersize=7,
                        linewidth=1.0, alpha=0.85,
                        label='Ratzenböck+2023', zorder=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label('Age (Myr)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    handles = []
    if has_cat:
        handles.append(Line2D([0], [0], marker='^', color='w',
                               markerfacecolor='red',
                               markersize=8, label='Catalog'))
    for ref_n, lbl in [(100, 'N=100'), (500, 'N=500')]:
        ref_s  = s_min + (s_max - s_min) * (np.sqrt(float(ref_n)) / denom)
        ref_ms = np.sqrt(np.clip(ref_s, s_min, s_max))
        handles.append(Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='#777777',
                               markersize=ref_ms,
                               label=lbl))
    ax.legend(handles=handles, fontsize=9, loc='best',
              framealpha=0.9, edgecolor='#aaaaaa',
              labelspacing=1.8, borderpad=1.2,
              handletextpad=0.8)
    ax.invert_xaxis()
    ax.set_xlabel('Galactic longitude $l$ (deg)', fontsize=12)
    ax.set_ylabel('Age (Myr)', fontsize=12)
    ax.set_title(f'Star formation history — {name_complex}  '
                 f'(PARSEC-{cmd})', fontsize=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.tight_layout()
    out = path_results + f'age_vs_l_{catalog_name}{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Age-dependent logZ threshold
# ---------------------------------------------------------------------------
def logz_threshold(age_myr):
    """
    Young clusters have intrinsically worse CMD fits due to PMS scatter,
    differential extinction and disk excess.
    """
    if age_myr < 10:
        return -20   # US-age: PMS scatter, very permissive
    elif age_myr < 20:
        return 0     # UCL/LCC-age: moderate
    elif age_myr < 35:
        return 5     # Her OB1-age: permissive — CMD scatter still present
    else:
        return 10    # background old population: strict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    bprp = load_results('BPRP')
    print(f"Loaded {len(bprp)} clusters (BPRP)")
    grp = None
    if args.cmd_compare:
        grp = load_results('GRP')
        print(f"Loaded {len(grp)} clusters (GRP)")

    # load per-star member table
    members = Table.read(path_hdbscan + f'hdbscan_clusters_{run_tag}.fits')
    members = members[members['cluster_id'] != -1]

    if args.cat is not None and os.path.exists(args.cat):
        print(f"Cross-matching with {args.cat} ...")
        cat  = Table.read(args.cat)
        global CLUSTER_NAMES
        CLUSTER_NAMES = build_cluster_names(cat, args.name_col, members)
        print(f"  Auto-named {len(CLUSTER_NAMES)} clusters:")
        for cid, name in sorted(CLUSTER_NAMES.items()):
            print(f"    Cluster {cid:3d} -> {name}")
        bprp = crossmatch_labels(bprp, cat, args.name_col)
        if grp is not None:
            grp = crossmatch_labels(grp, cat, args.name_col)
    elif args.cat is not None:
        print(f"Catalog not found: {args.cat} — skipping cross-match")

    bprp_good = apply_filters(bprp, args)
    print_summary(bprp_good, cmd='BPRP', compare=grp)
    plot_age_vs_l(bprp_good, args, cmd='BPRP')
    plot_age_map_stars(members, bprp, args, cmd='BPRP')
    plot_age_map_bubbles(members, bprp, args, cmd='BPRP')
    plot_age_map_stars_radec(members, bprp, args, cmd='BPRP')
    plot_age_map_bubbles_radec(bprp, args, cmd='BPRP')
    plot_age_map_stars_xyz(members, bprp, args, cmd='BPRP')
    plot_age_map_bubbles_xyz(members, bprp, args, cmd='BPRP')


if __name__ == '__main__':
    main()
