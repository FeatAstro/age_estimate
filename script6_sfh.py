#!/usr/bin/env python3
"""
Script 6 — Star formation history of the local Milky Way
=========================================================
Produces three complementary views of the star formation history:

  Fig 1 — Age vs Galactic longitude (l): shows the spatial age gradient
           across the complex on the sky, with Ratzenböck+2023 comparison.

  Fig 2 — Age vs distance: shows whether star formation propagated
           radially outward or inward from the Sun.

  Fig 3 — Age timeline: horizontal bar chart ordered by distance,
           one bar per cluster showing age ± 68% HDI. Like a Gantt
           chart of star formation events.
     
Usage
-----
    python3 script6_sfh.py
    python3 script6_sfh.py --logz-age
    python3 script6_sfh.py --logz-min 10 --age-max 25
    python3 script6_sfh.py --cat data/crossmatch/ScoCenGroups_Ratzenbock2023_crossmatch.fits \
                           --name-col SigMA
                                 
Input
-----
    outputs/<run_tag>/ages/ages_BPRP.fits
    outputs/hdbscan/hdbscan_clusters_<run_tag>.fits

Output
------
    outputs/<run_tag>/sfh/sfh_age_vs_l.png
    outputs/<run_tag>/sfh/sfh_age_vs_dist.png
    outputs/<run_tag>/sfh/sfh_timeline.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from astropy.table import Table, join

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
name_complex = 'Sco_Cen'
sky_tag      = 'ra100_300_dec-90_0' # choose
ms_tag       = '80'
mc_tag       = '20'
cv_tag       = '10' 

run_tag      = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_ages    = f'outputs/{run_tag}/ages/'
path_hdbscan = 'outputs/hdbscan/'
path_im      = f'outputs/{run_tag}/sfh/'

os.makedirs(path_im, exist_ok=True)

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

# # Full subgroup name mapping
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


# Cluster name mapping — built automatically from crossmatch if --cat is given,
# otherwise stays empty (cluster IDs are used as labels).
CLUSTER_NAMES = {}

def _resolve_cat_key(raw):
    """Convert a raw catalog value to the key type used in CAT_AGES/CAT_NAMES."""
    s = raw.decode().strip() if isinstance(raw, bytes) else str(raw).strip()
    try:
        return int(s), s
    except (ValueError, TypeError):
        return s, s


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
        vals, counts = np.unique(np.array(matched[name_col])[match_mask], return_counts=True)
        best_raw     = vals[np.argmax(counts)]
        best_key, best_str = _resolve_cat_key(best_raw)
        if counts.max() / n_total >= min_frac:
            names[int(cid)] = CAT_NAMES.get(best_key, best_str)
    return names	

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
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--cat', default=None,
                   help='Catalog FITS for crossmatch labels')
    p.add_argument('--name-col', default='SigMA')
    p.add_argument('--logz-min', type=float, default=10.0)
    p.add_argument('--logz-age', action='store_true',
                   help='Use age-dependent logZ threshold')
    p.add_argument('--age-max', type=float, default=None)
    p.add_argument('--cmd', choices=['BPRP', 'GRP'], default='BPRP')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------
def load_results(cmd='BPRP'):
    path = path_ages + f'ages_{cmd}.fits'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    return Table.read(path)


def apply_filters(results, args):
    mask = np.ones(len(results), dtype=bool)
    if args.logz_age:
        mask &= np.array([
            float(row['log_evidence']) > logz_threshold(float(row['age_map_myr']))
            for row in results
        ])
    else:
        mask &= np.array(results['log_evidence']) > args.logz_min
    if args.age_max is not None:
        mask &= np.array(results['age_map_myr']) <= args.age_max
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

        vals, counts = np.unique(np.array(matched[name_col])[mask], return_counts=True)
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
# Helpers
# ---------------------------------------------------------------------------
def get_label(cid, ids_arr=None):
    """Return physical name if available, else cluster ID string."""
    name = CLUSTER_NAMES.get(int(cid), '')
    return name if name else f'{int(cid)}'


def age_cmap_norm(ages, pad=2):
    age_min = max(0, float(ages.min()) - pad)
    age_max = float(ages.max()) + pad
    return plt.cm.RdBu_r, mcolors.Normalize(vmin=age_min, vmax=age_max)


# ---------------------------------------------------------------------------
# Fig 1 — Age vs Galactic longitude
# ---------------------------------------------------------------------------
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
    out = path_im + f'sfh_age_vs_l_{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 2 — Age vs distance
# ---------------------------------------------------------------------------
def plot_age_vs_dist(res, args, cmd='BPRP'):
    ages = np.array(res['age_map_myr'])
    lo   = np.array(res['age_lo_myr'])
    hi   = np.array(res['age_hi_myr'])
    dist = np.array(res['dist_mean'])
    l    = np.array(res['l_mean'])
    ids  = np.array(res['cluster_id'], dtype=int)
    yerr = np.vstack([np.maximum(0, ages - lo), np.maximum(0, hi - ages)])

    # heliocentric XY in the Galactic plane
    l_rad = np.deg2rad(l)
    X = dist * np.cos(l_rad)   # toward Galactic centre
    Y = dist * np.sin(l_rad)   # toward l=90

    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, float(ages.min()) - 2),
                             vmax=float(ages.max()) + 2)
    colors = cmap(norm(ages))

    sqrt_n = np.sqrt(np.array(res['n_members'], dtype=float))
    denom  = sqrt_n.max()
    s_min, s_max = 40, 300
    sizes  = (s_min + (s_max - s_min) * (sqrt_n / denom)
              if denom > 0 else np.full(len(sqrt_n), (s_min + s_max) / 2.0))

    fig, ax = plt.subplots(figsize=(10, 9))

    sc = ax.scatter(X, Y, c=ages, cmap=cmap, norm=norm,
                    s=sizes, alpha=0.85,
                    edgecolors='white', linewidths=0.8, zorder=3)

    for i in range(len(res)):
        label = get_label(ids[i])
        ax.annotate(label,
                    xy=(X[i], Y[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=6, color='green', zorder=5, fontweight='bold')

    # Sun at origin
    ax.scatter([0], [0], marker='*', s=300, c='gold',
               edgecolors='k', linewidths=0.8, zorder=6, label='Sun')
    ax.annotate('Sun', xy=(0, 0), xytext=(6, 6),
                textcoords='offset points', fontsize=8, color='goldenrod')

    # Galactic centre direction
    ax.annotate('', xy=(max(np.abs(X)) * 0.15, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(max(np.abs(X)) * 0.17, 0, 'GC', fontsize=8,
            color='gray', va='center')

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Age (Myr)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_aspect('equal')
    ax.set_xlabel('X (pc)  [toward Galactic centre]', fontsize=12)
    ax.set_ylabel('Y (pc)  [toward $l=90°$]', fontsize=12)
    ax.set_title(f'Heliocentric map — {name_complex}  (PARSEC-{cmd})', fontsize=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.invert_xaxis()   # Galactic convention: l increases to the left
    ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = path_im + f'sfh_age_vs_dist_{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 3 — Age timeline (Gantt-style)
# ---------------------------------------------------------------------------
def plot_timeline(res, args, cmd='BPRP'):
    ages  = np.array(res['age_map_myr'])
    lo    = np.array(res['age_lo_myr'])
    hi    = np.array(res['age_hi_myr'])
    dist  = np.array(res['dist_mean'])
    ids   = np.array(res['cluster_id'], dtype=int)
    n_mem = np.array(res['n_members'])

    # sort by distance (near to far)
    sort_idx = np.argsort(dist)
    ages  = ages[sort_idx]
    lo    = lo[sort_idx]
    hi    = hi[sort_idx]
    dist  = dist[sort_idx]
    ids   = ids[sort_idx]
    n_mem = n_mem[sort_idx]

    labels = [f"{get_label(cid)}  ({d:.0f} pc)"
                for cid, d in zip(ids, dist)]

    # plasma_r: dark purple (young) -> yellow (old), no white tones
    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, float(ages.min()) - 2),
                                vmax=float(ages.max()) + 2)
    colors = cmap(norm(ages))

    # bar thickness proportional to sqrt(N)
    bar_heights = 0.15 + 0.15 * (np.sqrt(n_mem) / np.sqrt(n_mem.max()))

    fig, ax = plt.subplots(figsize=(12, max(5, len(res) * 0.55 + 1.5)))

    for i, (age, age_lo, age_hi, col, bh) in enumerate(
            zip(ages, lo, hi, colors, bar_heights)):
            # HDI bar
        ax.barh(i, age_hi - age_lo, left=age_lo,
                height=bh, color=col, alpha=0.35,
                edgecolor='none', zorder=2)
        # horizontal line through center of bar
        ax.hlines(i, age_lo, age_hi,
                colors=col, linewidth=1.5, alpha=0.9, zorder=3)
        # MAP marker
        ax.plot(age, i, 'o', color=col, markersize=8,
                zorder=4)
        # age value text
        ax.text(age, i + bh / 2.0 + 0.08,
            f'{age:.1f} Myr',
            va='bottom', ha='center', fontsize=7.5,
            color='#333333', zorder=5)

    # Ratzenböck reference markers
    if 'ref_age_myr' in res.colnames:
        for i, row in enumerate(res[sort_idx]):
            ref_age = float(row['ref_age_myr'])
            ref_lo  = float(row['ref_age_lo'])
            ref_hi  = float(row['ref_age_hi'])
            if np.isfinite(ref_age):
                # error bar first (behind marker)
                ax.errorbar(ref_age, i,
                            xerr=[[ref_age - ref_lo], [ref_hi - ref_age]],
                            fmt='none',
                            ecolor='red', elinewidth=1.2,
                            capsize=4, capthick=1.2,
                            zorder=4)
                # triangle marker on top
                ax.plot(ref_age, i, '^', color='red',
                        markersize=7, zorder=5,
                        label='Ratzenböck+2023' if i == 0 else '')

    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Age (Myr)', fontsize=12)
    ax.set_title(f'Star formation timeline — {name_complex}  '
                    f'(ordered by distance, PARSEC-{cmd})',
                    fontsize=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label('Age (Myr)', fontsize=10)

    # legend
    handles = [
        Line2D([0], [0], marker='o', color='w',
                markerfacecolor='#888888',
                markersize=8, label='MAP age'),
        Line2D([0], [0], marker='s', color='#888888',
                alpha=0.35, markersize=12, label='68% HDI'),
    ]
    if 'ref_age_myr' in res.colnames:
        handles.append(
            Line2D([0], [0], marker='^', color='w',
                markerfacecolor='red',
                markersize=7, label='Catalog ± 68% HDI')
        )
    ax.legend(handles=handles, fontsize=8, loc='lower right',
                framealpha=0.9, edgecolor='#aaaaaa')

    plt.tight_layout()
    out = path_im + f'sfh_timeline_{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    results = load_results(args.cmd)
    print(f"Loaded {len(results)} clusters ({args.cmd})")

    if args.cat is not None and os.path.exists(args.cat):
        print(f"Cross-matching with {args.cat} ...")
        cat = Table.read(args.cat)
        hdb = Table.read(path_hdbscan + f'hdbscan_clusters_{run_tag}.fits')
        global CLUSTER_NAMES
        CLUSTER_NAMES = build_cluster_names(cat, args.name_col, hdb)
        print(f"  Auto-named {len(CLUSTER_NAMES)} clusters:")
        for cid, name in sorted(CLUSTER_NAMES.items()):
            print(f"    Cluster {cid:3d} -> {name}")
        results = crossmatch_labels(results, cat, args.name_col)
    elif args.cat is not None:
        print(f"Catalog not found: {args.cat} — skipping crossmatch")

    res = apply_filters(results, args)
    print(f"After filtering: {len(res)} clusters")

    if len(res) == 0:
        print("No clusters pass filters. Exiting.")
        return

    plot_age_vs_l(res, args, cmd=args.cmd)
    plot_age_vs_dist(res, args, cmd=args.cmd)
    plot_timeline(res, args, cmd=args.cmd)

    print(f"\nAll figures saved in {path_im}")


if __name__ == '__main__':
    main()