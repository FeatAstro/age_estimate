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
name_complex = 'Her_OB1'
sky_tag      = 'ra264_332_dec13_64'
ms_tag       = '70'
mc_tag       = '20'
cv_tag       = '8'

run_tag      = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_ages    = f'outputs/{run_tag}/ages/'
path_hdbscan = 'outputs/hdbscan/'
path_im      = f'outputs/{run_tag}/sfh/'

os.makedirs(path_im, exist_ok=True)

# Kerr+2024 reference ages per subgroup (PARSEC-BPRP, Table 3)
# format: 'ASSOC_ID': (age, age_lo, age_hi)
# age_lo = age - e_Age, age_hi = age + e_Age
# '>' flag: treated as lower limit, age_hi set to 99.9
CAT_AGES = {
    'ORPH_1' : (27.9, 25.5, 30.3),   'ORPH_2' : (26.5, 23.7, 29.3),
    'ORPH_3' : (39.0, 37.2, 40.8),   'ORPH_4' : (34.5, 32.0, 37.0),
    'ORPH_5' : (30.0, 28.1, 31.9),   'ORPH_6' : (33.7, 29.9, 37.5),
    'ORPH_7' : (39.8, 36.5, 43.1),   'ORPH_8' : (27.1, 25.8, 28.4),
    'ORPH_9' : (32.8, 30.0, 35.6),   'ORPH_10': (24.9, 23.0, 26.8),
    'ORPH_11': (32.4, 30.4, 34.4),   'ORPH_12': (27.7, 26.1, 29.3),
    'ORPH_13': (26.5, 22.7, 30.3),   'ORPH_14': (26.0, 23.7, 28.3),
    'ORPH_15': (26.5, 24.4, 28.6),   'ORPH_16': (34.5, 32.9, 36.1),
    'ORPH_17': (27.9, 25.1, 30.7),   'ORPH_18': (30.0, 28.6, 31.4),
    'ORPH_19': (27.3, 26.0, 28.6),   'ORPH_20': (32.6, 30.6, 34.6),
    'ORPH_21': (31.4, 29.5, 33.3),   'ORPH_22': (31.7, 29.0, 34.4),
    'ORPH_23': (30.1, 28.5, 31.7),   'ORPH_24': (27.4, 25.5, 29.3),
    'ORPH_25': (34.9, 32.3, 37.5),   'ORPH_26': (28.0, 25.3, 30.7),
    'ORPH_27': (28.7, 26.9, 30.5),   'ORPH_28': (32.1, 30.9, 33.3),
    'CINR_1' : (27.9, 26.0, 29.8),   'CINR_2' : (37.1, 32.6, 41.6),
    'CINR_3' : (29.3, 27.4, 31.2),   'CINR_4' : (38.5, 33.7, 43.3),
    'CINR_5' : (31.4, 30.0, 32.8),   'CINR_6' : (29.9, 28.3, 31.5),
    'CINR_7' : (32.9, 31.7, 34.1),   'CINR_8' : (39.3, 35.1, 43.5),
    'CINR_9' : (43.1, 39.1, 47.1),   'CINR_10': (35.7, 34.2, 37.2),
    'CUPV_1' : (68.2, 59.0, 77.4),   'CUPV_2' : (62.6, 50.8, 74.4),
    'CUPV_3' : (55.3, 53.0, 57.6),   'CUPV_4' : (68.4, 59.4, 77.4),
    'CUPV_5' : (71.0, 64.5, 77.5),   'CUPV_6' : (68.1, 61.4, 74.8),
    'CUPV_7' : (80.0, 80.0, 99.9),   'CUPV_8' : (55.5, 51.4, 59.6),
    'CUPV_9' : (53.8, 49.1, 58.5),   'CUPV_10': (58.4, 55.4, 61.4),
    'ROS6_1' : (80.0, 80.0, 99.9),   'ROS6_2' : (71.2, 64.8, 77.6),
}

# Full subgroup name mapping
CAT_NAMES = {
    'ORPH_1' : 'Orpheus 1',    'ORPH_2' : 'Orpheus 2',
    'ORPH_3' : 'Orpheus 3',    'ORPH_4' : 'Orpheus 4',
    'ORPH_5' : 'Orpheus 5',    'ORPH_6' : 'Orpheus 6',
    'ORPH_7' : 'Orpheus 7',    'ORPH_8' : 'Orpheus 8',
    'ORPH_9' : 'Orpheus 9',    'ORPH_10': 'Orpheus 10',
    'ORPH_11': 'Orpheus 11',   'ORPH_12': 'Orpheus 12',
    'ORPH_13': 'Orpheus 13',   'ORPH_14': 'Orpheus 14',
    'ORPH_15': 'Orpheus 15',   'ORPH_16': 'Orpheus 16',
    'ORPH_17': 'Orpheus 17',   'ORPH_18': 'Orpheus 18',
    'ORPH_19': 'Orpheus 19',   'ORPH_20': 'Orpheus 20',
    'ORPH_21': 'Orpheus 21',   'ORPH_22': 'Orpheus 22',
    'ORPH_23': 'Orpheus 23',   'ORPH_24': 'Orpheus 24',
    'ORPH_25': 'Orpheus 25',   'ORPH_26': 'Orpheus 26',
    'ORPH_27': 'Orpheus 27',   'ORPH_28': 'Orpheus 28',
    'CINR_1' : 'Cinyras 1',    'CINR_2' : 'Cinyras 2',
    'CINR_3' : 'Cinyras 3',    'CINR_4' : 'Cinyras 4',
    'CINR_5' : 'Cinyras 5',    'CINR_6' : 'Cinyras 6',
    'CINR_7' : 'Cinyras 7',    'CINR_8' : 'Cinyras 8',
    'CINR_9' : 'Cinyras 9',    'CINR_10': 'Cinyras 10',
    'CUPV_1' : 'Cupavo 1',     'CUPV_2' : 'Cupavo 2',
    'CUPV_3' : 'Cupavo 3',     'CUPV_4' : 'Cupavo 4',
    'CUPV_5' : 'Cupavo 5',     'CUPV_6' : 'Cupavo 6',
    'CUPV_7' : 'Cupavo 7',     'CUPV_8' : 'Cupavo 8',
    'CUPV_9' : 'Cupavo 9',     'CUPV_10': 'Cupavo 10',
    'ROS6_1' : 'Roslund 6-1',  'ROS6_2' : 'Roslund 6-2',
}

CLUSTER_NAMES = {}

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
def crossmatch_labels(results, cat, name_col):
	hdb_file = path_hdbscan + f'hdbscan_clusters_{run_tag}.fits'
	hdb = Table.read(hdb_file)

	cat_arr = np.array(cat['source_id'])
	_, keep = np.unique(cat_arr, return_index=True)
	cat_dedup = cat[keep]

	matched = join(hdb, cat_dedup[['source_id', name_col]],
			       keys='source_id', join_type='inner')

	ref_labels, ref_ages, ref_ages_lo, ref_ages_hi = [], [], [], []

	for row in results:
		k     = int(row['cluster_id'])
		mask  = np.array(matched['cluster_id']) == k
		n_tot = int(row['n_members'])

		if mask.sum() < 3:
			ref_labels.append('NEW')
			ref_ages.append(np.nan)
			ref_ages_lo.append(np.nan)
			ref_ages_hi.append(np.nan)
		else:
			names, counts = np.unique(
				np.array(matched[name_col])[mask], return_counts=True)
			best_raw = names[np.argmax(counts)]
			best_str = best_raw.decode().strip() if isinstance(best_raw, bytes) else str(best_raw).strip()
			frac     = counts.max() / n_tot * 100
			ref_labels.append(f'{best_str} ({frac:.0f}%)')
			cat_ages = CAT_AGES.get(best_str, (np.nan, np.nan, np.nan))
			ref_ages.append(cat_ages[0])
			ref_ages_lo.append(cat_ages[1])
			ref_ages_hi.append(cat_ages[2])

	results['ref_label']   = ref_labels
	results['ref_age_myr'] = np.array(ref_ages,    dtype=float)
	results['ref_age_lo']  = np.array(ref_ages_lo, dtype=float)
	results['ref_age_hi']  = np.array(ref_ages_hi, dtype=float)
	return results


def build_cluster_names(cat, name_col, hdb):
    from astropy.table import join as astropy_join

    # fix source_id type mismatch
    cat = cat.copy()
    hdb = hdb.copy()
    cat['source_id'] = np.array(cat['source_id']).astype(str)
    hdb['source_id'] = np.array(hdb['source_id']).astype(str)

    cat_arr = np.array(cat['source_id'])
    _, keep = np.unique(cat_arr, return_index=True)
    cat_dedup = cat[keep]

    try:
        matched = astropy_join(hdb, cat_dedup[['source_id', name_col]],
                               keys='source_id', join_type='inner')
    except Exception as e:
        print(f"  Warning: crossmatch join failed ({e})")
        return {}

    names = {}
    for cid in np.unique(np.array(hdb['cluster_id'])):
        hdb_mask   = np.array(hdb['cluster_id']) == cid
        n_total    = hdb_mask.sum()
        match_mask = np.array(matched['cluster_id']) == cid
        if match_mask.sum() < 3:
            continue

        str_ids      = np.array(matched[name_col])[match_mask].astype(str)
        vals, counts = np.unique(str_ids, return_counts=True)
        best_str     = vals[np.argmax(counts)]
        best_frac    = counts.max() / n_total

        if best_frac >= 0.30:
            names[int(cid)] = CAT_NAMES.get(best_str, best_str)

    return names


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

    # Ratzenböck reference
    has_ratz = False
    if 'ref_age_myr' in res.colnames:
        ref_mask = np.isfinite(np.array(res['ref_age_myr']))
        if ref_mask.sum() > 0:
            has_ratz = True
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
    if has_ratz:
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
    ids  = np.array(res['cluster_id'], dtype=int)
    yerr = np.vstack([np.maximum(0, ages - lo), np.maximum(0, hi - ages)])

    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, float(ages.min()) - 2),
                             vmax=float(ages.max()) + 2)
    colors = cmap(norm(ages))
    
    sqrt_n = np.sqrt(np.array(res['n_members'], dtype=float))
    denom  = sqrt_n.max()
    s_min, s_max = 40, 300
    sizes  = (s_min + (s_max - s_min) * (sqrt_n / denom)
              if denom > 0 else np.full(len(sqrt_n), (s_min + s_max) / 2.0))
    ms = np.sqrt(sizes)  # markersize = sqrt(area)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(res)):
        ax.errorbar(dist[i], ages[i],
                    yerr=[[yerr[0, i]], [yerr[1, i]]],
                    fmt='none',
                    capsize=3, capthick=1.2, elinewidth=1.0,
                    ecolor='#555555', zorder=4)
        ax.plot(dist[i], ages[i], 'o',
            color=colors[i], markersize=ms[i]*0.75,
            zorder=3)
        label = get_label(ids[i])
        ax.annotate(label,
                    xy=(dist[i], ages[i]),
                    xytext=(-10, 0), textcoords='offset points',
                    fontsize=6, ha='center', va='bottom',
                    color='green', zorder=5,
                    rotation=90, rotation_mode='anchor')

    # Sun marker
    ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.scatter([0], [0], marker='*', s=200, c='gold',
               edgecolors='k', linewidths=0.8, zorder=6, label='Sun')

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label('Age (Myr)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel('Distance (pc)', fontsize=12)
    ax.set_ylabel('Age (Myr)', fontsize=12)
    ax.set_title(f'Age vs distance — {name_complex}  '
                 f'(PARSEC-{cmd})', fontsize=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.2, linestyle='--')
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
