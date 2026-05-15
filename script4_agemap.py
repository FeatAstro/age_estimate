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
from astropy.table import Table, join

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
name_complex = 'Orion_OB1'
sky_tag      = 'ra75_90_dec-14_16'
ms_tag       = '37'
mc_tag       = '15'
cv_tag       = '6'

run_tag      = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_ages    = f'outputs/{run_tag}/ages/'
path_hdbscan = 'outputs/hdbscan/'
path_results = f'outputs/{run_tag}/agemap/'

os.makedirs(path_results, exist_ok=True)

# Kerr+2024 reference ages per subgroup (PARSEC-BPRP, Table 3)
# format: 'ASSOC_ID': (age, age_lo, age_hi)
# age_lo = age - e_Age, age_hi = age + e_Age
# '>' flag: treated as lower limit, age_hi set to 99.9

# Orion OB1 reference ages — not available, labels only
# fill from Briceño+2019 or Sanchez-Sanjuan (2024) when available
CAT_AGES = {
    1:  (np.nan, np.nan, np.nan),   # lambda Ori
    2:  (np.nan, np.nan, np.nan),   # Ori-North
    3:  (np.nan, np.nan, np.nan),   # Briceno-1A
    4:  (np.nan, np.nan, np.nan),   # Briceno-1B
    5:  (np.nan, np.nan, np.nan),   # Ori-East
    6:  (np.nan, np.nan, np.nan),   # OBP-Far
    7:  (np.nan, np.nan, np.nan),   # sigma Ori
    8:  (np.nan, np.nan, np.nan),   # OBP-b
    9:  (np.nan, np.nan, np.nan),   # OBP-d
    10: (np.nan, np.nan, np.nan),   # OBP-Near
    11: (np.nan, np.nan, np.nan),   # ONC
    12: (np.nan, np.nan, np.nan),   # Ori-South
    13: (np.nan, np.nan, np.nan),   # Orion Y
}

# Full subgroup name mapping
CAT_NAMES = {
    1:  'lambda Ori',   2:  'Ori-North',
    3:  'Briceno-1A',   4:  'Briceno-1B',
    5:  'Ori-East',     6:  'OBP-Far',
    7:  'sigma Ori',    8:  'OBP-b',
    9:  'OBP-d',        10: 'OBP-Near',
    11: 'ONC',          12: 'Ori-South',
    13: 'Orion Y',
}

# Cluster name mapping — built automatically from crossmatch if --cat is given,
# otherwise stays empty (cluster IDs are used as labels).
# You can also override manually: CLUSTER_NAMES = {31: 'US', 27: 'LCC', ...}
CLUSTER_NAMES = {}


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
        best_raw = vals[np.argmax(counts)]
        # try int lookup first, fall back to string
        try:
            best_key = int(best_raw)
        except (ValueError, TypeError):
            best_key = str(best_raw).strip()
        best_raw = vals[np.argmax(counts)]
        # try int lookup first, fall back to string
        try:
            best_key = int(best_raw)
        except (ValueError, TypeError):
            best_key = str(best_raw).strip()
        best_str = str(CAT_NAMES.get(best_key, best_raw)).strip()
        best_frac    = counts.max() / n_total

        if best_frac >= 0.30:
            names[int(cid)] = CAT_NAMES.get(best_str, best_str)

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
        mask &= np.array(results['age_map_myr']) <= args.age_max
    if args.l_min is not None:
        mask &= np.array(results['l_mean']) >= args.l_min
    if args.l_max is not None:
        mask &= np.array(results['l_mean']) <= args.l_max
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
			best_count  = counts.max()
			
			cat_subgroup_mask  = np.array(cat_dedup[name_col]).astype(str) == best_str
			n_subgroup_total   = cat_subgroup_mask.sum()
			frac_hdbscan     = best_count / n_tot * 100      # % of your cluster in subgroup
			frac_catalog     = best_count / n_subgroup_total * 100 if not np.isnan(n_subgroup_total) else np.nan
			ref_labels.append(
				f'{best_str} '
				f'({frac_hdbscan:.0f}% of HDBSCAN | '
				f'{frac_catalog:.0f}% of catalog)'
			)
			
			cat_ages = CAT_AGES.get(best_str, (np.nan, np.nan, np.nan))
			ref_ages.append(cat_ages[0])
			ref_ages_lo.append(cat_ages[1])
			ref_ages_hi.append(cat_ages[2])

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
    csv_path = path_results + f'summary_{cmd}.csv'
    hdr = 'cluster_id,n_members,l_mean,b_mean,dist_mean,age_map_myr,age_lo_myr,age_hi_myr,Av_map,log_evidence'
    if has_ref:
        hdr += ',ref_label,ref_age_myr,age_offset_myr'
    with open(csv_path, 'w') as f:
        f.write(hdr + '\n')
        for r in csv_rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print(f"\nCSV -> {csv_path}")



# ---------------------------------------------------------------------------
# Age map — white background, deep colormap, clean bubbles
# ---------------------------------------------------------------------------
def plot_age_map_stars(members, results, args, cmd='BPRP'):
    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    age_lookup = {int(row['cluster_id']): row['age_map_myr'] for row in res}
    l_mean = np.array(res['l_mean'])
    b_mean = np.array(res['b_mean'])
    ids    = np.array(res['cluster_id'], dtype=int)

    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=max(0, min(age_lookup.values()) - 2),
                             vmax=max(age_lookup.values()) + 2)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for cid, age in age_lookup.items():
        mask = np.array(members['cluster_id']) == cid
        if mask.sum() == 0:
            continue
        ax.scatter(np.array(members['l'])[mask],
                   np.array(members['b'])[mask],
                   color=cmap(norm(age)),
                   s=3, alpha=0.5, zorder=3, rasterized=True)

    for i in range(len(res)):
        cid  = ids[i]
        name = CLUSTER_NAMES.get(cid, '')
        ax.plot(l_mean[i], b_mean[i], '+',
                color='black', markersize=5, markeredgewidth=1.0, zorder=6)
        ax.annotate(f'{cid}',
                    xy=(l_mean[i], b_mean[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, fontweight='bold', color='black', zorder=7,
                    #bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none')
                    )
        if name:
            ax.annotate(name,
                        xy=(l_mean[i], b_mean[i]),
                        xytext=(5, -8), textcoords='offset points',
                        fontsize=6, fontstyle='italic', color='#1a6644', zorder=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Age (Myr)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('Galactic longitude $l$ (deg)', fontsize=12)
    ax.set_ylabel('Galactic latitude $b$ (deg)', fontsize=12)
    ax.set_title(rf'{name_complex} — Age map (stars)  (HDBSCAN $m_s$={ms_tag}, PARSEC-{cmd})',
                 fontsize=12, pad=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, linestyle='--', color='#aaaaaa')

    plt.tight_layout()
    out = path_results + f'agemap_stars_{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Age map (stars) -> {out}")


def plot_age_map_bubbles(results, args, cmd='BPRP'):
    res = apply_filters(results, args)
    if len(res) == 0:
        print("No clusters pass filters for age map.")
        return

    ages  = np.array(res['age_map_myr'])
    l     = np.array(res['l_mean'])
    b     = np.array(res['b_mean'])
    ids   = np.array(res['cluster_id'], dtype=int)

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

    order = np.argsort(sizes)[::-1]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i in order:
        ax.plot(l[i], b[i], 'o',
                color=colors[i], markersize=ms[i],
                alpha=0.75,
                markeredgecolor='white', markeredgewidth=1.2,
                zorder=3)
        cid  = ids[i]
        name = CLUSTER_NAMES.get(cid, '')
        ax.annotate(f'{cid}',
                    xy=(l[i], b[i]),
                    xytext=(0, 0), textcoords='offset points',
                    fontsize=7, fontweight='bold', color='black',
                    ha='center', va='center', zorder=5)
        if name:
            ax.annotate(name,
                        xy=(l[i], b[i]),
                        xytext=(0, -ms[i]*0.6), textcoords='offset points',
                        fontsize=6, fontstyle='italic', color='#1a6644',
                        ha='center', va='top', zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label('Age (Myr)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # size legend
    from matplotlib.lines import Line2D
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

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('Galactic longitude $l$ (deg)', fontsize=12)
    ax.set_ylabel('Galactic latitude $b$ (deg)', fontsize=12)
    ax.set_title(rf'{name_complex} — Age map (bubbles)  (HDBSCAN $m_s$={ms_tag}, PARSEC-{cmd})',
                 fontsize=12, pad=12)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.3, linestyle='--', color='#aaaaaa')

    plt.tight_layout()
    out = path_results + f'agemap_bubbles_{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Age map (bubbles) -> {out}")
    
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
    plot_age_map_stars(members, bprp, args, cmd='BPRP')
    plot_age_map_bubbles(bprp, args, cmd='BPRP')


if __name__ == '__main__':
    main()
