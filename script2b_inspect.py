#!/usr/bin/env python
"""
Script 2b — Visual inspection of HDBSCAN clusters
==================================================
For each cluster produces a 4-panel diagnostic figure:
  - CMD (G vs BP-RP)
  - Proper motion diagram (pml_corr vs pmb_corr)
  - Parallax histogram
  - Sky map (l vs b) with cluster highlighted

Usage
-----
  python3 script2b_inspect.py
  python3 script2b_inspect.py --clusters 0 1 5   # inspect specific clusters only
  
Input
-----
    outputs/hdbscan/hdbscan_clusters_<run_tag>.fits

Output
------
	outputs/<run_tag>/inspect_hdbscan/cluster_<cluster_id>_N<nb_stars>.png
    outputs/<run_tag>/inspect_hdbscan/cluster_summary_<run_tag>.fits
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.table import Table
from scipy.interpolate import interp1d

# ----------- Paths
name_complex = 'Orion_OB1'
sky_tag      = 'ra75_90_dec-14_16' # choose
ms_tag       = '37'
mc_tag       = '15'
cv_tag       = '6' 

run_tag      = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_out	 = f'outputs/hdbscan/'
path_im      = f'outputs/{run_tag}/inspect_hdbscan/'

os.makedirs(path_im, exist_ok=True)

# ----------- Args
parser = argparse.ArgumentParser()
parser.add_argument('--clusters', nargs='+', type=int, default=None,
                    help='Cluster IDs to inspect (default: all)')
args = parser.parse_args()

# ==============================================================================
# 0. Load PARSEC isochrone for CMD overlay (20 Myr, solar metallicity)
# ==============================================================================
import sys
sys.path.insert(0, '.')
try:
    from load_parsec import load_parsec, get_isochrone
    grid    = load_parsec('grids/parsec_solar_gaia.dat', label_max=3)
    iso_20  = get_isochrone(grid, log_age=7.3)   # 20 Myr
    # filter to MS+PMS only (label <= 3) to avoid giant branch fold
    label_mask    = np.array(iso_20['label']) <= 3
    iso_color_sorted = (np.array(iso_20['G_BPmag']) - np.array(iso_20['G_RPmag']))[label_mask]
    iso_mag_sorted   = np.array(iso_20['Gmag'])[label_mask]
    valid_iso = np.isfinite(iso_color_sorted) & np.isfinite(iso_mag_sorted)
    iso_color_sorted = iso_color_sorted[valid_iso]
    iso_mag_sorted   = iso_mag_sorted[valid_iso]
    HAS_ISO = True
    print("Loaded 20 Myr PARSEC isochrone for CMD overlay")
except Exception as e:
    HAS_ISO = False
    print(f"Could not load PARSEC isochrone ({e}), CMD overlay disabled")
 
# ==============================================================================
# 1. Load cluster catalog
# ==============================================================================
fits_path = path_out + f'hdbscan_clusters_{run_tag}.fits'
cat = Table.read(fits_path)
print(f"Loaded {len(cat)} members from {fits_path}")
 
cluster_ids = np.unique(cat['cluster_id'])
if args.clusters is not None:
    cluster_ids = np.array([c for c in args.clusters if c in cluster_ids])
    print(f"Inspecting {len(cluster_ids)} requested clusters: {cluster_ids}")
else:
    print(f"Inspecting all {len(cluster_ids)} clusters")
 
# ==============================================================================
# 2. Per-cluster diagnostics
# ==============================================================================
 
def sigma_clip_mask(x, nsigma=3, niter=3):
    mask = np.ones(len(x), dtype=bool)
    for _ in range(niter):
        med = np.median(x[mask])
        std = np.std(x[mask])
        mask = np.abs(x - med) < nsigma * std
    return mask
 
summary_rows = []
 
for cid in cluster_ids:
    mask = cat['cluster_id'] == cid
    c    = cat[mask]
    N    = len(c)
 
    G      = np.array(c['G_mag'])
    BP     = np.array(c['BP_mag'])
    RP     = np.array(c['RP_mag'])
    color  = BP - RP
    plx    = np.array(c['parallax_corrected'])
    pml    = np.array(c['pml_corr'])
    pmb    = np.array(c['pmb_corr'])
    l      = np.array(c['l'])
    b      = np.array(c['b'])
    dist   = np.array(c['distance'])
    prob   = np.array(c['probability'])
 
    # absolute magnitude using per-star distance (correct for extended associations)
    mean_dist = np.nanmedian(dist)
    M_G = G - 5 * np.log10(dist) + 5
 
    # kinematic dispersions
    plx_std  = np.std(plx)
    pml_std  = np.std(pml)
    pmb_std  = np.std(pmb)
    mean_plx = np.mean(plx)
 
    # ----------- Figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f'Cluster {cid}  |  N={N}  |  '
        f'dist={mean_dist:.0f} pc  |  '
        f'<plx>={mean_plx:.2f} mas  σ={plx_std:.2f}  |  '
        f'σ(pml)={pml_std:.2f}  σ(pmb)={pmb_std:.2f} mas/yr',
        fontsize=11, y=0.98
    )
 
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
 
    # --- Panel 1: CMD
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(color, M_G, c=prob, cmap='plasma', s=10, vmin=0, vmax=1, alpha=0.8)
    plt.colorbar(sc, ax=ax1, label='membership prob.')
    if HAS_ISO:
        ax1.plot(iso_color_sorted, iso_mag_sorted, 'k-', lw=1.5, alpha=0.7, label='20 Myr PARSEC')
        ax1.legend(fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel('BP - RP (mag)')
    ax1.set_ylabel('M_G (mag)')
    ax1.set_title('CMD')
    ax1.grid(True, alpha=0.3)
 
    # --- Panel 2: Proper motion diagram
    ax2 = fig.add_subplot(gs[0, 1])
    # all cluster stars in background for context
    all_pml = np.array(cat['pml_corr'])
    all_pmb = np.array(cat['pmb_corr'])
    ax2.scatter(all_pml, all_pmb, s=3, c='lightgray', alpha=0.3, zorder=1)
    sc2 = ax2.scatter(pml, pmb, c=prob, cmap='plasma', s=15, vmin=0, vmax=1, alpha=0.9, zorder=2)
    plt.colorbar(sc2, ax=ax2, label='membership prob.')
    # mark centroid
    ax2.scatter(np.mean(pml), np.mean(pmb), marker='+', s=100, c='red', zorder=3)
    ax2.set_xlabel('μ_l* (mas/yr) [LSR]')
    ax2.set_ylabel('μ_b (mas/yr) [LSR]')
    ax2.set_title('Proper motion')
    ax2.grid(True, alpha=0.3)
 
    # zoom to cluster + 3σ margin
    pm_margin = max(3 * pml_std, 3 * pmb_std, 2.0)
    ax2.set_xlim(np.mean(pml) - pm_margin, np.mean(pml) + pm_margin)
    ax2.set_ylim(np.mean(pmb) - pm_margin, np.mean(pmb) + pm_margin)
 
    # --- Panel 3: Parallax histogram
    ax3 = fig.add_subplot(gs[1, 0])
    plx_range = max(5 * plx_std, 0.5)
    bins = np.linspace(mean_plx - plx_range, mean_plx + plx_range, 40)
    ax3.hist(plx, bins=bins, color='steelblue', edgecolor='white', linewidth=0.5)
    ax3.axvline(mean_plx, color='red', lw=1.5, label=f'mean={mean_plx:.2f} mas')
    ax3.axvline(mean_plx - plx_std, color='red', lw=1, ls='--')
    ax3.axvline(mean_plx + plx_std, color='red', lw=1, ls='--', label=f'σ={plx_std:.2f} mas')
    ax3.set_xlabel('Parallax corrected (mas)')
    ax3.set_ylabel('N stars')
    ax3.set_title('Parallax distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
 
    # --- Panel 4: Sky map (RA/Dec)
    ax4 = fig.add_subplot(gs[1, 1])
    ra_all  = np.array(cat['ra'])
    dec_all = np.array(cat['dec'])
    ra_cl   = np.array(c['ra'])
    dec_cl  = np.array(c['dec'])
    ax4.scatter(ra_all, dec_all, s=2, c='lightgray', alpha=0.2, zorder=1)
    sc4 = ax4.scatter(ra_cl, dec_cl, c=prob, cmap='plasma', s=15, vmin=0, vmax=1, alpha=0.9, zorder=2)
    plt.colorbar(sc4, ax=ax4, label='membership prob.')
    ax4.invert_xaxis()
    ax4.set_xlabel('RA (deg)')
    ax4.set_ylabel('Dec (deg)')
    ax4.set_title('Sky position (RA/Dec)')
    ax4.grid(True, alpha=0.3)
 
    plt.savefig(path_im + f'cluster_{cid:03d}_N{N}.png', dpi=150, bbox_inches='tight')
    plt.close()
 
    # summary metrics
    summary_rows.append({
        'cluster_id'  : cid,
        'N'           : N,
        'mean_dist_pc': round(mean_dist, 1),
        'mean_plx'    : round(mean_plx, 3),
        'sigma_plx'   : round(plx_std, 3),
        'sigma_pml'   : round(pml_std, 3),
        'sigma_pmb'   : round(pmb_std, 3),
        'mean_prob'   : round(np.mean(prob), 3),
        'frac_prob90' : round(np.mean(prob > 0.9), 3),
        # quality flags — tuned for sparse OB associations
        'flag_plx_large' : plx_std > 1.0,
        'flag_pm_large'  : max(pml_std, pmb_std) > 3.0,
        'flag_few_stars' : N < 20,           # lowered from 50
        # combined quality: passes all three checks
        'quality_ok'     : (plx_std < 1.0) and (max(pml_std, pmb_std) < 3.0) and (N >= 20),
    })
 
    print(f"  Cluster {cid:3d}: N={N:4d}  dist={mean_dist:5.0f} pc  "
          f"σ_plx={plx_std:.2f}  σ_pm={max(pml_std,pmb_std):.2f}  "
          f"{'[PLX?]' if plx_std>1.0 else ''}  "
          f"{'[PM?]' if max(pml_std,pmb_std)>3.0 else ''}  "
          f"{'[FEW]' if N<20 else ''}")
 
# ==============================================================================
# 3. Summary table
# ==============================================================================
summary = Table(rows=summary_rows)
summary_path = path_out + f'cluster_summary_{run_tag}.fits'
summary.write(summary_path, format='fits', overwrite=True)
print(f"\nSummary saved -> {summary_path}")
 
# print flagged clusters
flagged = summary[summary['flag_plx_large'] | summary['flag_pm_large'] | summary['flag_few_stars']]
if len(flagged) > 0:
    print(f"\nFlagged clusters ({len(flagged)}):")
    flagged['cluster_id','N','sigma_plx','sigma_pml','sigma_pmb',
            'flag_plx_large','flag_pm_large','flag_few_stars'].pprint()
else:
    print("\nNo clusters flagged.")
 
# print quality-ok clusters ready for age fitting
good = summary[summary['quality_ok']]
print(f"\nQuality-OK clusters ({len(good)}) ready for age fitting:")
good['cluster_id','N','mean_dist_pc','sigma_plx','sigma_pml','sigma_pmb','mean_prob'].pprint()
 
print(f"\nDiagnostic figures saved in {path_im}")
 

