#!/usr/bin/env python3
"""
Script 5 — CMD + posterior diagnostic figures for fitted clusters
=================================================================

Usage
-----
    python script5_cmd_posteriors.py
    python script5_cmd_posteriors.py --clusters 23 31
    python script5_cmd_posteriors.py --logz-min 10
    python script5_cmd_posteriors.py --logz-age      # age-dependent threshold
    python script5_cmd_posteriors.py --cmd GRP
    
Input
-----
    outputs/<run_tag>/ages/ages_<cmd>.fits
   							posteriors/cluster_<ID>_<cmd>.npz
    outputs/hdbscan/hdbscan_clusters_<run_tag>.fits

Output
------
    outputs/<run_tag>/ages/cmd_cluster_<cluster_id>_<cmd>.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table

sys.path.insert(0, '.')
from load_parsec import load_parsec, get_isochrone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
name_complex = 'Orion_OB1'
sky_tag      = 'ra75_90_dec-14_16'
ms_tag       = '130'
mc_tag       = '15'
cv_tag       = '6'

run_tag      = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_ages    = f'outputs/{run_tag}/ages/'
path_hdbscan = 'outputs/hdbscan/'
path_grid    = 'grids/parsec_solar_gaia.dat'

os.makedirs(path_ages, exist_ok=True)

R_G  = 2.740
R_BP = 3.374
R_RP = 2.035
EXTINCTION = {
    'BPRP': {'R_color': R_BP - R_RP, 'R_mag': R_G},
    'GRP':  {'R_color': R_G  - R_RP, 'R_mag': R_G},
}

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
    p.add_argument('--clusters', nargs='+', type=int, default=None)
    p.add_argument('--logz-min', type=float, default=None,
                   help='Fixed logZ threshold')
    p.add_argument('--logz-age', action='store_true',
                   help='Use age-dependent logZ threshold')
    p.add_argument('--cmd', choices=['BPRP', 'GRP'], default='BPRP')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_extinction(iso_color, iso_mag, Av, cmd):
    R = EXTINCTION[cmd]
    return iso_color + R['R_color'] * Av, iso_mag + R['R_mag'] * Av


def get_cmd_arrays(iso, cmd, label_max=3):
    mask = np.array(iso['label']) <= label_max
    gmag = np.array(iso['Gmag'])[mask]
    bp   = np.array(iso['G_BPmag'])[mask]
    rp   = np.array(iso['G_RPmag'])[mask]
    if cmd == 'BPRP':
        return bp - rp, gmag
    else:
        return gmag - rp, gmag

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    ages_file = path_ages + f'ages_{args.cmd}.fits'
    ages = Table.read(ages_file)
    print(f"Loaded {len(ages)} fitted clusters from {ages_file}")

    hdb_file = path_hdbscan + f'hdbscan_clusters_{run_tag}.fits'
    cat = Table.read(hdb_file)

    grid = load_parsec(path_grid, label_max=3)
    print(f"Loaded PARSEC grid: log(age) = "
          f"{grid['log_ages'][0]:.2f} to {grid['log_ages'][-1]:.2f}")

    # filtering
    if args.logz_age:
        keep = [float(row['log_evidence']) > logz_threshold(float(row['age_map_myr']))
                for row in ages]
        ages = ages[np.array(keep)]
        print(f"After age-dependent logZ filter: {len(ages)} clusters")
        for row in ages:
            thresh = logz_threshold(float(row['age_map_myr']))
            print(f"  Cluster {int(row['cluster_id']):3d}: "
                  f"age={float(row['age_map_myr']):.1f} Myr  "
                  f"logZ={float(row['log_evidence']):.1f}  "
                  f"threshold={thresh}")
    elif args.logz_min is not None:
        ages = ages[np.array(ages['log_evidence']) > args.logz_min]
        print(f"After logZ > {args.logz_min}: {len(ages)} clusters")

    cluster_ids = np.array(ages['cluster_id'])
    if args.clusters is not None:
        cluster_ids = np.array([c for c in args.clusters if c in cluster_ids])
    print(f"Plotting {len(cluster_ids)} clusters")

    for cid in cluster_ids:
        post_file = path_ages + f'posteriors/cluster_{cid:04d}_{args.cmd}.npz'
        if not os.path.exists(post_file):
            print(f"  Cluster {cid:3d}: no posterior file, skipping")
            continue

        post        = np.load(post_file)
        color_obs   = post['color']
        mag_obs     = post['mag']
        log_age_map = float(post['log_age_map'])
        Av_map      = float(post['Av_map'])
        log_age_lo  = float(post['log_age_lo'])
        log_age_hi  = float(post['log_age_hi'])
        age_map_myr = 10**log_age_map / 1e6

        row_mask = np.array(ages['cluster_id']) == cid
        row = ages[row_mask][0]

        hdb_mask = np.array(cat['cluster_id']) == cid
        prob = np.array(cat['probability'])[hdb_mask]

        iso = get_isochrone(grid, log_age_map)
        iso_c, iso_m = get_cmd_arrays(iso, args.cmd, label_max=3)
        valid = np.isfinite(iso_c) & np.isfinite(iso_m)
        iso_c_ext, iso_m_ext = apply_extinction(
            iso_c[valid], iso_m[valid], Av_map, args.cmd)

        thresh   = logz_threshold(age_map_myr)
        logz_val = float(row['log_evidence'])
        logz_flag = '✓' if logz_val > thresh else '(marginal)'

        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(
            f'Cluster {cid}  |  '
            f'age = {age_map_myr:.1f} [{10**log_age_lo/1e6:.1f}, '
            f'{10**log_age_hi/1e6:.1f}] Myr  |  '
            f'Av = {Av_map:.2f}  |  '
            f'logZ = {logz_val:.1f} {logz_flag}  |  '
            f'N = {row["n_fit"]}  |  '
            f'dist = {row["dist_mean"]:.0f} pc',
            fontsize=11, y=1.01
        )

        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        # CMD
        ax1 = fig.add_subplot(gs[0, 0])
        sc = ax1.scatter(color_obs, mag_obs,
                         c=prob[:len(color_obs)] if len(prob) >= len(color_obs)
                         else np.ones(len(color_obs)),
                         cmap='plasma', vmin=0, vmax=1,
                         s=12, alpha=0.8, zorder=2, label='Members')
        plt.colorbar(sc, ax=ax1, label='P(member)', pad=0.02)
        ax1.plot(iso_c_ext, iso_m_ext, 'k-', lw=2, zorder=3,
                 label=f'{age_map_myr:.1f} Myr  Av={Av_map:.2f}')
        for log_a, ls in [(log_age_lo, '--'), (log_age_hi, '--')]:
            iso_u = get_isochrone(grid, log_a)
            ic, im = get_cmd_arrays(iso_u, args.cmd, label_max=3)
            v = np.isfinite(ic) & np.isfinite(im)
            ic_e, im_e = apply_extinction(ic[v], im[v], Av_map, args.cmd)
            ax1.plot(ic_e, im_e, 'k', lw=0.8, ls=ls, alpha=0.5, zorder=2)
        ax1.invert_yaxis()
        ax1.set_xlabel('BP $-$ RP (mag)' if args.cmd == 'BPRP' else 'G $-$ RP (mag)')
        ax1.set_ylabel('$M_G$ (mag)')
        ax1.set_title('CMD')
        ax1.legend(fontsize=8, loc='lower right')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.grid(True, alpha=0.2)

        # Age posterior
        ax2 = fig.add_subplot(gs[0, 1])
        log_age_sigma   = (log_age_hi - log_age_lo) / 2.0
        log_age_samples = np.random.normal(log_age_map, log_age_sigma, 5000)
        log_age_samples = np.clip(log_age_samples,
                                  max(log_age_lo - 2*log_age_sigma, 6.0),
                                  min(log_age_hi + 2*log_age_sigma, 7.95))
        age_samples_myr = 10**log_age_samples / 1e6
        ax2.hist(age_samples_myr, bins=40, color='steelblue',
                 edgecolor='white', linewidth=0.4, density=True)
        ax2.axvline(age_map_myr, color='k', lw=2,
                    label=f'MAP = {age_map_myr:.1f} Myr')
        ax2.axvline(10**log_age_lo/1e6, color='k', lw=1, ls='--', alpha=0.7)
        ax2.axvline(10**log_age_hi/1e6, color='k', lw=1, ls='--', alpha=0.7,
                    label='68% HDI')
        ax2.set_xlabel('Age (Myr)')
        ax2.set_ylabel('Posterior density')
        ax2.set_title('Age posterior')
        ax2.legend(fontsize=8)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(True, alpha=0.2)

        # Av posterior
        ax3 = fig.add_subplot(gs[0, 2])
        Av_lo    = float(row['Av_lo'])
        Av_hi    = float(row['Av_hi'])
        Av_sigma = max((Av_hi - Av_lo) / 2.0, 0.05)
        Av_samples = np.random.normal(Av_map, Av_sigma, 5000)
        Av_samples = np.clip(Av_samples,
                             max(Av_lo - 2*Av_sigma, 0.0),
                             min(Av_hi + 2*Av_sigma, 2.0))
        ax3.hist(Av_samples, bins=40, color='tomato',
                 edgecolor='white', linewidth=0.4, density=True)
        ax3.axvline(Av_map, color='k', lw=2, label=f'MAP = {Av_map:.2f}')
        ax3.axvline(Av_lo,  color='k', lw=1, ls='--', alpha=0.7)
        ax3.axvline(Av_hi,  color='k', lw=1, ls='--', alpha=0.7,
                    label='68% HDI')
        ax3.set_xlabel('$A_V$ (mag)')
        ax3.set_ylabel('Posterior density')
        ax3.set_title('Extinction posterior')
        ax3.legend(fontsize=8)
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        ax3.grid(True, alpha=0.2)

        plt.tight_layout()
        out_path = path_ages + f'cmd_cluster_{cid:03d}_{args.cmd}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Cluster {cid:3d}: age={age_map_myr:.1f} Myr  "
              f"Av={Av_map:.2f}  logZ={logz_val:.1f} {logz_flag}  -> {out_path}")

    print(f"\nDone. Figures saved in {path_ages}")


if __name__ == '__main__':
    main()
