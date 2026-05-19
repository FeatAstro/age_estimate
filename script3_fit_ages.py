#!/usr/bin/env python3
"""
Script 3 — Bayesian isochrone age fitting for HDBSCAN clusters
==============================================================
Loops over all clusters in the HDBSCAN output, applies photometric
quality cuts (Paper II Eq. 2-3), fits a PARSEC isochrone using the
skewed Cauchy likelihood of Ratzenböck et al. (2023), and saves
a summary table of ages and uncertainties.

Usage
-----
    # fit all clusters
    python3 script3_fit_ages.py

    # fit only cluster IDs 0, 3, 7 (for testing)
    python3 script3_fit_ages.py --clusters 0 3 7

    # resume: skip clusters that already have a result file
    python3 script3_fit_ages.py --resume

    # use more live points for better posterior resolution
    python3 script3_fit_ages.py --nlive 500

    # use GRP CMD instead of BPRP
    python3 script3_fit_ages.py --cmd GRP

Input
-----
    outputs/hdbscan/hdbscan_clusters_<run_tag>.fits

Output
------
    outputs/<run_tag>/ages/ages_<cmd>.fits          		  — one row per cluster
        					posteriors/cluster_<ID>_<cmd>.npz — full posterior samples
"""

import os
import sys
import argparse
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u
import dynesty

from load_parsec import load_parsec
from fit_isochrone import prepare_cmd, fit_cluster

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
name_complex = 'Sco_Cen'
sky_tag      = 'ra100_300_dec-90_0' # choose
ms_tag       = '80'
mc_tag       = '20'
cv_tag       = '10' 

path_hdbscan = 'outputs/hdbscan/'
path_grid    = 'grids/parsec_solar_gaia.dat'

run_tag  = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_run = f'outputs/{run_tag}/ages/'
path_post = path_run + 'posteriors/'

os.makedirs(path_run,  exist_ok=True)
os.makedirs(path_post, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--clusters', nargs='+', type=int, default=None,
                   help='Cluster IDs to fit (default: all)')
    p.add_argument('--resume', action='store_true',
                   help='Skip clusters that already have a posterior file')
    p.add_argument('--nlive', type=int, default=200,
                   help='Number of dynesty live points')
    p.add_argument('--cmd', choices=['BPRP', 'GRP'], default='BPRP',
                   help='Which Gaia CMD to fit')
    p.add_argument('--sample', default='rwalk',
                   help='dynesty sampling method')
    p.add_argument('--bound', default='multi',
                   help='dynesty bounding method')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load HDBSCAN catalog
    # ------------------------------------------------------------------
    hdbscan_file = (path_hdbscan +
                    f'hdbscan_clusters_{run_tag}.fits')
    print(f"Loading {hdbscan_file} ...", flush=True)
    cat = Table.read(hdbscan_file)
    print(f"  {len(cat)} member rows", flush=True)

    cluster_ids = np.unique(np.array(cat['cluster_id']))
    if args.clusters is not None:
        cluster_ids = np.array([k for k in args.clusters if k in cluster_ids])
        print(f"  Fitting only clusters: {cluster_ids.tolist()}", flush=True)
    print(f"  {len(cluster_ids)} clusters to process", flush=True)

    # ------------------------------------------------------------------
    # 2. Load PARSEC grid
    # ------------------------------------------------------------------
    print(f"Loading PARSEC grid from {path_grid} ...", flush=True)
    grid = load_parsec(path_grid, label_max=3)
    print(f"  {len(grid['log_ages'])} isochrones  "
          f"log(age) = {grid['log_ages'][0]:.2f} to {grid['log_ages'][-1]:.2f}",
          flush=True)

    # ------------------------------------------------------------------
    # 3. Loop over clusters
    # ------------------------------------------------------------------
    rows = []

    for k in cluster_ids:
        post_file = path_post + f'cluster_{k:04d}_{args.cmd}.npz'

        if args.resume and os.path.exists(post_file):
            print(f"  Cluster {k:4d}: skipping (posterior file exists)", flush=True)
            continue

        mask = np.array(cat['cluster_id']) == k
        sub  = cat[mask]
        n_members = len(sub)

        def _col(c):
            return c.filled(np.nan) if hasattr(c, 'filled') else np.array(c)

        G_app  = _col(sub['G_mag']).astype(float)
        BP_app = _col(sub['BP_mag']).astype(float)
        RP_app = _col(sub['RP_mag']).astype(float)

        flux_snr_G  = _col(sub['phot_g_mean_flux_over_error']).astype(float)
        flux_snr_BP = _col(sub['phot_bp_mean_flux_over_error']).astype(float)
        flux_snr_RP = _col(sub['phot_rp_mean_flux_over_error']).astype(float)

        plx     = np.array(sub['parallax_corrected'])
        dist_pc = np.array(sub['distance'])  # we can also use 1000.0 / np.where(plx > 0, plx, np.nan) as an approximation
        pmem    = _col(sub['probability']).astype(float) 

        excess = (_col(sub['phot_bp_rp_excess_factor']).astype(float)
                  if 'phot_bp_rp_excess_factor' in sub.colnames else None)
        if excess is not None and not np.any(np.isfinite(excess)):
            excess = None

        try:
            color, mag, quality_mask = prepare_cmd(
                G_app, BP_app, RP_app,
                flux_snr_G, flux_snr_BP, flux_snr_RP,
                dist_pc, pmem,
                excess_factor=excess,
                cmd=args.cmd,
                MG_max=12.0,
                debug=False,
            )
        except Exception as e:
            print(f"  Cluster {k:4d}: prepare_cmd failed ({e}), skipping",
                  flush=True)
            continue

        n_fit = len(color)

        if n_fit < 5:
            print(f"  Cluster {k:4d}: only {n_fit} stars after quality cuts, "
                  f"skipping (need >= 5)", flush=True)
            continue

        print(f"  Cluster {k:4d}: {n_members} members -> {n_fit} after cuts  "
              f"  fitting ...", end='', flush=True)

        try:
            result = fit_cluster(
                color, mag, grid,
                cmd=args.cmd,
                nlive=args.nlive,
                sample=args.sample,
                bound=args.bound,
            )
        except Exception as e:
            print(f"  FAILED ({e})", flush=True)
            continue

        map_v = result['map']
        lo_v  = result['hdi_lo']
        hi_v  = result['hdi_hi']

        log_age_map = map_v['log_age']
        log_age_lo  = lo_v['log_age']
        log_age_hi  = hi_v['log_age']

        age_map = 10**log_age_map / 1e6
        age_lo  = 10**log_age_lo  / 1e6
        age_hi  = 10**log_age_hi  / 1e6

        print(f"  age = {age_map:.1f} [{age_lo:.1f}, {age_hi:.1f}] Myr  "
              f"  Av = {map_v['Av']:.2f}  logZ = {result['log_evidence']:.1f}",
              flush=True)

        ra_vals = np.radians(np.array(sub['ra'], dtype=float))
        dec_vals = np.radians(np.array(sub['dec'], dtype=float))

        x = dist_pc * np.cos(dec_vals) * np.cos(ra_vals)
        y = dist_pc * np.cos(dec_vals) * np.sin(ra_vals)
        z = dist_pc * np.sin(dec_vals)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        z_mean = np.mean(z)
        dist_mean = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
        dist_std  = np.nanstd(dist_pc)

        center_icrs = SkyCoord(
            CartesianRepresentation(x_mean * u.pc,
                                    y_mean * u.pc,
                                    z_mean * u.pc),
            frame='icrs')
        ra_mean  = center_icrs.ra.deg
        dec_mean = center_icrs.dec.deg
        gal      = center_icrs.galactic
        l_mean   = gal.l.wrap_at(360 * u.deg).deg
        b_mean   = gal.b.deg

        rows.append({
            'cluster_id':       k,
            'n_members':        n_members,
            'n_fit':            n_fit,
            'log_age_map':      log_age_map,
            'log_age_lo':       log_age_lo,
            'log_age_hi':       log_age_hi,
            'age_map_myr':      age_map,
            'age_lo_myr':       age_lo,
            'age_hi_myr':       age_hi,
            'Av_map':           map_v['Av'],
            'Av_lo':            lo_v['Av'],
            'Av_hi':            hi_v['Av'],
            's_map':            map_v['s'],
            'a_map':            map_v['a'],
            'log_evidence':     result['log_evidence'],
            'log_evidence_err': result['log_evidence_err'],
            'ra_mean':          ra_mean,
            'dec_mean':         dec_mean,
            'l_mean':           l_mean,
            'b_mean':           b_mean,
            'x_mean':           x_mean,
            'y_mean':           y_mean,
            'z_mean':           z_mean,
            'dist_mean':        dist_mean,
            'dist_std':         dist_std,
        })

        np.savez(post_file, color=color, mag=mag,
                 log_age_map=log_age_map, Av_map=map_v['Av'],
                 log_age_lo=log_age_lo, log_age_hi=log_age_hi)

    # ------------------------------------------------------------------
    # 4. Save summary table
    # ------------------------------------------------------------------
    if not rows:
        print("No clusters fitted successfully.", flush=True)
        return

    out_table = Table(rows)
    out_file  = path_run + f'ages_{args.cmd}.fits'
    out_table.write(out_file, format='fits', overwrite=True)
    print(f"\nSaved {len(out_table)} cluster ages -> {out_file}", flush=True)

    ages = np.array(out_table['age_map_myr'])
    print(f"Age range: {ages.min():.1f} to {ages.max():.1f} Myr  "
          f"  median: {np.median(ages):.1f} Myr", flush=True)


if __name__ == '__main__':
    main()
