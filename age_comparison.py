#!/usr/bin/env python3
"""
age_comparison.py — Our fitted ages vs. a reference catalog
============================================================
Matches clusters by name (no crossmatch FITS required) and produces
an our-age vs. catalog-age scatter plot with a linear fit.

Usage
-----
    python3 age_comparison.py
    python3 age_comparison.py --cmd GRP
    python3 age_comparison.py --age-max 25

Output
------
    outputs/<run_tag>/agemap/age_comparison_<cmd>.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
name_complex = 'Orion_OB1'
sky_tag      = 'ra75_90_dec-14_16'
ms_tag       = '37'
mc_tag       = '15'
cv_tag       = '6'

catalog_name = 'Sanchez-Sanjuan+2024'

run_tag      = f'{name_complex}_{sky_tag}_ms{ms_tag}_mc{mc_tag}_cv{cv_tag}'
path_results = f'outputs/{run_tag}/agemap/'
os.makedirs(path_results, exist_ok=True)

# ---------------------------------------------------------------------------
# Our clusters  {hdbscan_cluster_id: name}  and  {id: (age, lo, hi)}
# From summary_BPRP_Orion_OB1_ra75_90_dec-14_16_ms37_mc15_cv6.csv
# Update when run_tag / ms_tag changes.
# ---------------------------------------------------------------------------

# Orion OB1
OUR_CLUSTER_NAMES = {
     5: 'Orion Y',
     7: 'Ori-North',
     9: 'Ori-South',
    10: 'Ori-North',
    12: 'OBP-Far',
    14: 'OBP-d',
    15: 'OBP-b',
    16: 'sigma Ori',
    17: 'ONC',
    18: 'OBP-Near',
    19: 'OBP-Near',
    20: 'Briceno-1B',
    21: 'Briceno-1A',
}

OUR_AGES = {
     5: (24.6, 23.7, 25.7),   # Orion Y
     7: (16.3, 13.3, 21.4),   # Ori-North
     9: ( 8.1,  7.8,  8.4),   # Ori-South
    10: (24.1, 21.1, 26.6),   # Ori-North
    12: (22.1, 16.8, 28.3),   # OBP-Far
    14: (13.7, 11.5, 15.0),   # OBP-d
    15: (22.4, 21.2, 23.6),   # OBP-b
    16: ( 3.2,  2.7,  4.1),   # sigma Ori
    17: ( 8.1,  7.8,  8.4),   # ONC
    18: ( 7.9,  7.6,  8.2),   # OBP-Near
    19: (14.0, 12.0, 15.0),   # OBP-Near
    20: (14.2, 13.6, 14.8),   # Briceno-1B
    21: (10.1,  9.4, 11.0),   # Briceno-1A
}

# # Sco-Cen
# OUR_CLUSTER_NAMES = {
#     17: 'Chamaeleon-1',
#     23: 'CrA-North',
#     24: 'V1062-Sco',
#     25: 'Lupus-1-4',
#     26: 'UPK606',
#     27: 'e Lup',
#     28: 'eta Lup',
#     29: 'nu Cen',
#     30: 'sigma Sco',
#     31: 'delta Sco',
#     32: 'Acrux',
#     33: 'sig Cen',
# }
#
# OUR_AGES = {
#     17: ( 3.072,  2.746,  3.349),   # Chamaeleon-1
#     23: (14.285, 13.748, 14.861),   # CrA-North
#     24: (17.661, 16.989, 18.371),   # V1062-Sco
#     25: ( 4.607,  3.795,  5.309),   # Lupus-1-4
#     26: (21.818, 16.813, 26.807),   # UPK606
#     27: (27.376, 24.290, 30.095),   # e Lup
#     28: (26.521, 23.859, 29.866),   # eta Lup
#     29: (17.626, 16.861, 18.444),   # nu Cen
#     30: (15.602, 13.392, 18.503),   # sigma Sco
#     31: ( 7.939,  7.648,  8.251),   # delta Sco
#     32: (14.561, 13.341, 17.298),   # Acrux
#     33: (17.742, 15.107, 20.167),   # sig Cen
# }

# ---------------------------------------------------------------------------
# Reference catalog ages  {group_id: (age_Myr, lo_Myr, hi_Myr)}
# ---------------------------------------------------------------------------

# Orion OB1 (Sanchez-Sanjuan et al. 2024)
CAT_AGES = {
1:  (4.7, 2.3, 11.0),   # lambda Ori
2:  (13.36, 8.73, 17.55),   # Ori-North
3:  (9.0, 5.6, 13.0),   # Briceno-1A
4:  (9.0, 5.6, 13.0),   # Briceno-1B
5:  (10.0,  8.0, 12.0),   # Ori-East
6:  (9.0,  7.0, 11.0),   # OBP-Far
7:  (2.5, 2.2, 2.8),   # sigma Ori
8:  (17.9, 11.4, 24.0),   # OBP-b
9:  (6.4, 2.1, 12.8),   # OBP-d
10: (6.8, 3.9, 10.4),   # OBP-Near
11: (2.0,  1.0,  3.0),   # ONC
12: (3.1, 1.0, 5.7),   # Ori-South
13: (18.2, 12.9, 28.0),   # Orion Y
}

CAT_NAMES = {
1:  'lambda Ori',   2:  'Ori-North',
3:  'Briceno-1A',   4:  'Briceno-1B',
5:  'Ori-East',     6:  'OBP-Far',
7:  'sigma Ori',    8:  'OBP-b',
9:  'OBP-d',        10: 'OBP-Near',
11: 'ONC',          12: 'Ori-South',
13: 'Orion Y',
}

# # Sco-Cen (Ratzenböck et al. 2023)
# CAT_AGES = {
#   #  ID   age   lo    hi
#      1: ( 3.8,  3.4,  4.2),    2: ( 5.8,  5.3,  7.6),    3: ( 9.8,  8.4, 11.0),
#      4: ( 7.6,  6.9,  8.4),    5: (10.0,  9.5, 11.0),    6: (12.7, 11.0, 13.1),
#      7: (13.7, 13.1, 15.0),    8: (14.7, 14.0, 15.5),    9: (19.1, 17.8, 21.5),
#     10: (15.0, 13.6, 15.9),   11: (17.2, 14.8, 18.1),   12: (20.0, 17.8, 22.5),
#     13: ( 6.0,  5.1,  6.6),   14: (15.3, 15.0, 15.9),   15: (16.9, 16.3, 17.8),
#     16: (42.1, 33.2, 51.1),   17: (20.9, 20.1, 21.6),   18: (13.4, 12.7, 14.8),
#     19: (14.4, 13.5, 14.8),   20: (15.7, 14.8, 16.0),   21: (15.5, 15.0, 16.1),
#     22: (11.2, 10.2, 12.2),   23: (10.2,  9.5, 11.2),   24: ( 8.8,  8.4,  9.4),
#     25: ( 9.4,  8.5, 10.8),   26: ( 3.4,  2.5,  6.5),   27: (15.9, 13.8, 17.5),
#     28: (15.4, 13.5, 16.2),   29: ( 8.5,  6.1, 10.5),   30: (11.6, 10.8, 12.1),
#     31: (14.5, 13.9, 15.1),   32: ( 8.5,  7.2,  9.6),   33: ( 3.8,  2.9,  5.7),
#     34: ( 2.8,  1.7,  3.5),   35: ( 9.6,  7.4, 11.3),   36: ( 9.2,  7.5, 12.5),
#     37: (19.1, 14.5, 25.7),
# }
# CAT_NAMES = {
#     1:'rho Oph/L1688', 2:'nu Sco',         3:'delta Sco',      4:'beta Sco',
#     5:'sigma Sco',     6:'Antares',        7:'rho Sco',        8:'Scorpio-Body',
#     9:'US-foreground', 10:'V1062-Sco',     11:'mu Sco',        12:'Libra-South',
#     13:'Lupus-1-4',    14:'eta Lup',       15:'phi Lup',       16:'Norma-North',
#     17:'e Lup',        18:'UPK606',        19:'rho Lup',       20:'nu Cen',
#     21:'sig Cen',      22:'Acrux',         23:'Musca-fgd',     24:'eps Cham',
#     25:'eta Cham',     26:'B59',           27:'Pipe-North',    28:'tet Oph',
#     29:'CrA-Main',     30:'CrA-North',     31:'Scorpio-Sting', 32:'Centaurus-Far',
#     33:'Chamaeleon-1', 34:'Chamaeleon-2',  35:'L134/L183',     36:'Oph SE',
#     37:'Oph NorthFar',
# }

# Sco-Cen (Kerr et al. 2021, ApJ 917 23) — re-key by HDBSCAN cluster_id when ready
# CAT_AGES = {
#         1: (4.6, 4.2, 5.0),     2: (13.3, 12.4, 14.2),   3: (18.8, 18.2, 19.4),
#         4: (5.4, 4.6, 6.2),     5: (16.9, 15.7, 18.1),   6: (22.8, 21.8, 23.8), 
#         7: (22.7, 21.6, 23.8),  8: (20.8, 19.7, 21.9),   9: (11.2, 10.3, 12.1),  
#         10: (10.2, 9.5, 10.9), 11: (14.6, 13.8, 15.4), 12: (18.5, 17.8, 19.2),
# }

# CAT_NAMES = {
#      1: 'Chamaeleon-1',  2:'CrA-North',  3:'V1062-Sco', 4:'Lupus-1-4', 
#      5:'UPK606',         6:'e Lup',      7:'eta Lup',   8:'nu Cen',
#      9:'sigma Sco',     10:'delta Sco', 11:'Acrux',    12:'sig Cen',
# }

# ---------------------------------------------------------------------------
# Matching — by cluster name, no crossmatch file needed
# ---------------------------------------------------------------------------

def build_matches(age_max=None):
    name_to_cat = {v: k for k, v in CAT_NAMES.items()}
    matches = []
    for cid, name in OUR_CLUSTER_NAMES.items():
        cat_key = name_to_cat.get(name)
        if cat_key is None or cat_key not in CAT_AGES:
            continue
        our_age, our_lo, our_hi = OUR_AGES[cid]
        cat_age, cat_lo, cat_hi = CAT_AGES[cat_key]
        if age_max is not None and our_age > age_max:
            continue
        matches.append({
            'cluster_id': cid,
            'name':       name,
            'our_age':    our_age,
            'our_lo':     our_lo,
            'our_hi':     our_hi,
            'cat_age':    cat_age,
            'cat_lo':     cat_lo,
            'cat_hi':     cat_hi,
        })
    return matches

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(matches, cmd):
    if not matches:
        print('No matched clusters — nothing to plot.')
        return

    cat_ages = np.array([m['cat_age'] for m in matches])
    cat_lo   = np.array([m['cat_age'] - m['cat_lo'] for m in matches])
    cat_hi   = np.array([m['cat_hi'] - m['cat_age'] for m in matches])
    our_ages = np.array([m['our_age'] for m in matches])
    our_lo   = np.array([m['our_age'] - m['our_lo'] for m in matches])
    our_hi   = np.array([m['our_hi'] - m['our_age'] for m in matches])

    slope, intercept, r, p_val, _ = linregress(cat_ages, our_ages)
    r2 = r ** 2

    amax  = max(cat_ages.max(), our_ages.max()) * 1.12
    lim   = (0.0, amax)
    x_fit = np.linspace(0, amax, 200)
    y_fit = slope * x_fit + intercept

    colors  = plt.cm.tab20.colors
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', 'p', '<', '>']

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, m in enumerate(matches):
        c  = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        ax.errorbar(m['cat_age'], m['our_age'],
                    xerr=[[m['cat_age'] - m['cat_lo']], [m['cat_hi'] - m['cat_age']]],
                    yerr=[[m['our_age'] - m['our_lo']], [m['our_hi'] - m['our_age']]],
                    fmt='none', color=c, lw=0.9, zorder=2, capsize=2)
        ax.scatter(m['cat_age'], m['our_age'],
                   s=70, color=c, marker=mk,
                   zorder=4, edgecolors='white', linewidths=0.5,
                   label=m['name'])

    ax.plot(lim, lim, color='#888888', lw=1, linestyle='--', zorder=1)

    sign  = '+' if intercept >= 0 else '-'
    label = f'y = {slope:.2f} x {sign} {abs(intercept):.1f} Myr'
    ax.plot(x_fit, y_fit, color='#c0392b', lw=1.8, zorder=3, label=label)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')
    ax.set_xlabel(f'{catalog_name} age (Myr)', fontsize=12)
    ax.set_ylabel('Our age (Myr)', fontsize=12)
    ax.set_title(
        f'{name_complex} — Age comparison (PARSEC-{cmd})\n'
        f'HDBSCAN $m_s$={ms_tag}  vs.  {catalog_name}',
        fontsize=11, pad=10)
    ax.legend(fontsize=8, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0,
              framealpha=0.9, edgecolor='#cccccc')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, alpha=0.25, linestyle='--', color='#aaaaaa')

    plt.tight_layout()
    fig.subplots_adjust(right=0.78)
    out = path_results + f'age_comparison_{catalog_name}_{cmd}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Age comparison -> {out}')

    print(f'\n{"Name":>16}  {"Cat age":>8}  {"Our age":>8}  {"Δ":>6}')
    print('-' * 46)
    for m in sorted(matches, key=lambda x: x['cat_age']):
        delta = m['our_age'] - m['cat_age']
        print(f"{m['name']:>16}  {m['cat_age']:>8.1f}  {m['our_age']:>8.1f}  {delta:>+6.1f}")
    print(f'\nFit: our = {slope:.3f} × catalog + {intercept:.2f} Myr  '
          f'(R² = {r2:.3f}, p = {p_val:.2e})')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cmd', default='BPRP', choices=['BPRP', 'GRP'])
    p.add_argument('--age-max', type=float, default=None,
                   help='Exclude clusters older than this (Myr)')
    return p.parse_args()


def main():
    args = parse_args()
    matches = build_matches(age_max=args.age_max)
    print(f'{len(matches)} clusters matched by name')
    plot_comparison(matches, cmd=args.cmd)


if __name__ == '__main__':
    main()
