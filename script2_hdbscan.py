#!/usr/bin/env python
"""
Script 2 — HDBSCAN clustering in 5D phase space
================================================
Usage
-----
  python3 script2_hdbscan.py
  
Input
-----
    data/processed/catalog_complete_<name_complex>.fits
    
Output
------
	outputs/hdbscan/hdbscan_clusters_<run_tag>.fits
"""

import os
import numpy as np
import time
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, GalacticLSR
import astropy.units as u
from sklearn.cluster import HDBSCAN

# ----------- Parameters
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES      = 150  # tune as needed
CV               = 6   # velocity scaling (Kerr+2023, Ratzenböck+2022) / more --> kinematic more important than spatial

# ----------- Paths
name_complex = 'Orion_OB1'
sky_tag   	 = 'ra75_90_dec-14_16' # choose

path_data    = 'data/processed/'
path_out     = 'outputs/hdbscan/'

run_tag 	 = f'{name_complex}_{sky_tag}_ms{MIN_SAMPLES}_mc{MIN_CLUSTER_SIZE}_cv{CV}'


os.makedirs(path_data, exist_ok=True)
os.makedirs(path_out, exist_ok=True)

# ==============================================================================
# 1. Load catalog
# ==============================================================================
t = Table.read(path_data + f'catalog_complete_{name_complex}_{sky_tag}.fits')
print(f"Loaded {len(t)} sources")

# ==============================================================================
# 2. Selection cuts
# ==============================================================================

# ----------- Astrometric quality: RUWE < 1.4 OR fidelity > 0.5
# Paper I Eq. 2: fidelity_v2 > 0.5 AND parallax_over_error > 4.5
# (parallax SNR already applied in Script 1; fidelity here)
ruwe     = np.array(t['ruwe'])
ruwe_cut = ruwe < 1.4
if 'fidelity' in t.colnames:
    fidelity       = np.array(t['fidelity'])
    astrometry_cut = ruwe_cut | (fidelity > 0.5)
    print("  Using RUWE < 1.4 OR fidelity > 0.5  (Paper I Eq. 2)")
else:
    astrometry_cut = ruwe_cut
    print("  fidelity column not found, using RUWE < 1.4 only")

# ----------- XYZ positions from corrected parallax (1000/parallax_corrected)
plx_corr = np.array(t['parallax_corrected'])
d_plx    = np.array(t['r_med_geo'])   # BailerJones geometric distance

l_r  = np.deg2rad(np.array(t['l']))
b_r  = np.deg2rad(np.array(t['b']))
X_all = d_plx * np.cos(b_r) * np.cos(l_r)
Y_all = d_plx * np.cos(b_r) * np.sin(l_r)
Z_all = d_plx * np.sin(b_r)

# ----------- Photometric quality cuts  (Paper II Eq. 1-2)
# Convert flux-over-error to magnitude error: err = 1.0857 / SNR_flux
G_err  = 1.0857 / np.array(t['phot_g_mean_flux_over_error'])
RP_err = 1.0857 / np.array(t['phot_rp_mean_flux_over_error'])
BP_err = 1.0857 / np.array(t['phot_bp_mean_flux_over_error'])

phot_err_cut = (G_err < 0.007) & (RP_err < 0.03) & (BP_err < 0.15)

# ----------- Flux excess factor C∗ cut
if 'phot_bp_rp_excess_factor' in t.colnames:
    BP_RP  = np.array(t['phot_bp_mean_mag']) - np.array(t['phot_rp_mean_mag'])
    G_mag  = np.array(t['phot_g_mean_mag'])
    C      = np.array(t['phot_bp_rp_excess_factor'])

    # Riello+2021 Table 2: expected excess f(BP-RP) by color range
    x = BP_RP
    f = np.where(
        x < 0.5,
        1.154360 + 0.033772*x + 0.032277*x**2,
        np.where(
            x < 4.0,
            1.162004 + 0.011464*x + 0.049255*x**2 - 0.005879*x**3,
            1.057572 + 0.140537*x
        )
    )

    # C* = C - f(BP-RP): corrected flux excess factor
    C_star = C - f

    # sigma_C* from Riello+2021 Eq. 18
    sigma_C_star = 0.0059898 + 8.817481e-12 * G_mag**7.618399

    # keep stars with |C*| < 5*sigma_C* (Paper II Eq. 2)
    excess_cut = (np.abs(C_star) < 5 * sigma_C_star) | (G_mag <= 5.0)
    print("  Applying corrected C* cut (Riello+2021 Table 2)")
else:
    excess_cut = np.ones(len(t), dtype=bool)
    print("  phot_bp_rp_excess_factor not found, skipping C* cut")

# ----------- Parallax cut
plx_cut = (plx_corr >= 2.0) & (plx_corr <= 3.6)  

# ----------- Absolute magnitude limit — automatic based on median distance
# Gaia reliable photometry limit G < 19.0; faint limit scales with distance
dist_pc  = d_plx   # BailerJones r_med_geo
M_G      = np.array(t['phot_g_mean_mag']) - 5 * np.log10(dist_pc) + 5

G_LIM    = 19.0
plx_selected = plx_corr[plx_cut & np.isfinite(plx_corr)]
d_ref    = np.nanmedian(1000.0 / plx_selected)
mu_ref   = 5 * np.log10(d_ref) - 5
MG_max   = G_LIM - mu_ref
print(f"  Reference distance: {d_ref:.0f} pc  μ={mu_ref:.2f}  Auto M_G limit: {MG_max:.1f}")

mg_cut   = M_G < MG_max

# ----------- CMD cut
# Sanchez-Sanjuan (2024) Eq. 1 — 30 Myr PARSEC linear approximation 
# Sanchez-Sanjuan show it matches isochrones from 6 Myr (λ Ori) to 38 Myr (NGC 2547) 
BP_RP = np.array(t['phot_bp_mean_mag']) - np.array(t['phot_rp_mean_mag'])
M_RP  = np.array(t['phot_rp_mean_mag']) - 5 * np.log10(dist_pc) + 5

cmd_cut = M_RP < (2.6 + 2.1 * BP_RP)

# ----------- Proper motion cut
# 1σ elliptical cut from Kerr+2023 velocity dispersion converted to mas/yr
d_kpc = 0.337
sigma_mu_major = 14.9 / (4.74 * d_kpc)  # ≈ 9.3 mas/yr
sigma_mu_minor =  3.1 / (4.74 * d_kpc)  # ≈ 1.9 mas/yr
n_sigma = 2

pmra_c  =  1.7
pmdec_c = -2.7

pmra  = np.array(t['pmra'])
pmdec = np.array(t['pmdec'])

pm_cut = (((pmra  - pmra_c) / (n_sigma * sigma_mu_major))**2 +
          ((pmdec - pmdec_c) / (n_sigma * sigma_mu_minor))**2) < 1
          
# ----------- Combine all cuts
selection = (
    astrometry_cut    &
#    phot_err_cut	  & 		# in prepare_cmd()
    mg_cut		   	  &
    plx_cut           &
#    excess_cut		  &			# in prepare_cmd()
    cmd_cut		  &
#    pm_cut            &
    np.isfinite(plx_corr)
)

print(f"  Astrometry (RUWE/fidelity):    {astrometry_cut.sum():7d}")
#print(f"  Photometric errors:            {phot_err_cut.sum():7d}")
print(f"  M_G < {MG_max:.1f} (auto):             {mg_cut.sum():7d}")
print(f"  Parallax cut (244-455 pc):     {plx_cut.sum():7d}")
#print(f"  Excess cut :                   {excess_cut.sum():7d}")
print(f"  CMD cut (< ~30 Myr):            {cmd_cut.sum():7d}")
#print(f"  Proper motion cut:             {pm_cut.sum():7d}")
print(f"  All cuts combined:             {selection.sum():7d} / {len(t)}")


t = t[selection]
# Save selection-cut catalog for diagnostic crossmatch
selection_out = path_out + f'catalog_selection_{run_tag}.fits'
t.write(selection_out, format='fits', overwrite=True)
print(f"Selection catalog saved -> {selection_out}  ({len(t)} sources)")

d_plx = d_plx[selection]
X_all = X_all[selection]; Y_all = Y_all[selection]; Z_all = Z_all[selection]
print(f"Kept {len(t)} sources after selection")

Glon = np.array(t['l']);    Glat = np.array(t['b'])
ra   = np.array(t['ra']);   dec  = np.array(t['dec'])
ra_error    = np.array(t['ra_error']);    dec_error   = np.array(t['dec_error'])
pmra        = np.array(t['pmra']);        pmdec       = np.array(t['pmdec'])
pmra_error  = np.array(t['pmra_error']); pmdec_error = np.array(t['pmdec_error'])
d   = d_plx   # 1000/parallax_corrected
d16 = np.array(t['r_lo_geo']);  d84 = np.array(t['r_hi_geo'])
ed  = (d84 - d16) / 2.0
e_Plx = np.array(t['parallax_error_corrected'])

X = X_all; Y = Y_all; Z = Z_all

# ==============================================================================
# 3. Proper motions -> Galactic frame 
# ==============================================================================
AG = np.matrix([[-0.0548755604162154,-0.8734370902348850,-0.4838350155487132],
                [ 0.4941094278755837,-0.4448296299600112, 0.7469822444972189],
                [-0.8676661490190047,-0.1980763734312015, 0.4559837761750669]])

pml_list, pmb_list, epml_list, epmb_list = [], [], [], []
for j in range(len(ra)):
    ra_r = ra[j]*np.pi/180;   dec_r = dec[j]*np.pi/180
    l_r  = Glon[j]*np.pi/180; b_r   = Glat[j]*np.pi/180

    p_icrs = [-np.sin(ra_r), np.cos(ra_r), 0]
    q_icrs = [-np.cos(ra_r)*np.sin(dec_r), -np.sin(ra_r)*np.sin(dec_r), np.cos(dec_r)]
    p_gal  = [-np.sin(l_r),  np.cos(l_r), 0]
    q_gal  = [-np.cos(l_r)*np.sin(b_r),  -np.sin(l_r)*np.sin(b_r), np.cos(b_r)]

    mu_icrs = np.matrix(p_icrs)*pmra[j] + np.matrix(q_icrs)*pmdec[j]
    mu_gal  = AG * mu_icrs.transpose()
    pml_list.append((np.matrix(p_gal)*mu_gal)[0,0])
    pmb_list.append((np.matrix(q_gal)*mu_gal)[0,0])

    G  = np.matrix([p_gal, q_gal]) * AG * np.matrix([[p_gal[i], q_gal[i]] for i in range(3)])
    J  = np.matrix([[G[0,0],G[0,1],0,0,0],[G[1,0],G[1,1],0,0,0],
                    [0,0,1,0,0],[0,0,0,G[0,0],G[0,1]],[0,0,0,G[1,0],G[1,1]]])
    C_cov = np.matrix(np.diag([ra_error[j]**2, dec_error[j]**2, e_Plx[j]**2,
                                pmra_error[j]**2, pmdec_error[j]**2]))
    Cg = J * C_cov * J.transpose()
    epml_list.append(np.sqrt(Cg[3,3]))
    epmb_list.append(np.sqrt(Cg[4,4]))

pml = np.array(pml_list);  pmb = np.array(pmb_list)
epml= np.array(epml_list); epmb= np.array(epmb_list)

# ==============================================================================
# 4. LSR correction + transverse velocities
# ==============================================================================
coord = SkyCoord(l=Glon*u.deg, b=Glat*u.deg, distance=d*u.pc,
                 pm_l_cosb=pml*u.mas/u.yr, pm_b=pmb*u.mas/u.yr,
                 radial_velocity=np.zeros(len(d))*u.km/u.s, frame='galactic')
LSR      = coord.transform_to(GalacticLSR)
pmlcorr  = LSR.pm_l_cosb.value
pmbcorr  = LSR.pm_b.value

Vl  = 4.74 * pmlcorr * d / 1000.
Vb  = 4.74 * pmbcorr * d / 1000.
eVl = np.sqrt((4.74*d/1000.*epml)**2 + (4.74*pmlcorr/1000.*ed)**2)
eVb = np.sqrt((4.74*d/1000.*epmb)**2 + (4.74*pmbcorr/1000.*ed)**2)

# ==============================================================================
# 5. HDBSCAN
# ==============================================================================
Coord = np.vstack((X, Y, Z, CV*Vl, CV*Vb)).T

print(f"Running HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}) ...")
t0 = time.time()
db1 = HDBSCAN(metric='euclidean', min_cluster_size=MIN_CLUSTER_SIZE,
              min_samples=MIN_SAMPLES, cluster_selection_method='leaf')
db1.fit(Coord)
labels        = db1.labels_
probabilities = db1.probabilities_
n_clusters    = len(set(labels)) - (1 if -1 in labels else 0)
print(f"  {time.time()-t0:.1f} s  |  {n_clusters} groups  |  {list(labels).count(-1)} noise")

# ==============================================================================
# 6. Build cluster catalog
# ==============================================================================
unique_labels    = sorted(set(labels))
core_samples_mask = np.zeros_like(labels, dtype=bool)
cluster_catalog  = []
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        continue

    class_member_mask = (labels == k)
    n = np.sum(class_member_mask)

    row = {
        'cluster_id' : np.full(n, k),
        'source_id'  : np.array(t['source_id'])[class_member_mask],
        'ra': ra[class_member_mask],   'dec': dec[class_member_mask],
        'l' : Glon[class_member_mask], 'b'  : Glat[class_member_mask],
        'X' : X[class_member_mask],  'Y': Y[class_member_mask],  'Z': Z[class_member_mask],
        'distance' : d[class_member_mask],
        'Vl': Vl[class_member_mask],  'Vb': Vb[class_member_mask],
        'pml_corr' : pmlcorr[class_member_mask],
        'pmb_corr' : pmbcorr[class_member_mask],
        'X_err': np.zeros(n), 'Y_err': np.zeros(n), 'Z_err': np.zeros(n),
        'Vl_err': eVl[class_member_mask], 'Vb_err': eVb[class_member_mask],
        'probability': probabilities[class_member_mask],
        'G_mag' : np.array(t['phot_g_mean_mag'])[class_member_mask],
        'BP_mag': np.array(t['phot_bp_mean_mag'])[class_member_mask],
        'RP_mag': np.array(t['phot_rp_mean_mag'])[class_member_mask],
        'phot_g_mean_flux_over_error' : np.array(t['phot_g_mean_flux_over_error'])[class_member_mask],
        'phot_bp_mean_flux_over_error': np.array(t['phot_bp_mean_flux_over_error'])[class_member_mask],
        'phot_rp_mean_flux_over_error': np.array(t['phot_rp_mean_flux_over_error'])[class_member_mask],
        'parallax_corrected'      : np.array(t['parallax_corrected'])[class_member_mask],
        'parallax_error_corrected': np.array(t['parallax_error_corrected'])[class_member_mask],
    }
    if 'phot_bp_rp_excess_factor' in t.colnames:
        row['phot_bp_rp_excess_factor'] = np.array(t['phot_bp_rp_excess_factor'])[class_member_mask]
    for col_name in ['j_m','j_msigcom','h_m','h_msigcom','ks_m','ks_msigcom',
                     'g_mean_psf_mag','g_mean_psf_mag_error',
                     'r_mean_psf_mag','r_mean_psf_mag_error',
                     'i_mean_psf_mag','i_mean_psf_mag_error',
                     'z_mean_psf_mag','z_mean_psf_mag_error',
                     'y_mean_psf_mag','y_mean_psf_mag_error']:
        if col_name in t.colnames:
            row[col_name] = np.array(t[col_name])[class_member_mask]

    cluster_catalog.append(Table(row))

# ==============================================================================
# 7. Save
# ==============================================================================
full_catalog = vstack(cluster_catalog)
out = path_out + f'hdbscan_clusters_{run_tag}.fits'
full_catalog.write(out, format='fits', overwrite=True)
print(f"Saved -> {out}  ({len(full_catalog)} members in {n_clusters} clusters)")

for k in sorted(set(labels)):
    if k == -1: continue
    mask = labels == k
    probs = probabilities[mask]
    print(f"Cluster {k:3d}: N={mask.sum():4d}  "
          f"l={Glon[mask].mean():6.1f} b={Glat[mask].mean():5.1f}  "
          f"dist={d[mask].mean():5.0f} pc  "
          f"mean_pmem={probs.mean():.2f}  pmem>0.9={np.sum(probs>0.9)}")
