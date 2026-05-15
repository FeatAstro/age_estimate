"""
fit_isochrone.py
================
Bayesian isochrone age fitting for a single stellar cluster using the
skewed Cauchy likelihood of Ratzenböck et al. (2023, Paper II).

This module is intentionally a pure function library with no I/O.
The main loop over clusters and all file handling lives in fit_ages.py.

Method summary (Ratzenböck et al. 2023, Appendix A)
----------------------------------------------------
For a cluster of N stars with observed CMD positions (color_n, mag_n), we
fit a PARSEC isochrone parameterised by:

    theta = (log_age, Av)          -- astrophysical parameters
    noise = (s, a)                 -- skewed Cauchy shape parameters

The signed perpendicular distance of star n from the isochrone is:

    d_n(theta) = sign * sqrt( (c_n - c(theta))^2 + (m_n - m(theta))^2 )

where (c(theta), m(theta)) is the nearest point on the isochrone to star n,
and the sign is +1 if the star is redder than the isochrone, -1 if bluer.

Each distance is modelled as drawn from a skewed Cauchy distribution:

    f(d; s, a) = 1 / [ s*pi * (1 + (d / (s*(1 + a*sign(d))))^2) ]

The full log-likelihood is the sum over all N stars:

    log L = sum_n  log f(d_n; s, a)

Priors (flat, from Paper II Table A.1):
    log_age in (6, 8)        -- 1 to 100 Myr
    Av      in (0, 2)        -- V-band extinction (mag)
    s_BPRP  in (0.15, 0.9)   -- scale for BP-RP CMD
    a       in (0.15, 0.9)   -- skewness (same for both CMDs)

Extinction is applied to the isochrone using the standard Gaia extinction
coefficients (Wang & Chen 2019): R_G=2.740, R_BP=3.374, R_RP=2.035,
derived for a typical stellar SED. These convert Av to magnitude shifts
in each Gaia band.

Sampling is done with dynesty nested sampling (Speagle 2020), which handles
the Av-age degeneracy better than MCMC for this problem. The posterior
summary (MAP + 68% HDI) is returned for each parameter.

Usage
-----
    from load_parsec import load_parsec, get_isochrone
    from fit_isochrone import prepare_cmd, fit_cluster

    grid = load_parsec("grids/parsec_solar_gaia.dat")

    color, mag, n_stars = prepare_cmd(
        G_app, BP_app, RP_app,
        flux_snr_G, flux_snr_BP, flux_snr_RP,
        dist_pc, excess_factor,
        cmd="BPRP"
    )

    result = fit_cluster(color, mag, grid, nlive=200)
    print(result["map"])     # {'log_age': ..., 'Av': ..., 's': ..., 'a': ...}
    print(result["hdi_lo"])
    print(result["hdi_hi"])
"""

import numpy as np
import dynesty
from load_parsec import get_isochrone

# ---------------------------------------------------------------------------
# Extinction coefficients (Wang & Chen 2019, Table 3, R_V=3.1)
# These convert Av (V-band extinction) to magnitude shifts in Gaia bands.
# Applied as: mag_band_extincted = mag_band_intrinsic + R_band * Av
# ---------------------------------------------------------------------------
R_G  = 2.740
R_BP = 3.374
R_RP = 2.035

# For the two CMDs:
#   BPRP: color = BP - RP,  mag = G
#     R_color = R_BP - R_RP = 3.374 - 2.035 = 1.339
#     R_mag   = R_G         = 2.740
#   GRP:  color = G - RP,   mag = G
#     R_color = R_G  - R_RP = 2.740 - 2.035 = 0.705
#     R_mag   = R_G         = 2.740

EXTINCTION = {
    "BPRP": {"R_color": R_BP - R_RP, "R_mag": R_G},
    "GRP":  {"R_color": R_G  - R_RP, "R_mag": R_G},
}

# ---------------------------------------------------------------------------
# Prior bounds (Paper II Table A.1)
# ---------------------------------------------------------------------------
PRIOR_BOUNDS = {
    "log_age": (6.0, 8.0),
    "Av":      (0.0, 2.0),
    "s":       (0.15, 0.9),
    "a":       (0.15, 0.9),
}


# ---------------------------------------------------------------------------
# 1. Photometric quality cuts and CMD preparation
# ---------------------------------------------------------------------------

def prepare_cmd(
	G_app, BP_app, RP_app,
	flux_snr_G, flux_snr_BP, flux_snr_RP,
	dist_pc,
	excess_factor=None,
	cmd="BPRP",
	MG_max=12.0,
	debug=False,
):
	"""
	Apply Paper II photometric quality cuts and return clean CMD arrays.

	Parameters
	----------
	G_app, BP_app, RP_app : array
		Apparent Gaia magnitudes for cluster members.
	flux_snr_G/BP/RP : array
		phot_*_mean_flux_over_error from Gaia DR3.
	dist_pc : array
		Per-star distances in pc.
	excess_factor : array or None
		phot_bp_rp_excess_factor. If None, C* cut is skipped.
	cmd : str
		"BPRP" or "GRP". Default "BPRP".
	MG_max : float
		Absolute magnitude faint limit. Default 10.0 (Paper II Eq. 3).
	debug : bool
		If True, print step-by-step star counts.

	Returns
	-------
	color, mag, mask
	"""
	G_app   = np.asarray(G_app,   dtype=float)
	BP_app  = np.asarray(BP_app,  dtype=float)
	RP_app  = np.asarray(RP_app,  dtype=float)
	dist_pc = np.asarray(dist_pc, dtype=float)

	if debug:
		print(f"        [pcmd] N input: {len(G_app)}")
		print(f"        [pcmd] BP finite: {np.isfinite(BP_app).sum()}")
		print(f"        [pcmd] RP finite: {np.isfinite(RP_app).sum()}")
		print(f"        [pcmd] dist finite: {np.isfinite(dist_pc).sum()}")

	# distance modulus per star
	mu     = 5.0 * np.log10(dist_pc) - 5.0
	G_abs  = G_app  - mu
	BP_abs = BP_app - mu
	RP_abs = RP_app - mu

	if debug:
		print(f"        [pcmd] G_abs range: {np.nanmin(G_abs):.2f} to {np.nanmax(G_abs):.2f}")
		print(f"        [pcmd] G_abs finite: {np.isfinite(G_abs).sum()}")

	# photometric error cuts (Paper II Eq. 2)
	G_err  = 1.0857 / np.asarray(flux_snr_G,  dtype=float)
	BP_err = 1.0857 / np.asarray(flux_snr_BP, dtype=float)
	RP_err = 1.0857 / np.asarray(flux_snr_RP, dtype=float)

	quality = (G_err < 0.007) & (RP_err < 0.03) & (BP_err < 0.15)

	if debug:
		print(f"        [pcmd] after SNR cuts: {quality.sum()}")
		print(f"        [pcmd]   G_err<0.007:  {(G_err<0.007).sum()}")
		print(f"        [pcmd]   RP_err<0.03:  {(RP_err<0.03).sum()}")
		print(f"        [pcmd]   BP_err<0.15:  {(BP_err<0.15).sum()}")

	# C* cut (Paper II Eq. 2, Riello+2021)
	# The flux SNR cuts above already remove stars with unreliable BP/RP
	# photometry. The Riello+2021 correction polynomial requires careful
	# calibration of the color-dependent expected C value.
	if excess_factor is not None:
		G_mag  = G_app - mu   # use apparent mag for Riello polynomial
		x      = BP_app - RP_app
		f      = np.where(
			x < 0.5,
			1.154360 + 0.033772*x + 0.032277*x**2,
			np.where(
				x < 4.0,
				1.162004 + 0.011464*x + 0.049255*x**2 - 0.005879*x**3,
				1.057572 + 0.140537*x
			)
		)
		C_star       = np.asarray(excess_factor, dtype=float) - f
		sigma_C_star = 0.0059898 + 8.817481e-12 * G_app**7.618399
		excess_cut   = (np.abs(C_star) < 5 * sigma_C_star) | (G_app <= 5.0)
		quality     &= excess_cut
	if debug:
		print(f"        [pcmd] after C* cut: {quality.sum()}")

	# absolute magnitude limit (Paper II Eq. 3)
	quality &= (G_abs < MG_max)
	if debug:
		print(f"        [pcmd] after MG<{MG_max}: {quality.sum()}")

	# remove NaN in any band
	quality &= (np.isfinite(G_abs) & np.isfinite(BP_abs) & np.isfinite(RP_abs))
	if debug:
		print(f"        [pcmd] after NaN filter: {quality.sum()}")

	# build CMD arrays
	if cmd == "BPRP":
		color = BP_abs - RP_abs
		mag   = G_abs
	elif cmd == "GRP":
		color = G_abs - RP_abs
		mag   = G_abs
	else:
		raise ValueError(f"cmd must be 'BPRP' or 'GRP', got '{cmd}'")

	return color[quality], mag[quality], quality


# ---------------------------------------------------------------------------
# 2. Isochrone distance computation
# ---------------------------------------------------------------------------

def _apply_extinction(iso_color, iso_mag, Av, cmd):
    """
    Shift isochrone in CMD by dust extinction.

    Extinction makes stars appear redder (positive color shift) and fainter
    (positive magnitude shift). The shifts are:
        delta_color = R_color * Av
        delta_mag   = R_mag   * Av
    where R_color and R_mag are the extinction coefficients for the chosen CMD.
    """
    R = EXTINCTION[cmd]
    return iso_color + R["R_color"] * Av, iso_mag + R["R_mag"] * Av


def _signed_distances(obs_color, obs_mag, iso_color, iso_mag):
    """
    For each observed star, compute its signed perpendicular distance to the
    nearest point on the isochrone curve.

    What this does
    --------------
    The isochrone is a discrete set of (color, mag) points ordered by mass.
    For each observed star we find the nearest isochrone point by Euclidean
    distance in CMD space, then record:
      - d  = Euclidean distance to that nearest point
      - sign = +1 if star is redder than the isochrone (positive color residual)
               -1 if star is bluer

    The sign convention follows Ratzenböck et al. (2023) Eq. A.2:
    positive d means the star is on the reddened/binary side (expected by the
    skewed Cauchy with a > 0), negative d means bluer than the isochrone.

    Parameters
    ----------
    obs_color, obs_mag : array of shape (N,)
        Observed CMD positions of N stars.
    iso_color, iso_mag : array of shape (M,)
        Isochrone CMD points ordered by mass.

    Returns
    -------
    d : array of shape (N,)
        Signed distances.
    """
    # broadcast: (N, 1) - (1, M) -> (N, M)
    dc = obs_color[:, None] - iso_color[None, :]
    dm = obs_mag[:, None]   - iso_mag[None, :]
    dist2 = dc**2 + dm**2

    # index of nearest isochrone point for each star
    idx = np.argmin(dist2, axis=1)

    d    = np.sqrt(dist2[np.arange(len(obs_color)), idx])
    sign = np.sign(obs_color - iso_color[idx])

    # stars exactly on the isochrone get sign +1 (arbitrary, measure zero)
    sign[sign == 0] = 1.0

    return sign * d


# ---------------------------------------------------------------------------
# 3. Skewed Cauchy log-PDF
# ---------------------------------------------------------------------------

def _skewed_cauchy_logpdf(d, s, a):
    """
    Log-PDF of the skewed Cauchy distribution (Ratzenböck+2023 Eq. A.1).

    f(d; s, a) = 1 / [ s*pi * (1 + (d / (s*(1 + a*sign(d))))^2) ]

    The skewness a > 0 makes the distribution wider on the positive side
    (redder/brighter than isochrone), which is physically motivated:
      - Unresolved binaries appear overluminous -> positive d
      - Extinction/reddening -> positive d
    The heavy Cauchy tails (vs Gaussian) down-weight true outliers
    (field star contaminants) without requiring an explicit outlier model.

    Parameters
    ----------
    d : array
        Signed distances from isochrone.
    s : float
        Scale parameter (width). Prior: uniform in (0.15, 0.9).
    a : float
        Skewness parameter. Prior: uniform in (0.15, 0.9).
        a=0 recovers the symmetric Cauchy; a>0 skews toward positive d.

    Returns
    -------
    logpdf : array, same shape as d.
    """
    s_eff = s * (1.0 + a * np.sign(d))
    # guard against s_eff <= 0 (shouldn't happen for a in (0,1) and s>0)
    s_eff = np.maximum(s_eff, 1e-10)
    return -np.log(s * np.pi * (1.0 + (d / s_eff)**2))


# ---------------------------------------------------------------------------
# 4. Likelihood and prior transform
# ---------------------------------------------------------------------------

def _make_loglike(obs_color, obs_mag, grid, cmd):
    """
    Return a log-likelihood function over theta = (log_age, Av, s, a).

    The closure captures the observed CMD data and the isochrone grid.
    dynesty calls loglike(theta) repeatedly during sampling.
    """
    def loglike(theta):
        log_age, Av, s, a = theta

        # get nearest isochrone from grid
        iso = get_isochrone(grid, log_age)

        if cmd == "BPRP":
            iso_color = iso["G_BPmag"] - iso["G_RPmag"]
        else:
            iso_color = iso["Gmag"]    - iso["G_RPmag"]
        iso_mag = iso["Gmag"]

        # apply extinction shift to isochrone
        iso_color_ext, iso_mag_ext = _apply_extinction(iso_color, iso_mag, Av, cmd)

        # remove any NaN points in the isochrone (can appear at edges)
        valid = np.isfinite(iso_color_ext) & np.isfinite(iso_mag_ext)
        if valid.sum() < 5:
            return -1e30

        # compute signed distances for all observed stars
        d = _signed_distances(obs_color, obs_mag, iso_color_ext[valid], iso_mag_ext[valid])

        return np.sum(_skewed_cauchy_logpdf(d, s, a))

    return loglike


def _make_ptform():
    """
    Return the prior transform function for dynesty.

    dynesty works in the unit hypercube u in [0,1]^4. ptform maps u to
    the physical parameter space by the inverse CDF of each prior.
    For flat priors this is simply a linear rescaling:
        theta_i = lo_i + u_i * (hi_i - lo_i)

    Parameter order: (log_age, Av, s, a)
    """
    lo = np.array([PRIOR_BOUNDS["log_age"][0],
                   PRIOR_BOUNDS["Av"][0],
                   PRIOR_BOUNDS["s"][0],
                   PRIOR_BOUNDS["a"][0]])
    hi = np.array([PRIOR_BOUNDS["log_age"][1],
                   PRIOR_BOUNDS["Av"][1],
                   PRIOR_BOUNDS["s"][1],
                   PRIOR_BOUNDS["a"][1]])

    def ptform(u):
        return lo + np.asarray(u) * (hi - lo)

    return ptform


# ---------------------------------------------------------------------------
# 5. Posterior summary
# ---------------------------------------------------------------------------

def _hdi(samples, credible_interval=0.68):
    """
    Compute the Highest Density Interval (HDI) from weighted posterior samples.

    The HDI is the shortest interval containing the specified probability mass.
    This is the same summary statistic used by Ratzenböck et al. (1σ = 68% HDI).

    Unlike a symmetric percentile interval, the HDI is not biased by
    asymmetric posteriors (which are common for Av and log_age when they
    hit the prior boundary or have a skewed likelihood ridge).
    """
    n = len(samples)
    sorted_samples = np.sort(samples)
    n_included = int(np.ceil(credible_interval * n))
    interval_widths = sorted_samples[n_included:] - sorted_samples[:n - n_included]
    min_idx = np.argmin(interval_widths)
    return sorted_samples[min_idx], sorted_samples[min_idx + n_included]


def _summarise_posterior(results):
    """
    Extract MAP and 68% HDI for each parameter from dynesty results.

    dynesty returns weighted samples. We resample them to equal weight
    before computing the HDI to avoid bias from the importance weights.

    Returns a dict with keys: map, hdi_lo, hdi_hi, log_evidence.
    Each of map/hdi_lo/hdi_hi is itself a dict keyed by parameter name.
    """
    param_names = ["log_age", "Av", "s", "a"]

    # equal-weight resampling
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    weights /= weights.sum()
    idx = np.random.choice(len(samples), size=5000, replace=True, p=weights)
    eq_samples = samples[idx]

    map_vals  = {}
    hdi_lo    = {}
    hdi_hi    = {}

    for i, name in enumerate(param_names):
        col = eq_samples[:, i]
        lo, hi = _hdi(col)
        hdi_lo[name] = lo
        hdi_hi[name] = hi
        # MAP = mode of the marginal, approximated as the weighted mean
        # of the highest-density region
        map_vals[name] = col[(col >= lo) & (col <= hi)].mean()

    return {
        "map":          map_vals,
        "hdi_lo":       hdi_lo,
        "hdi_hi":       hdi_hi,
        "log_evidence": float(results.logz[-1]),
        "log_evidence_err": float(results.logzerr[-1]),
    }


# ---------------------------------------------------------------------------
# 6. Main fitting function
# ---------------------------------------------------------------------------

def fit_cluster(
    obs_color,
    obs_mag,
    grid,
    cmd="BPRP",
    nlive=400,
    sample="rwalk",
    bound="multi",
    seed=42,
):
    """
    Fit a PARSEC isochrone to a cluster's CMD using nested sampling.

    Parameters
    ----------
    obs_color : array
        Observed color (BP-RP or G-RP) in absolute magnitudes, already
        quality-filtered by prepare_cmd().
    obs_mag : array
        Observed absolute G magnitude, quality-filtered.
    grid : dict
        Output of load_parsec(). Must contain the age range log(6,8).
    cmd : str
        "BPRP" or "GRP". Must match the CMD used to build obs_color.
    nlive : int
        Number of dynesty live points. 200 is fast (~1 min per cluster);
        500 gives better posterior resolution for publication.
    sample : str
        dynesty sampling method. "rwalk" (random walk) is robust for
        this 4D problem. "slice" is slower but more accurate.
    bound : str
        dynesty bounding method. "multi" (multi-ellipsoid) works well
        for this problem. "single" is faster but less accurate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        "map"              : dict of MAP values for each parameter
        "hdi_lo"           : dict of 68% HDI lower bounds
        "hdi_hi"           : dict of 68% HDI upper bounds
        "log_evidence"     : float, log Bayesian evidence
        "log_evidence_err" : float, uncertainty on log evidence
        "n_stars"          : int, number of stars used in fit
        "cmd"              : str, which CMD was used
    """
    obs_color = np.asarray(obs_color, dtype=float)
    obs_mag   = np.asarray(obs_mag,   dtype=float)

    if len(obs_color) < 5:
        raise ValueError(
            f"Only {len(obs_color)} stars after quality cuts — too few to fit. "
            "Check quality cuts or cluster membership."
        )

    np.random.seed(seed)

    loglike = _make_loglike(obs_color, obs_mag, grid, cmd)
    ptform  = _make_ptform()

    sampler = dynesty.NestedSampler(
        loglike, ptform, ndim=4,
        nlive=nlive,
        sample=sample,
        bound=bound,
    )
    sampler.run_nested(print_progress=False)
    results = sampler.results

    summary = _summarise_posterior(results)
    summary["n_stars"] = len(obs_color)
    summary["cmd"]     = cmd

    return summary
