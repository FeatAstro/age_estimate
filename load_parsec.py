"""
load_parsec.py
==============
Parse a PARSEC isochrone .dat file downloaded by download_parsec.py into
numpy arrays indexed by age, for use in CMD-based age fitting.

Usage
-----
    from load_parsec import load_parsec

    grid = load_parsec("grids/parsec_solar_gaia.dat")

    # available ages (log10 yr)
    print(grid["log_ages"])          # e.g. [5.0, 5.05, 5.1, ..., 8.0]

    # get one isochrone at log(age)=6.7  (~5 Myr)
    iso = grid["isochrones"][6.7]
    # iso is a dict with keys: "Mini", "label", "Gmag", "G_BPmag", "G_RPmag"
    # each value is a 1D numpy array over mass points

    # color and absolute magnitude arrays ready for CMD fitting
    color = iso["G_BPmag"] - iso["G_RPmag"]   # BP - RP
    mag   = iso["Gmag"]                         # absolute G

Returns
-------
dict with keys:
    "log_ages"    : sorted 1D array of log10(age/yr) values in the grid
    "isochrones"  : dict mapping log_age (float, rounded to 2 dp) -> isochrone dict
    "colnames"    : list of all column names parsed from the file header
    "raw"         : full pandas DataFrame of the entire file (all ages, all columns)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name aliases
# ---------------------------------------------------------------------------
# The CMD header uses "# Zini  MH  logAge  Mini ..." -- we rename a few
# columns to shorter, consistent names used in the fitting code.

RENAME = {
    "Zini":    "Zini",
    "MH":      "MH",
    "logAge":  "logAge",
    "Mini":    "Mini",
    "int_IMF": "int_IMF",
    "Mass":    "Mass",
    "logL":    "logL",
    "logTe":   "logTe",
    "logg":    "logg",
    "label":   "label",    # evolutionary stage: 0=PMS, 1=MS, ...
    "Gmag":    "Gmag",
    "G_BPmag": "G_BPmag",
    "G_RPmag": "G_RPmag",
}

# columns we keep in each per-age isochrone dict (everything else is in raw)
KEEP_COLS = ["Mini", "label", "Gmag", "G_BPmag", "G_RPmag", "logL", "logTe", "logg"]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_header_colnames(dat_path: Path) -> list[str]:
    """
    Extract column names from the header line starting with '# Zini'.
    Returns a list of strings, with the leading '#' stripped.
    """
    with open(dat_path) as f:
        for line in f:
            if line.startswith("# Zini"):
                # strip leading '# ' and split
                return line.lstrip("#").split()
    raise ValueError(
        f"Could not find column header line (starting with '# Zini') in {dat_path}.\n"
        "Make sure the file was downloaded with output_evstage=1."
    )


def load_parsec(
    dat_path: str | Path,
    label_max: int | None = None,
) -> dict:
    """
    Load a PARSEC isochrone grid into memory.

    Parameters
    ----------
    dat_path : str or Path
        Path to the .dat file produced by download_parsec.py.
    label_max : int or None
        If set, keep only rows with label <= label_max. Useful to restrict
        to PMS+MS (label <= 3) and exclude giant/AGB phases which are
        irrelevant for young cluster fitting and clutter interpolation.
        Default None = keep all evolutionary stages.

    Returns
    -------
    dict with keys:
        "log_ages"   : sorted numpy array of log10(age/yr)
        "isochrones" : dict  {log_age (float) -> {colname -> np.ndarray}}
        "colnames"   : list of column names from header
        "raw"        : full pandas DataFrame
    """
    dat_path = Path(dat_path)
    if not dat_path.exists():
        raise FileNotFoundError(dat_path)

    colnames = _parse_header_colnames(dat_path)

    # read all data rows (skip comment lines starting with #)
    df = pd.read_csv(
        dat_path,
        comment="#",
        sep=r"\s+",
        names=colnames,
        dtype=float,
    )

    if df.empty:
        raise ValueError(f"No data rows found in {dat_path}")

    # optional evolutionary stage filter
    if label_max is not None:
        df = df[df["label"] <= label_max].copy()

    # round logAge to 2 decimal places to avoid float comparison issues
    df["logAge"] = df["logAge"].round(2)

    log_ages = np.sort(df["logAge"].unique())

    # build per-age isochrone dicts
    isochrones = {}
    available = [c for c in KEEP_COLS if c in df.columns]
    missing   = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"[load_parsec] Warning: expected columns not found: {missing}")
        print(f"  Available columns: {list(df.columns)}")

    for age in log_ages:
        sub = df[df["logAge"] == age]
        isochrones[age] = {col: sub[col].values for col in available}

    return {
        "log_ages":   log_ages,
        "isochrones": isochrones,
        "colnames":   colnames,
        "raw":        df,
    }


# ---------------------------------------------------------------------------
# Convenience: get interpolated isochrone at arbitrary log_age
# ---------------------------------------------------------------------------

def get_isochrone(grid: dict, log_age: float) -> dict:
    """
    Return the isochrone closest to log_age in the grid.

    For the age resolution used here (0.05 dex steps), linear interpolation
    between adjacent isochrones is not necessary -- the nearest grid point
    is within ~6% in age, which is smaller than typical fitting uncertainties.
    If finer interpolation is ever needed, replace this with a mass-by-mass
    linear interpolation between the two bracketing isochrones.

    Parameters
    ----------
    grid : dict
        Output of load_parsec().
    log_age : float
        Desired log10(age/yr).

    Returns
    -------
    dict mapping column name -> np.ndarray for the nearest age in the grid.
    """
    ages = grid["log_ages"]
    nearest = ages[np.argmin(np.abs(ages - log_age))]
    return grid["isochrones"][nearest]


# ---------------------------------------------------------------------------
# Quick sanity check / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    path = sys.argv[1] if len(sys.argv) > 1 else "grids/parsec_solar_gaia.dat"

    print(f"Loading {path} ...")
    grid = load_parsec(path, label_max=3)   # PMS + MS + subgiant only

    ages = grid["log_ages"]
    print(f"  {len(ages)} isochrones: log(age) = {ages[0]:.2f} to {ages[-1]:.2f}")
    print(f"  Columns: {grid['colnames']}")

    # check a few isochrones
    for la in [5.0, 6.0, 6.7, 7.0, 8.0]:
        iso = get_isochrone(grid, la)
        n = len(iso["Mini"])
        print(f"  log(age)={la:.2f}  ({10**la/1e6:.2f} Myr)  {n} mass points  "
              f"  Mini=[{iso['Mini'].min():.2f}, {iso['Mini'].max():.2f}] Msun")

    # plot CMD for a few representative ages
    fig, ax = plt.subplots(figsize=(6, 8))
    colors_plot = plt.cm.plasma(np.linspace(0.1, 0.9, 6))
    for i, la in enumerate([5.5, 6.0, 6.5, 6.9, 7.2, 7.7]):
        iso = get_isochrone(grid, la)
        bp_rp = iso["G_BPmag"] - iso["G_RPmag"]
        ax.plot(bp_rp, iso["Gmag"], color=colors_plot[i],
                label=f"{10**la/1e6:.1f} Myr", lw=1.2)

    ax.invert_yaxis()
    ax.set_xlabel("$G_{BP} - G_{RP}$")
    ax.set_ylabel("$M_G$  (absolute)")
    ax.set_title("PARSEC isochrones (PMS+MS, solar Z)")
    ax.legend(fontsize=8, title="age")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(14, -4)
    plt.tight_layout()
    outfig = "parsec_isochrones_check.png"
    plt.savefig(outfig, dpi=150)
    print(f"\nCMD plot saved to {outfig}")
