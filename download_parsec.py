"""
download_parsec.py
==================
Download PARSEC isochrone grids from the Padova CMD web interface
(http://stev.oapd.inaf.it/cgi-bin/cmd) for use in isochronal age fitting
of young stellar clusters (Sco-Cen, Orion OB1, Vela OB2, ...).

The grid covers the pre-main-sequence (PMS) phase, which MIST/brutus does
not. It is the same model family used in Ratzenböck et al. (2023) for
Sco-Cen age fitting.

Usage
-----
    # default: Gaia DR3, solar metallicity, 0.1–100 Myr (covers Sco-Cen,
    #          Her OB1, Orion OB1, Vela OB2 in a single file)
    python download_parsec.py --out grids/parsec_solar_gaia.dat

    # custom age range (e.g. restrict to Sco-Cen ages only)
    python download_parsec.py --log-age-min 6.0 --log-age-max 7.5 \
                               --log-age-step 0.05 \
                               --out grids/parsec_solar_gaia_scocen.dat

    # 2MASS photometry
    python download_parsec.py --photsys 2mass \
                               --out grids/parsec_solar_2mass.dat

    # inspect live form fields if server changes
    python download_parsec.py --probe

Output
------
A plain-text isochrone table. Lines starting with # are header/comments.
Key columns (exact names depend on CMD version):
    logAge        log10(age / yr)
    Mini          initial mass (solar masses)
    label         evolutionary stage (PMS = 0, MS = 1, ...)
    Gmag          absolute Gaia G magnitude  (Vega, Av=0, distance modulus=0)
    G_BPmag       absolute Gaia BP magnitude
    G_RPmag       absolute Gaia RP magnitude
    Jmag, Hmag, Ksmag   (only if --photsys 2mass)

All magnitudes are ABSOLUTE with zero extinction. Distance modulus and
extinction are applied during fitting, not here.

Notes
-----
- Metallicity fixed at solar (Z=0.0152, [Fe/H]=0) following Ratzenböck et
  al. (2023) and appropriate for the solar neighborhood (<= 1 kpc).
- CMD interface version targeted: 3.9. If the server upgrades, run
  --probe to re-inspect current field names and update build_payload().
- The file is cached: re-running with the same --out skips the download
  unless --force is passed.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CMD 3.9 uses HTTPS and a versioned CGI path
CMD_URL  = "https://stev.oapd.inaf.it/cgi-bin/cmd_3.9"
CMD_BASE = "https://stev.oapd.inaf.it"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://stev.oapd.inaf.it/cgi-bin/cmd",
}

# Value of the photsys_file form field for each supported photometric system.
# These are filenames served by the CMD backend -- verify with --probe if
# the server is updated.
PHOTSYS_OPTIONS = {
    "gaia":  "YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat",
    "2mass": "YBC_tab_mag_odfnew/tab_mag_2mass_spitzer.dat",
}

# ---------------------------------------------------------------------------
# Form payload
# ---------------------------------------------------------------------------

def build_payload(
    log_age_min:   float,
    log_age_max:   float,
    log_age_step:  float,
    metallicity_z: float,
    photsys_file:  str,
) -> dict:
    """
    Build the POST payload for CMD 3.9.

    Field names and values verified against the live form in April 2026.
    Key changes from CMD 3.7:
      - cmd_version is now "3.9", URL is cmd_3.9 over HTTPS
      - metallicity fields renamed: isoc_zlow/isoc_zupp/isoc_dz
      - track_colibri default changed to parsec_CAF09_v1.2S_S_LMC_08_web
      - new required fields: extinction_av, extinction_coeff, extinction_curve,
        dust_sourceM, dust_sourceC, kind_LPV, kind_mag, kind_dust, lf_deltamag
    Run probe_cmd_form() if something breaks after a future server update.
    """
    return {
        # -- evolutionary tracks --
        "cmd_version":   "3.9",
        "track_parsec":  "parsec_CAF09_v1.2S",               # PARSEC 1.2S, includes PMS
        "track_colibri": "parsec_CAF09_v1.2S_S_LMC_08_web",  # TP-AGB (3.9 default)
        "track_postagb": "no",
        "n_inTPC":       "10",
        "eta_reimers":   "0.2",
        "kind_interp":   "1",
        "kind_postagb":  "-1",

        # -- photometric system --
        "photsys_file":    photsys_file,
        "photsys_version": "YBCnewVega",

        # -- dust (disabled; we apply extinction in the fitting code, not here) --
        "dust_sourceM": "nodustM",
        "dust_sourceC": "nodustC",

        # -- extinction (zero; applied per-cluster during fitting) --
        "extinction_av":    "0.0",
        "extinction_coeff": "constant",
        "extinction_curve": "cardelli",
        "kind_mag":         "2",
        "kind_dust":        "0",

        # -- LPV pulsation (keep server default) --
        "kind_LPV": "1",

        # -- IMF (controls mass sampling density only, not used in fitting) --
        "imf_file": "tab_imf/imf_chabrier_lognormal.dat",

        # -- age grid --
        "isoc_isagelog": "1",              # logarithmic age input
        "isoc_lagelow":  str(log_age_min),
        "isoc_lageupp":  str(log_age_max),
        "isoc_dlage":    str(log_age_step),

        # -- metallicity: single solar value (isoc_ismetlog=0 selects Z input) --
        "isoc_ismetlog": "0",
        "isoc_zlow":     str(metallicity_z),
        "isoc_zupp":     str(metallicity_z),
        "isoc_dz":       "0.0",

        # -- output --
        "output_kind":    "0",    # isochrones
        "output_evstage": "1",    # include evolutionary stage column
        "lf_maginf":      "-15",
        "lf_magsup":      "20",
        "lf_deltamag":    "0.5",
        "submit_form":    "Submit",
    }


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------

def _post_and_get_dat_url(payload: dict, timeout: int = 60) -> str:
    """
    POST the CMD form and parse the HTML response to find the .dat file URL.

    The server responds with an HTML page containing a link such as:
        <a href="/tmp/output_abc123.dat">here</a>
    We extract and return the full URL.
    """
    r = requests.post(CMD_URL, data=payload, headers=HEADERS, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".dat") and ("output" in href or "tmp" in href):
            return urljoin(CMD_URL, href)

    # fallback: regex search on raw HTML
    match = re.search(r'href=["\']([^"\']*\.dat)["\']', r.text)
    if match:
        href = match.group(1)
        return urljoin(CMD_URL, href)

    raise RuntimeError(
        "Could not find .dat link in the CMD server response.\n"
        "The server interface may have changed -- run --probe to inspect "
        "current form fields and update build_payload() accordingly.\n"
        f"Response snippet:\n{r.text[:1000]}"
    )


def _fetch_dat(url: str, timeout: int = 120) -> str:
    """Fetch the .dat file and return its content as a string."""
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def download_grid(
    out_path:      Path,
    log_age_min:   float = 5.0,
    log_age_max:   float = 8.0,
    log_age_step:  float = 0.05,
    metallicity_z: float = 0.0152,
    photsys:       str   = "gaia",
    force:         bool  = False,
    n_retries:     int   = 3,
) -> Path:
    """
    Download a PARSEC isochrone grid and save it to out_path.

    Parameters
    ----------
    out_path : Path
        Destination file. Parent directories are created automatically.
    log_age_min : float
        Lower bound of age grid in log10(yr). 5.0 = 0.1 Myr.
    log_age_max : float
        Upper bound of age grid in log10(yr). 8.0 = 100 Myr.
        Together, 5.0–8.0 covers Sco-Cen, Her OB1, Orion OB1, and Vela OB2.
    log_age_step : float
        Spacing in log10(yr). 0.05 dex gives ~12% age resolution.
    metallicity_z : float
        Absolute metallicity Z. Solar = 0.0152, matching [Fe/H] = 0.
    photsys : str
        Photometric system: 'gaia' (default) or '2mass'.
    force : bool
        Re-download even if out_path already exists.
    n_retries : int
        Number of attempts before raising an error.

    Returns
    -------
    Path
        Path to the saved file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"[download_parsec] File already exists: {out_path}  (--force to re-download)")
        return out_path

    if photsys not in PHOTSYS_OPTIONS:
        raise ValueError(f"photsys must be one of {list(PHOTSYS_OPTIONS)}, got '{photsys}'")

    payload = build_payload(
        log_age_min, log_age_max, log_age_step,
        metallicity_z, PHOTSYS_OPTIONS[photsys]
    )

    print(f"[download_parsec] Requesting grid from {CMD_URL}")
    print(f"  log(age/yr) : {log_age_min} to {log_age_max}  step {log_age_step}  ({10**log_age_min/1e6:.2g}–{10**log_age_max/1e6:.0f} Myr)")
    print(f"  Z           : {metallicity_z}  (solar [Fe/H]=0)")
    print(f"  photsys     : {photsys}")

    last_exc = None
    for attempt in range(1, n_retries + 1):
        try:
            dat_url = _post_and_get_dat_url(payload)
            print(f"[download_parsec] Fetching: {dat_url}")
            content = _fetch_dat(dat_url)
            break
        except Exception as exc:
            last_exc = exc
            print(f"[download_parsec] Attempt {attempt}/{n_retries} failed: {exc}")
            if attempt < n_retries:
                time.sleep(5 * attempt)   # back off a bit between retries
    else:
        raise RuntimeError(f"All {n_retries} attempts failed.") from last_exc

    out_path.write_text(content)
    n_lines = content.count("\n")
    print(f"[download_parsec] Saved {n_lines} lines -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Diagnostic: inspect the live form
# ---------------------------------------------------------------------------

def probe_cmd_form() -> None:
    """
    Print all form input names and current default values from the live CMD
    interface. Use this if a server upgrade changes field names and the
    payload in build_payload() needs updating.
    """
    print(f"GET {CMD_URL} ...")
    r = requests.get(CMD_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    form = soup.find("form")
    if not form:
        print("No <form> found -- server may require authentication or has changed.")
        return
    print(f"Form action : {form.get('action')}\n")
    print(f"{'type':12s}  {'name':45s}  current_default")
    print("-" * 80)
    for inp in form.find_all(["input", "select", "textarea"]):
        print(
            f"{inp.get('type', 'select'):12s}  "
            f"{inp.get('name', '?'):45s}  "
            f"{inp.get('value', '')[:60]}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download a PARSEC PMS-inclusive isochrone grid from the Padova CMD interface "
            "(http://stev.oapd.inaf.it/cgi-bin/cmd) for young-cluster age fitting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out", default="grids/parsec_solar_gaia.dat",
        help="Output path for the isochrone .dat file",
    )
    p.add_argument(
        "--log-age-min", type=float, default=5.0,
        help="Lower age bound in log10(yr)  [5.0 = 0.1 Myr]",
    )
    p.add_argument(
        "--log-age-max", type=float, default=8.0,
        help="Upper age bound in log10(yr)  [8.0 = 100 Myr]",
    )
    p.add_argument(
        "--log-age-step", type=float, default=0.05,
        help="Step size in log10(yr)",
    )
    p.add_argument(
        "--metallicity-z", type=float, default=0.0152,
        help="Absolute metallicity Z (solar = 0.0152)",
    )
    p.add_argument(
        "--photsys", choices=["gaia", "2mass"], default="gaia",
        help="Photometric system",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-download even if output file already exists",
    )
    p.add_argument(
        "--probe", action="store_true",
        help="Print current form fields from live server instead of downloading",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.probe:
        probe_cmd_form()
        sys.exit(0)
    download_grid(
        out_path=Path(args.out),
        log_age_min=args.log_age_min,
        log_age_max=args.log_age_max,
        log_age_step=args.log_age_step,
        metallicity_z=args.metallicity_z,
        photsys=args.photsys,
        force=args.force,
    )
