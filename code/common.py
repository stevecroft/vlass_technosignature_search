"""Shared paths, constants and helpers for the VLASS wideband technosignature search.

Import this module FIRST in every script: it redirects astropy's config/cache
directories into the project (astropy otherwise fails on some systems with
PermissionError on ~/.astropy) before astropy is imported anywhere.
"""
import os, io, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DER = os.path.join(ROOT, "data", "derived")
RES = os.path.join(ROOT, "results")
for d in (RAW, DER, RES, os.path.join(ROOT, ".astropy", "config"), os.path.join(ROOT, ".astropy", "cache")):
    os.makedirs(d, exist_ok=True)
os.environ["XDG_CONFIG_HOME"] = os.path.join(ROOT, ".astropy", "config")
os.environ["XDG_CACHE_HOME"] = os.path.join(ROOT, ".astropy", "cache")
os.environ.setdefault("ASTROPY_CONFIGDIR", os.environ["XDG_CONFIG_HOME"])
os.environ.setdefault("ASTROPY_CACHE_DIR", os.environ["XDG_CACHE_HOME"])

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- services
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
VIZIER_ASU = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"   # CfA mirror (CDS primary was down, Nov 2026)
SIMBAD_TAP = "https://simbad.cfa.harvard.edu/simbad/sim-tap"     # CfA mirror
CADC_TAP = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus"
CADC_DATA = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VLASS/"
VLASS_CAT = "J/ApJS/255/30"                                       # Gordon et al. 2021, VLASS Epoch 1 QL (CIRADA)

# ---------------------------------------------------------------- constants / conventions
PC_M = 3.0857e16          # parsec in metres
QL_FLUX_BIAS = 0.87       # QL fluxes ~13% low: divide catalogue/image fluxes by this
Z95 = 1.645               # one-sided 95%
DET_SIGMA = 5.0           # PyBDSF detection threshold (sigma)
BAND_HZ = 2e9             # VLASS 2-4 GHz: EIRP = L_nu * 2 GHz for a flat band-filling signal
BPRP_SPLIT = 1.84         # ~K7/M0 boundary in Gaia BP-RP
SHELLS = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
OFFSET_ARCMIN = 3.0       # background-control offsets
N_OFFSETS = 10
MATCH_R_ARCSEC = 10.0


def vizier_tsv_to_df(text):
    """Parse a VizieR ASU-TSV response (header, units, dashes, data). Returns None if no data block."""
    lines = text.splitlines()
    dash = [k for k, l in enumerate(lines) if l.startswith("---")]
    if not dash:
        return None
    i = dash[0]
    hdr = lines[i - 2].split("\t")
    body = [l for l in lines[i + 1:] if l.strip() and not l.startswith("#")]
    return pd.read_csv(io.StringIO("\n".join(body)), sep="\t", names=hdr, skipinitialspace=True, low_memory=False)


def viz_get(params, timeout=1800, tries=5, wait=60):
    """GET from the VizieR mirror with retries on its intermittent 'STDB_ERR / no connection' responses."""
    import requests
    for t in range(tries):
        try:
            r = requests.get(VIZIER_ASU, params=params, timeout=timeout, headers={"Accept-Encoding": "gzip"})
            if r.status_code == 200 and "\n---" in r.text:
                return r.text
            if r.status_code == 200 and ("STDB_ERR" not in r.text and "No server" not in r.text
                                         and "#Title:" not in r.text.strip().splitlines()[-1]):
                return r.text           # valid response with zero rows
        except Exception:
            pass
        time.sleep(wait)
    raise RuntimeError(f"VizieR query failed after {tries} tries: {params}")


def propagate(ra, dec, pmra, pmdec, epoch, ref_epoch=2016.0):
    """Linear proper-motion propagation (deg, deg, mas/yr) from Gaia DR3 J2016.0; parallax ignored (<=0.55")."""
    dt = np.asarray(epoch) - ref_epoch
    pmra = np.nan_to_num(np.asarray(pmra, float)); pmdec = np.nan_to_num(np.asarray(pmdec, float))
    return (np.asarray(ra) + pmra * dt / 3.6e6 / np.cos(np.deg2rad(dec)),
            np.asarray(dec) + pmdec * dt / 3.6e6)


def lum(flux_mjy, dist_pc):
    """Spectral luminosity L_nu [W/Hz] from flux density [mJy] and distance [pc]."""
    return 4 * np.pi * (np.asarray(dist_pc) * PC_M) ** 2 * np.asarray(flux_mjy) * 1e-29


def poisson_ul95(k):
    """95% upper limit on a Poisson mean given k observed events (3.0 for k=0)."""
    from scipy.stats import chi2
    return 0.5 * chi2.ppf(0.95, 2 * (np.asarray(k) + 1))


def set_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    return plt


def panel_letter(ax, letter):
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
