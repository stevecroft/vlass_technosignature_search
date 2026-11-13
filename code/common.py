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


# The Gaia download is stored as two row-halves so that every tracked file stays far below the ~40 MiB GitHub
# blob-API limit used by Overleaf's GitHub sync (the single file was 44 MB).
GAIA_PARTS = [os.path.join(RAW, f"gaia_dr3_plx10_part{i}.parquet") for i in (1, 2)]


def read_gaia(columns=None):
    """The Gaia DR3 sample of 01_fetch_gaia.py (parts concatenated in their original row order)."""
    return pd.concat([pd.read_parquet(p, columns=columns) for p in GAIA_PARTS], ignore_index=True)


def write_gaia(df):
    h = (len(df) + 1) // 2
    for p, part in zip(GAIA_PARTS, (df.iloc[:h], df.iloc[h:])):
        part.to_parquet(p, index=False, compression="zstd")


def image_mjd(wq):
    """MJD for each row of the CADC QL image table. CAOM lacks time bounds for some images (mostly epoch 4.1);
    fill them from the FITS-header dates recovered by 08b_fill_image_dates.py, then from the median of the same
    epoch and tile (tiles are observed together; median within-tile range 0.06 d), then the epoch median."""
    m = wq.mjd.copy()
    f = os.path.join(RAW, "vlass_ql_dates_header.csv")
    if os.path.exists(f):
        h = pd.read_csv(f).dropna(subset=["mjd"]).drop_duplicates("uri").set_index("uri").mjd
        m = m.fillna(wq.uri.map(h))
    tile = wq.uri.str.extract(r"ql\.(T\d+t\d+)\.")[0]
    m = m.fillna(m.groupby([wq.epoch, tile]).transform("median"))
    m = m.fillna(m.groupby(wq.epoch).transform("median"))
    n = int(wq.mjd.isna().sum() - m.isna().sum())
    if n: print(f"image_mjd: filled {n} missing MJDs ({int(m.isna().sum())} still missing)", flush=True)
    return m


def earth_bary_xyz(jyear):
    """Earth's heliocentric position (au, ICRS equatorial) from the low-precision solar ephemeris of the
    Astronomical Almanac. Neglecting the Sun's ~0.01 au barycentric offset errs by ~1% of the parallax
    (<=6 mas for Barnard's star; checked against astropy GCRS), far below the 0.1" astrometric floor."""
    n = (np.asarray(jyear, float) - 2000.0) * 365.25 - 0.0       # days from J2000.0 (TT ~ UTC here)
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360)
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360)
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    R = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    eps = np.deg2rad(23.439 - 4.0e-7 * n)
    xs, ys, zs = R * np.cos(lam), R * np.cos(eps) * np.sin(lam), R * np.sin(eps) * np.sin(lam)   # geocentric Sun
    return -xs, -ys, -zs


def propagate(ra, dec, pmra, pmdec, epoch, ref_epoch=2016.0, plx=None):
    """Propagate Gaia DR3 J2016.0 positions (deg, deg, mas/yr) to `epoch` (Julian year): linear proper motion
    plus, if `plx` (mas) is given, the geocentric parallax displacement at that date (up to 0.55" here)."""
    dt = np.asarray(epoch, float) - ref_epoch
    pmra = np.nan_to_num(np.asarray(pmra, float)); pmdec = np.nan_to_num(np.asarray(pmdec, float))
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    da, dd = pmra * dt, pmdec * dt                                   # mas
    if plx is not None:
        X, Y, Z = earth_bary_xyz(epoch)
        a, d = np.deg2rad(ra), np.deg2rad(dec)
        p = np.nan_to_num(np.asarray(plx, float))
        da = da + p * (X * np.sin(a) - Y * np.cos(a))
        dd = dd + p * (X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d))
    return ra + da / 3.6e6 / np.cos(np.deg2rad(dec)), dec + dd / 3.6e6


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
