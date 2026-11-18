"""21 - Cross-check of the forced-photometry results against the per-epoch VLASS component catalogues.

The main analysis (17-19) associates stars with the Epoch-1 QL catalogue (Gordon et al. 2021) and measures every
epoch by forced photometry at the star's predicted position. This script repeats the key steps with catalogues made
independently (PyBDSF on full images) for each later half-epoch, and with the QL median-stack catalogue:

  E1                   : CIRADA Epoch-1 QL catalogue (J/ApJS/255/30), as used by 03-18
  QL3.1, QL3.2, QL4.1  : NRAO interim QL component lists (VLASS Memo 22; M. Lacy), per-component MJD, all-sky
  SE2.1 ... SE3.2      : NRAO Single Epoch Continuum Imaging (SECI) component lists, versions 2026-06: calibrated
                         (not Quick Look) images with per-image astrometric corrections; partial sky coverage
  QLMED                : NRAO interim component list for the median stack of the Epoch 1-3 QL images (no MJD)
  QL2                  : CIRADA Epoch-2 QL catalogue, used if data/raw/ql_catalogs_full/CIRADA_VLASS2QL*components*.csv
                         exists (cirada.ca was unreachable when this was written)
Full lists: NRAO SharePoint folder VLASS_catalogs/{QL,SECI} (M. Lacy), linked from the NRAO VLASS pages while the
vlass-dl.nrao.edu cache is offline. They are git-ignored; step A keeps what this analysis needs.

Steps
  A. Reduce each full catalogue to components within 60" of any sample star (positions propagated to the catalogue's
     median date) -> data/raw/vlass_ql_catalogs_near_stars.parquet (tracked), and record every image (QL subtile or
     SE pointing; both on the same 1-degree grid) present in each catalogue -> data/raw/vlass_catalog_images.parquet,
     which defines coverage. Overlap duplicates (< 1" apart, different images) keep the one nearer its image centre;
     median stack (no image column): the higher S/N.
  B. Each star-epoch is matched to the catalogue(s) of its own half-epoch: star propagated (proper motion + parallax)
     to the component's MJD; nearest component within R_POS = 4" recorded, a match is <= R_MATCH = 2.5" and S/N >= 5.
     Chance: the same at the 8 control positions 14" away used by 18. SE: only star-epochs inside an SE image.
  C. Flux scale: forced S_mJy x peak_bias vs QL Peak_flux (both QL image scale) -> per-epoch kappa;
     forced S_mJy (corrected to the true scale) vs SE Peak_flux -> check of the QL bias correction.
  D. Own-epoch confusion veto: predicted contamination at the measurement position from that epoch's catalogue
     (components within 15", elliptical Gaussians, Maj/Min floored at the beam), excluding components within
     R_MATCH of the measurement position -- for stars and controls alike, so chance coincidences stay in b.
     Epoch 1 keeps the adopted veto (03's association decisions); Epoch 2 uses SE2 where covered, else E1.
     The steady and single-epoch tests of 18 are recomputed for the adopted (E1), own-epoch and union vetoes.
  E. Candidate astrometry: catalogue positions of the 160 candidate stars in every epoch (SE preferred where it
     detects the source); offset from the star, and co-moving vs stationary where >= 2 epochs are detected.
  F. Median stack: stars that move < 1" across their Epoch 1-3 images, matched at the median of their three
     propagated positions; compared with the all-epoch forced coadd (steady test).

Inputs : data/raw/ql_catalogs_full/*.fits (only for step A), data/raw/vlass_e1ql_components.parquet,
         data/derived/ms_sample_vlass.parquet, data/derived/alleepoch_forced{,_controls}.parquet,
         data/derived/forced_calibration.json, results/alleepoch_candidates.csv, results/vlass_detections_multiepoch.csv
Outputs: data/raw/vlass_ql_catalogs_near_stars.parquet, data/raw/vlass_catalog_images.parquet,
         data/derived/epoch_catalog_crosscheck.json, data/derived/epoch_catalog_matches.parquet,
         results/epoch_catalog_candidates.csv, results/medstack_star_matches.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import glob, json, argparse
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2, poisson
from scipy.optimize import brentq

ap = argparse.ArgumentParser(); ap.add_argument("--reduce", action="store_true", help="rebuild the reduced catalogue")
ap.add_argument("--offline", action="store_true", help="use only cached SIMBAD / X-ray counterparts")
a = ap.parse_args()
FULL = os.path.join(C.RAW, "ql_catalogs_full")
RED = os.path.join(C.RAW, "vlass_ql_catalogs_near_stars.parquet")
IMG = os.path.join(C.RAW, "vlass_catalog_images.parquet")
R_KEEP, R_DUP, R_MATCH, R_POS, R_SEARCH, VETO_SNR, THR, R_CTRL, N_CTRL = 60.0, 1.0, 2.5, 4.0, 15.0, 2.5, 5.0, 14.0, 8
IMG_HALF = 0.5          # half-width (deg) of a QL subtile / SE pointing image on the 1-degree grid
PATTERNS = {"QL3.1": "QL3.1_components*.fits", "QL3.2": "QL3.2_components*.fits", "QL4.1": "QL4.1_components*.fits",
            "QLMED": "QL_medstack_components*.fits", "SE2.1": "SECI2.1_components_*.fits", "SE2.2": "SECI2.2_components_*.fits",
            "SE3.1": "SECI3.1_components_*.fits", "SE3.2": "SECI3.2_components_*.fits", "QL2": "CIRADA_VLASS2QL*components*.csv"}
KEEP = ["Component_name", "RA", "DEC", "E_RA", "E_DEC", "Total_flux", "Peak_flux", "E_Peak_flux", "Isl_rms", "SNR",
        "Maj", "Min", "PA", "DC_Maj", "DC_Min", "S_Code", "Flag", "MJD", "Subtile", "BMAJ", "BMIN"]


def zech_ul(k, b, cl=0.95):
    """Conditional (Zech 1989) upper limit on the signal mean, as in 18."""
    f = lambda mu: poisson.cdf(k, mu + b) / poisson.cdf(k, b) - (1 - cl)
    return brentq(f, 0, 100 + 10 * k)


def xyz(ra, de):
    ra, de = np.deg2rad(ra), np.deg2rad(de); return np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]


def chord(r_arcsec):
    return 2 * np.sin(np.deg2rad(r_arcsec / 3600) / 2)


def image_centre(name):
    s = pd.Series(name).astype(str).str.extract(r"J(\d\d)(\d\d)(\d\d)([+-])(\d\d)(\d\d)(\d\d)")
    ra = 15 * (s[0].astype(float) + s[1].astype(float) / 60 + s[2].astype(float) / 3600)
    de = (s[4].astype(float) + s[5].astype(float) / 60 + s[6].astype(float) / 3600) * np.where(s[3] == "-", -1, 1)
    return ra.values, de.values


def sep_arcsec(ra1, de1, ra2, de2):
    d = xyz(ra1, de1) - xyz(ra2, de2); return np.rad2deg(2 * np.arcsin(np.linalg.norm(d, axis=1) / 2)) * 3600


def jyear_from_mjd(m):
    return 2000.0 + (np.asarray(m, float) - 51544.5) / 365.25


def dedup(K, by_image):
    """Drop overlap duplicates: pairs < R_DUP apart; keep the one nearer its image centre (or the higher S/N)."""
    t = cKDTree(xyz(K.RA.values, K.DEC.values)); pr = t.query_pairs(chord(R_DUP), output_type="ndarray")
    if not len(pr): return K
    if by_image:
        cra, cde = image_centre(K.Subtile.values); score = sep_arcsec(K.RA.values, K.DEC.values, cra, cde)
        pr = pr[K.Subtile.values[pr[:, 0]] != K.Subtile.values[pr[:, 1]]]
    else:
        score = -K.SNR.values
    drop = np.where(score[pr[:, 0]] > score[pr[:, 1]], pr[:, 0], pr[:, 1])
    return K.drop(K.index[np.unique(drop)])


def reduce_catalogues(sv):
    from astropy.io import fits
    out, imgs = [], []
    for cat, pat in PATTERNS.items():
        fl = sorted(glob.glob(os.path.join(FULL, pat)))
        if not fl: print("missing", pat); continue
        p = fl[-1]                                                     # latest version
        if p.endswith(".fits"):
            d = fits.open(p, memmap=True)[1].data; names = d.columns.names
            K = pd.DataFrame({c: (np.asarray(d[c]).astype(str) if d[c].dtype.kind in "SU"
                                  else np.asarray(d[c]).astype(np.asarray(d[c]).dtype.newbyteorder("=")))
                              for c in KEEP + ["Pointing"] if c in names})
        else:   # CIRADA csv (Epoch-2 QL): same PyBDSF column names; drop non-preferred duplicates
            K = pd.read_csv(p); K = K[K.Duplicate_flag < 2] if "Duplicate_flag" in K else K
            if "SNR" not in K: K["SNR"] = K.Peak_flux / K.Isl_rms
            K = K[[c for c in KEEP if c in K.columns]]
        if "Pointing" in K: K = K.rename(columns={"Pointing": "Subtile"})          # SE image name (same grid)
        if "Flag" not in K: K["Flag"] = 0
        n0 = len(K)
        if "Subtile" in K:
            im = K.groupby("Subtile").size().rename("n_comp").reset_index(); im.insert(0, "cat", cat); imgs.append(im)
        jy = jyear_from_mjd(np.nanmedian(K.MJD)) if "MJD" in K else 2021.0
        ra, de = C.propagate(sv.ra.values, sv.dec.values, sv.pmra.values, sv.pmdec.values, np.full(len(sv), jy))
        t = cKDTree(xyz(K.RA.values, K.DEC.values))
        hits = t.query_ball_point(xyz(ra, de), chord(R_KEEP))
        idx = np.unique(np.concatenate([np.asarray(h, int) for h in hits if len(h)]))
        K = K.iloc[idx].copy(); n1 = len(K)
        K = dedup(K, by_image=("Subtile" in K))
        K.insert(0, "cat", cat); K.insert(1, "file", os.path.basename(p)); out.append(K)
        print(f"{cat} ({os.path.basename(p)}): {n0} components, {n1} within {R_KEEP:.0f}\" of a star, {len(K)} after de-duplication", flush=True)
    R = pd.concat(out, ignore_index=True)
    R["E_RA"] = R.E_RA * 3600 * np.cos(np.deg2rad(R.DEC)); R["E_DEC"] = R.E_DEC * 3600   # deg -> on-sky arcsec (as in 03)
    R.to_parquet(RED, index=False, compression="zstd")
    pd.concat(imgs, ignore_index=True).to_parquet(IMG, index=False, compression="zstd")
    return R


def e1_catalogue():
    v = C.e1_components(columns=["RAJ2000", "DEJ2000", "e_RAJ2000", "e_DEJ2000", "Ftot", "Fpeak", "e_Fpeak", "Maj", "Min", "PA",
                                 "Islrms", "SCode", "DupFlag", "QualFlag", "BMaj", "BMin"])
    v = v[v.DupFlag < 2]
    return pd.DataFrame(dict(cat="E1", RA=v.RAJ2000.values, DEC=v.DEJ2000.values,
                             E_RA=v.e_RAJ2000.values * 3600 * np.cos(np.deg2rad(v.DEJ2000.values)), E_DEC=v.e_DEJ2000.values * 3600,
                             Total_flux=v.Ftot.values, Peak_flux=v.Fpeak.values, E_Peak_flux=v.e_Fpeak.values, Isl_rms=v.Islrms.values,
                             SNR=(v.Fpeak / v.Islrms).values, Maj=v.Maj.values, Min=v.Min.values, PA=v.PA.values, S_Code=v.SCode.values,
                             Flag=v.QualFlag.values, MJD=np.nan, BMAJ=v.BMaj.values, BMIN=v.BMin.values))


def covered(ra, de, im):
    """True where (ra, de) lies inside one of the catalogue's images (1-degree boxes on the image grid)."""
    if im is None: return np.ones(len(ra), bool)
    cra, cde = image_centre(im.Subtile.values); t = cKDTree(xyz(cra, cde))
    hits = t.query_ball_point(xyz(ra, de), chord(3600 * IMG_HALF * 1.5))
    out = np.zeros(len(ra), bool)
    for i, h in enumerate(hits):
        if not h: continue
        h = np.asarray(h); dra = ((ra[i] - cra[h] + 180) % 360 - 180) * np.cos(np.deg2rad(de[i]))
        out[i] = np.any((np.abs(dra) <= IMG_HALF) & (np.abs(de[i] - cde[h]) <= IMG_HALF))
    return out


def contam(ra, de, K, tree):
    """Summed component contribution (mJy/beam, catalogue scale) at (ra, de), excluding components within R_MATCH."""
    hits = tree.query_ball_point(xyz(ra, de), chord(R_SEARCH))
    ii = np.repeat(np.arange(len(ra)), [len(h) for h in hits]); out = np.zeros(len(ra))
    if not len(ii): return out
    jj = np.concatenate([np.asarray(h, int) for h in hits])
    dE = (K.RA.values[jj] - ra[ii]) * 3600 * np.cos(np.deg2rad(de[ii])); dN = (K.DEC.values[jj] - de[ii]) * 3600
    far = np.hypot(dE, dN) > R_MATCH
    pa = np.deg2rad(K.PA.values[jj]); u = dE * np.sin(pa) + dN * np.cos(pa); w = dE * np.cos(pa) - dN * np.sin(pa)
    val = K.Peak_flux.values[jj] * np.exp(-4 * np.log(2) * (u ** 2 / K.MajV.values[jj] ** 2 + w ** 2 / K.MinV.values[jj] ** 2))
    np.add.at(out, ii, np.where(far, val, 0.0)); return out


def match(X, K, t):
    """Nearest component within R_POS of each star-epoch in X, star re-propagated to the component's MJD."""
    hits = t.query_ball_point(xyz(X.ra_e.values, X.de_e.values), chord(R_POS + 15))
    ii = np.repeat(np.arange(len(X)), [len(h) for h in hits])
    jj = np.concatenate([np.asarray(h, int) for h in hits]) if len(ii) else np.array([], int)
    jyc = np.where(np.isfinite(K.MJD.values[jj]), jyear_from_mjd(K.MJD.values[jj]), X.ep.values[ii])
    rs, ds = C.propagate(X.ra.values[ii], X.dec.values[ii], X.pmra.values[ii], X.pmdec.values[ii], jyc, plx=X.parallax.values[ii])
    dE = (K.RA.values[jj] - rs) * 3600 * np.cos(np.deg2rad(ds)); dN = (K.DEC.values[jj] - ds) * 3600
    P = pd.DataFrame(dict(row=X.index.values[ii], comp=jj, dE=dE, dN=dN, sep=np.hypot(dE, dN)))
    P = P[P.sep <= R_POS].sort_values("sep").drop_duplicates("row")
    for c in ["Peak_flux", "E_Peak_flux", "Isl_rms", "SNR", "Total_flux", "E_RA", "E_DEC", "S_Code", "Flag", "MJD", "DC_Maj"]:
        P[c] = K[c].values[P.comp.values] if c in K else np.nan
    return P.set_index("row")


_ACTIVE = None


def active_ids():
    """Stars with activity evidence (04b); as in 18, the background is counted only around stars without it."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = set(pd.read_parquet(os.path.join(C.DER, "activity_all.parquet")).query("active").source_id)
    return _ACTIVE


def tests(LT, CT, veto, cveto):
    """Steady (coadd) and single-epoch tests of 18 for a given star veto and control-veto matrix."""
    keep = (~LT.detected.values) & (~veto)
    U = LT[keep].copy(); Cv = cveto[keep]; Cs = CT[keep]
    U["w"] = 1 / U.sigma_mJy ** 2; g = U.groupby("source_id")
    CO = pd.DataFrame({"S": (U.S_mJy * U.w).groupby(U.source_id).sum() / g.w.sum(), "sig": g.w.sum() ** -0.5})
    CO["snr"] = CO.S / CO.sig
    co_hits = co_tot = ep_hits = ep_tot = 0
    inact = ~U.source_id.isin(active_ids()).values
    for j in range(N_CTRL):
        ok = ~Cv[:, j] & np.isfinite(Cs[:, j]) & inact; z = Cs[ok, j]; s = U.sigma_mJy.values[ok]; sid = U.source_id.values[ok]
        ep_hits += int((z >= THR).sum()); ep_tot += int(ok.sum())
        zc = pd.Series(z / s).groupby(sid).sum() / np.sqrt(pd.Series(1 / s ** 2).groupby(sid).sum())
        co_hits += int((zc >= THR).sum()); co_tot += len(zc)
    b_s = co_hits / co_tot * int((~CO.index.isin(active_ids())).sum()); b_e = ep_hits / ep_tot * int(inact.sum())
    return U, dict(n_trials_steady=int(len(CO)), n_trials_snap=int(len(U)), b_steady=b_s, b_snap=b_e, n_vetoed=int(veto.sum())), \
        set(CO.index[CO.snr >= THR]), set(U.source_id[U.snr >= THR])


def rstd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                         columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "dist_pc"])
    if a.reduce or not os.path.exists(RED): R = reduce_catalogues(sv)
    else: R = pd.read_parquet(RED)
    IM = pd.read_parquet(IMG)
    cats = {c: g.reset_index(drop=True) for c, g in R.groupby("cat")}
    cats["E1"] = e1_catalogue()
    for c, K in cats.items():
        K["MajV"] = np.fmax(K.Maj.values, K.BMAJ.fillna(0).values); K["MinV"] = np.fmax(K.Min.values, K.BMIN.fillna(0).values)
        K["MajV"] = np.where(K.MajV > 0, K.MajV, 2.5); K["MinV"] = np.where(K.MinV > 0, K.MinV, 2.5)
    trees = {c: cKDTree(xyz(K.RA.values, K.DEC.values)) for c, K in cats.items()}
    have_ql2 = "QL2" in cats
    # catalogues searched for each half-epoch; the first listed is the one used for the own-epoch veto
    HALF = {"1.1": ["E1"], "1.2": ["E1"], "2.1": (["QL2"] if have_ql2 else []) + ["SE2.1"], "2.2": (["QL2"] if have_ql2 else []) + ["SE2.2"],
            "3.1": ["QL3.1", "SE3.1"], "3.2": ["QL3.2", "SE3.2"], "4.1": ["QL4.1"]}
    PARTIAL = {c for c in cats if c.startswith("SE")}
    res = dict(catalogues={c: dict(n_near_stars=int(len(K)), file=str(K.file.iloc[0]) if "file" in K else "J/ApJS/255/30")
                           for c, K in cats.items()}, have_QL2=have_ql2, R_MATCH=R_MATCH, R_POS=R_POS)
    print({c: v["n_near_stars"] for c, v in res["catalogues"].items()}, flush=True)

    # ---------------- B. star-epoch matching
    LT = pd.read_parquet(os.path.join(C.DER, "alleepoch_forced.parquet"))
    CTp = pd.read_parquet(os.path.join(C.DER, "alleepoch_forced_controls.parquet"))
    assert (CTp.source_id.values == LT.source_id.values).all() and (CTp.tag.values == LT.tag.values).all()
    CT = CTp[[f"c{j}" for j in range(N_CTRL)]].values; CV18 = CTp[[f"cv{j}" for j in range(N_CTRL)]].values.astype(bool)
    LT = LT.merge(sv, on="source_id", how="left", validate="many_to_one")
    LT["half_epoch"] = LT.half_epoch.astype(str)
    cal = json.load(open(os.path.join(C.DER, "forced_calibration.json"))); kappa = cal["kappa"]
    me = pd.read_csv(os.path.join(C.RES, "vlass_detections_multiepoch.csv"))
    fscale = me.groupby("epoch_label").flux_scale.first(); fscale.index = fscale.index.map(lambda v: f"{v:.1f}")
    LT["fs"] = LT.half_epoch.map(fscale).astype(float).values
    LT["bp"] = C.peak_bias(LT.half_epoch.values)
    LT["sig_cat"] = LT.sigma_mJy * LT.bp * kappa      # = eta * sigma_raw (QL image scale)
    chk = LT[LT.pred_contam > 0]; res["contam_snr_check_max_dev"] = float(np.abs(kappa * chk.pred_contam / chk.sig_cat - chk.contam_snr).max())
    LT["ra_e"], LT["de_e"] = C.propagate(LT.ra.values, LT.dec.values, LT.pmra.values, LT.pmdec.values, LT.ep.values, plx=LT.parallax.values)
    LT["veto_own"] = LT.veto.values; CVown = CV18.copy(); LT["veto_cat"] = "E1(adopted)"; LT["own_contam_snr"] = LT.contam_snr
    long, per_cat = [], {}
    for half, clist in HALF.items():
        idx = LT.index[LT.half_epoch == half]
        if not len(idx): continue
        for ci, cat in enumerate(clist):
            if cat not in cats: continue
            K, t = cats[cat], trees[cat]; X = LT.loc[idx]
            cov = covered(X.ra_e.values, X.de_e.values, IM[IM.cat == cat] if cat in PARTIAL else None)
            Xc = X[cov]
            P = match(Xc, K, t)
            Lc = Xc[["source_id", "tag", "half_epoch", "ep", "S_mJy", "sigma_mJy", "snr", "fs", "bp", "veto", "detected"]].copy()
            Lc["cat"] = cat; Lc = Lc.join(P.drop(columns=["comp"]).add_prefix("m_"))
            nch = np.zeros(len(Xc), int)
            for j in range(N_CTRL):
                th = j * 2 * np.pi / N_CTRL
                rc = Xc.ra_e.values - R_CTRL * np.cos(th) / 3600 / np.cos(np.deg2rad(Xc.de_e.values)); dc = Xc.de_e.values + R_CTRL * np.sin(th) / 3600
                d, jx = t.query(xyz(rc, dc), distance_upper_bound=chord(R_MATCH)); ok = np.isfinite(d)
                nch += ok & (K.SNR.values[np.where(ok, jx, 0)] >= THR)
            Lc["n_ctrl_match"] = nch
            long.append(Lc)
            per_cat.setdefault(cat, dict(n_star_epochs_in_half=0, n_covered=0))
            per_cat[cat]["n_star_epochs_in_half"] += int(len(X)); per_cat[cat]["n_covered"] += int(cov.sum())
            if ci == 0 and not half.startswith("1"):
                # own-epoch veto with this catalogue; uncovered star-epochs keep the adopted (E1) veto
                own = contam(Xc.ra_e.values, Xc.de_e.values, K, t) * kappa / Xc.sig_cat.values
                LT.loc[Xc.index, "own_contam_snr"] = own; LT.loc[Xc.index, "veto_own"] = own >= VETO_SNR; LT.loc[Xc.index, "veto_cat"] = cat
                pos = LT.index.get_indexer(Xc.index)
                for j in range(N_CTRL):
                    th = j * 2 * np.pi / N_CTRL
                    rc = Xc.ra_e.values - R_CTRL * np.cos(th) / 3600 / np.cos(np.deg2rad(Xc.de_e.values)); dc = Xc.de_e.values + R_CTRL * np.sin(th) / 3600
                    CVown[pos, j] = contam(rc, dc, K, t) * kappa / Xc.sig_cat.values >= VETO_SNR
    ML = pd.concat(long); ML["row"] = ML.index; ML = ML.reset_index(drop=True)
    ML["cmatch"] = (ML.m_sep <= R_MATCH) & (ML.m_SNR >= THR)

    # ---------------- C. per-catalogue concordance, flux scale, astrometry
    for cat, X in ML.groupby("cat"):
        g = X[np.isfinite(X.snr)]; cm = g[g.cmatch]; fd = g[g.snr >= THR]
        pc = X.n_ctrl_match.sum() / (N_CTRL * len(X))
        d = per_cat[cat]
        d.update(n_cat_match=int(len(cm)), chance_per_position=float(pc), chance_expected=float(pc * len(X)),
                 n_cat_match_secure=int(cm.detected.sum()),
                 n_forced_ge5=int(len(fd)), n_forced_ge5_unvetoed=int((~fd.veto).sum()), n_forced_ge5_with_cat=int(fd.cmatch.sum()),
                 n_forced_ge5_unvetoed_with_cat=int((fd.cmatch & ~fd.veto).sum()),
                 n_cat_forced_lt5=int((cm.snr < THR).sum()), n_cat_forced_lt3=int((cm.snr < 3).sum()),
                 n_cat_forced_lt5_flagged=int(((cm.snr < THR) & (cm.m_Flag > 0)).sum()),
                 n_cat_forced_lt5_sep_gt1p5=int(((cm.snr < THR) & (cm.m_sep > 1.5)).sum()),
                 n_cat_forced_lt5_unvetoed_nonsecure=int(((cm.snr < THR) & ~cm.veto & ~cm.detected).sum()))
        q = cm[(cm.m_SNR >= 10) & (cm.m_Flag == 0) & (cm.m_sep <= 1.0)]
        if len(q) >= 3:
            rq = q.S_mJy * q.bp / q.m_Peak_flux        # QL-image-scale ratio (-> kappa for QL catalogues)
            rt = q.S_mJy / q.m_Peak_flux                                # true-scale ratio (-> QL bias check for SE)
            d.update(flux_n=int(len(q)), ratio_qlscale=float(np.median(rq)), ratio_qlscale_err=float(1.2533 * rstd(rq) / np.sqrt(len(q))),
                     ratio_true=float(np.median(rt)), ratio_true_err=float(1.2533 * rstd(rt) / np.sqrt(len(q))),
                     kappa_implied=float(kappa * np.median(rq)), ql_bias_implied=float(np.median(q.bp) * np.median(rt)))
        q2 = cm[cm.m_SNR >= 10]
        if len(q2) >= 5:
            for ax, e in (("dE", "m_E_RA"), ("dN", "m_E_DEC")):
                d[f"{ax}_median"] = float(np.median(q2["m_" + ax])); d[f"{ax}_rstd"] = float(rstd(q2["m_" + ax]))
                d[f"{ax}_sys"] = float(np.sqrt(max(rstd(q2["m_" + ax]) ** 2 - np.median(q2[e]) ** 2, 0)))
            d["astrom_n"] = int(len(q2))
    # SE vs QL for the same star-epoch (Epoch 3): direct flux and position comparison
    A = ML[ML.cat.str.startswith("QL3")].set_index("row"); B = ML[ML.cat.str.startswith("SE3")].set_index("row")
    J = A[A.cmatch].join(B[B.cmatch][["m_Peak_flux", "m_dE", "m_dN", "m_SNR"]], rsuffix="_se", how="inner")
    J = J[(J.m_SNR >= 10) & (J.m_SNR_se >= 10)]
    if len(J) >= 3:
        res["SE3_vs_QL3"] = dict(n=int(len(J)), peak_ratio_SE_over_QL=float(np.median(J.m_Peak_flux_se / J.m_Peak_flux)),
                                 peak_ratio_err=float(1.2533 * rstd(J.m_Peak_flux_se / J.m_Peak_flux) / np.sqrt(len(J))),
                                 pos_diff_rstd_E=float(rstd(J.m_dE_se - J.m_dE)), pos_diff_rstd_N=float(rstd(J.m_dN_se - J.m_dN)))
    res["per_catalogue"] = per_cat

    # ---------------- D. tests with the own-epoch and union vetoes
    Cand = pd.read_csv(os.path.join(C.RES, "alleepoch_candidates.csv")); expl = Cand.set_index("source_id").explained
    variants = {"adopted_E1": (LT.veto.values, CV18), "own_epoch": (LT.veto_own.values.astype(bool), CVown),
                "union": (LT.veto.values | LT.veto_own.values.astype(bool), CV18 | CVown)}
    # stars that only the catalogues (or only the own-epoch veto) bring in: classify them exactly as 18 does
    sec_ids = set(LT.source_id[LT.detected])
    near = ML[(ML.m_sep <= 1.0) & (ML.m_SNR >= THR) & ~ML.source_id.isin(sec_ids | set(expl.index))]
    catonly = near.groupby("source_id").agg(n_cat=("cat", "size"), n_epochs=("half_epoch", lambda x: x.str[0].nunique()),
                                            cats=("cat", lambda x: ",".join(x)), seps=("m_sep", lambda x: ",".join(f"{v:.2f}" for v in x)),
                                            peaks=("m_Peak_flux", lambda x: ",".join(f"{v:.2f}" for v in x)),
                                            forced_snr=("snr", lambda x: ",".join(f"{v:.1f}" for v in x)), vetoed=("veto", "any"))
    extra = set(catonly.index[catonly.n_epochs >= 2])
    for name, (vv, cv) in variants.items():
        _, _, cco, cep = tests(LT, CT, vv, cv); extra |= (cco | cep) - set(expl.index)
    extra = sorted(extra)
    if extra:
        import importlib
        P04 = importlib.import_module("04_fetch_counterparts"); P07 = importlib.import_module("07_classify_and_limits")
        fsb, fx = os.path.join(C.RAW, "simbad_forced_candidates.csv"), os.path.join(C.RAW, "xray_forced_candidates.csv")
        sb = pd.read_csv(fsb); xr = pd.read_csv(fx)
        svf = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                              columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "dist_pc", "bp_rp", "MG", "dMG", "ruwe", "non_single_star"])
        todo = [i for i in extra if i not in set(xr.source_id)]
        if todo and not a.offline:
            s0 = svf.set_index("source_id").loc[todo]; xrows = []
            for sid, r in s0.iterrows():
                x1 = P04.viz_cone("J/A+A/588/A103/cat2rxs", r.ra, r.dec, 40, "2RXS,CRate,e_CRate,ExiML")
                x2 = P04.viz_cone("J/A+A/682/A34/erass1-m", r.ra, r.dec, 20, "IAUName,MLFlux1,DETLIKE1")
                gv = lambda x, k: (x[k] if x is not None and k in x.index else np.nan)
                xrows.append(dict(source_id=sid, rxs_r=gv(x1, "_r"), rxs_crate=gv(x1, "CRate"), era_r=gv(x2, "_r"), era_F=gv(x2, "MLFlux1")))
            xr = pd.concat([xr, pd.DataFrame(xrows)], ignore_index=True); xr.to_csv(fx, index=False)
            sbn = P04.simbad_by_gaia(np.array(todo)); sb = pd.concat([sb, sbn], ignore_index=True).drop_duplicates("source_id"); sb.to_csv(fsb, index=False)
        S = svf[svf.source_id.isin(extra)].merge(xr, on="source_id", how="left").merge(sb[["source_id", "main_id", "otype", "otypes"]], on="source_id", how="left")
        S["LX"] = np.where(S.era_F.notna() | S.rxs_crate.notna(), 1.0, np.nan)
        ev = S.apply(P07.evidence, axis=1); S["multiplicity_evidence"] = [e[0] for e in ev]; S["activity_evidence"] = [e[1] for e in ev]
        S["explained"] = S.activity_evidence.str.len() > 0
        expl_all = pd.concat([expl, S.set_index("source_id").explained])
        X0 = S.set_index("source_id")[["main_id", "dist_pc", "otypes", "multiplicity_evidence", "activity_evidence", "explained"]]
        catonly = catonly.join(X0, how="left")
        res["extra_classified"] = X0.reset_index().to_dict(orient="records")
    else:
        expl_all = expl
    catonly.reset_index().to_csv(os.path.join(C.RES, "epoch_catalog_only_stars.csv"), index=False)
    res["catalogue_only"] = dict(n_stars_any=int(len(catonly)), n_stars_ge2_epochs=int((catonly.n_epochs >= 2).sum()),
                                 ge2_epochs=[int(s) for s in catonly.index[catonly.n_epochs >= 2]])
    expl160 = expl; expl = expl_all
    vres = {}
    for name, (vv, cv) in variants.items():
        U, d, cco, cep = tests(LT, CT, vv, cv)
        allc = cco | cep; new = sorted(allc - set(expl.index)); unexp = {s for s in allc if s in expl.index and not expl[s]}
        k_s = len(cco & unexp); k_e = int(((U.snr >= THR) & U.source_id.isin(unexp)).sum())
        d.update(n_cand_stars=len(allc), n_cand_coadd=len(cco), n_cand_epoch=len(cep), n_unexplained=len(unexp),
                 n_new_unclassified=len(new), new_unclassified=[int(s) for s in new], k_steady=k_s, k_snap=k_e,
                 mu95_steady=zech_ul(k_s, d["b_steady"]), mu95_snap=zech_ul(k_e, d["b_snap"]),
                 lost_vs_adopted=[int(s) for s in sorted(set(expl160.index) - allc)],
                 lost_unexplained=[int(s) for s in sorted(set(expl160.index[~expl160]) - allc)],
                 added_vs_adopted=[int(s) for s in sorted(allc - set(expl160.index))],
                 added_unexplained=[int(s) for s in sorted(unexp - set(expl160.index))])
        vres[name] = d
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, list)},
              "new:", new, "lost:", d["lost_vs_adopted"], flush=True)
    res["veto_variants"] = vres
    vo = LT.veto_own.astype(bool)
    res["star_epochs_veto"] = dict(E1=int(LT.veto.sum()), own=int(vo.sum()), own_not_E1=int((vo & ~LT.veto).sum()),
                                   E1_not_own=int((LT.veto & ~vo).sum()),
                                   veto_catalogue_counts=LT.veto_cat.value_counts().to_dict())

    # ---------------- E. candidate astrometry and co-motion
    sysd = {c: float(np.hypot(per_cat.get(c, {}).get("dE_sys", 0.3), per_cat.get(c, {}).get("dN_sys", 0.3)) / np.sqrt(2)) for c in per_cat}
    cid = set(Cand.source_id)
    X = ML[ML.source_id.isin(cid)].copy(); X["sys"] = X.cat.map(sysd)
    X["is_se"] = X.cat.str.startswith("SE")
    X = X.merge(sv[["source_id", "ra", "dec", "pmra", "pmdec", "parallax"]], on="source_id", how="left")
    det = X[(X.m_sep <= R_POS) & (X.m_SNR >= THR)].sort_values(["is_se"], ascending=False)
    best = det.drop_duplicates("row")                                    # one position per star-epoch, SE preferred
    rows = []
    for sid in sorted(cid):
        g = X[X.source_id == sid]; h = best[best.source_id == sid]; L1 = LT[LT.source_id == sid]
        r = dict(source_id=sid, n_epochs=int(L1.shape[0]), n_epochs_with_cat=int(g.row.nunique()), n_cat_det=int(len(h)),
                 n_forced_ge5=int((L1.snr >= THR).sum()),
                 n_forced_ge5_cat=int(h.row.isin(L1.index[L1.snr >= THR]).sum()),
                 n_veto_E1=int(L1.veto.sum()), n_veto_own=int(L1.veto_own.astype(bool).sum()),
                 cats_det=";".join(f"{c}:{s:.1f}\"/{p:.2f}mJy/{z:.0f}sig" for c, s, p, z in zip(h.cat, h.m_sep, h.m_Peak_flux, h.m_SNR)))
        if len(h):
            e1 = h.m_E_RA.fillna(0) ** 2 + h.sys ** 2; e2 = h.m_E_DEC.fillna(0) ** 2 + h.sys ** 2
            r.update(mean_sep=float(h.m_sep.mean()), chi2_at_star=float(np.sum(h.m_dE ** 2 / e1 + h.m_dN ** 2 / e2)), dof_at_star=2 * len(h))
            r["p_at_star"] = float(chi2.sf(r["chi2_at_star"], r["dof_at_star"]))
        if len(h) >= 2:
            w1, w2 = 1 / e1, 1 / e2
            cE = np.sum(h.m_dE * w1) / w1.sum(); cN = np.sum(h.m_dN * w2) / w2.sum()
            r["chi2_comoving"] = float(np.sum((h.m_dE - cE) ** 2 * w1 + (h.m_dN - cN) ** 2 * w2))
            jy = np.where(np.isfinite(h.m_MJD), jyear_from_mjd(h.m_MJD), h.ep)
            rs, ds = C.propagate(h.ra.values, h.dec.values, h.pmra.values, h.pmdec.values, jy, plx=h.parallax.values)
            E = (rs - rs[0]) * 3600 * np.cos(np.deg2rad(ds)) + h.m_dE.values; N = (ds - ds[0]) * 3600 + h.m_dN.values
            sE = np.sum(E * w1) / w1.sum(); sN = np.sum(N * w2) / w2.sum()
            r.update(chi2_stationary=float(np.sum((E - sE) ** 2 * w1 + (N - sN) ** 2 * w2)), dof_2=2 * len(h) - 2,
                     motion_span_arcsec=float(np.hypot(np.ptp((rs - rs[0]) * 3600 * np.cos(np.deg2rad(ds))), np.ptp((ds - ds[0]) * 3600))))
        rows.append(r)
    E_ = pd.DataFrame(rows).merge(Cand[["source_id", "main_id", "dist_pc", "explained", "multiplicity_evidence", "snr_coadd",
                                        "cand_coadd", "cand_epoch", "epoch_snr"]], on="source_id")
    E_.sort_values(["explained", "snr_coadd"], ascending=[True, False]).to_csv(os.path.join(C.RES, "epoch_catalog_candidates.csv"), index=False)
    ue = E_[~E_.explained]
    res["candidates"] = dict(n=int(len(E_)), n_with_cat_det=int((E_.n_cat_det > 0).sum()), n_unexplained=int(len(ue)),
                             n_unexplained_with_cat_det=int((ue.n_cat_det > 0).sum()), n_unexplained_multi_cat=int((ue.n_cat_det >= 2).sum()),
                             n_p_at_star_lt_0p01=int((E_.p_at_star < 0.01).sum()), n_unexplained_p_at_star_lt_0p01=int((ue.p_at_star < 0.01).sum()),
                             forced_ge5_epochs=int(E_.n_forced_ge5.sum()), forced_ge5_epochs_with_cat=int(E_.n_forced_ge5_cat.sum()),
                             astrometric_sys_arcsec=sysd)

    # ---------------- F. median stack
    if "QLMED" in cats:
        K = cats["QLMED"]; t = trees["QLMED"]
        e3 = LT[LT.tag.isin(["ep1", "ep2", "ep3"])].copy()
        e3["x"] = e3.ra_e * 3600 * np.cos(np.deg2rad(e3.de_e)); e3["y"] = e3.de_e * 3600
        g = e3.groupby("source_id")
        P = pd.DataFrame({"n": g.size(), "ra_m": g.ra_e.median(), "de_m": g.de_e.median(),
                          "span": np.hypot(g.x.max() - g.x.min(), g.y.max() - g.y.min())})
        P = P[(P.n == 3) & (P.span < 1.0)]
        d, j = t.query(xyz(P.ra_m.values, P.de_m.values), distance_upper_bound=chord(R_MATCH))
        ok = np.isfinite(d); jj = np.where(ok, j, 0)
        P["med_sep"] = np.where(ok, np.rad2deg(2 * np.arcsin(np.where(ok, d, 0) / 2)) * 3600, np.nan)
        for c in ["Peak_flux", "E_Peak_flux", "Isl_rms", "SNR"]: P["med_" + c] = np.where(ok, K[c].values[jj], np.nan)
        P["med_S_Code"] = np.where(ok, K.S_Code.values[jj], "")
        P["med_det"] = ok & (P.med_SNR >= THR)
        nch = 0
        for jc in range(N_CTRL):
            th = jc * 2 * np.pi / N_CTRL
            rc = P.ra_m.values - R_CTRL * np.cos(th) / 3600 / np.cos(np.deg2rad(P.de_m.values)); dc = P.de_m.values + R_CTRL * np.sin(th) / 3600
            dd, jx = t.query(xyz(rc, dc), distance_upper_bound=chord(R_MATCH)); okc = np.isfinite(dd)
            nch += int((okc & (K.SNR.values[np.where(okc, jx, 0)] >= THR)).sum())
        Ua = LT[~LT.veto].copy(); Ua["w"] = 1 / Ua.sigma_mJy ** 2; ga = Ua.groupby("source_id")
        CA = pd.DataFrame({"S_coadd": (Ua.S_mJy * Ua.w).groupby(Ua.source_id).sum() / ga.w.sum(), "sig_coadd": ga.w.sum() ** -0.5})
        CA["snr_coadd"] = CA.S_coadd / CA.sig_coadd
        P = P.join(CA); sec = set(LT.source_id[LT.detected])
        # status from the 160 forced-photometry candidates (Table of unexplained candidates); "catonly" = classified here
        P["status"] = np.where(P.index.isin(sec), "secure", np.where(P.index.isin(expl160.index[expl160]), "explained",
                               np.where(P.index.isin(expl160.index[~expl160]), "unexplained",
                                        np.where(P.index.isin(expl.index), "catonly", "none"))))
        D = P[P.med_det]
        D.reset_index().merge(sv[["source_id", "dist_pc"]], on="source_id").to_csv(os.path.join(C.RES, "medstack_star_matches.csv"), index=False)
        rate = nch / (N_CTRL * len(P)); st = ["secure", "explained", "unexplained", "catonly", "none"]
        res["medstack"] = dict(n_stars_tested=int(len(P)), n_det=int(len(D)), chance_rate=float(rate), chance_expected=float(rate * len(P)),
                               by_status={s: int((D.status == s).sum()) for s in st}, tested_by_status={s: int((P.status == s).sum()) for s in st},
                               none_snr_coadd_median=float(D[D.status == "none"].snr_coadd.median()) if (D.status == "none").any() else None,
                               none_snr_coadd_ge3=int(((D.status == "none") & (D.snr_coadd >= 3)).sum()),
                               none_peak_median_mJy=float(D[D.status == "none"].med_Peak_flux.median()) if (D.status == "none").any() else None,
                               Isl_rms_median_near_stars=float(np.nanmedian(K.Isl_rms)),
                               unexplained_detected=[int(s) for s in D.index[D.status == "unexplained"]],
                               secure_not_detected=[int(s) for s in P.index[(P.status == "secure") & ~P.med_det]])
        print("medstack", {k: v for k, v in res["medstack"].items() if not isinstance(v, list)}, flush=True)
    keepc = ["row", "source_id", "tag", "half_epoch", "cat", "snr", "S_mJy", "sigma_mJy", "veto", "detected", "n_ctrl_match"] + \
            [c for c in ML.columns if c.startswith("m_")]
    ML.loc[ML.m_sep.notna() | ML.source_id.isin(cid), keepc].to_parquet(os.path.join(C.DER, "epoch_catalog_matches.parquet"), index=False, compression="zstd")
    json.dump(res, open(os.path.join(C.DER, "epoch_catalog_crosscheck.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "veto_variants"}, indent=1, default=float))
