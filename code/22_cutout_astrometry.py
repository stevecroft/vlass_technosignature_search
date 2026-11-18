"""22 - Per-epoch astrometry of faint steady emitters from the 41" Quick Look cutouts, and a co-motion test.

For a forced detection with too few catalog positions (G 133-48 has one), the position in each epoch is measured
directly from the cutout: an elliptical Gaussian with the local restoring beam (BMaj/BMin/BPA of the nearest Epoch-1
catalog component) and free amplitude and centre is fitted to the pixels within 5" of the star. The position error
at that signal-to-noise ratio is calibrated empirically: the same Gaussian is injected at the star position of
INJ random cutouts of the same epoch with similar noise and re-fitted (correlated noise and sidelobes included); a
per-epoch systematic term (the QL-catalog astrometric scatter from 21, or 0.1" per axis) is added in quadrature.
The CIRADA Epoch-1 catalog lacks the w-term correction (Kimball et al. 2026), but the Epoch-1 images served by CADC
carry the corrected WCS: for the secure natural emitters (S/N > 10) the Epoch-1 cutout offsets from the star have an
rms of ~0.13" without correction and ~0.26" if the per-subtile catalog offsets of 02b are applied again, so no
correction is applied to the cutout positions (APPLY_E1_CORR = False; the validation numbers are written to the JSON).
Three models for the offsets (radio minus Gaia star position at each image epoch):
  at_star    : offset zero in every epoch (emission from the star);
  comoving   : a constant offset (a companion or a source moving with the star);
  stationary : a fixed sky position (a background source); the offsets then follow minus the star's motion.
Inputs : data/raw/vlass_<tag>_ql_cutouts_41px.npz, data/derived/forced_<tag>.parquet, data/derived/ms_sample_vlass.parquet,
         data/raw/vlass_e1ql_components.parquet, data/raw/vlass_e1ql_position_offsets.csv, results/vlass_star_matches_classified.csv,
         data/derived/epoch_catalog_crosscheck.json
Outputs: results/cutout_astrometry.csv (per star-epoch fits), data/derived/cutout_astrometry.json (model chi^2)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import chi2

TARGETS = {349716170186042880: "G 133-48", 2080884927342950272: "PM J20156+4321", 2071061615589030400: "TYC 3573-1391-1",
           2137474969751117312: "LSPM J1943+5337 (validation)"}
# Gaia DR3 2563195186142791552 is not fitted: its forced detection is the wing of an extended double source whose
# Epoch-1 components lie 7.0" and 8.2" away, beyond the 5" fitting region, so the fit runs to its bounds.
HB, R_FIT, INJ, SYS_DEFAULT = 20, 5.0, 300, 0.1
APPLY_E1_CORR = False
TAGS = ["ep1", "ep2", "ep3", "ep4"]


def xyz(ra, de):
    ra, de = np.deg2rad(ra), np.deg2rad(de); return np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]


def gauss(X, Y, A, x0, y0, bmaj, bmin, bpa):
    dE = -(X - x0); dN = Y - y0                                  # QL cutouts: +x = west, +y = north, 1"/px
    u = dE * np.sin(bpa) + dN * np.cos(bpa); w = dE * np.cos(bpa) - dN * np.sin(bpa)
    return A * np.exp(-4 * np.log(2) * (u ** 2 / bmaj ** 2 + w ** 2 / bmin ** 2))


def fit(img, x0, y0, beam):
    n = img.shape[0]; Y, X = np.mgrid[:n, :n]; m = np.hypot(X - x0, Y - y0) <= R_FIT
    f = lambda p: (gauss(X[m], Y[m], p[0], p[1], p[2], *beam) - img[m])
    A0 = max(float(np.nanmax(img[m])), 1e-3)
    r = least_squares(f, [A0, x0, y0], bounds=([0, x0 - 3, y0 - 3], [10 * A0 + 1, x0 + 3, y0 + 3]))
    return r.x                                                   # A, x, y


if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                         columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "subtile"]).set_index("source_id")
    comp = C.e1_components(columns=["RAJ2000", "DEJ2000", "BMaj", "BMin", "BPA", "DupFlag"])
    ctree = cKDTree(xyz(comp.RAJ2000.values, comp.DEJ2000.values))
    off = pd.read_csv(C.E1_OFFSETS).set_index("Subtile")
    pc = json.load(open(os.path.join(C.DER, "epoch_catalog_crosscheck.json")))["per_catalogue"]
    sysq = {c: float(np.hypot(v.get("dE_sys", SYS_DEFAULT), v.get("dN_sys", SYS_DEFAULT)) / np.sqrt(2)) for c, v in pc.items() if "dE_sys" in v}
    SYS = {"ep1": SYS_DEFAULT, "ep2": SYS_DEFAULT, "ep3": float(np.mean([sysq.get("QL3.1", SYS_DEFAULT), sysq.get("QL3.2", SYS_DEFAULT)])),
           "ep4": sysq.get("QL4.1", SYS_DEFAULT)}
    Tm = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); sec = Tm[Tm.secure & (Tm["class"] != "background (multi-epoch astrometry)")]
    rg = np.random.default_rng(11); rows, val = [], []
    for tag in TAGS:
        F = pd.read_parquet(os.path.join(C.DER, f"forced_{tag}.parquet"), columns=["source_id", "ep", "x0", "y0", "sig_raw", "S_raw", "good", "half_epoch"])
        z = np.load(os.path.join(C.RAW, f"vlass_{tag}_ql_cutouts_41px.npz")); cut = z["cut"]
        pos = pd.Series(np.arange(len(F)), index=F.source_id.values)
        def beam_of(sid):
            _, j = ctree.query(xyz(sv.ra[sid], sv.dec[sid])); return (comp.BMaj.values[j], comp.BMin.values[j], np.deg2rad(comp.BPA.values[j]))
        def e1corr(sid):
            if tag != "ep1": return 0.0, 0.0
            s_ = str(sv.subtile[sid]).strip()
            return (float(off.dE.get(s_, 0.0)), float(off.dN.get(s_, 0.0))) if s_ in off.index else (0.0, 0.0)
        # w-term validation on the secure natural emitters (Epoch 1 only)
        if tag == "ep1":
            for sid in sec.source_id.unique():
                if sid not in pos.index: continue
                i = pos[sid]; img = cut[i].astype(float) * 1e3; A, x, y = fit(img, F.x0.values[i], F.y0.values[i], beam_of(sid))
                cE, cN = e1corr(sid)
                val.append(dict(source_id=sid, dE_raw=-(x - F.x0.values[i]), dN_raw=y - F.y0.values[i], corrE=cE, corrN=cN,
                                snr=A / F.sig_raw.values[i]))
        good = np.where(F.good.values)[0]
        for sid, name in TARGETS.items():
            if sid not in pos.index: continue
            i = pos[sid]; img = cut[i].astype(float) * 1e3; beam = beam_of(sid); x0, y0 = F.x0.values[i], F.y0.values[i]
            A, x, y = fit(img, x0, y0, beam); cE, cN = e1corr(sid) if APPLY_E1_CORR else (0.0, 0.0)
            dE, dN = -(x - x0) - cE, (y - y0) - cN
            at_bound = bool(max(abs(x - x0), abs(y - y0)) > 2.9)
            # empirical position error: inject the fitted Gaussian into INJ random cutouts with similar noise
            sim = good[np.abs(np.log(F.sig_raw.values[good] / F.sig_raw.values[i])) < 0.15]
            pick = rg.choice(sim, min(INJ, len(sim)), replace=False); ex, ey = [], []
            n = cut.shape[1]; Yg, Xg = np.mgrid[:n, :n]
            for k in pick:
                xk, yk = F.x0.values[k], F.y0.values[k]
                im2 = cut[k].astype(float) * 1e3 + gauss(Xg, Yg, A, xk, yk, *beam)
                _, xf, yf = fit(np.nan_to_num(im2), xk, yk, beam); ex.append(-(xf - xk)); ey.append(yf - yk)
            rs = lambda v: 1.4826 * np.median(np.abs(np.asarray(v) - np.median(v)))
            eE, eN = np.hypot(rs(ex), SYS[tag]), np.hypot(rs(ey), SYS[tag])
            ra_e, de_e = C.propagate(sv.ra[sid], sv.dec[sid], sv.pmra[sid], sv.pmdec[sid], F.ep.values[i], plx=sv.parallax[sid])
            rows.append(dict(source_id=sid, name=name, tag=tag, half_epoch=F.half_epoch.values[i], ep=float(F.ep.values[i]),
                             snr_fit=A / F.sig_raw.values[i], dE=dE, dN=dN, eE=eE, eN=eN, w_term_dE=cE, w_term_dN=cN, at_bound=at_bound,
                             star_ra=float(ra_e), star_dec=float(de_e)))
        del cut, z
        print(tag, "done", flush=True)
    A_ = pd.DataFrame(rows); A_.to_csv(os.path.join(C.RES, "cutout_astrometry.csv"), index=False)
    V = pd.DataFrame(val); V = V[V.snr > 10]
    out = dict(w_term_validation=dict(n=int(len(V)),
                                      rms_dN_raw=float(np.sqrt(np.mean(V.dN_raw ** 2))), rms_dN_corrected=float(np.sqrt(np.mean((V.dN_raw - V.corrN) ** 2))),
                                      rms_dE_raw=float(np.sqrt(np.mean(V.dE_raw ** 2))), rms_dE_corrected=float(np.sqrt(np.mean((V.dE_raw - V.corrE) ** 2))),
                                      median_corrN=float(np.median(V.corrN))), sys_per_tag=SYS, n_inject=INJ, stars={})
    for sid, g in A_.groupby("source_id"):
        g = g[(g.snr_fit >= 3) & ~g.at_bound].sort_values("ep")   # epochs with a measurable source
        if len(g) < 2: continue
        wE, wN = 1 / g.eE ** 2, 1 / g.eN ** 2
        c_at = float(np.sum(g.dE ** 2 * wE + g.dN ** 2 * wN))
        cE, cN = np.sum(g.dE * wE) / wE.sum(), np.sum(g.dN * wN) / wN.sum()
        c_co = float(np.sum((g.dE - cE) ** 2 * wE + (g.dN - cN) ** 2 * wN))
        mE = (g.star_ra - g.star_ra.iloc[0]) * 3600 * np.cos(np.deg2rad(g.star_dec)); mN = (g.star_dec - g.star_dec.iloc[0]) * 3600
        E, N = g.dE + mE, g.dN + mN                          # radio position relative to the star at the first epoch
        sE, sN = np.sum(E * wE) / wE.sum(), np.sum(N * wN) / wN.sum()
        c_st = float(np.sum((E - sE) ** 2 * wE + (N - sN) ** 2 * wN)); k = len(g)
        out["stars"][str(sid)] = dict(name=TARGETS[sid], n_epochs=int(k), epochs=";".join(g.half_epoch.astype(str)),
                                      motion_arcsec=float(np.hypot(np.ptp(mE), np.ptp(mN))),
                                      chi2_at_star=c_at, dof_at_star=2 * k, p_at_star=float(chi2.sf(c_at, 2 * k)),
                                      chi2_comoving=c_co, chi2_stationary=c_st, dof_2=2 * k - 2,
                                      p_comoving=float(chi2.sf(c_co, 2 * k - 2)), p_stationary=float(chi2.sf(c_st, 2 * k - 2)),
                                      offset_comoving=[float(cE), float(cN)])
    json.dump(out, open(os.path.join(C.DER, "cutout_astrometry.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
