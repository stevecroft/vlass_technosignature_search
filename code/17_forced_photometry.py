"""17 - Forced photometry of every sample star in every VLASS Quick Look epoch.

For each epoch tag (ep1 = 1.1/1.2, ep2 = 2.1/2.2, ep3 = 3.1/3.2, ep4 = 4.1) every star has one 41x41 px (1"/px) QL
cutout from 09_download_cutouts.py, centred on its proper-motion-propagated position. Here we
  * add the parallax displacement at the image epoch (cutouts were centred without it),
  * measure the flux density by bilinear interpolation of the image at the star's position (an unbiased linear
    estimator; it loses some peak for a source off a pixel centre, calibrated below as kappa),
  * measure the noise as the robust (MAD) rms of pixels more than 10" from the cutout centre,
  * repeat the same estimator at 8 control positions 14" from the star (same sub-pixel phase), which gives the
    empirical noise normalisation eta (std of S/sigma at blank positions) and the empirical >=5 sigma false-positive
    rate per trial, including sidelobes and unrelated sources,
  * correct for the QL peak bias of each half-epoch (0.85 in 1.1, 0.92 later; Kimball et al. 2026).

Calibration (written to data/derived/forced_calibration.json):
  kappa = peak-loss factor of the bilinear estimator, S_true = S_bilinear / kappa. Adopted from injection-recovery:
          elliptical Gaussians with the local restoring beam (BMaj/BMin/BPA of the nearest Epoch-1 catalogue component)
          injected at the star positions of 24,000 random stars and re-measured with this exact code. Cross-check:
          median(bilinear / catalogue Fpeak) for Epoch-1 secure associations with catalogue SNR > 10 (kappa_cat).
          The injection table (detected fraction at 2-15 sigma vs Phi(level - 5)) validates the p_i model of 19.
  eta   = robust std of S/sigma at control positions (per tag); sigma_eff = eta * sigma_MAD.

Inputs : data/raw/vlass_<tag>_ql_cutouts_41px.npz (or data/raw/cutouts_<tag>/shard_*.npz while downloading),
         data/derived/ms_sample_vlass.parquet, data/raw/vlass_ql_wcs_cadc.parquet,
         results/vlass_detections_multiepoch.csv (per-half-epoch flux scales), results/vlass_star_matches_classified.csv
Outputs: data/derived/forced_<tag>.parquet, data/derived/forced_controls_<tag>.npz, data/derived/forced_calibration.json
Usage  : python 17_forced_photometry.py [--tags ep1,ep2,ep3,ep4]
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import glob, json, argparse
import numpy as np, pandas as pd
from astropy.time import Time

ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="ep1,ep2,ep3,ep4")
ap.add_argument("--inject", type=int, default=24000, help="stars per tag in the injection-recovery test")
a = ap.parse_args()
LEVELS = np.array([2, 3, 4, 5, 6, 8, 10, 15])   # injected true peak in units of the effective noise
HB = 20                      # cutout half-size (px); star nominally at pixel (HB + off)
R_NOISE = 10.0               # noise from pixels beyond this radius (arcsec = px)
R_CTRL, N_CTRL = 14.0, 8     # control positions
CHUNK = 20000


def load_tag(tag):
    f = os.path.join(C.RAW, f"vlass_{tag}_ql_cutouts_41px.npz")
    if os.path.exists(f):
        z = np.load(f); d = {k: z[k] for k in z.files}; d["partial"] = False
    else:
        sh = sorted(glob.glob(os.path.join(C.RAW, f"cutouts_{tag}", "shard_*.npz")))
        if not sh: return None
        zs = [np.load(s) for s in sh]
        st = np.concatenate([z["star"] for z in zs]); _, iu = np.unique(st, return_index=True)
        d = dict(star_index=st[iu], cut=np.concatenate([z["cut"] for z in zs])[iu],
                 off=np.concatenate([z["off"] for z in zs])[iu], ep=np.concatenate([z["ep"] for z in zs])[iu], partial=True)
    return d


def bilinear(cut, x, y):
    """cut: (N, n, n); x, y: (N,) pixel coordinates (0-based, x = column). Returns (N,) interpolated values."""
    n = cut.shape[1]
    ix = np.clip(np.floor(x).astype(int), 0, n - 2); iy = np.clip(np.floor(y).astype(int), 0, n - 2)
    fx = x - ix; fy = y - iy; k = np.arange(len(x))
    return ((1 - fx) * (1 - fy) * cut[k, iy, ix] + fx * (1 - fy) * cut[k, iy, ix + 1] +
            (1 - fx) * fy * cut[k, iy + 1, ix] + fx * fy * cut[k, iy + 1, ix + 1])


def half_epoch_of(ep, ranges):
    lab = np.full(len(ep), "", dtype=object)
    for e, (lo, hi) in ranges.items():
        lab[(ep >= lo - 0.01) & (ep <= hi + 0.01)] = e
    return lab


def measure(d, sv):
    st = d["star_index"]; s = sv.iloc[st]
    ep = d["ep"] if "ep" in d else s.obs_jyear.values           # Epoch-1 array predates the `ep` field
    a0, d0 = C.propagate(s.ra.values, s.dec.values, s.pmra.values, s.pmdec.values, ep)
    a1, d1 = C.propagate(s.ra.values, s.dec.values, s.pmra.values, s.pmdec.values, ep, plx=s.parallax.values)
    dxp = -(a1 - a0) * 3600 * np.cos(np.deg2rad(s.dec.values)); dyp = (d1 - d0) * 3600   # QL: cd11=-1", cd22=+1", no rotation
    x0 = HB + d["off"][:, 0] + dxp; y0 = HB + d["off"][:, 1] + dyp
    n = d["cut"].shape[1]; Y, X = np.mgrid[:n, :n]; ring = np.hypot(X - HB, Y - HB) > R_NOISE
    core = np.hypot(X - HB, Y - HB) <= 1.5
    th = np.arange(N_CTRL) * 2 * np.pi / N_CTRL
    out = {k: np.empty(len(st)) for k in ["S", "sig", "maxcore"]}; ctrl = np.empty((len(st), N_CTRL), np.float32)
    for i in range(0, len(st), CHUNK):
        c = d["cut"][i:i + CHUNK].astype(np.float64) * 1e3                                   # mJy/beam
        v = c[:, ring]; med = np.nanmedian(v, axis=1)
        out["sig"][i:i + CHUNK] = 1.4826 * np.nanmedian(np.abs(v - med[:, None]), axis=1)
        xs, ys = x0[i:i + CHUNK], y0[i:i + CHUNK]
        out["S"][i:i + CHUNK] = bilinear(np.nan_to_num(c), xs, ys)
        out["maxcore"][i:i + CHUNK] = np.nanmax(np.where(core, c, -np.inf).reshape(len(c), -1), axis=1)
        for j, t in enumerate(th):
            ctrl[i:i + CHUNK, j] = bilinear(np.nan_to_num(c), xs + R_CTRL * np.cos(t), ys + R_CTRL * np.sin(t))
    good = np.isfinite(out["sig"]) & (out["sig"] > 0) & np.isfinite(d["cut"][:, HB, HB])
    return dict(star_index=st, source_id=s.source_id.values, ep=ep, x0=x0, y0=y0, plx_dx=dxp, plx_dy=dyp,
                S_raw=out["S"], sig_raw=out["sig"], maxcore_raw=out["maxcore"], good=good), ctrl



def inject(d, F, cal, tag, n, comp_tree, comp):
    """Inject elliptical Gaussians with the local restoring beam at the star positions of n random good stars and
    re-measure with the identical estimator. Returns a table and the injected peak-loss factor."""
    from scipy.spatial import cKDTree
    rg = np.random.default_rng(7); good = np.where(F.good.values)[0]; pick = np.sort(rg.choice(good, min(n, len(good)), replace=False))
    sv_ = SV.iloc[F.star_index.values[pick]]
    xyz = lambda r, de: np.c_[np.cos(np.deg2rad(de)) * np.cos(np.deg2rad(r)), np.cos(np.deg2rad(de)) * np.sin(np.deg2rad(r)), np.sin(np.deg2rad(de))]
    _, jb = comp_tree.query(xyz(sv_.ra.values, sv_.dec.values))
    bmaj, bmin, bpa = comp.BMaj.values[jb], comp.BMin.values[jb], np.deg2rad(comp.BPA.values[jb])
    lev = LEVELS[np.arange(len(pick)) % len(LEVELS)]
    eta, kappa = cal[tag]["eta"], cal["kappa"]
    cut = d["cut"][pick].astype(np.float64) * 1e3; x0, y0 = F.x0.values[pick], F.y0.values[pick]
    sig_true = eta * F.sig_raw.values[pick] / kappa; A = lev * sig_true          # true peak, catalogue scale (mJy/beam)
    n_ = cut.shape[1]; Y, X = np.mgrid[:n_, :n_]
    dE = -(X[None] - x0[:, None, None]); dN = Y[None] - y0[:, None, None]          # +x = west
    u = dE * np.sin(bpa)[:, None, None] + dN * np.cos(bpa)[:, None, None]; w = dE * np.cos(bpa)[:, None, None] - dN * np.sin(bpa)[:, None, None]
    G = np.exp(-4 * np.log(2) * (u ** 2 / bmaj[:, None, None] ** 2 + w ** 2 / bmin[:, None, None] ** 2))
    c2 = cut + A[:, None, None] * G
    ring = np.hypot(X - HB, Y - HB) > R_NOISE
    v = c2[:, ring]; med = np.nanmedian(v, axis=1); sig2 = 1.4826 * np.nanmedian(np.abs(v - med[:, None]), axis=1)
    S2 = bilinear(np.nan_to_num(c2), x0, y0)
    T = pd.DataFrame(dict(source_id=F.source_id.values[pick], level=lev, A=A, bmaj=bmaj, bmin=bmin, S_before=F.S_raw.values[pick],
                          S_after=S2, snr_before=F.S_raw.values[pick] / (eta * F.sig_raw.values[pick]), snr_after=S2 / (eta * sig2)))
    T["kappa_inj"] = (T.S_after - T.S_before) / T.A
    return T

if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                         columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "obs_jyear"]); SV = sv
    wq = pd.read_parquet(os.path.join(C.RAW, "vlass_ql_wcs_cadc.parquet")); wq["mjd"] = C.image_mjd(wq)
    wq["jy"] = Time(wq.mjd.values, format="mjd").jyear
    ranges = {str(e): (g.jy.min(), g.jy.max()) for e, g in wq.groupby("epoch")}
    me = pd.read_csv(os.path.join(C.RES, "vlass_detections_multiepoch.csv"))
    fscale = me.groupby("epoch_label").flux_scale.first(); fscale.index = fscale.index.map(lambda v: f"{v:.1f}")
    T = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); sec = T[T.secure].drop_duplicates("source_id")
    calf = os.path.join(C.DER, "forced_calibration.json")
    cal = json.load(open(calf)) if os.path.exists(calf) else {}

    comp_tree = comp = None
    for tag in a.tags.split(","):
        d = load_tag(tag)
        if d is None: print(tag, ": no cutouts yet"); continue
        if "source_id" in d:   # authoritative: the Epoch-1 array's star_index refers to an older row order of the sample
            idx = np.searchsorted(sv.source_id.values, d["source_id"])
            assert np.array_equal(sv.source_id.values[idx], d["source_id"]), "cutout source_id not in sample"
            d["star_index"] = idx
        m, ctrl = measure(d, sv); F = pd.DataFrame(m)
        F["half_epoch"] = half_epoch_of(F.ep.values, ranges)
        assert (F.half_epoch != "").all(), "cutout epoch outside every half-epoch range"
        if tag == "ep1":    # peak-loss calibration against the catalogue, same images (cross-check)
            S = sec.merge(F, on="source_id"); S = S[(S.Fpeak / S.Islrms > 10) & S.good]
            r = S.S_raw / S.Fpeak
            cal["kappa_cat"] = float(np.median(r)); cal["kappa_cat_err"] = float(1.2533 * 1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(len(r)))
            cal["kappa_cat_n"] = int(len(r))
        g = F.good.values; z = ctrl[g] / F.sig_raw.values[g, None]; z = z[np.isfinite(z)]
        eta = float(1.4826 * np.median(np.abs(z - np.median(z))))
        cal[tag] = dict(eta=eta)
        if comp_tree is None:
            from scipy.spatial import cKDTree
            comp = C.e1_components(columns=["RAJ2000", "DEJ2000", "BMaj", "BMin", "BPA"])
            cx = np.c_[np.cos(np.deg2rad(comp.DEJ2000)) * np.cos(np.deg2rad(comp.RAJ2000)), np.cos(np.deg2rad(comp.DEJ2000)) * np.sin(np.deg2rad(comp.RAJ2000)), np.sin(np.deg2rad(comp.DEJ2000))]
            comp_tree = cKDTree(cx)
        if tag == "ep1":    # adopted peak-loss factor: injection of the restoring beam through this exact estimator
            cal["kappa"] = cal["kappa_cat"]
            IJ0 = inject(d, F, cal, tag, a.inject, comp_tree, comp)
            cal["kappa"] = float(np.median(IJ0.kappa_inj)); cal["kappa_err"] = float(1.2533 * IJ0.kappa_inj.std() / np.sqrt(len(IJ0)))
        kappa = cal["kappa"]
        IJ = inject(d, F, cal, tag, a.inject, comp_tree, comp)       # levels now in units of the adopted effective noise
        IJ.to_parquet(os.path.join(C.DER, f"forced_inject_{tag}.parquet"), index=False)
        from scipy.stats import norm
        rec = IJ.groupby("level").apply(lambda q: pd.Series(dict(frac_det=np.mean(q.snr_after >= 5), model=norm.cdf(q.name - 5),
                                                                 kappa_inj=np.median(q.kappa_inj), n=len(q))), include_groups=False)
        bp = C.peak_bias(F.half_epoch.astype(str).values)                    # QL peak bias (Kimball et al. 2026)
        F["S_mJy"] = F.S_raw / kappa / bp
        F["sigma_mJy"] = eta * F.sig_raw / kappa / bp
        F["snr"] = F.S_raw / (eta * F.sig_raw)
        F["snr_maxcore"] = F.maxcore_raw / (eta * F.sig_raw)
        zc = z / eta
        cal[tag] = dict(n_stars=int(len(F)), n_good=int(g.sum()), partial=bool(d["partial"]), eta=eta,
                        ctrl_mean=float(np.mean(zc)), ctrl_std=float(np.std(zc)),
                        fp_rate_ge5=float(np.mean(zc >= 5)), fp_rate_ge5_err=float(np.sqrt(max((zc >= 5).sum(), 1)) / len(zc)),
                        fp_rate_le_m5=float(np.mean(zc <= -5)),
                        star_mean_snr=float(F.snr[g].mean()), star_rate_ge5=float(np.mean(F.snr[g] >= 5)),
                        half_epochs=F.half_epoch.value_counts().to_dict(),
                        inject=dict(kappa_inj=float(np.median(IJ.kappa_inj)), table=rec.reset_index().to_dict(orient="list")))
        F.to_parquet(os.path.join(C.DER, f"forced_{tag}.parquet"), index=False)
        np.savez_compressed(os.path.join(C.DER, f"forced_controls_{tag}.npz"), star_index=F.star_index.values,
                            ctrl_snr=(ctrl / (eta * F.sig_raw.values[:, None])).astype(np.float32))
        print(tag, "eta %.3f ctrl_std %.3f fp(>=5) %.2e" % (eta, cal[tag]["ctrl_std"], cal[tag]["fp_rate_ge5"]), flush=True)
        print(rec.round(3).to_string(), flush=True)
    json.dump(cal, open(calf, "w"), indent=1)
    print("kappa adopted (injection) = %.3f +- %.3f; catalogue cross-check %.3f +- %.3f (N=%d)" % (
        cal["kappa"], cal["kappa_err"], cal["kappa_cat"], cal["kappa_cat_err"], cal["kappa_cat_n"]))
