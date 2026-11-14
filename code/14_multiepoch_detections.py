"""14 - Multi-epoch VLASS photometry and astrometry of the securely detected stars (Garrett 2026 diagnostics:
variability and co-motion with the host star).

For each secure association (R > 0.5) we take every Quick Look image (latest version per epoch and subtile) that
covers the star's epoch-propagated position in epochs 1.1-4.1, and fetch one multi-SUB request per image: box 0 is
the target, boxes 1.. are bright, compact, isolated Epoch-1 catalog components within 0.45 deg (frame references).

Per image
  * frame tie: median (fit - catalog) offset of the references; its scatter / sqrt(n) is the tie uncertainty
  * flux scale: median (fit peak / catalog Fpeak) of the references, an empirical per-epoch scale relative to the
    Epoch-1 catalog, kept as a check; fluxes are divided by the QL peak bias of their half-epoch (0.85 in 1.1,
    0.92 later; Kimball et al. 2026). Reference positions are the w-term-corrected Epoch-1 positions (02b).
  * target: 2-D Gaussian fit if the peak within 3" of the star is >= 5 sigma (robust sigma from r > 10"); otherwise
    a forced peak measurement at the star's position
Per star
  * light curve, chi2 about the weighted mean, modulation index, detection fraction
  * astrometry: after a global Gaia tie (median radio - star offset over all detections), chi2 of the
    'moves with the star' model and of a 'stationary' model (fixed weighted-mean position), and the proper-motion
    lever arm |mu| * (t_last - t_first)

Inputs : results/vlass_star_matches_classified.csv, data/derived/ms_sample_vlass.parquet,
         data/raw/vlass_e1ql_components.parquet, data/raw/vlass_ql_wcs_cadc.parquet
Outputs: data/raw/multiepoch_detections/<source_id>_<image>.fits (cached multi-extension cutouts)
         results/vlass_detections_multiepoch.csv (one row per star and image)
         results/vlass_detections_variability.csv (one row per star)
Usage  : python 14_multiepoch_detections.py [--threads 2] [--limit N]
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import io, time, argparse
import numpy as np, pandas as pd, requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--candidates", action="store_true", help="measure the unexplained all-epoch candidates of 18 instead of the secure associations")
a = ap.parse_args()
CACHE = os.path.join(C.RAW, "multiepoch_detections"); os.makedirs(CACHE, exist_ok=True)
HB = 30                     # half-box in pixels (1"/px): 61x61 cutouts
DET = 5.0                   # detection threshold (sigma) for a fitted measurement
SYS_POS = 0.10              # per-axis systematic position floor (arcsec) added in quadrature
BEAM = 2.5                  # nominal QL beam FWHM (arcsec) for the SNR-based position error


def g2d(xy, A, x0, y0, sx, sy, th, c):
    x, y = xy; ct, st = np.cos(th), np.sin(th)
    aa = ct**2 / (2*sx**2) + st**2 / (2*sy**2); b = st*ct*(1/(2*sy**2) - 1/(2*sx**2)); cc = st**2/(2*sx**2) + ct**2/(2*sy**2)
    return A*np.exp(-(aa*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + cc*(y-y0)**2)) + c


def mkwcs(rw):
    w = WCS(naxis=2); w.wcs.ctype = ["RA---SIN", "DEC--SIN"]; w.wcs.crpix = [rw.crpix1, rw.crpix2]
    w.wcs.crval = [rw.crval1, rw.crval2]; w.wcs.cd = [[rw.cd11, rw.cd12], [rw.cd21, rw.cd22]]; return w


def robust_sigma(d):
    d = d[np.isfinite(d)]; return 1.4826 * np.median(np.abs(d - np.median(d))) if d.size else np.nan


def measure(d, x0, y0, search=3.0, box=10, force=False):
    """d: cutout (Jy/beam) with the expected position at pixel (x0, y0). Returns a dict (mJy, pixels)."""
    Y, X = np.mgrid[:d.shape[0], :d.shape[1]]; r = np.hypot(X - x0, Y - y0)
    sig = robust_sigma(d[r > 10]) * 1e3; dm = d * 1e3
    near = np.where(r <= search, dm, -np.inf); jy, jx = np.unravel_index(np.argmax(near), dm.shape); pk = dm[jy, jx]
    xi, yi = int(round(x0)), int(round(y0))
    forced = float(dm[yi, xi]) if 0 <= yi < dm.shape[0] and 0 <= xi < dm.shape[1] else np.nan
    out = dict(sigma=sig, forced_peak=forced, max_peak=float(pk), detected=bool(np.isfinite(sig) and pk >= DET * sig))
    if not (out["detected"] or force): return out
    sl = (slice(max(jy - box, 0), jy + box + 1), slice(max(jx - box, 0), jx + box + 1))
    st = dm[sl]; Ys, Xs = Y[sl], X[sl]; ok = np.isfinite(st)
    try:
        p, cov = curve_fit(g2d, (Xs[ok], Ys[ok]), st[ok], p0=[pk, jx, jy, 1.1, 0.9, 0, 0], maxfev=20000)
        e = np.sqrt(np.abs(np.diag(cov)))
        out.update(fit_peak=p[0], fit_x=p[1], fit_y=p[2], fit_ex=e[1], fit_ey=e[2], fwhm_a=2.355 * abs(p[3]), fwhm_b=2.355 * abs(p[4]),
                   fit_ok=bool(np.isfinite(e[1]) and abs(p[1] - jx) < 3 and abs(p[2] - jy) < 3))
    except Exception:
        out.update(fit_ok=False)
    return out


def main():
    if a.candidates:      # unexplained all-epoch candidates (18); same measurement, frame tie taken from the secure run
        T = pd.read_csv(os.path.join(C.RES, "alleepoch_candidates.csv")); T = T[~T.explained.astype(bool)].copy()
        T["class"] = "unexplained candidate"; T["rel"] = np.nan; T["RAJ2000"] = np.nan; T["DEJ2000"] = np.nan
        T = T.drop(columns=["dist_pc"]).reset_index(drop=True)
    else:
        T = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv"))
        T = T[T.rel > 0.5].drop_duplicates("source_id").reset_index(drop=True)
    SUF = "_candidates" if a.candidates else ""
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "dist_pc"])
    T = T[["source_id", "main_id", "class", "rel", "RAJ2000", "DEJ2000"]].merge(sv, on="source_id", how="left")
    if a.limit: T = T.head(a.limit)
    v = C.e1_components(columns=["RAJ2000", "DEJ2000", "Fpeak", "DCMaj", "NNdist", "SCode", "DupFlag"])
    v = v[(v.DupFlag < 2) & (v.SCode == "S") & (v.Fpeak > 4) & (v.DCMaj < 1.5) & (v.NNdist > 30)].reset_index(drop=True)
    vc = SkyCoord(v.RAJ2000.values * u.deg, v.DEJ2000.values * u.deg)
    wq = pd.read_parquet(os.path.join(C.RAW, "vlass_ql_wcs_cadc.parquet"))
    wq["mjd"] = C.image_mjd(wq)          # fill CAOM records without time bounds (see 08b_fill_image_dates.py)
    wq["subtile"] = wq.uri.str.extract(r"ql\.(T\d+t\d+\.J\d+[+-]\d+)\.")[0]; wq["ver"] = wq.uri.str.extract(r"\.v(\d+)\.")[0].astype(int)
    img = wq.sort_values("ver").groupby(["epoch", "subtile"]).tail(1).reset_index(drop=True)
    ic = SkyCoord(img.crval1.values * u.deg, img.crval2.values * u.deg)

    jobs = []
    for s in T.itertuples():
        sc = SkyCoord(s.ra * u.deg, s.dec * u.deg)
        refs = v[vc.separation(sc).deg < 0.45]
        refs = refs.assign(d=SkyCoord(refs.RAJ2000.values * u.deg, refs.DEJ2000.values * u.deg).separation(sc).deg)
        refs = refs[refs.d > 0.01].nsmallest(6, "d")
        for k in np.where(ic.separation(sc).deg < 1.0)[0]:
            rw = img.iloc[k]; ep = Time(rw.mjd, format="mjd").jyear
            ra, de = C.propagate(np.array([s.ra]), np.array([s.dec]), np.array([s.pmra]), np.array([s.pmdec]), ep, plx=np.array([s.parallax]))
            w = mkwcs(rw); x, y = [float(q) for q in w.world_to_pixel(SkyCoord(ra[0] * u.deg, de[0] * u.deg))]
            if not (HB + 1 <= x + 1 <= rw.nx - HB and HB + 1 <= y + 1 <= rw.ny - HB): continue
            rx, ry = w.world_to_pixel(SkyCoord(refs.RAJ2000.values * u.deg, refs.DEJ2000.values * u.deg))
            inb = (rx + 1 >= HB + 1) & (rx + 1 <= rw.nx - HB) & (ry + 1 >= HB + 1) & (ry + 1 <= rw.ny - HB)
            jobs.append(dict(star=s, rw=rw, ep=ep, x=x, y=y, refs=refs[inb], rx=np.atleast_1d(rx)[inb], ry=np.atleast_1d(ry)[inb]))
    print(f"{len(T)} stars, {len(jobs)} images", flush=True)
    sess = requests.Session()

    def run(j):
        s, rw = j["star"], j["rw"]; name = rw.uri.split("/")[-1].replace(".fits", "")
        f = os.path.join(CACHE, f"{s.source_id}_{name}.fits")
        pix = [(j["x"], j["y"])] + list(zip(j["rx"], j["ry"]))
        boxes = [(int(round(px)) + 1, int(round(py)) + 1) for px, py in pix]
        if not os.path.exists(f):
            subs = [f"[{X-HB}:{X+HB},{Y-HB}:{Y+HB},*,*]" for X, Y in boxes]
            for t in range(5):
                try:
                    r = sess.get(C.CADC_DATA + rw.uri.split("/")[-1], params=[("SUB", q) for q in subs], timeout=300)
                    if r.status_code != 200 or r.content[:6] != b"SIMPLE": raise RuntimeError(f"HTTP {r.status_code}")
                    open(f, "wb").write(r.content); break
                except Exception as e:
                    err = str(e); time.sleep(10 * (t + 1))
            else:
                print("FAILED", name, err, flush=True); return None
        ext = [h for h in fits.open(f) if h.data is not None]
        hs = [np.squeeze(h.data).astype(float) for h in ext]
        # 0-based pixel origin of each cutout in the full image, from its own header (robust to cached cutouts
        # whose box was rounded differently from the current expected position)
        org = [(rw.crpix1 - h.header["CRPIX1"], rw.crpix2 - h.header["CRPIX2"]) for h in ext]
        if len(hs) != len(boxes): return None
        w = mkwcs(rw); ps = abs(rw.cd22) * 3600
        # references: frame offset (arcsec) and flux-scale ratio relative to the Epoch-1 catalog
        dE, dN, fr = [], [], []
        for k, rr in enumerate(j["refs"].itertuples(), start=1):
            ox, oy = org[k]; mx, my = j["rx"][k - 1] - ox, j["ry"][k - 1] - oy
            m = measure(hs[k], mx, my, search=2.0, box=8)
            if not (m["detected"] and m.get("fit_ok")): continue
            c_ = w.pixel_to_world(m["fit_x"] + ox, m["fit_y"] + oy)
            dE.append((c_.ra.deg - rr.RAJ2000) * 3600 * np.cos(np.deg2rad(rr.DEJ2000))); dN.append((c_.dec.deg - rr.DEJ2000) * 3600)
            fr.append(m["fit_peak"] / rr.Fpeak)
        nref = len(dE); dE, dN = np.array(dE), np.array(dN)
        oE, oN = (np.median(dE), np.median(dN)) if nref else (np.nan, np.nan)
        sE = 1.4826 * np.median(np.abs(dE - oE)) / np.sqrt(nref) if nref >= 3 else np.nan
        sN = 1.4826 * np.median(np.abs(dN - oN)) / np.sqrt(nref) if nref >= 3 else np.nan
        ox, oy = org[0]; mx, my = j["x"] - ox, j["y"] - oy
        m = measure(hs[0], mx, my)
        row = dict(source_id=s.source_id, main_id=s.main_id, cls=s._3, image=name, epoch_label=rw.epoch, epoch=j["ep"],
                   n_ref=nref, frame_E=oE, frame_N=oN, frame_eE=sE, frame_eN=sN, flux_scale_ref=np.median(fr) if fr else np.nan,
                   sigma_mJy=m["sigma"], forced_peak_mJy=m["forced_peak"], detected=m["detected"])
        if m["detected"] and m.get("fit_ok"):
            c_ = w.pixel_to_world(m["fit_x"] + ox, m["fit_y"] + oy)
            sa, sd = C.propagate(np.array([s.ra]), np.array([s.dec]), np.array([s.pmra]), np.array([s.pmdec]), j["ep"], plx=np.array([s.parallax]))
            cosd = np.cos(np.deg2rad(sd[0])); snr = m["fit_peak"] / m["sigma"]
            ep_ = np.hypot(np.hypot(m["fit_ex"], m["fit_ey"]) / np.sqrt(2) * ps, BEAM / (2 * snr))
            row.update(peak_mJy=m["fit_peak"], snr=snr, fwhm_a=m["fwhm_a"] * ps, fwhm_b=m["fwhm_b"] * ps,
                       dE_raw=(c_.ra.deg - sa[0]) * 3600 * cosd, dN_raw=(c_.dec.deg - sd[0]) * 3600, fit_ok=True,
                       ra_fit=c_.ra.deg, dec_fit=c_.dec.deg, pos_err=ep_)
        return row

    with ThreadPoolExecutor(a.threads) as ex:
        rows = [r for r in ex.map(run, jobs) if r is not None]
    M = pd.DataFrame(rows).sort_values(["source_id", "epoch"]).reset_index(drop=True)
    # overlapping subtiles of one half-epoch image the same visibilities: keep the lowest-noise image per star and half-epoch
    n0 = len(M); M = M.sort_values("sigma_mJy").groupby(["source_id", "epoch_label"]).head(1).sort_values(["source_id", "epoch"]).reset_index(drop=True)
    print(f"kept {len(M)} of {n0} images (one per star and half-epoch)", flush=True)

    for c in ["fit_ok", "dE_raw", "dN_raw", "pos_err", "peak_mJy"]:
        if c not in M: M[c] = np.nan
    M["fit_ok"] = M.fit_ok.fillna(False).astype(bool); M["detected"] = M.detected & M.fit_ok
    # images without usable references: per-epoch median frame offset, scatter as its uncertainty
    for c, e in [("frame_E", "frame_eE"), ("frame_N", "frame_eN")]:
        med = M.groupby("epoch_label")[c].transform("median")
        scat = M.groupby("epoch_label")[c].transform(lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))
        miss = M[c].isna(); M.loc[miss, e] = scat[miss]; M.loc[miss, c] = med[miss]
        last = M[c].isna(); M.loc[last, c] = 0.0; M.loc[last, e] = 0.2        # no references anywhere in the epoch
        M.loc[M[e].isna() & ~last, e] = 0.2
    M["dE_star"] = M.dE_raw - M.frame_E; M["dN_star"] = M.dN_raw - M.frame_N
    # per-epoch flux scale from all reference measurements, relative to the Epoch-1 catalog
    scale = M.groupby("epoch_label").flux_scale_ref.median().fillna(1.0); M["flux_scale"] = M.epoch_label.map(scale)
    for c in ["peak_mJy", "forced_peak_mJy", "sigma_mJy"]:
        M[c.replace("_mJy", "_corr_mJy")] = M[c] / C.peak_bias(M.epoch_label)
    # global Gaia tie: median radio - star offset over all detections (most detected stars are genuine)
    det = M[M.detected & M.dE_star.notna()]
    gE, gN = det.dE_star.median(), det.dN_star.median()
    if a.candidates:      # use the tie measured on the secure associations
        Ms = pd.read_csv(os.path.join(C.RES, "vlass_detections_multiepoch.csv")); Ms = Ms[Ms.detected]
        gE, gN = (Ms.dE_raw - Ms.frame_E).median(), (Ms.dN_raw - Ms.frame_N).median()
    M["dE_star"] -= gE; M["dN_star"] -= gN
    M["eE"] = np.sqrt(M.pos_err**2 + M.frame_eE.fillna(0.1)**2 + SYS_POS**2)
    M["eN"] = np.sqrt(M.pos_err**2 + M.frame_eN.fillna(0.1)**2 + SYS_POS**2)
    M.to_csv(os.path.join(C.RES, f"vlass_detections_multiepoch{SUF}.csv"), index=False)

    S = []
    for sid, g in M.groupby("source_id"):
        s = T[T.source_id == sid].iloc[0]; d = g[g.detected & g.dE_star.notna()]
        f, e = g.peak_corr_mJy.where(g.detected, g.forced_peak_corr_mJy), g.sigma_corr_mJy
        wm = np.sum(f / e**2) / np.sum(1 / e**2); chi2f = np.sum(((f - wm) / e)**2); dof_f = len(g) - 1
        row = dict(source_id=sid, main_id=s.main_id, cls=s["class"], dist_pc=s.dist_pc, pm_masyr=np.hypot(s.pmra, s.pmdec),
                   n_images=len(g), n_det=len(d), epochs=",".join(g.epoch_label.astype(str)),
                   S_mean_mJy=wm, S_max_mJy=f.max(), S_min_mJy=f.min(), mod_index=np.std(f) / np.mean(f) if np.mean(f) > 0 else np.nan,
                   chi2_flux=chi2f, dof_flux=dof_f, p_const_flux=stats.chi2.sf(chi2f, dof_f) if dof_f > 0 else np.nan)
        if len(d) >= 2:
            chi_star = np.sum((d.dE_star / d.eE)**2 + (d.dN_star / d.eN)**2); dof_star = 2 * len(d)
            # stationary model: fixed sky position = weighted mean of the radio positions
            cosd = np.cos(np.deg2rad(d.dec_fit.mean()))
            xE = (d.ra_fit - d.ra_fit.mean()) * 3600 * cosd; xN = (d.dec_fit - d.dec_fit.mean()) * 3600
            wE, wN = 1 / d.eE**2, 1 / d.eN**2
            chi_stat = np.sum(wE * (xE - np.sum(wE * xE) / wE.sum())**2 + wN * (xN - np.sum(wN * xN) / wN.sum())**2); dof_stat = 2 * len(d) - 2
            # co-moving with a constant offset (e.g. emission from a companion): star-track residuals about their weighted mean
            chi_com = np.sum(wE * (d.dE_star - np.sum(wE * d.dE_star) / wE.sum())**2 + wN * (d.dN_star - np.sum(wN * d.dN_star) / wN.sum())**2)
            offE, offN = np.sum(wE * d.dE_star) / wE.sum(), np.sum(wN * d.dN_star) / wN.sum()
            lever = np.hypot(s.pmra, s.pmdec) / 1e3 * (d.epoch.max() - d.epoch.min())
            row.update(chi2_star=chi_star, dof_star=dof_star, p_star=stats.chi2.sf(chi_star, dof_star),
                       chi2_stationary=chi_stat, dof_stationary=dof_stat, p_stationary=stats.chi2.sf(chi_stat, dof_stat) if dof_stat > 0 else np.nan,
                       chi2_comove_offset=chi_com, offset_E=offE, offset_N=offN,
                       dchi2_stationary_minus_comove=chi_stat - chi_com,
                       lever_arm_arcsec=lever, median_pos_err=np.median(np.hypot(d.eE, d.eN)))
        S.append(row)
    S = pd.DataFrame(S).sort_values("dist_pc")
    # verdicts: |dchi2| > 9 between the co-moving (constant offset) and stationary models, which have equal dof
    dc = S.get("dchi2_stationary_minus_comove", pd.Series(np.nan, index=S.index))
    S["motion"] = np.select([S.n_det < 2, dc > 9, dc < -9], ["single detection", "co-moving", "stationary"], "inconclusive")
    S["variable"] = S.p_const_flux < 0.01
    S.to_csv(os.path.join(C.RES, f"vlass_detections_variability{SUF}.csv"), index=False)
    print("per-epoch flux scale vs Epoch-1 catalog:", scale.round(3).to_dict())
    print(f"global Gaia tie: dE={gE:+.3f}\" dN={gN:+.3f}\"; images with detections {int(M.detected.sum())}/{len(M)}")
    print(S[["main_id", "dist_pc", "pm_masyr", "n_images", "n_det", "S_mean_mJy", "S_max_mJy", "mod_index", "p_const_flux",
             "lever_arm_arcsec", "chi2_star", "chi2_comove_offset", "chi2_stationary", "dof_stationary", "dchi2_stationary_minus_comove",
             "offset_E", "offset_N"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
