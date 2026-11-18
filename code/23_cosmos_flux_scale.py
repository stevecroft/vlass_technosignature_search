"""23 - Sub-threshold flux scale of the forced photometry, tested on the VLA-COSMOS 3 GHz catalog.

The injection tests of 17 add sources to the restored images, so they test the estimator but not the imaging (snapshot
surveys can recover faint flux incompletely; White et al. 2007). VLA-COSMOS 3 GHz (Smolcic et al. 2017; 0.75"
resolution, 2.3 uJy rms, same band) provides 10,830 sources with known flux densities in 2 deg^2 of VLASS sky.
Compact (unresolved at 0.75", single-component) sources with no other catalog source within R_ISO are measured in
half-epochs 1.1, 2.1 and 3.1 (one per epoch) with exactly the estimator of 17 (bilinear S at the source, MAD sigma beyond 10",
eta, kappa, QL peak bias). The inverse-variance-weighted mean ratio S_VLASS / S_COSMOS in bins of S_COSMOS, from
~30 uJy (0.2 sigma per epoch) to a few mJy, tests whether faint emission is recovered with the calibrated flux scale.
Inputs : HEASARC TAP vlacos3ghz (cached), data/raw/vlass_ql_wcs_cadc.parquet, data/derived/forced_calibration.json
Outputs: data/raw/vla_cosmos_3ghz.csv, data/raw/cosmos_forced_parts/*.csv (per-image measurements), results/cosmos_forced.csv,
         results/cosmos_flux_scale.csv, data/derived/cosmos_flux_scale.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import io, json, time
import numpy as np, pandas as pd, requests, pyvo
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree

HB, R_NOISE, R_ISO, CHUNK = 20, 10.0, 8.0, 40      # CADC returns HTTP 500 for ~150 SUBs in one request
TAG = {"1.1": "ep1", "1.2": "ep1", "2.1": "ep2", "2.2": "ep2", "3.1": "ep3", "3.2": "ep3", "4.1": "ep4"}
BINS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 30.0]


def mkwcs(rw):
    w = WCS(naxis=2); w.wcs.ctype = ["RA---SIN", "DEC--SIN"]; w.wcs.crpix = [rw.crpix1, rw.crpix2]
    w.wcs.crval = [rw.crval1, rw.crval2]; w.wcs.cd = [[rw.cd11, rw.cd12], [rw.cd21, rw.cd22]]; return w


def xyz(ra, de):
    ra, de = np.deg2rad(ra), np.deg2rad(de); return np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]


if __name__ == "__main__":
    fc = os.path.join(C.RAW, "vla_cosmos_3ghz.csv")
    if not os.path.exists(fc):
        t = pyvo.dal.TAPService("https://heasarc.gsfc.nasa.gov/xamin/vo/tap").run_sync(
            "SELECT name, ra, dec, flux_3_ghz, flux_3_ghz_error, rms_3_ghz, snr_3_ghz, resolved_flag, multi_component_flag FROM vlacos3ghz", maxrec=100000)
        t.to_table().to_pandas().to_csv(fc, index=False)
    K = pd.read_csv(fc)
    tr = cKDTree(xyz(K.ra.values, K.dec.values)); nn = tr.query(xyz(K.ra.values, K.dec.values), k=2)[0][:, 1]
    K["nn_arcsec"] = np.rad2deg(2 * np.arcsin(nn / 2)) * 3600
    T = K[(K.resolved_flag == 0) & (K.multi_component_flag == 0) & (K.nn_arcsec > R_ISO)].reset_index(drop=True)
    print("COSMOS sources", len(K), "compact isolated", len(T), flush=True)
    wq = pd.read_parquet(os.path.join(C.RAW, "vlass_ql_wcs_cadc.parquet")); wq["mjd"] = C.image_mjd(wq)
    wq["subtile"] = wq.uri.str.extract(r"ql\.(T\d+t\d+\.J\d+[+-]\d+)\.")[0]; wq["ver"] = wq.uri.str.extract(r"\.v(\d+)\.")[0].astype(int)
    img = wq.sort_values("ver").groupby(["epoch", "subtile"]).tail(1).reset_index(drop=True)
    img = img[(np.abs(img.crval2 - 2.2) < 2.5) & (np.abs(img.crval1 - 150.1) < 2.5)].reset_index(drop=True)
    cal = json.load(open(os.path.join(C.DER, "forced_calibration.json"))); kappa = cal["kappa"]
    HALVES = ["1.1", "2.1", "3.1"]                 # one half-epoch per epoch (COSMOS lies in the 1.1 half of the sky)
    MEAS = os.path.join(C.RAW, "cosmos_forced_parts"); os.makedirs(MEAS, exist_ok=True)   # per-image measurements (resumable)
    src = SkyCoord(T.ra.values * u.deg, T.dec.values * u.deg)
    Yg, Xg = np.mgrid[:2 * HB + 1, :2 * HB + 1]; ring = np.hypot(Xg - HB, Yg - HB) > R_NOISE
    from concurrent.futures import ThreadPoolExecutor
    def get_chunk(fn, bb):
        sess = requests.Session()
        for t_ in range(3):
            try:
                r = sess.get(C.CADC_DATA + fn, params=[("SUB", f"[{X-HB}:{X+HB},{Y-HB}:{Y+HB},*,*]") for _, X, Y, _, _ in bb], timeout=180)
                if r.status_code != 200 or r.content[:6] != b"SIMPLE": raise RuntimeError(f"HTTP {r.status_code}")
                arr = np.array([np.squeeze(h.data).astype(np.float32) for h in fits.open(io.BytesIO(r.content)) if h.data is not None])
                if arr.shape != (len(bb), 2 * HB + 1, 2 * HB + 1): raise RuntimeError("shape")
                return bb, arr
            except Exception:
                time.sleep(5 * (t_ + 1))
        print("failed", fn, bb[0][0], flush=True); return bb, None
    for half in HALVES:
        I = img[img.epoch == half].reset_index(drop=True)
        if not len(I): continue
        ii, d2, _ = src.match_to_catalog_sky(SkyCoord(I.crval1.values * u.deg, I.crval2.values * u.deg))
        eta = cal[TAG[half]]["eta"]; bp = float(C.peak_bias(np.array([half]))[0])
        for k in np.unique(ii[d2.deg < 0.72]):
            rw = I.iloc[k]; fn = rw.uri.split("/")[-1]; fo = os.path.join(MEAS, f"{half}_{fn.split('.ql.')[1][:22]}.csv")
            if os.path.exists(fo): continue
            w = mkwcs(rw); sel = np.where((ii == k) & (d2.deg < 0.72))[0]
            xs, ys = w.world_to_pixel(src[sel]); boxes = []
            for s_, x, y in zip(sel, np.atleast_1d(xs), np.atleast_1d(ys)):
                X, Y = int(round(float(x))) + 1, int(round(float(y))) + 1
                if X - HB < 1 or Y - HB < 1 or X + HB > rw.nx or Y + HB > rw.ny: continue
                boxes.append((s_, X, Y, float(x) + 1 - X, float(y) + 1 - Y))
            rows = []
            with ThreadPoolExecutor(3) as ex:
                for bb, arr in ex.map(lambda c0: get_chunk(fn, boxes[c0:c0 + CHUNK]), range(0, len(boxes), CHUNK)):
                    if arr is None: continue
                    for (s_, X, Y, ox, oy), c in zip(bb, arr):
                        c = c.astype(float) * 1e3
                        if not np.isfinite(c[HB, HB]): continue
                        v = c[ring]; sig = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v)))
                        x0, y0 = HB + ox, HB + oy; ix, iy = int(np.floor(x0)), int(np.floor(y0)); fx, fy = x0 - ix, y0 - iy
                        cc = np.nan_to_num(c)
                        S = (1 - fx) * (1 - fy) * cc[iy, ix] + fx * (1 - fy) * cc[iy, ix + 1] + (1 - fx) * fy * cc[iy + 1, ix] + fx * fy * cc[iy + 1, ix + 1]
                        rows.append(dict(name=T.name[s_], half_epoch=half, S_cosmos=T.flux_3_ghz[s_], S_mJy=S / kappa / bp,
                                         sigma_mJy=eta * sig / kappa / bp, snr=S / (eta * sig)))
            pd.DataFrame(rows).to_csv(fo, index=False); print(half, fn, len(boxes), "boxes", len(rows), "measured", flush=True)
    import glob
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(MEAS, "*.csv"))) if os.path.getsize(f) > 10], ignore_index=True)
    D.to_csv(os.path.join(C.RES, "cosmos_forced.csv"), index=False)
    D = D[D.sigma_mJy < 3 * D.sigma_mJy.median()]                 # drop sidelobe-dominated cutouts
    out_rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        x = D[(D.S_cosmos >= lo) & (D.S_cosmos < hi)]
        if len(x) < 5: continue
        w = 1 / x.sigma_mJy ** 2; mS = np.sum(w * x.S_mJy) / w.sum(); eS = w.sum() ** -0.5
        mC = np.sum(w * x.S_cosmos) / w.sum()
        # error of the ratio also from a bootstrap over sources (includes source variability and confusion)
        rg = np.random.default_rng(1)
        a_ = (w * x.S_mJy).groupby(x.name).sum().values; b_ = (w * x.S_cosmos).groupby(x.name).sum().values
        bs = []
        for _ in range(1000):
            i_ = rg.integers(0, len(a_), len(a_)); bs.append(a_[i_].sum() / b_[i_].sum())
        out_rows.append(dict(S_lo=lo, S_hi=hi, n_src=int(x.name.nunique()), n_meas=int(len(x)), mean_S_cosmos=mC, mean_S_vlass=mS,
                             mean_S_vlass_err=eS, ratio=mS / mC, ratio_err=float(np.std(bs)), median_snr_per_epoch=float(np.median(x.snr))))
    O = pd.DataFrame(out_rows); O.to_csv(os.path.join(C.RES, "cosmos_flux_scale.csv"), index=False)
    faint = D[D.S_cosmos < 0.5]; w = 1 / faint.sigma_mJy ** 2
    summ = dict(n_cosmos=int(len(K)), n_compact_isolated=int(len(T)), n_measurements=int(len(D)), n_sources_measured=int(D.name.nunique()),
                ratio_below_0p5mJy=float(np.sum(w * faint.S_mJy) / np.sum(w * faint.S_cosmos)),
                ratio_below_0p5mJy_err=float(np.sum(w) ** -0.5 / (np.sum(w * faint.S_cosmos) / np.sum(w))),
                median_sigma_mJy=float(D.sigma_mJy.median()))
    json.dump(summ, open(os.path.join(C.DER, "cosmos_flux_scale.json"), "w"), indent=1)
    print(O.round(3).to_string(index=False)); print(json.dumps(summ, indent=1))
