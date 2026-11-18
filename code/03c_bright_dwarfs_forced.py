"""03c - Forced photometry of the bright nearby dwarfs missing from the Gaia DR3 sample (03b), at their SIMBAD
(Hipparcos-based) positions, with the estimator of 17.

For each star (results/bright_dwarfs_missing.csv) the SIMBAD J2000 position, proper motion and parallax are propagated
to each Quick Look image epoch; a 41x41 px CADC cutout is taken from the image whose centre is nearest (within 0.72 deg),
one per half-epoch (1.1-4.1), and S (bilinear at the star), sigma (MAD rms beyond 10"), the S/N with the per-epoch
noise factor eta, the peak-loss factor kappa and the QL peak bias are computed exactly as in 17. The coadd over
half-epochs gives a steady limit S_lim = (5 + 1.645) sigma_coadd, or S_coadd + 1.645 sigma_coadd if larger.
These stars are not added to the population analysis (their selection differs from the Gaia sample); the per-star
limits close the gap left by Gaia's incompleteness for the brightest stars.
Inputs : results/bright_dwarfs_missing.csv, SIMBAD TAP (proper motions; cached), data/raw/vlass_ql_wcs_cadc.parquet,
         data/derived/forced_calibration.json
Outputs: data/raw/bright_dwarfs_simbad_pm.csv, data/raw/bright_dwarfs_cutouts.npz,
         results/bright_dwarfs_forced.csv (per half-epoch), results/bright_dwarfs_limits.csv (per star)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import io, json, time
import numpy as np, pandas as pd, requests, pyvo
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

HB, R_NOISE, THR, Z = 20, 10.0, C.DET_SIGMA, C.Z95
TAG = {"1.1": "ep1", "1.2": "ep1", "2.1": "ep2", "2.2": "ep2", "3.1": "ep3", "3.2": "ep3", "4.1": "ep4"}


def mkwcs(rw):
    w = WCS(naxis=2); w.wcs.ctype = ["RA---SIN", "DEC--SIN"]; w.wcs.crpix = [rw.crpix1, rw.crpix2]
    w.wcs.crval = [rw.crval1, rw.crval2]; w.wcs.cd = [[rw.cd11, rw.cd12], [rw.cd21, rw.cd22]]; return w


def bilinear(c, x, y):
    ix, iy = int(np.floor(x)), int(np.floor(y)); fx, fy = x - ix, y - iy
    return ((1 - fx) * (1 - fy) * c[iy, ix] + fx * (1 - fy) * c[iy, ix + 1] + (1 - fx) * fy * c[iy + 1, ix] + fx * fy * c[iy + 1, ix + 1])


if __name__ == "__main__":
    B = pd.read_csv(os.path.join(C.RES, "bright_dwarfs_missing.csv"))
    fpm = os.path.join(C.RAW, "bright_dwarfs_simbad_pm.csv")
    if not os.path.exists(fpm):
        names = ",".join("'" + n.replace("'", "''") + "'" for n in B.main_id)
        q = f"SELECT main_id, ra, dec, pmra, pmdec, plx_value FROM basic WHERE main_id IN ({names})"
        pyvo.dal.TAPService(C.SIMBAD_TAP).run_sync(q).to_table().to_pandas().to_csv(fpm, index=False)
    PM = pd.read_csv(fpm).drop_duplicates("main_id").set_index("main_id")
    B = B.join(PM[["pmra", "pmdec"]], on="main_id"); B[["pmra", "pmdec"]] = B[["pmra", "pmdec"]].fillna(0.0)
    wq = pd.read_parquet(os.path.join(C.RAW, "vlass_ql_wcs_cadc.parquet")); wq["mjd"] = C.image_mjd(wq)
    wq["subtile"] = wq.uri.str.extract(r"ql\.(T\d+t\d+\.J\d+[+-]\d+)\.")[0]; wq["ver"] = wq.uri.str.extract(r"\.v(\d+)\.")[0].astype(int)
    img = wq.sort_values("ver").groupby(["epoch", "subtile"]).tail(1).reset_index(drop=True)
    cal = json.load(open(os.path.join(C.DER, "forced_calibration.json"))); kappa = cal["kappa"]
    fcut = os.path.join(C.RAW, "bright_dwarfs_cutouts.npz")
    cache = dict(np.load(fcut, allow_pickle=True)) if os.path.exists(fcut) else {}
    sess = requests.Session(); rows, store = [], {}
    for _, s in B.iterrows():
        for half in sorted(img.epoch.unique()):
            I = img[img.epoch == half]
            d = SkyCoord(I.crval1.values * u.deg, I.crval2.values * u.deg).separation(SkyCoord(s.ra * u.deg, s.dec * u.deg)).deg
            if d.min() > 0.72: continue
            rw = I.iloc[int(np.argmin(d))]; ep = Time(rw.mjd, format="mjd").jyear if np.isfinite(rw.mjd) else 2018.5
            ra, de = C.propagate(np.array([s.ra]), np.array([s.dec]), np.array([s.pmra]), np.array([s.pmdec]), np.array([ep]),
                                 ref_epoch=2000.0, plx=np.array([s.plx_value]))
            x, y = [float(v) for v in mkwcs(rw).world_to_pixel(SkyCoord(ra[0] * u.deg, de[0] * u.deg))]
            X, Y = int(round(x)) + 1, int(round(y)) + 1
            if X - HB < 1 or Y - HB < 1 or X + HB > rw.nx or Y + HB > rw.ny: continue
            key = f"{s.main_id}|{half}"
            if key in cache: cut = cache[key]
            else:
                for t in range(4):
                    try:
                        r = sess.get(C.CADC_DATA + rw.uri.split("/")[-1], params=[("SUB", f"[{X-HB}:{X+HB},{Y-HB}:{Y+HB},*,*]")], timeout=300)
                        if r.status_code != 200 or r.content[:6] != b"SIMPLE": raise RuntimeError(f"HTTP {r.status_code}")
                        cut = np.squeeze([h.data for h in fits.open(io.BytesIO(r.content)) if h.data is not None][0]).astype(np.float32); break
                    except Exception as e:
                        cut = None; time.sleep(5 * (t + 1))
                if cut is None: print("failed", key, flush=True); continue
            store[key] = cut
            c = cut.astype(float) * 1e3; n = c.shape[0]; Yg, Xg = np.mgrid[:n, :n]
            if not np.isfinite(c[HB, HB]): continue
            v = c[np.hypot(Xg - HB, Yg - HB) > R_NOISE]; sig = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v)))
            x0, y0 = HB + (x + 1 - X), HB + (y + 1 - Y); S = bilinear(np.nan_to_num(c), x0, y0)
            eta = cal[TAG[half]]["eta"]; bp = float(C.peak_bias(np.array([half]))[0])
            rows.append(dict(main_id=s.main_id, V=s.V, plx=s.plx_value, sp_type=s.sp_type, status=s.status, half_epoch=half, ep=ep,
                             S_mJy=S / kappa / bp, sigma_mJy=eta * sig / kappa / bp, snr=S / (eta * sig)))
        print(s.main_id, "done", flush=True)
    np.savez(fcut, **store)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(C.RES, "bright_dwarfs_forced.csv"), index=False)
    D["w"] = 1 / D.sigma_mJy ** 2; g = D.groupby("main_id")
    P = pd.DataFrame({"n_ep": g.size(), "S": (D.S_mJy * D.w).groupby(D.main_id).sum() / g.w.sum(), "sig": g.w.sum() ** -0.5,
                      "max_snr": g.snr.max()}).join(B.set_index("main_id")[["V", "plx_value", "sp_type", "status"]])
    P["snr_coadd"] = P.S / P.sig; P["dist_pc"] = 1000 / P.plx_value
    P["S_lim"] = np.maximum((THR + Z) * P.sig, P.S + Z * P.sig); P["L_lim_WHz"] = C.lum(P.S_lim, P.dist_pc)
    P.reset_index().sort_values("V").to_csv(os.path.join(C.RES, "bright_dwarfs_limits.csv"), index=False)
    print(P.sort_values("V").round(3).to_string())
