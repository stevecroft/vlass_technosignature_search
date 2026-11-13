"""09 - Resumable bulk download of 41x41-pixel (1"/px) VLASS QL cutouts at every sample star, using CADC
multi-SUB pixel cutouts (one request per image, one FITS extension per star). Stars are propagated to each image's
epoch (caom2 MJD, or FITS DATE-OBS where missing) using proper motion only, as for Epoch 1; the parallax shift
(<=0.55") is applied at the photometry stage through the stored sub-pixel offsets. Shards are written every 500 images; finished images are skipped on restart.
Then --consolidate merges the shards into one array.

Usage:
  python 09_download_cutouts.py --epochs 1.1,1.2 --tag ep1 [--threads 4] [--limit 20]
  python 09_download_cutouts.py --tag ep1 --consolidate
Epoch 1 (35,300 images, 208,573 stars) took 7.8 h at 4 threads with 0 failures (Nov 2026).
CADC pitfalls: ?CIRCLE= is silently ignored (full 55 MB image returned); ?SUB=[x1:x2,y1:y2,*,*] is 1-based, inclusive.

Inputs : data/derived/ms_sample_vlass.parquet, data/raw/vlass_ql_wcs_cadc.parquet
Outputs: data/raw/cutouts_<tag>/shard_*.npz  ->  data/raw/vlass_<tag>_ql_cutouts_41px.npz
         (arrays: star_index [row in ms_sample_vlass], source_id, cut [Jy/beam, N x 41 x 41], off [sub-pixel x,y], ep [image epoch])
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import io, time, glob, argparse
import numpy as np, pandas as pd, requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor, as_completed

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", default="1.1,1.2"); ap.add_argument("--tag", default="ep1")
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--half", type=int, default=20); ap.add_argument("--consolidate", action="store_true")
a = ap.parse_args(); HB = a.half
OUT = os.path.join(C.RAW, f"cutouts_{a.tag}"); os.makedirs(OUT, exist_ok=True)


def mkwcs(rw):
    w = WCS(naxis=2); w.wcs.ctype = ["RA---SIN", "DEC--SIN"]; w.wcs.crpix = [rw.crpix1, rw.crpix2]
    w.wcs.crval = [rw.crval1, rw.crval2]; w.wcs.cd = [[rw.cd11, rw.cd12], [rw.cd21, rw.cd22]]; return w


def consolidate():
    zs = [np.load(f) for f in sorted(glob.glob(f"{OUT}/shard_*.npz"))]
    st = np.concatenate([z["star"] for z in zs]); _, iu = np.unique(st, return_index=True)
    stars = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id"])
    cut = np.concatenate([z["cut"] for z in zs])[iu]; off = np.concatenate([z["off"] for z in zs])[iu]
    ep = np.concatenate([z["ep"] for z in zs])[iu]; st = st[iu]
    f = os.path.join(C.RAW, f"vlass_{a.tag}_ql_cutouts_41px.npz")
    np.savez(f, star_index=st, source_id=stars.source_id.values[st], cut=cut, off=off, ep=ep)
    print(f"wrote {f}: {len(st)} stars")


if __name__ == "__main__":
    if a.consolidate:
        consolidate(); raise SystemExit
    stars = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id", "ra", "dec", "pmra", "pmdec"])
    wq = pd.read_parquet(os.path.join(C.RAW, "vlass_ql_wcs_cadc.parquet"))
    wq["mjd"] = C.image_mjd(wq)          # fill CAOM records without time bounds (see 08b_fill_image_dates.py)
    wq["subtile"] = wq.uri.str.extract(r"ql\.(T\d+t\d+\.J\d+[+-]\d+)\.")[0]; wq["ver"] = wq.uri.str.extract(r"\.v(\d+)\.")[0].astype(int)
    img = wq[wq.epoch.isin(a.epochs.split(","))].sort_values("ver").groupby(["epoch", "subtile"]).tail(1).reset_index(drop=True)
    cc = SkyCoord(img.crval1.values * u.deg, img.crval2.values * u.deg)
    i_, d2_, _ = SkyCoord(stars.ra.values * u.deg, stars.dec.values * u.deg).match_to_catalog_sky(cc); ok = d2_.deg < 0.72
    groups = pd.DataFrame({"star": np.where(ok)[0], "img": i_[ok]}).groupby("img").star.apply(np.array)
    done = set()
    for f in glob.glob(f"{OUT}/shard_*.npz"):
        done.update(np.load(f)["img_idx"].tolist())
    todo = [k for k in groups.index if k not in done]
    if a.limit: todo = todo[:a.limit]
    print(f"images total {len(groups)}, done {len(done)}, todo {len(todo)}", flush=True)
    sess = requests.Session()

    def fetch(k):
        rw = img.iloc[k]; w = mkwcs(rw); st = groups[k]
        ep = Time(rw.mjd, format="mjd").jyear if np.isfinite(rw.mjd) else 2018.5
        s = stars.iloc[st]; ra, de = C.propagate(s.ra.values, s.dec.values, s.pmra.values, s.pmdec.values, ep)  # proper motion only; 17 applies parallax via `off`
        xs, ys = w.world_to_pixel(SkyCoord(ra * u.deg, de * u.deg)); subs, keep, offs = [], [], []
        for j, (x, y) in enumerate(zip(np.atleast_1d(xs), np.atleast_1d(ys))):
            X, Y = int(round(float(x))) + 1, int(round(float(y))) + 1
            if X - HB < 1 or Y - HB < 1 or X + HB > rw.nx or Y + HB > rw.ny: continue
            subs.append(f"[{X-HB}:{X+HB},{Y-HB}:{Y+HB},*,*]"); keep.append(st[j]); offs.append((float(x) + 1 - X, float(y) + 1 - Y))
        if not subs:
            return k, np.array([], int), np.zeros((0, 2*HB+1, 2*HB+1), np.float32), np.zeros((0, 2), np.float32), ep
        fn = rw.uri.split("/")[-1]; err = ""
        for t in range(5):
            try:
                r = sess.get(C.CADC_DATA + fn, params=[("SUB", x) for x in subs], timeout=300)
                if r.status_code != 200 or r.content[:6] != b"SIMPLE": raise RuntimeError(f"HTTP {r.status_code}")
                arr = [np.squeeze(h.data).astype(np.float32) for h in fits.open(io.BytesIO(r.content)) if h.data is not None]
                if len(arr) != len(subs) or any(x.shape != (2*HB+1, 2*HB+1) for x in arr): raise RuntimeError("shape/count mismatch")
                return k, np.array(keep), np.array(arr), np.array(offs, np.float32), ep
            except Exception as e:
                err = str(e); time.sleep(10 * (t + 1))
        print(f"FAILED img {k} {fn}: {err}", flush=True); return k, None, None, None, ep

    buf = {"img_idx": [], "star": [], "cut": [], "off": [], "ep": []}
    nshard = len(glob.glob(f"{OUT}/shard_*.npz")); t0 = time.time(); n = 0; nfail = 0
    def flush():
        global buf, nshard
        if not buf["img_idx"]: return
        np.savez(f"{OUT}/shard_{nshard:05d}.npz", img_idx=np.array(buf["img_idx"]), star=np.concatenate(buf["star"]),
                 cut=np.concatenate(buf["cut"]), off=np.concatenate(buf["off"]), ep=np.concatenate(buf["ep"]))
        nshard += 1; buf = {"img_idx": [], "star": [], "cut": [], "off": [], "ep": []}
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(fetch, k) for k in todo]
        for f in as_completed(futs):
            k, st, cut, off, ep = f.result(); n += 1
            if st is None: nfail += 1; continue
            for key, val in zip(["img_idx", "star", "cut", "off", "ep"], [k, st, cut, off, np.full(len(st), ep)]):
                buf[key].append(val)
            if len(buf["img_idx"]) >= 500: flush()
            if n % 500 == 0: print(f"{n}/{len(todo)} images, {nfail} failed, {(time.time()-t0)/n:.2f} s/img", flush=True)
    flush(); print(f"DONE {n} images, {nfail} failed, wall {(time.time()-t0)/3600:.2f} h", flush=True)
