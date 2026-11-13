"""08b - Recover observation dates for QL images whose CAOM record has no time bound (caom2.Plane.time_bounds_lower).

7,722 of the 122,698 QL images (latest versions) lack an MJD in CAOM: 14 in epoch 1.1, 133 in 3.1, 256 in 3.2 and
7,319 in 4.1 (Nov 2026). Their FITS headers do carry DATE-OBS ("First scan started"). A 1-pixel SUB cutout returns the
full header (~1 s per image). Run once after 08; common.image_mjd() then fills missing MJDs from this table.

Output: data/raw/vlass_ql_dates_header.csv (uri, date_obs, mjd); resumable.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import re, time
import numpy as np, pandas as pd, requests

OUT = os.path.join(C.RAW, "vlass_ql_dates_header.csv")


def iso_to_mjd(s):
    from astropy.time import Time
    return Time(s, format="isot", scale="utc").mjd


if __name__ == "__main__":
    W = pd.read_parquet(os.path.join(C.RAW, "vlass_ql_wcs_cadc.parquet"))
    todo = W[W.mjd.isna()].uri.tolist()
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["uri", "date_obs", "mjd"])
    todo = [u_ for u_ in todo if u_ not in set(done.uri)]
    print(f"images without MJD: {W.mjd.isna().sum()}, still to fetch: {len(todo)}", flush=True)
    sess = requests.Session(); rows = []; t0 = time.time()
    for i, uri in enumerate(todo):
        fn = uri.split("/")[-1]; d = None
        for t in range(4):
            try:
                r = sess.get(C.CADC_DATA + fn, params=[("SUB", "[1:1,1:1,*,*]")], timeout=120)
                m = re.search(rb"DATE-OBS= '([0-9T:\-\.]+)'", r.content[:28800])
                if r.status_code == 200 and m: d = m.group(1).decode(); break
                raise RuntimeError(f"HTTP {r.status_code}")
            except Exception:
                time.sleep(5 * (t + 1))
        rows.append(dict(uri=uri, date_obs=d, mjd=iso_to_mjd(d) if d else np.nan))
        if len(rows) == 200 or i == len(todo) - 1:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True); done.to_csv(OUT, index=False); rows = []
            print(f"{i+1}/{len(todo)}  {(time.time()-t0)/3600:.2f} h", flush=True)
    print("missing after fetch:", int(done.mjd.isna().sum()))
