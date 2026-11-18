"""04b - Activity evidence for every sample star, not only the detected ones.

A forced detection is removed from k (18) when its host star has activity evidence (07.evidence: a 2RXS counterpart within
40", an eRASS1 counterpart within 20", or a SIMBAD X-ray/flare/BY Dra/RS CVn/emission-line/young/CV/magnetic-CP type).
That rule also removes chance background coincidences, and any transmitter, at such stars. To keep k, the chance
background b, and N_eff on the same footing, the same evidence is needed for every star in the sample:
  * 18 counts control-position detections only around stars without activity evidence;
  * 19 and 07 give stars with activity evidence p_i = 0 in N_eff (a transmitter there would be classified as natural).

Inputs : data/derived/ms_sample_vlass.parquet
Outputs: data/raw/xray_2rxs_positions.parquet, data/raw/xray_erass1_positions.parquet (full-catalog positions from
         HEASARC TAP, cached),
         data/raw/simbad_otypes_all.parquet (SIMBAD activity object types by Gaia DR3 ID, cached),
         data/derived/activity_all.parquet (source_id, rxs_r, era_r, otypes, activity_evidence, active)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import time, importlib
import numpy as np, pandas as pd, pyvo
from scipy.spatial import cKDTree

P07 = importlib.import_module("07_classify_and_limits")
HEASARC_TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap"   # same catalogs as VizieR J/A+A/588/A103 and J/A+A/682/A34
CATS = {"2rxs": ("rass2rxs", 40.0), "erass1": ("erass1main", 20.0)}   # HEASARC table, match radius (arcsec)


def xyz(ra, de):
    a, d = np.radians(ra), np.radians(de)
    return np.c_[np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]


def _strip(svc, tb, lo, hi):
    """Positions with lo <= dec < hi; strips that reach the service's 100,000-row limit are split in two."""
    for k in range(4):
        try:
            t = svc.run_sync(f"SELECT ra, dec FROM {tb} WHERE dec >= {lo} AND dec < {hi}", maxrec=100_000).to_table(); break
        except Exception as e:
            print("retry", tb, lo, str(e)[:100], flush=True); time.sleep(30)
    else:
        raise RuntimeError(f"HEASARC query failed: {tb} {lo}")
    if len(t) >= 100_000:
        mid = 0.5 * (lo + hi); return _strip(svc, tb, lo, mid) + _strip(svc, tb, mid, hi)
    return [t.to_pandas().astype(float)]


def fetch_catalog(name):
    tb, _ = CATS[name]; f = os.path.join(C.RAW, f"xray_{name}_positions.parquet")
    svc = pyvo.dal.TAPService(HEASARC_TAP)
    n_exp = int(svc.run_sync(f"SELECT COUNT(*) AS n FROM {tb}").to_table()["n"][0])
    if os.path.exists(f):
        out = pd.read_parquet(f)
        if len(out) == n_exp: return out
        print(name, "cache incomplete:", len(out), "of", n_exp, "- refetching", flush=True)
    parts = []
    for lo in range(-90, 90, 10):
        parts += _strip(svc, tb, lo, lo + 10)
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == n_exp, f"{name}: fetched {len(out)} of {n_exp}"
    out.to_parquet(f, index=False)
    return out


def nearest(sv, cat, r_arcsec):
    d, _ = cKDTree(xyz(cat.ra.values, cat.dec.values)).query(xyz(sv.ra.values, sv.dec.values), k=1,
                                                              distance_upper_bound=2 * np.sin(np.radians(r_arcsec / 3600) / 2))
    sep = 2 * np.degrees(np.arcsin(np.clip(d / 2, 0, 1))) * 3600
    return np.where(np.isfinite(d), sep, np.nan)


ACT_OTYPES = ("X", "Er*", "Fl*", "BY*", "RS*", "TT*", "Y*O", "CV*", "a2*", "Em*")   # the activity types of 07.evidence


def simbad_otypes():
    """SIMBAD activity object types for every object with a Gaia DR3 identifier and parallax > 9 mas.

    Querying by object type is far cheaper than looking up 210,000 identifiers (a TAP upload join times out on the
    mirror). Only the activity types are returned, which is all the k/b/N_eff filter needs."""
    f = os.path.join(C.RAW, "simbad_otypes_all.parquet")
    if os.path.exists(f): return pd.read_parquet(f)
    lst = ",".join(f"'{t}'" for t in ACT_OTYPES)
    q = ("SELECT i.id AS gid, b.main_id, o.otype FROM otypes AS o JOIN basic AS b ON b.oid = o.oidref "
         f"JOIN ident AS i ON i.oidref = o.oidref WHERE o.otype IN ({lst}) AND b.plx_value > 9 AND i.id LIKE 'Gaia DR3 %'")
    for k in range(4):
        try:
            r = pyvo.dal.TAPService(C.SIMBAD_TAP).run_sync(q, maxrec=2_000_000).to_table().to_pandas(); break
        except Exception as e:
            print("retry", str(e)[:120], flush=True); time.sleep(30)
    else:
        raise RuntimeError("SIMBAD activity-type query failed")
    r["source_id"] = r.gid.str.replace("Gaia DR3 ", "").astype("int64")
    g = r.groupby("source_id").agg(main_id=("main_id", "first"), otypes=("otype", lambda x: "|".join(sorted(set(map(str, x))))))
    g = g.reset_index(); g.to_parquet(f, index=False)
    return g


if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                         columns=["source_id", "ra", "dec", "dMG", "ruwe", "non_single_star"])
    A = sv.copy()
    for name in CATS:
        cat = fetch_catalog(name); A[f"{'rxs' if name == '2rxs' else 'era'}_r"] = nearest(sv, cat, CATS[name][1])
        print(name, len(cat), "sources;", int(A[f"{'rxs' if name == '2rxs' else 'era'}_r"].notna().sum()), "sample stars matched")
    sb = simbad_otypes()
    A = A.merge(sb, on="source_id", how="left")
    A["LX"] = np.where(A.rxs_r.notna() | A.era_r.notna(), 1.0, np.nan)
    ev = A.apply(P07.evidence, axis=1)
    A["activity_evidence"] = [e[1] for e in ev]; A["active"] = A.activity_evidence.str.len() > 0
    A[["source_id", "rxs_r", "era_r", "main_id", "otypes", "activity_evidence", "active"]].to_parquet(
        os.path.join(C.DER, "activity_all.parquet"), index=False, compression="zstd")
    print("stars:", len(A), "| with activity evidence:", int(A.active.sum()), f"({100 * A.active.mean():.2f}%)",
          "| X-ray:", int(A.LX.notna().sum()), "| SIMBAD type only:", int((A.active & A.LX.isna()).sum()))
