"""04 - External counterparts for every star with a VLASS component within 2.5" (the ~73 candidates):
ROSAT 2RXS (40" radius) and eROSITA eRASS1 main (20"), via the CfA VizieR mirror; SIMBAD main_id/otype/sp_type
and all otypes via the CfA SIMBAD TAP mirror. Also SIMBAD names for the 12 nearest sample stars.
NOTE: SIMBAD is a live database; object types may change slightly over time. The as-fetched tables are saved.

Inputs : data/derived/matches_10arcsec.parquet, data/derived/ms_sample_vlass.parquet
Outputs: data/raw/xray_counterparts.csv, data/raw/simbad_candidates.csv, data/raw/simbad_nearest.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import os, time
import numpy as np, pandas as pd, pyvo


def viz_cone(cat, ra, dec, r_arcsec, cols):
    txt = C.viz_get({"-source": cat, "-c": f"{ra:.6f} {dec:+.6f}", "-c.rs": r_arcsec, "-out": cols,
                     "-out.add": "_r", "-sort": "_r", "-out.max": 5}, timeout=60, tries=3, wait=10)
    df = C.vizier_tsv_to_df(txt)
    return None if df is None or len(df) == 0 else df.iloc[0]


def simbad_by_gaia(ids):
    sim = pyvo.dal.TAPService(C.SIMBAD_TAP); out = []
    for chunk in np.array_split(np.asarray(ids), max(1, len(ids) // 20)):
        lst = ",".join(f"'Gaia DR3 {x}'" for x in chunk)
        q = f"SELECT i.id AS gid, b.main_id, b.otype, b.sp_type, b.oid FROM ident AS i JOIN basic AS b ON i.oidref=b.oid WHERE i.id IN ({lst})"
        for t in range(3):
            try:
                out.append(sim.search(q).to_table().to_pandas()); break
            except Exception:
                time.sleep(10)
    sb = pd.concat(out, ignore_index=True)
    sb["source_id"] = sb.gid.str.replace("Gaia DR3 ", "").astype("int64")
    if not len(sb):                                   # none of these stars is in SIMBAD
        return sb.assign(otypes=pd.Series(dtype=str))
    oids = ",".join(str(int(x)) for x in sb.oid)
    ot = sim.search(f"SELECT oidref, otype FROM otypes WHERE oidref IN ({oids})").to_table().to_pandas()
    ot = ot.groupby("oidref").otype.apply(lambda x: "|".join(sorted(set(map(str, x))))).rename("otypes")
    return sb.merge(ot, left_on="oid", right_index=True, how="left")


if __name__ == "__main__":
    A = pd.read_parquet(os.path.join(C.DER, "matches_10arcsec.parquet"))
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"))
    cand = A[A.sep < 2.5].copy()
    stars = sv.iloc[cand.istar.unique()]
    print("candidate stars:", len(stars))
    # reuse the as-fetched tables (SIMBAD is live; re-querying unchanged stars would let classifications drift);
    # query only stars not yet cached. Delete the two csv files to force a full refresh.
    fx, fsb = os.path.join(C.RAW, "xray_counterparts.csv"), os.path.join(C.RAW, "simbad_candidates.csv")
    xr0 = pd.read_csv(fx) if os.path.exists(fx) else pd.DataFrame(columns=["source_id"])
    sb0 = pd.read_csv(fsb) if os.path.exists(fsb) else pd.DataFrame(columns=["source_id"])
    new_x = stars[~stars.source_id.isin(xr0.source_id)]; new_s = stars[~stars.source_id.isin(sb0.source_id)]
    print("not yet cached: X-ray", len(new_x), " SIMBAD", len(new_s))
    rows = []
    for _, r in new_x.iterrows():
        a = viz_cone("J/A+A/588/A103/cat2rxs", r.ra, r.dec, 40, "2RXS,CRate,e_CRate,ExiML")
        b = viz_cone("J/A+A/682/A34/erass1-m", r.ra, r.dec, 20, "IAUName,MLFlux1,DETLIKE1")
        gv = lambda x, k: (x[k] if x is not None and k in x.index else np.nan)
        rows.append(dict(source_id=r.source_id, rxs_r=gv(a, "_r"), rxs_crate=gv(a, "CRate"), rxs_name=gv(a, "2RXS"),
                         era_r=gv(b, "_r"), era_F=gv(b, "MLFlux1"), era_name=gv(b, "IAUName")))
        time.sleep(0.3)
    xr = pd.concat([xr0, pd.DataFrame(rows)], ignore_index=True) if rows else xr0
    xr.to_csv(fx, index=False)
    print("2RXS:", int(xr.rxs_r.notna().sum()), " eRASS1:", int(xr.era_r.notna().sum()))
    sb = pd.concat([sb0, simbad_by_gaia(new_s.source_id.values)], ignore_index=True) if len(new_s) else sb0
    sb.to_csv(fsb, index=False)
    print("SIMBAD identifications:", len(sb))
    near = sv.nsmallest(12, "dist_pc")
    sim = pyvo.dal.TAPService(C.SIMBAD_TAP)
    lst = ",".join(f"'Gaia DR3 {x}'" for x in near.source_id)
    nm = sim.search(f"SELECT i.id AS gid, b.main_id FROM ident AS i JOIN basic AS b ON i.oidref=b.oid WHERE i.id IN ({lst})").to_table().to_pandas()
    nm["source_id"] = nm.gid.str.replace("Gaia DR3 ", "").astype("int64")
    nm[["source_id", "main_id"]].to_csv(os.path.join(C.RAW, "simbad_nearest.csv"), index=False)
