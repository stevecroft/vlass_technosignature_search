"""03b - Which bright nearby dwarfs are missing from the Gaia DR3 sample?

Gaia DR3 is incomplete for the brightest stars (G <~ 3): many lack a source, a parallax, or BP-RP photometry, and some
early-type stars fall outside the empirical main-sequence window. This lists SIMBAD stars with parallax > 10 mas, V < 4,
dec > -40 deg and a dwarf (luminosity class V) spectral type, and records why each is not in the sample.

Inputs : SIMBAD TAP; data/derived/ms_sample_vlass.parquet; data/raw/gaia_dr3_plx10_part{1,2}.parquet
Output : results/bright_dwarfs_missing.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import pandas as pd, pyvo

if __name__ == "__main__":
    q = ("SELECT b.main_id, b.plx_value, b.sp_type, b.ra, b.dec, a.V, i.id AS gid FROM basic AS b JOIN allfluxes AS a ON a.oidref = b.oid "
         "LEFT JOIN ident AS i ON i.oidref = b.oid AND i.id LIKE 'Gaia DR3 %' WHERE b.plx_value > 10 AND a.V < 4 AND b.dec > -40")
    r = pyvo.dal.TAPService(C.SIMBAD_TAP).run_sync(q, maxrec=10000).to_table().to_pandas().drop_duplicates("main_id")
    S = set(pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id"]).source_id)
    G = C.read_gaia(columns=["source_id", "bp_rp"]).set_index("source_id")

    def status(x):
        g = str(x.gid).replace("Gaia DR3 ", "").strip()
        if not g.isdigit(): return "no Gaia DR3 source"
        s = int(g)
        if s in S: return "in sample"
        if s not in G.index: return "no DR3 parallax > 10 mas at S/N > 10"
        if pd.isna(G.loc[s, "bp_rp"]): return "no BP-RP"
        return "outside main-sequence window or footprint"
    r["status"] = r.apply(status, axis=1)
    sp = r.sp_type.astype(str)
    r["dwarf"] = sp.str.contains("V") & ~sp.str.contains("III") & ~sp.str.contains("II") & ~sp.str.contains("IV")
    out = r[r.dwarf & (r.status != "in sample")].sort_values("V")
    out[["main_id", "V", "plx_value", "sp_type", "ra", "dec", "gid", "status"]].to_csv(os.path.join(C.RES, "bright_dwarfs_missing.csv"), index=False)
    print(len(out), "bright dwarfs missing;", out.status.value_counts().to_dict())
