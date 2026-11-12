"""02 - VLASS Epoch 1 Quick Look catalogue (Gordon et al. 2021, VizieR J/ApJS/255/30): components + subtiles.
Downloaded in declination strips from the CfA VizieR mirror (one bulk pass rather than ~10^5 cone searches).
Outputs: data/raw/vlass_e1ql_components.parquet (expected 3,381,277 rows)
         data/raw/vlass_e1ql_subtiles.parquet   (expected 35,285 rows)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import os, glob
import pandas as pd

COLS = ("RAJ2000,DEJ2000,e_RAJ2000,e_DEJ2000,Ftot,e_Ftot,Fpeak,e_Fpeak,Maj,Min,PA,Islrms,DCMaj,e_DCMaj,DCMin,e_DCMin,"
        "DCPA,SCode,Subtile,DupFlag,QualFlag,NNdist,PeakToRing,NVSSdist,FIRSTdist,MainSample,BMaj,BMin,BPA")
SUBCOLS = "Subtile,Tile,Epoch,Bmaj,Bmin,Crval1,Crval2,Obs,DateImg,MeanIslrms,SDIslrms,Ncomp"
TMP = os.path.join(C.RAW, "vlass_strips"); os.makedirs(TMP, exist_ok=True)

if __name__ == "__main__":
    # subtile (image) table
    st = C.vizier_tsv_to_df(C.viz_get({"-source": f"{C.VLASS_CAT}/subtile", "-out": SUBCOLS, "-out.max": "unlimited"}, timeout=600))
    st["Subtile"] = st.Subtile.str.strip()
    st.to_parquet(os.path.join(C.RAW, "vlass_e1ql_subtiles.parquet")); print("subtiles:", len(st))
    # component table in 1-degree strips (5-degree strips occasionally time out server-side)
    for lo in range(-45, 90):
        f = os.path.join(TMP, f"comp_{lo:+03d}.parquet")
        if os.path.exists(f):
            continue
        txt = C.viz_get({"-source": f"{C.VLASS_CAT}/comp", "-out": COLS, "-out.max": "unlimited", "DEJ2000": f"{lo}..{lo+1}"})
        df = C.vizier_tsv_to_df(txt)
        (df if df is not None else pd.DataFrame(columns=COLS.split(","))).to_parquet(f)
        print(f"dec {lo}..{lo+1}: {0 if df is None else len(df)}", flush=True)
    v = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TMP, "comp_*.parquet")))], ignore_index=True)
    v = v.dropna(subset=["RAJ2000"])
    v["SCode"] = v.SCode.astype(str).str.strip(); v["Subtile"] = v.Subtile.astype(str).str.strip()
    v = v.drop_duplicates(subset=["RAJ2000", "DEJ2000", "Subtile"]).reset_index(drop=True)   # strip edges are inclusive
    v.to_parquet(os.path.join(C.RAW, "vlass_e1ql_components.parquet"))
    print("components:", len(v), " MainSample==1:", int((v.MainSample == 1).sum()))
