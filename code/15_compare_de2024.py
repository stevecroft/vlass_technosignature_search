"""15 - Comparison with the VLASS-Gaia DR3 stellar radio census of De et al. (2024, AJ 168, 288; VizieR J/AJ/168/288).

De et al. matched VLASS QL Epochs 1 and 2 to Gaia DR3 stars within 500 pc (1" after proper-motion correction).
We compare their table with our sample and associations: which of their stars are in our main-sequence <= 100 pc
sample, which of those we associate securely, and why the others differ.

Inputs : data/raw/de2024_vlass_gaia_table1.csv (downloaded here if missing), data/derived/ms_sample_vlass.parquet,
         data/derived/matches_10arcsec.parquet, results/vlass_star_matches_classified.csv,
         results/vlass_detections_variability.csv (from 14, optional)
Outputs: results/de2024_overlap.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import numpy as np, pandas as pd, requests

F = os.path.join(C.RAW, "de2024_vlass_gaia_table1.csv")
if not os.path.exists(F):
    r = requests.get(C.VIZIER_ASU, params={"-source": "J/AJ/168/288/table1", "-out.max": "unlimited", "-out.all": ""}, timeout=300)
    C.vizier_tsv_to_df(r.text).to_csv(F, index=False)
D = pd.read_csv(F).rename(columns={"GaiaDR3": "source_id"})
sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id", "dist_pc", "bp_rp"])
A = pd.read_parquet(os.path.join(C.DER, "matches_10arcsec.parquet"))
T = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv"))
best = A.sort_values("sep").drop_duplicates("source_id")[["source_id", "sep", "rel"]] if "rel" in A else \
       T.sort_values("sep").drop_duplicates("source_id")[["source_id", "sep", "rel"]]
sec = set(T[T.rel > 0.5].source_id)

D["in_sample"] = D.source_id.isin(sv.source_id)
D = D.merge(best, on="source_id", how="left").merge(T.drop_duplicates("source_id")[["source_id", "class"]], on="source_id", how="left")
D["ours_secure"] = D.source_id.isin(sec)
ep1 = D.Flux1.notna() & (D.Flux1 > 0)
D["why_not_secure"] = np.select(
    [~D.in_sample & (D.Dist > 100), ~D.in_sample, D.ours_secure, ~ep1, D.rel.isna()],
    ["beyond 100 pc", "not in main-sequence sample (CMD/colour/parallax cut or outside footprint)", "",
     "no Epoch-1 flux in De et al. (Epoch-2-only source)", "no Epoch-1 catalog component within 10\" in our match"],
    "Epoch-1 component present but our reliability R<=0.5")
D.to_csv(os.path.join(C.RES, "de2024_overlap.csv"), index=False)

print(f"De et al. rows: {len(D)}; within 100 pc: {(D.Dist <= 100).sum()}; in our sample: {D.in_sample.sum()}")
print(f"our secure associations: {len(sec)}; of these in De et al.: {D.ours_secure.sum()}")
print("De et al. stars in our sample, by reason they are not secure here:")
print(D[D.in_sample].why_not_secure.replace("", "secure in both").value_counts().to_string())
print("De et al. category of our secure stars:", D[D.ours_secure].Category.value_counts().to_dict())
missing = sorted(sec - set(D.source_id))
print("our secure stars absent from De et al.:", len(missing))
print(T[T.source_id.isin(missing)].drop_duplicates("source_id")[["main_id", "dist_pc", "rel", "Fpeak", "class"]].to_string(index=False))
print(D[D.in_sample & ~D.ours_secure][["source_id", "Dist", "Flux1", "Flux2", "Category", "sep", "rel", "why_not_secure"]].round(3).to_string(index=False))
