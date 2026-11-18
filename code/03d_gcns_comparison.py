"""03d - Completeness and contamination of the sample relative to the Gaia Catalogue of Nearby Stars (GCNS).

GCNS (Gaia Collaboration, Smart et al. 2021) assigns a probability of lying within 100 pc to every EDR3 source, including
sources with low-S/N parallaxes, and flags white dwarfs. EDR3 and DR3 share source identifiers and astrometry. For GCNS
objects with dist_50 <= 100 pc (the column is in kpc), wd_prob < 0.5 and dec > -41 deg we record whether each is in the search sample and, if
not, the first cut it fails: parallax > 10 mas and parallax/error > 10 (01); finite BP-RP; the main-sequence window
(03); the VLASS footprint. Contamination: sample stars that GCNS does not place within 100 pc.
Inputs : Gaia archive TAP external.gaiaedr3_gcns_main_1 (cached), data/raw/gaia_dr3_plx10_part{1,2}.parquet,
         data/derived/ms_sample_vlass.parquet
Outputs: data/raw/gcns_dec41.parquet, results/gcns_comparison.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json
import numpy as np, pandas as pd, pyvo

if __name__ == "__main__":
    f = os.path.join(C.RAW, "gcns_dec41.parquet")
    if not os.path.exists(f):
        tap = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap"); parts = []
        edges = list(range(-41, 91, 12)) + [90]                    # strips: anonymous jobs are limited to 60 s
        for lo, hi in zip(edges[:-1], edges[1:]):
            q = (f"SELECT source_id, ra, dec, parallax, parallax_error, dist_50, phot_g_mean_mag, gcns_prob, wd_prob "
                 f"FROM external.gaiaedr3_gcns_main_1 WHERE dec >= {lo} AND dec < {hi}")
            parts.append(tap.run_sync(q, maxrec=200000, timeout=300).to_table().to_pandas())
        pd.concat(parts).to_parquet(f, index=False, compression="zstd")
    G = pd.read_parquet(f)
    G = G[(G.dist_50 <= 0.1) & (G.wd_prob < 0.5)]                # dist_50 is in kpc
    raw = C.read_gaia(columns=["source_id", "bp_rp", "dec"]).set_index("source_id")
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id"])
    S = set(sv.source_id)
    from importlib import import_module
    ms = import_module("03_sample_crossmatch").main_sequence(C.read_gaia().sort_values("source_id").reset_index(drop=True)).set_index("source_id")
    def why(sid):
        if sid in S: return "in sample"
        if sid not in raw.index: return "parallax <= 10 mas or parallax/error <= 10"
        if not np.isfinite(raw.bp_rp.get(sid, np.nan)): return "no BP-RP"
        if sid in ms.index and not ms.is_MS[sid]: return "outside main-sequence window"
        return "outside VLASS footprint"
    G["status"] = [why(s) for s in G.source_id.values]
    gset = set(G.source_id)
    out = dict(n_gcns_nonWD_100pc_dec41=int(len(G)), status_counts=G.status.value_counts().to_dict(),
               frac_in_sample=float((G.status == "in sample").mean()),
               status_counts_G_lt_15=G[G.phot_g_mean_mag < 15].status.value_counts().to_dict(),
               n_sample=int(len(S)), n_sample_not_gcns_100pc=int(len(S - gset)), frac_sample_not_gcns=float(len(S - gset) / len(S)),
               dist_50_pc_median_of_plx_cut_losses=float(1000 * G[G.status.str.startswith("parallax")].dist_50.median()),
               n_gcns_100pc_dec41_all=int(len(pd.read_parquet(f).query("dist_50 <= 0.1"))))
    json.dump(out, open(os.path.join(C.RES, "gcns_comparison.json"), "w"), indent=1); print(json.dumps(out, indent=1))
