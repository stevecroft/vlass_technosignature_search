"""16 - Comparison with the Sydney Radio Star Catalogue (SRSC; Driessen et al. 2024, PASA 41, e084; radiostars.org).

Which of our secure associations are catalogued radio stars, from which surveys and frequencies, and whether any has
a circularly polarized (Stokes V) detection. Also: SRSC stars in our sample that we do not detect.

Inputs : data/raw/srsc/{ReadMe,stars.dat,radio.dat} (downloaded here if missing), results/vlass_star_matches_classified.csv,
         data/derived/ms_sample_vlass.parquet, results/vlass_perstar_limits_100pc_MS.parquet
Outputs: results/srsc_overlap.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import numpy as np, pandas as pd, requests
from astropy.io import ascii

D = os.path.join(C.RAW, "srsc"); os.makedirs(D, exist_ok=True)
for f in ["ReadMe", "stars.dat", "radio.dat"]:
    if not os.path.exists(os.path.join(D, f)):
        open(os.path.join(D, f), "wb").write(requests.get(f"https://radiostars.org/data/download/{f}/", timeout=120).content)
S = ascii.read(os.path.join(D, "stars.dat"), readme=os.path.join(D, "ReadMe"), format="cds").to_pandas()
R = ascii.read(os.path.join(D, "radio.dat"), readme=os.path.join(D, "ReadMe"), format="cds").to_pandas()
# Gaia DR3 IDs exceed 2**53: parse them as exact int64 (a float64 key silently merges stars with nearby IDs, e.g. binary
# components). SRSC entries without a DR3 ID (DR2 ID or none) are matched by position (2", star propagated to the SRSC epoch).
S["source_id"] = pd.array([int(x) if isinstance(x, str) else pd.NA for x in S.Gaia.astype(str).str.extract(r"Gaia DR3 (\d+)")[0]],
                          dtype="Int64")
_sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id", "ra", "dec", "pmra", "pmdec"])
_m = S.source_id.isna() & S.RAdeg.notna()
if _m.any():
    from astropy.coordinates import SkyCoord; import astropy.units as u
    ep = pd.to_numeric(S.loc[_m, "Epoch"].astype(str).str.lstrip("J"), errors="coerce").fillna(2016.0).values
    cs = SkyCoord(S.loc[_m, "RAdeg"].values * u.deg, S.loc[_m, "DEdeg"].values * u.deg)
    cg = SkyCoord(_sv.ra.values * u.deg, _sv.dec.values * u.deg)
    j, d, _ = cs.match_to_catalog_sky(cg)
    dt = ep - 2016.0; jj = j
    ra2 = _sv.ra.values[jj] + _sv.pmra.values[jj] * dt / 3.6e6 / np.cos(np.deg2rad(_sv.dec.values[jj])); de2 = _sv.dec.values[jj] + _sv.pmdec.values[jj] * dt / 3.6e6
    d2 = cs.separation(SkyCoord(ra2 * u.deg, de2 * u.deg)).arcsec
    ok = d2 < 2.0
    S.loc[S.index[_m][ok], "source_id"] = _sv.source_id.values[jj][ok]
    print(f"SRSC entries without a DR3 ID: {int(_m.sum())}; positionally matched to the sample: {int(ok.sum())}")
agg = R.groupby("Identifier").agg(n_radio=("Freq", "size"), freqs=("Freq", lambda x: ",".join(f"{v:g}" for v in sorted(set(x)))),
                                  surveys=("Survey", lambda x: ",".join(sorted(set(map(str, x.dropna()))))),
                                  n_stokesV=("SpeakV", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
                                  refs=("Ref", lambda x: ",".join(sorted(set(map(str, x))))))
S = S.merge(agg, left_on="Identifier", right_index=True, how="left")
S = S.dropna(subset=["source_id"]).groupby("source_id").agg(
    Identifier=("Identifier", lambda x: ",".join(map(str, x))), Simbad=("Simbad", "first"), n_radio=("n_radio", "sum"),
    freqs=("freqs", lambda x: ",".join(sorted(set(",".join(map(str, x.dropna())).split(",")) - {""}, key=float))),
    surveys=("surveys", lambda x: ",".join(sorted(set(",".join(map(str, x.dropna())).split(",")) - {""}))),
    n_stokesV=("n_stokesV", "sum"), refs=("refs", lambda x: ",".join(map(str, x.dropna())))).reset_index()   # one row per star

T = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); T = T[T.rel > 0.5].drop_duplicates("source_id")
sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id", "dist_pc"])
L = pd.read_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet"), columns=["source_id", "S_lim_mJy", "detected", "confused"])
O = T[["source_id", "main_id", "class", "dist_pc", "rel"]].merge(S[["source_id", "Identifier", "Simbad", "n_radio", "freqs", "surveys", "n_stokesV", "refs"]],
                                                                  on="source_id", how="left")
O["in_srsc"] = O.Identifier.notna()
O.to_csv(os.path.join(C.RES, "srsc_overlap.csv"), index=False)
print(f"SRSC stars: {len(S)} ({S.source_id.notna().sum()} with Gaia DR3 IDs); our secure: {len(O)}; in SRSC: {int(O.in_srsc.sum())}; "
      f"with a Stokes V detection in SRSC: {int((O.n_stokesV > 0).sum())}")
print(O[["main_id", "class", "dist_pc", "in_srsc", "freqs", "n_stokesV"]].to_string(index=False))
ins = S[S.source_id.isin(sv.source_id)].merge(L, on="source_id", how="left")
print(f"SRSC stars in our sample: {len(ins)}; securely detected by us: {int(ins.source_id.isin(T.source_id).sum())}; "
      f"undetected: {int((~ins.source_id.isin(T.source_id)).sum())}")
ins["detected_by_us"] = ins.source_id.isin(T.source_id)
ins.to_csv(os.path.join(C.RES, "srsc_in_sample.csv"), index=False)
