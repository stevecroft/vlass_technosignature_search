"""02b - Per-image astrometric correction of the CIRADA Epoch-1 QL catalogue.

The Epoch-1 QL images were made without w-term corrections; the resulting position offsets (up to ~1.5" at low
elevation) were later removed from the QL image metadata, but the CIRADA Epoch-1 catalogue (J/ApJS/255/30) was built
from the uncorrected images (Kimball et al. 2026). The later-epoch QL lists (NRAO, VLASS Memo 22) come from corrected
images and agree with Gaia to ~0.05" (script 21). We therefore measure, for every Epoch-1 subtile, the median
(Epoch-1 minus Epoch-3) position of compact, isolated, high-S/N components, and subtract it from the Epoch-1
positions (common.e1_components applies the table).

Reference pairs: Epoch-1 components with DupFlag < 2, S_Code S, Fpeak/Islrms > 20, DCMaj < 1.5", NNdist > 20",
matched within 2" to QL3.1/QL3.2 components with S/N > 20 and DC_Maj < 1.5".
Offset per subtile: median over its pairs if n >= N_MIN; otherwise the median over the subtile's VLASS tile
(observed in a single pass, so at one hour angle); otherwise a running median in declination (1-degree bins).

Inputs : data/raw/vlass_e1ql_components.parquet, data/raw/vlass_e1ql_subtiles.parquet,
         data/raw/ql_catalogs_full/QL3.{1,2}_components.fits (not tracked; see README)
Output : data/raw/vlass_e1ql_position_offsets.csv (Subtile, Tile, dec, n, dE, dN [arcsec, E1 minus true],
         eE, eN [uncertainty of the adopted offset], source), data/derived/e1_astrometry_summary.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

N_MIN = 10
FULL = os.path.join(C.RAW, "ql_catalogs_full")


def xyz(ra, de):
    ra, de = np.deg2rad(ra), np.deg2rad(de); return np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]


if __name__ == "__main__":
    from astropy.io import fits
    e1 = pd.read_parquet(os.path.join(C.RAW, "vlass_e1ql_components.parquet"),
                         columns=["RAJ2000", "DEJ2000", "Fpeak", "Islrms", "DCMaj", "SCode", "DupFlag", "NNdist", "Subtile"])
    st = pd.read_parquet(os.path.join(C.RAW, "vlass_e1ql_subtiles.parquet"), columns=["Subtile", "Tile", "Crval2"])
    st["Subtile"] = st.Subtile.str.strip(); st["Tile"] = st.Tile.str.strip()
    e1["Subtile"] = e1.Subtile.astype(str).str.strip()
    r = e1[(e1.DupFlag < 2) & (e1.SCode.astype(str).str.strip() == "S") & (e1.Fpeak / e1.Islrms > 20) & (e1.DCMaj < 1.5) & (e1.NNdist > 20)]
    pairs = []
    for fn in ["QL3.1_components.fits", "QL3.2_components.fits"]:
        d = fits.open(os.path.join(FULL, fn), memmap=True)[1].data
        q = pd.DataFrame({c: np.asarray(d[c]).astype(np.asarray(d[c]).dtype.newbyteorder("=")) for c in ["RA", "DEC", "Peak_flux", "Isl_rms", "DC_Maj"]})
        q = q[(q.Peak_flux / q.Isl_rms > 20) & (q.DC_Maj < 1.5)]
        t = cKDTree(xyz(q.RA.values, q.DEC.values))
        dd, j = t.query(xyz(r.RAJ2000.values, r.DEJ2000.values), distance_upper_bound=2 * np.sin(np.deg2rad(2 / 3600) / 2))
        ok = np.isfinite(dd); a, b = r[ok], q.iloc[j[ok]]
        pairs.append(pd.DataFrame(dict(Subtile=a.Subtile.values, dec=b.DEC.values,
                                       dE=(a.RAJ2000.values - b.RA.values) * 3600 * np.cos(np.deg2rad(b.DEC.values)),
                                       dN=(a.DEJ2000.values - b.DEC.values) * 3600)))
        print(fn, "pairs", int(ok.sum()), flush=True)
    P = pd.concat(pairs).merge(st[["Subtile", "Tile"]], on="Subtile", how="left")
    sub = P.groupby("Subtile").agg(n=("dN", "size"), dE=("dE", "median"), dN=("dN", "median"))
    til = P.groupby("Tile").agg(n_t=("dN", "size"), dE_t=("dE", "median"), dN_t=("dN", "median"))
    P["dbin"] = np.floor(P.dec).astype(int)
    db = P.groupby("dbin").agg(dE_d=("dE", "median"), dN_d=("dN", "median")).rolling(5, center=True, min_periods=1).median()
    O = st.merge(sub, left_on="Subtile", right_index=True, how="left").merge(til, left_on="Tile", right_index=True, how="left")
    O["dbin"] = np.floor(O.Crval2).astype(int); O = O.merge(db, left_on="dbin", right_index=True, how="left")
    O["n"] = O.n.fillna(0).astype(int)
    use_s = O.n >= N_MIN; use_t = ~use_s & (O.n_t.fillna(0) >= N_MIN)
    O["source"] = np.where(use_s, "subtile", np.where(use_t, "tile", "declination"))
    O["dE_corr"] = np.where(use_s, O.dE, np.where(use_t, O.dE_t, O.dE_d))
    O["dN_corr"] = np.where(use_s, O.dN, np.where(use_t, O.dN_t, O.dN_d))
    # uncertainty of the adopted offset, per axis: error of the subtile median for own offsets; for tile and
    # declination fallbacks, the scatter of own-offset subtiles about their tile median (or about the declination
    # running median), which includes the true image-to-image variation
    rstd = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    se = P.groupby("Subtile").agg(eE=("dE", lambda x: 1.2533 * rstd(x) / np.sqrt(len(x))), eN=("dN", lambda x: 1.2533 * rstd(x) / np.sqrt(len(x))))
    O = O.merge(se, left_on="Subtile", right_index=True, how="left")
    own = O[use_s]
    tE = float(np.median(own.groupby("Tile").dE.std().dropna())); tN = float(np.median(own.groupby("Tile").dN.std().dropna()))
    dres = own.assign(rE=own.dE - own.dE_d, rN=own.dN - own.dN_d)
    dE_s, dN_s = float(rstd(dres.rE.dropna())), float(rstd(dres.rN.dropna()))
    O["eE"] = np.where(use_s, O.eE, np.where(use_t, tE, dE_s)); O["eN"] = np.where(use_s, O.eN, np.where(use_t, tN, dN_s))
    print(f"correction uncertainty: tile fallback ({tE:.3f}, {tN:.3f}), declination fallback ({dE_s:.3f}, {dN_s:.3f}), "
          f"own-offset median ({O.eE[use_s].median():.3f}, {O.eN[use_s].median():.3f})")
    O = O[["Subtile", "Tile", "Crval2", "n", "dE_corr", "dN_corr", "eE", "eN", "source"]].rename(
        columns={"Crval2": "dec", "dE_corr": "dE", "dN_corr": "dN"})
    O[["dE", "dN"]] = O[["dE", "dN"]].fillna(0.0).round(3); O[["eE", "eN"]] = O[["eE", "eN"]].round(3)
    O.to_csv(os.path.join(C.RAW, "vlass_e1ql_position_offsets.csv"), index=False)
    # residual check: pairs after subtracting each subtile's adopted offset
    Q = P.merge(O[["Subtile", "dE", "dN"]].rename(columns={"dE": "cE", "dN": "cN"}), on="Subtile")
    Q["rE"], Q["rN"] = Q.dE - Q.cE, Q.dN - Q.cN
    print(O.source.value_counts().to_dict())
    print("pairs:", len(P), " raw median (dE, dN) by dec band and residual after correction:")
    bands = {}
    for lo, hi in [(-41, -20), (-20, 0), (0, 30), (30, 60), (60, 90)]:
        s = (Q.dec >= lo) & (Q.dec < hi)
        rs = 1.4826 * np.median(np.abs(Q.rN[s] - Q.rN[s].median()))
        bands[f"{lo}_{hi}"] = dict(n=int(s.sum()), dE=float(Q.dE[s].median()), dN=float(Q.dN[s].median()), resid_rstd_N=float(rs))
        print(f"  [{lo:3d},{hi:3d}) n={s.sum():6d}  raw ({Q.dE[s].median():+.2f}, {Q.dN[s].median():+.2f})"
              f"  residual ({Q.rE[s].median():+.3f}, {Q.rN[s].median():+.3f})  residual robust std N {rs:.2f}")
    import json
    json.dump(dict(n_pairs=int(len(P)), n_subtiles=int(len(O)), source_counts=O.source.value_counts().to_dict(), N_MIN=N_MIN,
                   bands=bands, subtile_offset_pct_50_90_99=[float(x) for x in np.percentile(np.hypot(sub.dE[sub.n >= N_MIN], sub.dN[sub.n >= N_MIN]), [50, 90, 99])]),
              open(os.path.join(C.DER, "e1_astrometry_summary.json"), "w"), indent=1)
