"""12 - Compare VLASS continuum limits with the Breakthrough Listen narrowband limits of Price et al. 2020
(AJ 159, 86; doi:10.3847/1538-3881/ab65f1; arXiv:1906.07750).
Price et al. quote EIRP_min = 4 pi d^2 S_min * (1 Hz) at d = 50 pc: 2.1e12 W (GBT L/S, 1.10-2.80 GHz) and 9.1e12 W
(Parkes 10-cm, 2.60-3.45 GHz), before the factor-of-3.3 correction of Choza et al. (2024) applied below; occurrence limits < 0.45% (L, 882 stars), < 0.37% (S), < 2.0% (Parkes), assuming P_det = 0.5.

Inputs : results/vlass_perstar_limits_100pc_MS.parquet
Output : results/vlass_vs_BL_Price2020_comparison.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json
import numpy as np, pandas as pd

BL = {"GBT": dict(EIRP=2.1e12, band="1.10-2.80 GHz", frac="<0.45% (L), <0.37% (S) [paper, P_det=0.5]", N=882, dwell="3x5 min ON"),
      "Parkes": dict(EIRP=9.1e12, band="2.60-3.45 GHz", frac="<2.0% [paper]", N=189, dwell="multi-epoch")}
# Choza et al. (2024) found that the turboSETI S/N used by Price et al. (2020) is overestimated by a factor of ~3.3, so the
# published EIRP limits are too low by that factor. We apply it here, as in Figure 9 (seti_limits_py3 uses m = 10 x 3.3 = 33).
CHOZA = 3.3

if __name__ == "__main__":
    fa = os.path.join(C.RES, "alleepoch_perstar_limits.parquet")
    if os.path.exists(fa):     # all-epoch steady limits (19); mu95 from the candidate statistics (18)
        L = pd.read_parquet(fa).rename(columns={"S_lim_steady": "S_lim_mJy", "L_lim_steady": "Lnu_lim_WHz"}).dropna(subset=["S_lim_mJy"])
        MU = json.load(open(os.path.join(C.DER, "alleepoch_candidates.json")))["mu95_hidden"]; label = "This work VLASS QL 1.1-4.1 (continuum)"
    else:
        L = pd.read_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet")); MU = 3.0; label = "This work VLASS QL E1 (continuum)"
    A50 = 4 * np.pi * (50 * C.PC_M) ** 2
    S50 = np.median(L.S_lim_mJy[(L.dist_pc > 45) & (L.dist_pc <= 50)])
    E50 = A50 * S50 * 1e-29 * C.BAND_HZ
    L50 = L[L.dist_pc <= 50]; Lstar = np.percentile(L50.Lnu_lim_WHz, 95)
    act = set(pd.read_parquet(os.path.join(C.DER, "activity_all.parquet")).query("active").source_id)   # 04b; p_i = 0 there
    N50 = int(((L50.Lnu_lim_WHz < Lstar) & ~L50.source_id.isin(act)).sum())
    rows = [dict(survey=f"Price+2020 {k} (narrowband)", band=v["band"], signal_type="<=~3 Hz, |drift|<=4 Hz/s", dwell=v["dwell"],
                 S_lim_Jy_at50pc=CHOZA * v["EIRP"] / A50 / 1e-26, EIRP_lim_W_at50pc=CHOZA * v["EIRP"], EIRP_published_W=v["EIRP"], fraction_UL95=v["frac"], N_stars_constraining=v["N"]) for k, v in BL.items()]
    rows.append(dict(survey=label, band="2-4 GHz", signal_type="total in-band power of a steady signal",
                     dwell="~5 s snapshots", S_lim_Jy_at50pc=S50 * 1e-3, EIRP_lim_W_at50pc=E50, EIRP_published_W=np.nan,
                     fraction_UL95=f"<{MU/N50:.1e} (P_det=1); <{MU/(0.5*N50):.1e} (P_det=0.5)", N_stars_constraining=N50))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(C.RES, "vlass_vs_BL_Price2020_comparison.csv"), index=False)
    print(out.to_string(index=False))
    print(f"EIRP ratio VLASS/BL-GBT = {E50/(CHOZA*2.1e12):.2e} (corrected; {E50/2.1e12:.2e} published); "
          f"flux-density ratio BL/VLASS = {(CHOZA*2.1e12/A50)/(S50*1e-29):.0f}")
