"""07 - Classify the VLASS-matched stars, apply background tests, and compute per-star and population limits.

Classification of the secure associations (R > 0.5), in priority order: literature overrides
(data/raw/classification_overrides.csv), CV*, a2* (magnetic CP), RS*/RS?, TT*/Y*O, BY*, Er*/Fl* (flare), X-ray-active star,
else UNEXPLAINED. Multiplicity evidence (SIMBAD binary types, Gaia NSS > 0, RUWE > 1.4, dMG < -0.5) is recorded in its own
column but does not explain an association, the same rule as for the all-epoch candidates (18).
Background tests for UNEXPLAINED matches:
  * NVSS/FIRST proper-motion test: a radio source that is still at the VLASS position in NVSS (~1994.5) or FIRST (~2000)
    although the star moved more than twice the radio offset since then is stationary -> background.
  * Multi-epoch VLASS astrometry (script 06) for LSPM J1943+5337.
Limits: non-detections S_lim = (5 + 1.645) * subtile MeanIslrms / peak_bias (95% completeness; QL peak bias 0.85 in
half-epoch 1.1, 0.92 in 1.2, Kimball et al. 2026); confused stars (unrelated component < 3") use max(S_lim, that
component's flux); detected stars use (F + 1.645 sigma) / int_bias (integrated bias 0.90 in 1.1, 1.00 in 1.2).
Population: f < 3.0 / N_eff(L*) (95%, zero unexplained detections), N_eff = stars whose limit is below L*.

Inputs : data/derived/{ms_sample_vlass, matches_10arcsec}.parquet, data/raw/{xray_counterparts, simbad_candidates,
         simbad_nearest}.csv, data/derived/lspm1943_astrometry.json (optional)
Outputs: results/vlass_star_matches_classified.csv, results/vlass_perstar_limits_100pc_MS.parquet,
         results/vlass_population_limit_curve.csv, results/nearest_star_limits.csv
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json
import numpy as np, pandas as pd

PC_CM = 3.0857e18
ECF_2RXS = 6e-12            # erg/cm2/s per ct/s, coronal spectrum, 0.1-2.4 keV
NVSS_EPOCH, FIRST_EPOCH = 1994.5, 2000.0
LSPM = 2137474969751117312


BINARY_OTYPES = {"EB*", "SB*", "RS*", "RS?", "CV*", "**", "El*", "Al*", "bL*", "WU*"}
OVR = pd.read_csv(os.path.join(C.RAW, "classification_overrides.csv")).set_index("source_id") \
    if os.path.exists(os.path.join(C.RAW, "classification_overrides.csv")) else pd.DataFrame()


def evidence(r):
    """Multiplicity and activity evidence, recorded separately from the class (a review recommendation)."""
    ol = set((r.otypes if isinstance(r.otypes, str) else "").split("|"))
    single = r.source_id in OVR.index and bool(OVR.loc[r.source_id, "single"])
    mult = []
    if not single:
        if ol & BINARY_OTYPES: mult.append("SIMBAD " + "/".join(sorted(ol & BINARY_OTYPES)))
        if r.non_single_star > 0: mult.append("Gaia NSS")
        if r.ruwe > 1.4: mult.append(f"RUWE {r.ruwe:.1f}")
        if r.dMG < -0.5: mult.append(f"over-luminous ({r.dMG:+.2f} mag)")
    act = []
    if np.isfinite(r.LX): act.append("X-ray")
    for t, lab in [("Er*", "flare"), ("Fl*", "flare"), ("BY*", "BY Dra"), ("RS*", "RS CVn"), ("TT*", "young"), ("Y*O", "young"),
                   ("CV*", "CV"), ("a2*", "magnetic CP")]:
        if t in ol and lab not in act: act.append(lab)
    return "; ".join(mult), "; ".join(act)


def classify(r):
    if r.source_id in OVR.index: return OVR.loc[r.source_id, "class"]
    ol = set((r.otypes if isinstance(r.otypes, str) else "").split("|"))
    if "CV*" in ol: return "cataclysmic variable"
    if "a2*" in ol: return "magnetic CP star"
    if "RS*" in ol or "RS?" in ol: return "RS CVn binary"
    if "TT*" in ol or "Y*O" in ol: return "young star"
    if "BY*" in ol: return "BY Dra variable"
    if "Er*" in ol or "Fl*" in ol: return "flare star"
    # Multiplicity is recorded as evidence (column multiplicity_evidence, figure groups) but is not a radio-emission
    # mechanism, so it does not explain an association; the same rule as for the all-epoch candidates (18).
    if np.isfinite(r.LX): return "X-ray-active star"
    return "UNEXPLAINED"


def group(r):
    """Coarse grouping for figures; multiplicity is evidence, not a radio-emission mechanism."""
    if r["class"].startswith("background"): return "Background (not coincident with the star)"
    if r["class"] == "UNEXPLAINED": return "Unexplained"
    if r["class"] in ("cataclysmic variable", "magnetic CP star", "young star"): return "Young, CV or magnetic CP"
    if r.multiplicity_evidence: return "Active, with multiplicity evidence"
    return "Active, no multiplicity evidence"


if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"))
    A = pd.read_parquet(os.path.join(C.DER, "matches_10arcsec.parquet"))
    xr = pd.read_csv(os.path.join(C.RAW, "xray_counterparts.csv"))
    sb = pd.read_csv(os.path.join(C.RAW, "simbad_candidates.csv"))

    # ---------------- classification of the secure associations (rel > 0.5), the same set used by every result
    T = A[A.rel > 0.5].copy()
    T = T.merge(sv[["source_id", "ra", "dec", "pmra", "pmdec", "dist_pc", "bp_rp", "MG", "dMG", "ruwe", "non_single_star",
                    "phot_variable_flag"]], on="source_id", how="left")
    T = T.merge(xr, on="source_id", how="left").merge(sb[["source_id", "main_id", "otype", "sp_type", "otypes"]], on="source_id", how="left")
    T["S_corr_mJy"] = T.Ftot / C.int_bias(T.half); T["Lnu_R_WHz"] = C.lum(T.S_corr_mJy, T.dist_pc)
    T["FX_cgs"] = np.where(T.era_F.notna(), T.era_F, T.rxs_crate * ECF_2RXS)
    T["LX_erg_s"] = T.FX_cgs * 4 * np.pi * (T.dist_pc * PC_CM) ** 2
    T["logGB"] = np.log10(T.LX_erg_s / (T.Lnu_R_WHz * 1e7))
    T["LX"] = T.LX_erg_s
    T["class"] = T.apply(classify, axis=1); T["secure"] = T.rel > 0.5; T["note"] = ""
    ev = T.apply(evidence, axis=1); T["multiplicity_evidence"] = [e[0] for e in ev]; T["activity_evidence"] = [e[1] for e in ev]
    T["override_ref"] = T.source_id.map(OVR["reference"]) if len(OVR) else ""
    pm = np.hypot(T.pmra, T.pmdec)
    shN = pm * (T.comp_epoch - NVSS_EPOCH) / 1000; shF = pm * (T.comp_epoch - FIRST_EPOCH) / 1000
    stat = (T["class"] == "UNEXPLAINED") & (((T.NVSSdist < 5) & (shN > 2 * T.NVSSdist)) | ((T.FIRSTdist < 1.5) & (shF > 2 * T.FIRSTdist)))
    T.loc[stat, "class"] = "background (proper-motion test)"
    T.loc[stat, "note"] = [f"radio source stationary: star moved {a:.1f}\" since NVSS / {b:.1f}\" since FIRST; NVSSdist={c:.1f}\", FIRSTdist={d:.1f}\""
                           for a, b, c, d in zip(shN[stat], shF[stat], T.NVSSdist[stat], T.FIRSTdist[stat])]
    jf = os.path.join(C.DER, "lspm1943_astrometry.json")
    if os.path.exists(jf) and json.load(open(jf))["verdict"] == "not coincident":
        aj = json.load(open(jf)); m = T.source_id == LSPM
        T.loc[m, "class"] = "background (multi-epoch astrometry)"
        T.loc[m, "note"] = f"radio source not coincident with the star in VLASS 1.2/2.2/3.2; chi2(at star, N) = {aj['tgt_N_chi2_at_star']:.1f} for 3 dof"
    T["group"] = T.apply(group, axis=1)
    T.sort_values("rel", ascending=False).to_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv"), index=False)
    print("secure matches:", int(T.secure.sum()))
    print(T[T.secure]["class"].value_counts().to_string())
    print("Gudel-Benz log(LX/Lnu) secure, X-ray detected:", np.percentile(T.logGB[T.secure & T.logGB.notna()], [0, 50, 100]).round(2))

    # ---------------- per-star limits
    L = sv[["source_id", "ra", "dec", "parallax", "dist_pc", "bp_rp", "MG", "dMG", "phot_g_mean_mag", "ruwe", "non_single_star",
            "sub_rms", "subtile", "vepoch", "obs_jyear"]].copy()
    L["S_lim_mJy"] = (C.DET_SIGMA + C.Z95) * L.sub_rms / C.peak_bias(L.vepoch)
    near = A[(A.sep_star_epoch < 3.0) & (A.rel < 0.5)].copy(); near["F"] = near.Ftot / C.int_bias(near.half)
    conf = near.groupby("istar").F.max()
    L["confused"] = False; L.loc[conf.index, "confused"] = True
    L.loc[conf.index, "S_lim_mJy"] = np.maximum(L.loc[conf.index, "S_lim_mJy"], conf.values)
    det = A[A.rel > 0.5]
    L["detected"] = False; L.loc[det.istar.values, "detected"] = True
    # secure associations whose emission is explained (natural emitter, or a background source not at the star);
    # unexplained secure associations are treated like any other star in the all-epoch candidate test (18, 19)
    L["explained_detection"] = L.source_id.isin(T.source_id[T.secure & (T["class"] != "UNEXPLAINED")])
    L.loc[det.istar.values, "S_lim_mJy"] = (det.Ftot.values + C.Z95 * det.e_Ftot.values) / C.int_bias(det.half.values)
    L["Lnu_lim_WHz"] = C.lum(L.S_lim_mJy, L.dist_pc); L["EIRP_lim_W_2GHz"] = L.Lnu_lim_WHz * C.BAND_HZ
    # stars with any non-secure catalogued component within 10" (excluded from stacking in script 10)
    L["has_nearby_component"] = False; L.loc[A[A.rel <= 0.5].istar.unique(), "has_nearby_component"] = True
    L.to_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet"), index=False)
    print("N stars:", len(L), " confused:", int(L.confused.sum()), " detected:", int(L.detected.sum()))
    print("S_lim mJy 5/50/95%:", np.percentile(L.S_lim_mJy, [5, 50, 95]).round(2), " Lnu_lim median: %.3g W/Hz" % np.median(L.Lnu_lim_WHz))

    # ---------------- population limit (no unexplained detections remain)
    k_unexpl = int(((T["class"] == "UNEXPLAINED") & T.secure).sum())
    Lgrid = np.logspace(5, 10.5, 300); Neff = np.searchsorted(np.sort(L.Lnu_lim_WHz.values), Lgrid)
    fUL = np.where(Neff > 0, C.poisson_ul95(k_unexpl) / np.maximum(Neff, 1), np.nan)
    pd.DataFrame({"Lnu_star_WHz": Lgrid, "EIRP_2GHz_W": Lgrid * C.BAND_HZ, "N_eff": Neff, "k_unexplained": k_unexpl, "f_UL95": fUL}).to_csv(
        os.path.join(C.RES, "vlass_population_limit_curve.csv"), index=False)
    for x in [1e7, 1e8, 1e9, 3e9]:
        i = np.argmin(np.abs(Lgrid - x)); print(f"L*={x:.0e} W/Hz: N_eff={Neff[i]}, f<{fUL[i]:.2e}")

    nm = pd.read_csv(os.path.join(C.RAW, "simbad_nearest.csv"))
    L.nsmallest(12, "dist_pc").merge(nm, on="source_id", how="left")[["main_id", "source_id", "dist_pc", "bp_rp", "S_lim_mJy", "Lnu_lim_WHz",
                                                                        "EIRP_lim_W_2GHz", "detected", "confused"]].to_csv(os.path.join(C.RES, "nearest_star_limits.csv"), index=False)
