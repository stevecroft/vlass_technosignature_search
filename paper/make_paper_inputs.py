"""Generate numbers.tex (\newcommand macros) and tables/*.tex from ../results and ../data/derived, so that every
quantity quoted in ms.tex is produced by the analysis code. Also copies figures into figures/.
Run after the pipeline:  python make_paper_inputs.py"""
import os, sys, json, shutil, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RES, DER, RAW = (os.path.join(ROOT, d) for d in ("results", "data/derived", "data/raw"))
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True); os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
macros = {}


def sci(x, d=1):
    """LaTeX scientific notation (math mode content)."""
    if x == 0 or not np.isfinite(x): return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10: m /= 10; e += 1
    return (f"{m:.{d}f}\\times10^{{{e}}}") if e != 0 else f"{m:.{d}f}"


def fas(x, d=1):
    """AAS fractional arcseconds: 0.25 -> 0\\farcs25"""
    return f"{x:.{d}f}".replace(".", "\\farcs")


def C_PEAK(label):
    return 0.85 if label == "1.1" else 0.92          # QL peak bias (Kimball et al. 2026), as code/common.peak_bias


def com(n): return f"{int(n):,}".replace(",", "{,}")
def put(name, val): macros[name] = val


def clean_name(s, sid):
    if not isinstance(s, str) or not s.strip(): return f"Gaia DR3 {sid}"
    s = re.sub(r"^(V\*|\*|NAME)\s+", "", s.strip()); s = re.sub(r"\s+", " ", s)
    s = s.replace("_", r"\_").replace("&", r"\&").replace("#", r"\#")
    # brace the name: a cell starting with "[" right after "\\" would be read as \\[<dimen>]
    return "{" + s + "}"


if __name__ == "__main__":
    cnt = json.load(open(os.path.join(DER, "sample_counts.json"))); fit = json.load(open(os.path.join(DER, "reliability_fit.json")))
    for k, n in [("NGaia", "n_gaia"), ("NFinite", "n_finite_bprp"), ("NMS", "n_ms"), ("NMSdec", "n_ms_dec41"), ("NSample", "n_in_vlass"),
                 ("NComp", "n_components"), ("NMainSample", "n_mainsample"), ("NUsedComp", "n_used_components"), ("NSubtiles", "n_subtiles"),
                 ("NRealOne", "real_pairs_lt1")]:
        put(k, com(cnt[n]))
    put("NOffOne", f"{cnt['offset_pairs_lt1_per_realisation']:.1f}")
    put("EpochMin", f"{cnt['obs_epoch_min']:.1f}"); put("EpochMax", f"{cnt['obs_epoch_max']:.1f}")
    put("ErrScaleK", f"{fit.get('err_scale_k', 1.0):.2f}"); put("EcorrMed", f"{fit.get('e_corr_median', 0.0):.2f}")
    ea = json.load(open(os.path.join(DER, "e1_astrometry_summary.json")))
    put("EOnePairs", com(ea["n_pairs"])); put("EOneDNsouth", f"{ea['bands']['-41_-20']['dN']:.2f}")
    put("EOneDNnorth", f"{abs(ea['bands']['60_90']['dN']):.2f}")
    for k_, v_ in [("EOneNsub", "subtile"), ("EOneNtile", "tile"), ("EOneNdec", "declination")]: put(k_, com(ea["source_counts"].get(v_, 0)))
    put("SigSys", f"{fit['sigma_sys_arcsec']:.2f}"); put("NTrue", f"{fit['N_true']:.0f}"); put("NSecure", str(fit["n_rel_gt_0p5"]))
    put("NRelNine", str(fit["n_rel_gt_0p9"])); put("NFalseSecure", f"{fit['expected_false_rel_gt_0p5']:.1f}")

    # --- classification
    T = pd.read_csv(os.path.join(RES, "vlass_star_matches_classified.csv")); xr = pd.read_csv(os.path.join(RAW, "xray_counterparts.csv"))
    # stars with an Epoch-1 component within 2.5" in the *current* cross-match (the cached X-ray table can hold stars that
    # dropped out after the astrometric correction, so it is restricted to these)
    _m = pd.read_parquet(os.path.join(DER, "matches_10arcsec.parquet"), columns=["source_id", "sep", "rel"])
    _m = _m[_m.sep <= 2.5].sort_values("rel", ascending=False).drop_duplicates("source_id")
    xr = xr[xr.source_id.isin(_m.source_id)]; assert len(xr) == len(_m), "xray_counterparts.csv lacks current 2.5-arcsec matches; rerun 04"
    put("NCand", str(len(_m))); put("NRxs", str(int(xr.rxs_r.notna().sum()))); put("NEra", str(int(xr.era_r.notna().sum())))
    _ns = _m[_m.rel <= 0.5]; _xs = xr.set_index("source_id").reindex(_ns.source_id)
    put("NCandNonSec", str(len(_ns))); put("NonSecSepMed", f"{_ns.sep.median():.1f}"); put("SecSepMed", f"{_m[_m.rel > 0.5].sep.median():.2f}")
    put("NNonSecRxs", str(int(_xs.rxs_r.notna().sum())))
    _o = pd.read_parquet(os.path.join(DER, "offset_pairs.parquet"), columns=["istar", "sep", "k"])
    put("NCandChance", f"{_o[_o.sep <= 2.5].groupby('k').istar.nunique().sum() / _o.k.nunique():.0f}")
    S = T[T.secure].copy(); vc = S["class"].value_counts()
    CLS = [("UNEXPLAINED", "Unexpl", "U"), ("emission-line star", "Em", "Em"), ("flare star", "Flare", "Fl"), ("RS CVn binary", "RS", "RS"), ("BY Dra variable", "BY", "BY"),
           ("young star", "Young", "Y"), ("cataclysmic variable", "CV", "CV"), ("X-ray-active star", "XA", "X"), ("magnetic CP star", "CP", "CP")]
    for c, k, _ in CLS: put(f"NC{k}", str(int(vc.get(c, 0))))
    gv = S["group"].value_counts()
    put("NGrpMult", str(int(gv.get("Active, with multiplicity evidence", 0)))); put("NGrpSingle", str(int(gv.get("Active, no multiplicity evidence", 0))))
    put("NGrpOther", str(int(gv.get("Young, CV or magnetic CP", 0))))
    put("NMultSimbad", str(int(S.multiplicity_evidence.fillna("").str.contains("SIMBAD").sum())))
    # radio-star confirmation evidence (scripts 14-16)
    so = pd.read_csv(os.path.join(RES, "srsc_overlap.csv")); de = pd.read_csv(os.path.join(RES, "de2024_overlap.csv"))
    va = pd.read_csv(os.path.join(RES, "vlass_detections_variability.csv"))
    put("NSRSC", str(int(so.in_srsc.sum()))); put("NSRSCV", str(int((so.n_stokesV > 0).sum())))
    put("NDeOverlap", str(int(de.ours_secure.sum()))); put("NDeTotal", str(len(de))); put("NDeInSample", str(int(de.in_sample.sum())))
    put("NDeEpTwoOnly", str(int((de.why_not_secure == "no Epoch-1 flux in De et al. (Epoch-2-only source)").sum())))
    put("NVariable", str(int(va.variable.sum()))); put("NComoving", str(int((va.motion == "co-moving").sum())))
    put("NInconclusive", str(int((va.motion == "inconclusive").sum()))); put("NSingleDet", str(int((va.motion == "single detection").sum())))
    put("NStationary", str(int((va.motion == "stationary").sum())))
    si = pd.read_csv(os.path.join(RES, "srsc_in_sample.csv"))
    put("NSRSCinSample", str(len(si))); put("NSRSCdet", str(int(si.detected_by_us.sum()))); put("NSRSCundet", str(int((~si.detected_by_us).sum())))
    _u = si[~si.detected_by_us]; _f = _u.freqs.fillna("").apply(lambda q: [float(x) for x in str(q).split(",") if x])   # MHz
    _PA = pd.read_parquet(os.path.join(RES, "alleepoch_perstar_limits.parquet")).set_index("source_id")
    _cand = _u.source_id.map(_PA.candidate).fillna(False).astype(bool); _vl = _f.apply(lambda L: any(2000 <= x <= 4000 for x in L))
    put("NSRSClow", str(int(_f.apply(lambda L: len(L) > 0 and max(L) < 2000).sum()))); put("NSRSCvlass", str(int(_vl.sum())))
    put("NSRSCvlassCand", str(int((_vl & _cand).sum()))); put("NSRSCcand", str(int(_cand.sum())))
    me = pd.read_csv(os.path.join(RES, "vlass_detections_multiepoch.csv"))
    put("NMultiImg", str(len(me))); put("NMultiDetImg", str(int(me.detected.sum())))
    put("NImgPerStarMin", str(int(va.n_images.min()))); put("NImgPerStarMax", str(int(va.n_images.max())))
    fs = me.groupby("epoch_label").flux_scale.first()
    put("FluxScaleMin", f"{fs.min():.2f}"); put("FluxScaleMax", f"{fs.max():.2f}")
    # the reference-source scale (later epoch / Epoch-1 catalog) predicted by the Kimball et al. biases: the x.1
    # half-epochs re-observe the 1.1 sky (peak bias 0.85), the x.2 half-epochs the 1.2 sky (0.92)
    lab = fs.index.map(lambda v: f"{v:.1f}")
    pred = np.array([C_PEAK(l) / (C_PEAK("1.1") if l.endswith(".1") else C_PEAK("1.2")) for l in lab])
    dev = np.abs(fs.values / pred - 1)[np.array([not l.startswith("1.") for l in lab])]
    put("FsAgree", f"{100 * dev.max():.0f}")
    _o = va.set_index("source_id"); _off = lambda sid: float(np.hypot(_o.offset_E[sid], _o.offset_N[sid]))
    put("GNineOff", f"{_off(_o.index[_o.main_id.astype(str).str.replace(' ', '') == 'G9-38B'][0]):.1f}")
    put("CDOff", f"{_off(5970189504209690496):.1f}")
    cm = va[va.motion == "co-moving"]
    put("LeverMin", f"{cm.lever_arm_arcsec.min():.1f}"); put("LeverMax", f"{cm.lever_arm_arcsec.max():.1f}")
    put("LeverMed", f"{cm.lever_arm_arcsec.median():.1f}")
    put("NSecureX", str(int(S.LX.notna().sum())))
    gb = S.logGB.dropna(); put("GBmed", f"{gb.median():.2f}"); put("GBmin", f"{gb.min():.2f}"); put("GBmax", f"{gb.max():.2f}")
    put("NUnexplained", str(int((S["class"] == "UNEXPLAINED").sum())))
    put("NExplSecure", str(int((S["class"] != "UNEXPLAINED").sum())))
    put("NXAMult", str(int(((S["class"] == "X-ray-active star") & (S.multiplicity_evidence.fillna("") != "")).sum())))
    _pc = pd.read_csv(os.path.join(RES, "vlass_population_limit_curve.csv"))
    _k1 = int(_pc.k_unexplained.iloc[0]); put("KEOne", str(_k1))
    from scipy.stats import chi2 as _c2
    put("MuEOne", f"{0.5 * _c2.ppf(0.95, 2 * (_k1 + 1)):.2f}")
    # the secure association explained only by its SIMBAD emission-line type (if exactly one)
    _U = S[S["class"] == "emission-line star"]
    if len(_U) == 1:
        _u = _U.iloc[0]
        put("EmSecName", str(_u.main_id).replace("2MASS ", "2MASS~"))
        put("EmSecDist", f"{_u.dist_pc:.1f}"); put("EmSecS", f"{_u.S_corr_mJy:.2f}"); put("EmSecRel", f"{_u.rel:.2f}")
        put("EmSecRUWE", f"{_u.ruwe:.1f}"); put("EmSecdMG", f"{-_u.dMG:.2f}")
        _F = pd.read_parquet(os.path.join(DER, "alleepoch_forced.parquet"), columns=["source_id", "tag", "snr"])
        _F = _F[_F.source_id == _u.source_id].set_index("tag").snr
        put("EmSecSNRone", f"{_F.get('ep1', np.nan):.1f}"); put("EmSecSNRlater", f"{_F.drop('ep1', errors='ignore').max():.1f}")
    put("NNatural", str(int((~S["class"].str.startswith("background") & (S["class"] != "UNEXPLAINED")).sum())))
    put("NBackground", str(int(S["class"].str.startswith("background").sum())))

    # --- per-star limits and population
    L = pd.read_parquet(os.path.join(RES, "vlass_perstar_limits_100pc_MS.parquet"))
    put("EarthLeakAU", com(round(np.sqrt(400.0 / (4 * np.pi * np.median(L.S_lim_mJy) * 1e-29)) / 1.496e11, -2)))  # Earth L_nu ~ 400 W/Hz
    p5, p50, p95 = np.percentile(L.S_lim_mJy, [5, 50, 95]); put("SlimMed", f"{p50:.2f}"); put("SlimLo", f"{p5:.2f}"); put("SlimHi", f"{p95:.2f}")
    put("LlimMed", sci(np.median(L.Lnu_lim_WHz))); put("EIRPlimMed", sci(np.median(L.EIRP_lim_W_2GHz)))
    put("NConfused", str(int(L.confused.sum()))); put("NDetected", str(int(L.detected.sum())))
    put("NExcluded", str(int((L.detected | L.confused | L.has_nearby_component).sum())))
    put("NNearComp", str(int((L.has_nearby_component & ~L.detected & ~L.confused).sum())))
    near = pd.read_csv(os.path.join(RES, "nearest_star_limits.csv"))
    b = near.iloc[0]; put("BarnardL", sci(b.Lnu_lim_WHz)); put("BarnardEIRP", sci(b.EIRP_lim_W_2GHz)); put("BarnardS", f"{b.S_lim_mJy:.2f}")
    pop = pd.read_csv(os.path.join(RES, "vlass_population_limit_curve.csv"))
    for x, k in [(1e7, "Seven"), (1e8, "Eight"), (1e9, "Nine")]:
        i = np.argmin(np.abs(pop.Lnu_star_WHz - x)); put(f"Fpop{k}", sci(pop.f_UL95[i])); put(f"Npop{k}", com(pop.N_eff[i]))
    put("FpopFloor", sci(pop.f_UL95.min())); put("NpopFloor", com(pop.N_eff.max()))
    put("OneInNine", com(round(1 / pop.f_UL95[np.argmin(np.abs(pop.Lnu_star_WHz - 1e9))], -3)))

    # --- LSPM J1943+5337
    aj = json.load(open(os.path.join(DER, "lspm1943_astrometry.json"))); ast = pd.read_csv(os.path.join(RES, "lspm1943_vlass_multiepoch_astrometry.csv"))
    put("LspmChi", f"{aj['tgt_N_chi2_at_star']:.1f}"); put("LspmMu", f"{aj['tgt_N_mu_mas_yr']:.0f}"); put("LspmMuErr", f"{aj['tgt_N_mu_err']:.0f}")
    put("FrameTieE", f"{aj['frame_tie_dE']:+.2f}"); put("FrameTieN", f"{aj['frame_tie_dN']:+.2f}")
    se3 = ast[ast.image == "SE 3.2"].iloc[0]; put("LspmOffThree", f"{np.hypot(se3.tgt_E - se3.star_E, se3.tgt_N - se3.star_N):.1f}")
    put("LspmSigThree", f"{abs(se3.tgt_N - se3.star_N) / se3.err_N:.0f}")
    lim = pd.read_csv(os.path.join(RES, "lspm1943_imaging_limits.csv")).set_index("image")
    put("PossIO", f"{lim.loc['POSS-I O','mag_limit_3sigma']:.1f}"); put("PossIE", f"{lim.loc['POSS-I E','mag_limit_3sigma']:.1f}")
    put("PossIIB", f"{lim.loc['POSS-II B_J','mag_limit_3sigma']:.1f}"); put("PossIIR", f"{lim.loc['POSS-II R_F','mag_limit_3sigma']:.1f}")
    put("LspmOffPossI", f"{lim.loc['POSS-I O','star_offset_arcsec']:.1f}"); put("LspmOffPossII", f"{lim.loc['POSS-II B_J','star_offset_arcsec']:.1f}")
    put("LspmOffPS", f"{lim.loc['PS1 g','star_offset_arcsec']:.1f}")
    _ge = pd.read_csv(os.path.join(DER, "lspm1943_gaia_position_errors.csv")); _ge["s"] = _ge[["sig_ra_mas", "sig_dec_mas"]].max(axis=1)
    put("GaiaErrStar", f"{_ge[_ge.object == 'star'].s.max():.1f}"); put("GaiaErrComp", f"{_ge[_ge.object == 'companion'].s.max():.0f}")
    lspm = T[T.source_id == 2137474969751117312].iloc[0]
    put("LspmChiTwoD", f"{aj['chi2_at_star_2d']:.0f}"); put("LspmDofTwoD", f"{aj['dof_at_star_2d']}")
    put("LspmChiRAstat", f"{aj['tgt_E_chi2_stationary']:.0f}"); put("LspmChiRAcomove", f"{aj['tgt_E_chi2_comoving']:.0f}")
    put("LspmChiDecStat", f"{aj['tgt_N_chi2_stationary']:.1f}")
    put("LspmMuE", f"{aj['tgt_E_mu_mas_yr']:.0f}"); put("LspmMuEErr", f"{aj['tgt_E_mu_err']:.0f}")
    q12 = ast[ast.image == "QL 1.2"].iloc[0]; s12 = ast[ast.image == "SE 1.2"].iloc[0]
    put("LspmQLSEdE", f"{abs(q12.tgt_E - s12.tgt_E):.2f}"); put("LspmQLSEdN", f"{abs(q12.tgt_N - s12.tgt_N):.2f}")
    put("LspmPA", f"{lspm.DCPA:.0f}")
    _L = 4 * np.pi * (lspm.dist_pc * 3.0857e16) ** 2 * lspm.S_corr_mJy * 1e-29
    _e = int(np.floor(np.log10(_L))); put("LspmL", f"{_L / 10**_e:.1f}\\times10^{{{_e}}}")
    _v = pd.read_csv(os.path.join(RES, "vlass_detections_variability.csv")).set_index("source_id").loc[2137474969751117312]
    put("LspmOffAU", f"{np.hypot(_v.offset_E, _v.offset_N) * lspm.dist_pc:.0f}")
    put("LspmOffArcsec", f"{np.hypot(_v.offset_E, _v.offset_N):.1f}")
    put("LspmS", f"{lspm.S_corr_mJy:.2f}")   # flux-scale corrected, as in Table 1
    put("LspmSep", f"{lspm.sep:.2f}"); put("LspmRel", f"{lspm.rel:.3f}"); put("LspmDist", f"{lspm.dist_pc:.0f}")
    put("LspmMaj", f"{lspm.DCMaj:.1f}"); put("LspmMin", f"{lspm.DCMin:.1f}")

    # --- stacking
    F = pd.read_csv(os.path.join(RES, "vlass_ep1_stack_results.csv")); fv = pd.read_csv(os.path.join(RES, "vlass_fraction_vs_power_limits.csv"))
    m24 = F[(F.shell_pc == "20-40") & (F.type == "M")].iloc[0]
    put("StackMSL", sci(m24.Lmean_WHz, 1)); put("StackMSLerr", sci(m24.L_err_WHz, 1)); put("StackMSLsnr", f"{m24.L_SNR:.1f}")
    put("StackMS", f"{m24.S_mean_uJy:.1f}"); put("StackMSerr", f"{m24.S_err_uJy:.1f}"); put("StackMSNR", f"{m24.S_SNR:.1f}"); put("StackMN", com(m24.N))
    put("StackML", sci(m24.Lmean_WHz)); put("StackMMed", f"{m24.S_median_uJy:.1f}"); put("StackMMedSNR", f"{m24.S_median_uJy / m24.S_median_err_uJy:.1f}")
    put("StackFGKmaxSNR", f"{F[F.type == 'FGK'].L_SNR.max():.1f}"); put("NStackStars", com(F.N.sum()))
    Lg = fv.Lnu_T_WHz.values
    for t in ["FGK", "M"]:
        put(f"Reach{t}", sci(Lg[np.nanargmax(fv[f'f_UL95_stack_{t}'] < 1)], 2))
        put(f"DetReach{t}", sci(Lg[np.nanargmax(np.nan_to_num(fv[f'f_UL95_detection_{t}'], nan=9) < 1)], 1))
        i6 = np.argmin(np.abs(Lg - 1e6)); put(f"FStackSix{t}", f"{fv[f'f_UL95_stack_{t}'][i6]:.2f}")
    iB = np.argmin(np.abs(np.log(Lg) - np.log(4e5)))      # Garrett (2026) Type I leakage benchmark, L_nu ~ 4e5 W/Hz
    for t in ["FGK", "M"]: put(f"FBrat{t}", f"{fv[f'f_UL95_stack_{t}'][iB]:.2f}")
    trp = pd.read_csv(os.path.join(RES, "vlass_transmitter_rate_points.csv"))          # from code/13_transmitter_rate.py
    put("TRfloor", sci(trp[trp.kind == "individual"].transmitter_rate.min(), 1))
    put("ReachEIRP", sci(min(Lg[np.nanargmax(fv[f'f_UL95_stack_{t}'] < 1)] for t in ['FGK', 'M']) * 2e9))
    z = np.load(os.path.join(RAW, "vlass_ep1_ql_cutouts_41px.npz"), mmap_mode="r"); put("NCutouts", com(len(z["source_id"])))

    # --- BL comparison
    bl = pd.read_csv(os.path.join(RES, "vlass_vs_BL_Price2020_comparison.csv")); v = bl.iloc[-1]
    put("VlassFiftyS", f"{v.S_lim_Jy_at50pc*1e3:.2f}"); put("VlassFiftyEIRP", sci(v.EIRP_lim_W_at50pc)); g_ = bl.iloc[0]   # BL GBT row, EIRP and S include the Choza et al. (2024) factor of 3.3 (script 12)
    put("BLratio", sci(v.EIRP_lim_W_at50pc / g_.EIRP_lim_W_at50pc)); put("BLratioPub", sci(v.EIRP_lim_W_at50pc / g_.EIRP_published_W))
    put("BLEIRPcorr", sci(g_.EIRP_lim_W_at50pc)); put("BLEIRPpub", sci(g_.EIRP_published_W)); put("BLSJy", f"{g_.S_lim_Jy_at50pc:.0f}")
    put("FluxRatio", com(round(g_.S_lim_Jy_at50pc / (v.S_lim_Jy_at50pc), -2))); put("NFifty", com(v.N_stars_constraining))
    MU = float(v.mu95) if "mu95" in bl.columns else 3.0      # no background subtraction; steady detections within 50 pc (script 12)
    put("KFifty", str(int(v.k_det)) if "k_det" in bl.columns else "0"); put("MuFifty", f"{MU:.1f}")
    put("FFifty", sci(MU / v.N_stars_constraining)); put("FFiftyHalf", sci(2 * MU / v.N_stars_constraining))
    put("BLfCommon", f"{100 * g_.f95_common:.2f}"); put("NGBTS", com(int(g_.N_stars_constraining)))

    # --- all-epoch analysis (scripts 17-19)
    ALL = None
    if os.path.exists(os.path.join(DER, "alleepoch_summary.json")):
        cal = json.load(open(os.path.join(DER, "forced_calibration.json")))
        cj = json.load(open(os.path.join(DER, "alleepoch_candidates.json")))
        sm = json.load(open(os.path.join(DER, "alleepoch_summary.json")))
        tg = sorted(k for k in cal if k.startswith("ep"))
        put("KappaInj", f"{cal['kappa']:.3f}"); put("KappaCat", f"{cal['kappa_cat']:.3f}"); put("KappaCatErr", f"{cal['kappa_cat_err']:.3f}")
        put("KappaCatN", str(cal["kappa_cat_n"])); put("KappaSys", f"{100 * abs(cal['kappa_cat'] / cal['kappa'] - 1):.0f}")
        _e0, _e1 = f"{min(cal[t]['eta'] for t in tg):.2f}", f"{max(cal[t]['eta'] for t in tg):.2f}"
        put("EtaRange", _e0 if _e0 == _e1 else f"{_e0}$--${_e1}")
        put("CtrlStdMax", f"{max(cal[t]['ctrl_std'] for t in tg):.2f}")
        put("FPRawMin", sci(min(cal[t]["fp_rate_ge5"] for t in tg))); put("FPRawMax", sci(max(cal[t]["fp_rate_ge5"] for t in tg)))
        tb = pd.DataFrame(cal["ep1"]["inject"]["table"]).set_index("level")
        put("InjFour", f"{tb.frac_det[4]:.2f}"); put("InjFive", f"{tb.frac_det[5]:.2f}"); put("InjSix", f"{tb.frac_det[6]:.2f}")
        put("NInject", com(int(tb.n.sum()))); put("NEpochTags", str(len(tg)))
        put("NStarEpochs", com(sum(cal[t]["n_good"] for t in tg)))
        # sample accounting: cutouts per epoch, discarded cutouts, stars with no usable or no unvetoed epoch
        put("NCutoutsAll", com(sum(cal[t]["n_stars"] for t in tg))); put("NBadCut", com(sum(cal[t]["n_stars"] - cal[t]["n_good"] for t in tg)))
        put("NCutEpList", ", ".join(com(cal[t]["n_stars"]) for t in tg[:-1]) + ", and " + com(cal[tg[-1]]["n_stars"]))
        _P = pd.read_parquet(os.path.join(RES, "alleepoch_perstar_limits.parquet"), columns=["source_id", "n", "n_ep_all"])
        _dec = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["source_id", "dec"]).set_index("source_id").dec
        _none = _P.n_ep_all == 0
        put("NNoEpoch", com(_none.sum())); put("NNoEpochPct", f"{100 * _none.mean():.1f}")
        put("NNoEpochSouth", com((_dec.reindex(_P.source_id[_none]) < -38).sum()))
        put("NAllVeto", str(int((_P.n.isna() & (_P.n_ep_all > 0)).sum()))); put("NWithForced", com((~_none).sum()))
        from scipy.stats import poisson as _po
        # round-2 referee response: background diagnostics, host-level snapshot test, background-free limits
        put("NCtrlAssoc", str(cj["n_ctrl_assoc_exempt"])); put("NCoaddVeto", str(cj["n_coadd_vetoed"]))
        _pa = cj["ctrl_hits_per_angle_steady"]; put("CtrlAngleMin", str(min(_pa))); put("CtrlAngleMax", str(max(_pa)))
        put("PRankSteady", f"{cj['p_rank_steady']:.2f}")
        put("BSteadyErrBoot", f"{cj['b_steady_err_boot']:.1f}"); put("BSnapErrBoot", f"{cj['b_snap_err_boot']:.1f}")
        put("KHost", str(cj["k_host"])); put("BHost", f"{cj['b_host']:.1f}"); put("MuHost", f"{cj['mu95_host']:.1f}")
        put("MuSteadyBz", f"{cj['mu95_steady_b0']:.1f}")
        _cd = cj["catalog_density"]
        put("CatStarIn", str(_cd["star_1_2p5"])); put("CatCtrlIn", f"{_cd['ctrl_1_2p5']:.1f}"); put("CatPLow", f"{_cd['p_low_1_2p5']:.2f}")
        put("CatStarZero", str(_cd["star_0_1"])); put("CatCtrlZero", f"{_cd['ctrl_0_1']:.1f}")
        _bf = [_cd["by_G"][k] for k in ["15-17", "17-30"]]; _bb = [_cd["by_G"][k] for k in ["0-13", "13-15"]]
        put("CatFaintStar", str(sum(x[0] for x in _bf))); put("CatFaintCtrl", f"{sum(x[1] for x in _bf):.1f}")
        put("CatBrightStar", str(sum(x[0] for x in _bb))); put("CatBrightCtrl", f"{sum(x[1] for x in _bb):.1f}")
        put("NCoaddStars", com(sm["n_with_coadd"])); put("NInactPhot", com(sm["n_eligible_with_photometry"]))
        for _x, _k in [(1e7, "Seven"), (1e8, "Eight"), (1e9, "Nine")]: put(f"FBz{_k}", sci(sm[f"f95_b0_1e{int(np.log10(_x))}"]))
        put("FBzFloor", sci(sm["f95_b0_floor"])); put("FDeltaFloorBz", sci(sm["fdelta95_b0_floor"])); put("FDeltaEpochCount", sci(sm["fdelta95_epochcount_floor"]))
        put("FDutyOneBz", sci(sm["f95_duty1_b0_floor"])); put("FDutyThreeBz", sci(sm["f95_duty0.3_b0_floor"])); put("FDutyTenthBz", sci(sm["f95_duty0.1_b0_floor"]))
        _PP = pd.read_csv(os.path.join(RES, "alleepoch_population_limits.csv")); _okp = np.isfinite(_PP.f95_b0) & (_PP.f95_b0 < 1)
        put("AddDiffPct", f"{100 * ((_PP.f95_b0_add / _PP.f95_b0)[_okp].max() - 1):.2f}")   # S_i + 5 sigma_i rule at candidates vs 5 sigma_i
        put("ReachIndBz", sci(sm["f95_b0_reach1"], 1)); put("ReachIndEIRP", sci(float(f"{sm['f95_b0_reach1']:.1e}") * 2e9, 1))
        _Lf = float(_PP.L_WHz[_PP.N_eff >= 0.999 * _PP.N_eff.max()].min())   # all stars without activity evidence contribute
        put("FloorL", sci(_Lf, 1)); put("FloorEIRP", sci(float(f"{_Lf:.1e}") * 2e9, 1))
        put("FDutyOne", sci(sm["f95_duty1_floor"])); put("FDutyThree", sci(sm["f95_duty0.3_floor"])); put("FDutyTenth", sci(sm["f95_duty0.1_floor"]))
        put("BzFactor", f"{sm['f95_b0_floor'] / sm['f95_floor']:.1f}")
        _xv = json.load(open(os.path.join(DER, "epoch_catalog_crosscheck.json")))["veto_variants"]["own_epoch"]
        put("XcOwnAngleMin", str(min(_xv["ctrl_hits_per_angle_steady"]))); put("XcOwnAngleMax", str(max(_xv["ctrl_hits_per_angle_steady"])))
        put("XcOwnPLow", f"{_po.cdf(_xv['k_steady'], _xv['b_steady']):.4f}"); put("XcOwnBErrBoot", f"{_xv['b_steady_err_boot']:.1f}")
        _ad = os.path.join(RES, "activity_by_distance.csv")
        if os.path.exists(_ad):
            _A = pd.read_csv(_ad).set_index("shell")
            for _sh, _k in [("0-10", "Ten"), ("10-20", "Twenty"), ("20-40", "Forty"), ("80-100", "Hundred")]: put(f"PctAct{_k}", f"{100 * _A.f_active[_sh]:.0f}")
        _gc = os.path.join(RES, "gcns_comparison.json")
        if os.path.exists(_gc):
            _g = json.load(open(_gc)); _sc = _g["status_counts"]
            put("NGCNS", com(_g["n_gcns_nonWD_100pc_dec41"])); put("PctGCNS", f"{100 * _g['frac_in_sample']:.0f}")
            put("NGCNSmsw", com(_sc.get("outside main-sequence window", 0))); put("PctGCNSmsw", f"{100 * _sc.get('outside main-sequence window', 0) / _g['n_gcns_nonWD_100pc_dec41']:.1f}")
            put("NGCNSplx", com(_sc.get("parallax <= 10 mas or parallax/error <= 10", 0))); put("NGCNSfoot", com(_sc.get("outside VLASS footprint", 0)))
            put("NContam", com(_g["n_sample_not_gcns_100pc"])); put("PctContam", f"{100 * _g['frac_sample_not_gcns']:.1f}")
        _bl = os.path.join(RES, "bright_dwarfs_limits.csv")
        if os.path.exists(_bl):
            _B = pd.read_csv(_bl).set_index("main_id"); put("NBrightMeas", str(len(_B))); _det = _B[_B.snr_coadd >= 5]
            put("NBrightDet", str(len(_det))); put("NBrightUndet", str(len(_B) - len(_det)))
            put("AlgolS", f"{_B.S['* bet Per']:.0f}"); put("ThetAurSNR", f"{_B.snr_coadd['* tet Aur']:.1f}"); put("XiTauSNR", f"{_B.snr_coadd['* ksi Tau']:.1f}")
            for _n, _k in [("* alf CMa", "Sirius"), ("* alf Aql", "Altair"), ("* alf Lyr", "Vega"), ("* alf PsA", "Fomalhaut")]: put(f"BrightL{_k}", sci(_B.L_lim_WHz[_n]))
            put("BrightUndetMaxSNR", f"{_B.snr_coadd[_B.snr_coadd < 5].max():.1f}")
        _ca = os.path.join(DER, "cutout_astrometry.json")
        if os.path.exists(_ca):
            _st = json.load(open(_ca))["stars"]; g1 = _st["349716170186042880"]; l1 = _st["2137474969751117312"]; t1 = _st["2071061615589030400"]
            put("NInjAstro", str(json.load(open(_ca)).get("n_inject", 300)))
            put("GoteChiStar", f"{g1['chi2_at_star']:.1f}"); put("GoteDofStar", str(g1["dof_at_star"])); put("GoteChiStat", f"{g1['chi2_stationary']:.1f}")
            put("GoteDofStat", str(g1["dof_2"])); put("GotePStat", f"{g1['p_stationary']:.3f}"); put("GoteMotion", f"{g1['motion_arcsec']:.2f}")
            put("LspmQLChiStat", f"{l1['chi2_stationary']:.1f}"); put("LspmQLChiCo", f"{l1['chi2_comoving']:.1f}"); put("LspmQLDof", str(l1["dof_2"]))
            put("LspmQLPCo", f"{l1['p_comoving']:.3f}"); put("LspmQLPStat", f"{l1['p_stationary']:.2f}")
            put("TycMotion", f"{t1['motion_arcsec']:.2f}"); put("TycPStat", f"{t1['p_stationary']:.2f}"); put("TycPCo", f"{t1['p_comoving']:.2f}")
            put("TycOffset", f"{np.hypot(*t1['offset_comoving']):.1f}")
        _cs = os.path.join(DER, "cosmos_flux_scale.json")
        if os.path.exists(_cs):
            _c = json.load(open(_cs)); _O = pd.read_csv(os.path.join(RES, "cosmos_flux_scale.csv"))
            put("CosN", com(_c["n_sources_measured"])); put("CosNmeas", com(_c["n_measurements"])); put("CosNiso", com(_c["n_compact_isolated"])); put("CosNcat", com(_c["n_cosmos"]))
            put("CosRatioFaint", f"{_c['ratio_below_0p5mJy']:.2f}"); put("CosRatioFaintErr", f"{_c['ratio_below_0p5mJy_err']:.2f}")
            _lo = _O.iloc[0]; put("CosLoBin", f"{1e3 * _lo.S_lo:.0f}--{1e3 * _lo.S_hi:.0f}"); put("CosLoRatio", f"{_lo.ratio:.2f}"); put("CosLoErr", f"{_lo.ratio_err:.2f}")
            _r = _O[(_O.S_lo >= 0.05) & (_O.S_hi <= 1.0)]; put("CosRatioMin", f"{_r.ratio.min():.2f}"); put("CosRatioMax", f"{_r.ratio.max():.2f}")
        _sv = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["ra", "dec"])
        from scipy.spatial import cKDTree as _KD
        _xyz = np.c_[np.cos(np.deg2rad(_sv.dec)) * np.cos(np.deg2rad(_sv.ra)), np.cos(np.deg2rad(_sv.dec)) * np.sin(np.deg2rad(_sv.ra)), np.sin(np.deg2rad(_sv.dec))]
        _pr = _KD(_xyz).query_pairs(2 * np.sin(np.deg2rad(2.5 / 3600) / 2)); put("NPairsBeam", com(len(_pr))); put("NStarsPairs", com(len({i for q in _pr for i in q})))
        _co = pd.read_csv(os.path.join(RES, "epoch_catalog_only_stars.csv")).merge(pd.read_parquet(os.path.join(DER, "activity_all.parquet"), columns=["source_id", "active", "eligible"]), on="source_id", how="left")
        _c1 = _co[_co.n_epochs == 1]; put("NCatOnlySingle", str(len(_c1))); put("NCatOnlySingleAct", str(int(_c1.active.astype(bool).sum())))
        put("NCatOnlySingleElig", str(int(_c1.eligible.fillna(False).astype(bool).sum())))
        _KB = pd.read_csv(os.path.join(RES, "alleepoch_k_b_threshold.csv")).set_index("threshold")
        put("KCoFour", str(int(_KB.k_steady[4.0]))); put("BCoFour", f"{_KB.b_steady[4.0]:.1f}")
        put("KCoSeven", str(int(_KB.k_steady[7.0]))); put("BCoSeven", f"{_KB.b_steady[7.0]:.1f}")
        put("KCoNegThree", str(int(_KB.k_steady_neg[3.0]))); put("BCoNegThree", f"{_KB.b_steady_neg[3.0]:.1f}")
        # SCR J0612-1546B, the brightest single-epoch forced detection without activity evidence
        _LT = pd.read_parquet(os.path.join(DER, "alleepoch_forced.parquet"), columns=["source_id", "half_epoch", "sigma_mJy", "snr", "S_mJy", "veto", "detected"])
        _x = _LT[_LT.source_id == 2992871647243229824].sort_values("snr")
        put("SCRsnr", f"{_x.snr.iloc[-1]:.1f}"); put("SCRS", f"{_x.S_mJy.iloc[-1]:.1f}"); put("SCRhalf", str(_x.half_epoch.iloc[-1]))
        put("SCRnoiseRatio", f"{_x.sigma_mJy.iloc[-1] / _x.sigma_mJy.iloc[:-1].median():.1f}")
        _CT = pd.read_parquet(os.path.join(DER, "alleepoch_forced_controls.parquet"))
        _ok = (~_LT.detected & ~_LT.veto & ~_LT.source_id.isin(set(pd.read_parquet(os.path.join(DER, "activity_all.parquet")).query("active").source_id))).values
        _h = [(_CT[f"c{j}"].values >= _x.snr.iloc[-1]) & ~_CT[f"cv{j}"].values & _ok for j in range(8)]
        _h5 = [(_CT[f"c{j}"].values >= 5) & ~_CT[f"cv{j}"].values & _ok for j in range(8)]
        put("SCRctrlAbove", str(int(sum(x.sum() for x in _h)))); put("SCRctrlTot", str(int(sum(x.sum() for x in _h5))))
        _bm = os.path.join(RES, "bright_dwarfs_missing.csv")
        if os.path.exists(_bm): put("NBrightMissing", str(len(pd.read_csv(_bm))))
        put("PctInactive", f"{100 * (1 - cj['f_active_sample']):.0f}")                 # (fraction of stars that are eligible)
        # chance probability of an unrelated X-ray source within the match radius, from the full-catalog source densities
        for _nm, _r, _area, _key in [("2rxs", 40.0, 4 * np.pi, "PChanceRxs"), ("erass1", 20.0, 2 * np.pi, "PChanceEra")]:
            _f = os.path.join(RAW, f"xray_{_nm}_flux.parquet")
            if os.path.exists(_f):
                _dens = len(pd.read_parquet(_f)) / _area                       # per steradian (eRASS1: half the sky)
                put(_key, sci(1 - np.exp(-_dens * np.pi * np.radians(_r / 3600) ** 2), 0))
        put("NActive", com(cj["n_presence_active"])); put("PctActive", f"{100 * cj['n_presence_active'] / cnt['n_in_vlass']:.1f}")   # presence of activity evidence (04b)

        put("BSteadyErr", f"{cj['b_steady_err']:.1f}"); put("BSnapErr", f"{cj['b_snap_err']:.1f}")
        put("NCtrlSteady", str(cj["ctrl_hits_steady"])); put("NCtrlSnap", str(cj["ctrl_hits_snap"]))
        put("MuSteadyMarg", f"{cj['mu95_steady_marg']:.1f}"); put("MuSnapMarg", f"{cj['mu95_snap_marg']:.1f}")
        put("BSteadyAll", f"{cj['b_steady_unfiltered']:.1f}"); put("BSnapAll", f"{cj['b_snap_unfiltered']:.1f}")
        put("BSteadyNoEmit", f"{cj['b_steady_noemit']:.1f}"); put("BSnapNoEmit", f"{cj['b_snap_noemit']:.1f}")
        put("PSteadyLowAll", f"{_po.cdf(cj['k_steady'], cj['b_steady_unfiltered']):.3f}")
        put("PSteadyLow", f"{_po.cdf(cj['k_steady'], cj['b_steady']):.3f}"); put("PSnapLow", f"{_po.cdf(cj['k_snap'], cj['b_snap']):.2f}"); put("PSnapHigh", f"{_po.sf(cj['k_snap'] - 1, cj['b_snap']):.2f}")
        # eligibility variants (18): X-ray threshold -3.5 (LxA) / -4.5 (LxB), and AFGK multiples eligible at all L (WithMult)
        for _t, _k in [("lx35", "LxA"), ("lx45", "LxB"), ("withmult", "WithMult")]:
            _ks, _bs, _ke, _be = cj[f"k_steady_{_t}"], cj[f"b_steady_{_t}"], cj[f"k_snap_{_t}"], cj[f"b_snap_{_t}"]
            put(f"KSteady{_k}", str(_ks)); put(f"BSteady{_k}", f"{_bs:.1f}"); put(f"KSnap{_k}", str(_ke)); put(f"BSnap{_k}", f"{_be:.1f}")
            put(f"PSteady{_k}", sci(_po.sf(_ks - 1, _bs), 0) if _po.sf(_ks - 1, _bs) < 0.01 else f"{_po.sf(_ks - 1, _bs):.2f}")
            put(f"PSnap{_k}", sci(_po.sf(_ke - 1, _be), 0) if _po.sf(_ke - 1, _be) < 0.01 else f"{_po.sf(_ke - 1, _be):.2f}")
            put(f"NElig{_k}", com(cj[f"n_eligible_{_t}"]))
        # ---- technosignature eligibility (04b)
        _EL = pd.read_parquet(os.path.join(DER, "activity_all.parquet"))
        _sv2 = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["source_id", "dist_pc", "phot_g_mean_mag"])
        _EL = _EL.merge(_sv2, on="source_id", how="left")
        _hot = (_EL.log_lx_lbol >= -4.0).fillna(False)
        put("NAFGK", com(_EL.afgk.sum())); put("NMdw", com((~_EL.afgk).sum())); put("NElig", com(_EL.eligible.sum()))
        put("PctElig", f"{100 * _EL.eligible.mean():.0f}")
        put("NExclClassAFGK", com((_EL.afgk & _EL.excl_class).sum())); put("NExclHot", com((_EL.afgk & ~_EL.excl_class & _hot).sum()))
        put("NAFGKMult", com((_EL.afgk & ~_EL.excl_class & ~_hot & _EL.mult).sum()))
        put("PctAFGKMult", f"{100 * (_EL.afgk & ~_EL.excl_class & ~_hot & _EL.mult).sum() / (_EL.afgk & ~_EL.excl_class & ~_hot).sum():.0f}")
        put("NEnv", com(_EL.env.sum())); put("NEnvM", com((_EL.env & ~_EL.afgk).sum()))
        put("NExclM", com((~_EL.afgk & _EL.excl_class).sum())); put("NExclAll", com((~_EL.eligible & ~_EL.env).sum()))
        put("NAboveEnv", com((_EL.eligible | _EL.env).sum()))
        put("NEBall", com(_EL.excl_reason.fillna("").str.contains("eclipsing binary").sum()))
        put("NEBafgk", com((_EL.afgk & _EL.excl_reason.fillna("").str.contains("eclipsing binary")).sum()))
        # completeness of the X-ray cut: lowest detectable log L_X/L_bol for 2RXS-like (2e-13) and eRASS1-like (5e-14
        # erg cm^-2 s^-1) flux limits at each AFGK star (L_bol from M_G with BC_G = 0, as in 04b)
        _Lb = 3.828e33 * 10 ** (-0.4 * (_EL.phot_g_mean_mag - 5 * np.log10(_EL.dist_pc / 10) - 4.67))
        _a4 = 4 * np.pi * (_EL.dist_pc * 3.0857e18) ** 2
        for _fl, _k in [(2e-13, "Rxs"), (5e-14, "Era")]:
            _rm = np.log10(_fl * _a4 / _Lb)[_EL.afgk]
            put(f"XLim{_k}Med", f"{_rm.median():.1f}"); put(f"XLim{_k}Hi", f"{_rm.quantile(0.99):.1f}")
            put(f"XLim{_k}PctFour", f"{100 * (_rm <= -4.0).mean():.0f}")
        for (_lo, _hi), _k in [((0, 10), "Ten"), ((80, 100), "Hundred")]:
            _sh = (_EL.dist_pc >= _lo) & (_EL.dist_pc < _hi)
            put(f"PctEligAFGK{_k}", f"{100 * _EL.eligible[_sh & _EL.afgk].mean():.0f}")
            put(f"PctMultAFGK{_k}", f"{100 * (_EL.mult[_sh & _EL.afgk & ~_EL.excl_class & ~_hot]).mean():.0f}")
        put("LEnv", "10^{10}"); put("LDGCVn", sci(4.5e9))
        # like-for-like comparison with the earlier rule (every star with activity evidence excluded), AFGK stars only
        for (_lo, _hi), _k in [((0, 10), "Ten"), ((80, 100), "Hundred")]:
            _sh = (_EL.dist_pc >= _lo) & (_EL.dist_pc < _hi) & _EL.afgk
            put(f"NAFGK{_k}", com(_sh.sum())); put(f"NOldEligAFGK{_k}", com((_sh & ~_EL.active).sum()))
            put(f"PctOldEligAFGK{_k}", f"{100 * (_sh & ~_EL.active).sum() / _sh.sum():.0f}")
        # brightest detections (S/N >= 5, unvetoed star-epochs, any epoch): M dwarfs, and any star
        _LTm = pd.read_parquet(os.path.join(DER, "alleepoch_forced.parquet"), columns=["source_id", "S_mJy", "snr", "veto"])
        _LTm = _LTm[~_LTm.veto & (_LTm.snr >= 5)].merge(_EL[["source_id", "afgk", "dist_pc", "main_id"]], on="source_id")
        _LTm["L"] = 4 * np.pi * (_LTm.dist_pc * 3.0857e16) ** 2 * _LTm.S_mJy * 1e-29
        put("MaxMdwL", sci(_LTm.L[~_LTm.afgk].max())); put("MaxStarL", sci(_LTm.L.max()))
        put("MaxStarName", clean_name(_LTm.loc[_LTm.L.idxmax(), "main_id"], "").strip("{}"))
        # X-ray depth table: AFGK stars outside the known-emitter classes, by distance shell
        _ra = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["source_id", "ra", "dec"]).set_index("source_id")
        _X = _EL[_EL.afgk & ~_EL.excl_class].copy()
        _a, _d = np.radians(_ra.ra.reindex(_X.source_id).values), np.radians(_ra.dec.reindex(_X.source_id).values)
        _Mg = np.array([[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445], [-0.8676661490, -0.1980763734, 0.4559837762]])
        _v = _Mg @ np.array([np.cos(_d) * np.cos(_a), np.cos(_d) * np.sin(_a), np.sin(_d)]); _l = np.degrees(np.arctan2(_v[1], _v[0])) % 360
        _era = (_l > 179.944) & (_l < 359.944)                                   # eRASS1 (German) half of the sky
        _Lb2 = 3.828e33 * 10 ** (-0.4 * (_X.phot_g_mean_mag.values - 5 * np.log10(_X.dist_pc.values / 10) - 4.67))
        _rmin = np.log10(np.where(_era, 5e-14, 2e-13) * 4 * np.pi * (_X.dist_pc.values * 3.0857e18) ** 2 / _Lb2)
        _hotX = (_X.log_lx_lbol >= -4.0).fillna(False).values
        _below = ~_hotX & (_X.log_lx_lbol.notna().values | (_rmin <= -4.0)); _unc = ~_hotX & ~_below
        _rows = []
        for _lo, _hi in [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
            _m = ((_X.dist_pc >= _lo) & (_X.dist_pc < _hi)).values
            _rows.append(f"{_lo}--{_hi} & {com(_m.sum())} & {com((_m & _hotX).sum())} & {com((_m & _below).sum())} & {com((_m & _unc).sum())} & {100 * (_m & _unc).sum() / _m.sum():.0f} \\\\")
        put("PctXUnc", f"{100 * _unc.mean():.0f}"); put("PctXUncHundred", f"{100 * (_unc & (_X.dist_pc >= 80).values).sum() / (_X.dist_pc >= 80).sum():.0f}")
        open(os.path.join(HERE, "tables", "xray_depth.tex"), "w").write(r"""\begin{deluxetable}{lrrrrr}
\tablecaption{X-ray constraints on the activity level of the AFGK stars outside the known radio-emitter classes.\label{tab:xray}}
\tablehead{\colhead{Shell} & \colhead{$N$} & \colhead{$\ge-4$} & \colhead{$<-4$} & \colhead{Unconstrained} & \colhead{Unc.} \\
\colhead{(pc)} & \colhead{} & \colhead{} & \colhead{} & \colhead{} & \colhead{(\%)}}
\startdata
""" + "\n".join(_rows) + r"""
\enddata
\tablecomments{Columns give the number of stars with measured $\log L_X/L_{\rm bol}\ge-4$ (excluded), with $\log L_X/L_{\rm bol}<-4$ established (an X-ray counterpart below the threshold, or no counterpart where the survey limit corresponds to $\log L_X/L_{\rm bol}\le-4$), and unconstrained (no counterpart, and a survey limit above the threshold). Survey limits are approximated as $5\times10^{-14}$\,erg\,cm$^{-2}$\,s$^{-1}$ in the eRASS1 half of the sky and $2\times10^{-13}$ (2RXS) elsewhere, with $L_{\rm bol}$ from $M_G$. Unconstrained stars are treated as eligible, so an active star among them would be counted in $k$.}
\end{deluxetable}
""")
        # RUWE of the brightest stars: fraction of AFGK stars outside the classes with RUWE > 1.4, G < 6 vs 6-13
        _G3 = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["source_id", "ruwe"])
        _E3 = _EL.merge(_G3, on="source_id"); _E3 = _E3[_E3.afgk & ~_E3.excl_class]
        put("PctRuweBright", f"{100 * (_E3.ruwe[_E3.phot_g_mean_mag < 6] > 1.4).mean():.0f}")
        put("PctRuweFaint", f"{100 * (_E3.ruwe[(_E3.phot_g_mean_mag >= 6) & (_E3.phot_g_mean_mag < 13)] > 1.4).mean():.0f}")
        put("EpsEriRUWE", f"{_E3.set_index('source_id').ruwe[5164707970261890560]:.1f}")
        _CAe = pd.read_csv(os.path.join(RES, "alleepoch_candidates.csv"))
        _Tse = pd.read_csv(os.path.join(RES, "vlass_star_matches_classified.csv")); _Tse = _Tse[_Tse.secure]
        put("MaxNatL", sci(max(_CAe.L_max_epoch.max(), _Tse.Lnu_R_WHz.max())))
        put("MaxForcedML", sci(_CAe[_CAe.bp_rp >= 1.84].L_max_epoch.max()))
        _afc = _CAe[_CAe.eligible | (_CAe.env & (_CAe.bp_rp < 1.84))]
        put("NAFGKCand", str(len(_afc))); put("NAFGKCandMult", str(int(_afc.mult.sum())))
        put("NAFGKCandSteady", str(int((_afc.cand_coadd).sum())))
        put("NCandAFGKExcl", str(int(((_CAe.bp_rp < 1.84) & ~_CAe.eligible & ~_CAe.env).sum())))
        put("NCandM", str(int((_CAe.bp_rp >= 1.84).sum())))
        put("NEligPhot", com(sm["n_eligible_with_photometry"])); put("NEnvPhot", com(sm["n_env_with_photometry"]))
        put("FBzElig", sci(sm["f95_b0_elig_floor"])); put("NeffAboveEnv", com(sm["Neff_max"]))
        put("FBzAtEnv", sci(sm["f95_b0_1e10"]))
        put("KMall", str(sm["k_Mall"])); put("KMallSec", str(sm["k_Mall_secure"])); put("NMallCoadd", com(sm["n_Mall_coadd"]))
        put("MuMall", f"{sm['mu95_Mall']:.1f}")
        for _e, _k in [(7, "Seven"), (8, "Eight"), (9, "Nine")]:
            put(f"FMall{_k}", sci(sm[f"f95_b0_Mall_1e{_e}"])); put(f"NeffMall{_k}", com(round(sm[f"Neff_Mall_1e{_e}"])))
        put("MuWithMult", f"{sm['mu95_withmult']:.1f}"); put("NWithMultSteady", com(sm["Neff_withmult_max"]))
        put("FBzWithMult", sci(sm["f95_b0_withmult_floor"]))
        put("FAllElig", sci(cj["mu95_steady"] / sm["Neff_elig_max"]))
        put("FBzEnvKZero", sci(3.0 / sm["Neff_max"]))
        _CA = pd.read_csv(os.path.join(RES, "alleepoch_candidates.csv")); _ex = _CA[_CA.activity_evidence.fillna("") != ""]   # with activity evidence
        put("NCandActEv", str(len(_ex))); put("NCandNoActEv", str(int((_CA.activity_evidence.fillna("") == "").sum())))
        put("PSteadyHigh", f"{_po.sf(cj['k_steady'] - 1, cj['b_steady']):.2f}")
        put("TycSNR", f"{_CA.set_index('source_id').snr_coadd[2071061615589030400]:.1f}")
        put("NEligSteady", com(sm["Neff_elig_max"]))
        _ev = _ex.activity_evidence.fillna("").str.split("; "); _tc = _ev.explode().value_counts()
        _xc = _ev.apply(lambda L: "X-ray" in L)
        put("NCandXcat", str(int(_xc.sum()))); put("NCandSimbadOnly", str(int((~_xc).sum())))
        for _k, _t in [("TX", "X-ray (SIMBAD)"), ("TFl", "flare"), ("TEm", "emission-line"), ("TBY", "BY Dra"), ("TRS", "RS CVn"), ("TY", "young"), ("TCP", "magnetic CP")]:
            put("NCand" + _k, str(int(_tc.get(_t, 0))))
        put("NCandStars", str(cj["n_candidate_stars"])); put("NCandExpl", str(cj["n_explained"])); put("NCandMult", str(cj["n_multiplicity_only"]))
        put("KUnexpl", str(cj["k_unexplained"])); put("BExp", f"{cj['b_expected']:.1f}"); put("MuHidden", f"{cj['mu95_hidden']:.1f}")
        put("KSteady", str(cj["k_steady"])); put("BSteady", f"{cj['b_steady']:.1f}"); put("MuSteady", f"{cj['mu95_steady']:.1f}")
        put("KSnap", str(cj["k_snap"])); put("BSnap", f"{cj['b_snap']:.1f}"); put("MuSnap", f"{cj['mu95_snap']:.1f}")
        put("NCandCoadd", str(cj["n_cand_coadd"])); put("NCandEpoch", str(cj["n_cand_epoch"]))
        put("CtrlRateCo", sci(cj["ctrl_rate_coadd_per_star"])); put("NStarEpTrials", com(cj["n_star_epochs_trials"]))
        put("NVetoStarEp", str(cj["n_vetoed_star_epochs"])); put("NStarsVeto", str(cj["n_stars_any_veto"]))
        put("CtrlRateEp", sci(cj["ctrl_rate_per_epoch"])); put("CtrlRateAny", sci(cj["ctrl_rate_coadd_per_star"]))
        put("MedEpochs", f"{sm['median_epochs']:.0f}")
        _E = pd.read_parquet(os.path.join(RES, "alleepoch_perstar_limits.parquet"), columns=["natural", "candidate"])
        _em = _E.natural | _E.candidate          # stars whose limit is set by their own (natural or unexplained) emission
        assert int(_em.sum()) == int(_E.natural.sum()) + int((_E.candidate & ~_E.natural).sum())
        put("NEmitters", str(int(_em.sum()))); put("NEmitPct", f"{100 * _em.mean():.1f}")
        PA = pd.read_parquet(os.path.join(RES, "alleepoch_perstar_limits.parquet")); PA = PA[PA.S_lim_steady.notna()]
        q5, q50, q95 = np.percentile(PA.S_lim_steady, [5, 50, 95]); put("SlimAllMed", f"{q50:.2f}"); put("SlimAllLo", f"{q5:.2f}"); put("SlimAllHi", f"{q95:.2f}")
        put("LlimAllMed", sci(PA.L_lim_steady.median())); put("EIRPlimAllMed", sci(PA.L_lim_steady.median() * 2e9))
        put("KardRatio", f"{PA.L_lim_steady.median() * 2e9 / 1e17:.0f}")      # Kardashev Type I level of Figure 9, 1e17 W
        put("SlimAnyMed", f"{PA.S_lim_anyepoch.median():.2f}")
        put("SlimBestMed", f"{PA.S_lim_bestepoch.median():.2f}")
        bn = PA.nsmallest(1, "dist_pc").iloc[0]; put("BarnardLAll", sci(bn.L_lim_steady)); put("BarnardEIRPAll", sci(bn.L_lim_steady * 2e9))
        put("BarnardSAll", f"{bn.S_lim_steady:.2f}"); put("BarnardNep", str(int(bn.n)))
        put("EarthLeakAUAll", com(round(np.sqrt(400.0 / (4 * np.pi * PA.S_lim_steady.median() * 1e-29)) / 1.496e11, -2)))
        put("NBelowBrat", com(int((PA.L_lim_steady < 4e5).sum())))
        _s = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["bp_rp"]); put("NAstars", com(int((_s.bp_rp < 0.33).sum())))
        put("FFiftyRatio", f"{g_.f95_common / (2 * MU / v.N_stars_constraining):.0f}")
        PP = pd.read_csv(os.path.join(RES, "alleepoch_population_limits.csv"))
        fi = lambda L0, col="f95": float(np.interp(np.log10(L0), np.log10(PP.L_WHz), PP[col]))
        for x, k in [(1e7, "Seven"), (1e8, "Eight"), (1e9, "Nine")]:
            put(f"FAll{k}", sci(fi(x))); put(f"NAll{k}", com(fi(x, "N_eff")))
        put("FAllFloor", sci(PP.f95.min())); put("NAllFloor", com(PP.N_eff.max())); put("FDeltaFloor", sci(PP.fdelta95.min()))
        # Zuckerman (2026) benchmark: 60 MW beamed at Earth from 200 pc -> ~1e10 Jy in a few-Hz (3 Hz) channel, continuous.
        # Band-averaged over our 2 GHz: S = 1e10 Jy * 3 Hz / 2 GHz; L_nu = 4 pi d^2 S (toward Earth); EIRP = L_nu * 2 GHz.
        from scipy.stats import chi2 as _chi2
        _PC = 3.0857e16; _Sz = 1e10 * 3.0 / 2e9; _Lz = _Sz * 1e-26 * 4 * np.pi * (200 * _PC) ** 2
        _floor1 = lambda x: (lambda e, m: (f"{m}\\times" if m > 1 else "") + f"10^{{{e}}}")(int(np.floor(np.log10(x))), int(x / 10 ** int(np.floor(np.log10(x)))))   # round DOWN, 1 sig. fig.
        _nc = PA["class"].value_counts(); put("NRatioMA", f"{_nc['M'] / _nc['AFGK']:.1f}"); put("SqrtRatioMA", f"{np.sqrt(_nc['M'] / _nc['AFGK']):.1f}")
        put("ZuckSband", f"{_Sz:.0f}"); put("ZuckL", sci(_Lz)); put("ZuckEIRP", sci(_Lz * 2e9))
        put("ZuckChan", sci(_Sz * 2e9 / 2e6, 1))          # the same signal concentrated in one 2 MHz VLASS channel (Jy)
        put("ZuckMarginMed", _floor1(_Lz / PA.L_lim_steady.median())); put("ZuckMarginWorst", _floor1(_Lz / PA.L_lim_steady.max()))
        put("LlimAllMax", sci(PA.L_lim_steady.max()))
        # Zuckerman's target class F6-M2 (Pecaut & Mamajek 2013 table v2022.04.16: Bp-Rp 0.640 F6V, 2.23 M2V), apparently single
        _BPlo, _BPhi = 0.64, 2.23; put("BPRPlo", f"{_BPlo:.2f}"); put("BPRPhi", f"{_BPhi:.2f}")
        _G = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["source_id", "ruwe", "non_single_star", "dMG"])
        _Z = PA.merge(_G, on="source_id")
        _Z = _Z[(_Z.bp_rp >= _BPlo) & (_Z.bp_rp <= _BPhi) & (_Z.ruwe < 1.4) & (_Z.non_single_star == 0) & (_Z.dMG > -0.5)]
        _ku = int(_Z.unexplained_candidate.sum())                        # forced detections without activity evidence
        _CAz = pd.read_csv(os.path.join(RES, "alleepoch_candidates.csv")); _cjz = json.load(open(os.path.join(DER, "alleepoch_candidates.json")))
        _stz = set(_CAz.source_id[_CAz.cand_coadd & ~_CAz.explained & ~_CAz.source_id.isin(_cjz["coadd_vetoed_ids"])])   # the k of 18
        _ks = int(_Z.source_id.isin(_stz).sum()); _mu = 0.5 * _chi2.ppf(0.95, 2 * (_ks + 1))  # steady test (continuous signal), b = 0
        put("NSolarSteady", str(_ks))
        assert _Lz > 1e10        # far above L_ENV: eligible and envelope stars both count (04b)
        _actZ = set(pd.read_parquet(os.path.join(DER, "activity_all.parquet")).query("~eligible & ~env").source_id)
        _inst = _Z.n.notna() & ~_Z.source_id.isin(_cjz["coadd_vetoed_ids"])   # in the steady test (not vetoed in every epoch or in the coadd)
        _nin = int((~_Z.source_id.isin(_actZ) & _inst).sum())             # stars without activity evidence in the steady test (p_i = 1 here)
        put("NSolarSingle", com(len(_Z))); put("NSolarNat", str(int((_Z.natural & ~_Z.unexplained_candidate).sum())))
        put("NSolarActive", com(len(_Z) - _nin)); put("NSolarInactive", com(_nin))
        put("NSolarUnexpl", str(_ku)); put("FSolarSingle", sci(_mu / _nin))
        put("FAllZeroFloor", sci(PP.f95_k0b0.min()))
        put("OneInNineAll", com(round(1 / fi(1e9), -2)))
        put("ReachIndAll", sci(PP.L_WHz[np.argmax(PP.f95.values < 1)], 1))
        ST = pd.read_csv(os.path.join(RES, "alleepoch_stacks.csv")); SA = ST[(ST.scope == "all") & (ST["pop"] == "all")]
        SAi, SAa = ST[(ST.scope == "all") & (ST["pop"] == "inactive")], ST[(ST.scope == "all") & (ST["pop"] == "active")]
        for _pp, _T in [("Act", SAa), ("In", SAi)]:
            _m = _T[(_T.shell == "20-40") & (_T.cls == "M")].iloc[0]
            put(f"AllMSL{_pp}", sci(_m.meanL)); put(f"AllMSLErr{_pp}", sci(_m.meanL_err)); put(f"AllMSNR{_pp}", f"{_m.meanL_snr:.1f}"); put(f"AllMN{_pp}", com(_m.N_clean))
        _ma, _mi = [_T[(_T.shell == "20-40") & (_T.cls == "M")].iloc[0].meanL for _T in (SAa, SAi)]
        put("ActInRatio", f"{_ma / _mi:.1f}")                          # active / inactive mean luminosity, M dwarfs at 20-40 pc
        put("ActFracMForty", f"{100 * SAa[(SAa.shell == '20-40') & (SAa.cls == 'M')].N_clean.iloc[0] / SA[(SA.shell == '20-40') & (SA.cls == 'M')].N_clean.iloc[0]:.0f}")
        put("InMmaxSNR", f"{SAi[SAi.cls == 'M'].meanL_snr.max():.1f}"); put("ActAFGKmaxSNR", f"{SAa[SAa.cls == 'AFGK'].meanL_snr.max():.1f}")
        put("InAFGKmaxSNR", f"{SAi[SAi.cls == 'AFGK'].meanL_snr.max():.1f}")
        m = SA[(SA.shell == "20-40") & (SA.cls == "M")].iloc[0]
        put("AllMSL", sci(m.meanL)); put("AllMSLErr", sci(m.meanL_err)); put("AllMSNR", f"{m.meanL_snr:.1f}"); put("AllMN", com(m.N_clean))
        put("AllMNeffW", com(round(m.Neff_w, -1))); put("AllMDmed", f"{m.dmed_z:.3f}"); put("AllMDmedErr", f"{m.dmed_z_err:.3f}")
        put("AllMDmedSNR", f"{m.dmed_z / m.dmed_z_err:.1f}")
        put("AllAFGKmaxSNR", f"{SA[SA.cls == 'AFGK'].meanL_snr.max():.1f}")
        put("AllAFGKmaxShell", SA[SA.cls == "AFGK"].sort_values("meanL_snr").iloc[-1].shell.replace("-", "--"))
        put("NMShellsFour", str(int((SA[SA.cls == "M"].meanL_snr >= 4).sum())))
        fc = os.path.join(RES, "vlass_detections_variability_candidates.csv")
        if os.path.exists(fc):
            vc_ = pd.read_csv(fc)
            _sec = set(pd.read_csv(os.path.join(RES, "vlass_star_matches_classified.csv")).source_id)   # secure ones have their own astrometry (Sections 4.2-4.3)
            multi = vc_[(vc_.n_det >= 2) & ~vc_.source_id.isin(_sec)]
            put("NCandAstro", str(len(multi)))
            if len(multi):
                r_ = multi.sort_values("S_mean_mJy").iloc[-1]
                put("CandAstroName", r_.main_id.replace("  ", " ")); put("CandAstroNdet", str(int(r_.n_det)))
                put("CandAstroOff", f"{np.hypot(r_.offset_E, r_.offset_N):.1f}"); put("CandAstroChi", f"{r_.chi2_star:.0f}")
                put("CandAstroDof", str(int(r_.dof_star))); put("CandAstroS", f"{r_.S_mean_mJy:.2f}")
        big = SA[SA.N_clean >= 1000]; put("TopTenMax", f"{100 * big.top10_wfrac.max():.1f}")
        put("NStackAll", com(SA.N_clean.sum()))
        SLm = pd.read_csv(os.path.join(RES, "alleepoch_stack_limits.csv"))
        def _cross(s):
            # lowest L_T at which f95_mean < 1, interpolated in log L between the bracketing grid points (f95_mean is
            # NaN where even f = 1 is allowed)
            ok = s.f95_mean.values < 1; i = int(np.nanargmax(ok))
            if i == 0 or np.isfinite(s.f95_mean[i - 1]): return s.L_WHz[i]
            y0 = s.f95_mean_nosamp[i - 1]; y1 = s.f95_mean[i]
            t = np.log10(y0) / (np.log10(y0) - np.log10(y1)) if y0 > 1 > y1 else 1.0
            return 10 ** (np.log10(s.L_WHz[i - 1]) + t * (np.log10(s.L_WHz[i]) - np.log10(s.L_WHz[i - 1])))
        for t in ["AFGK"]:                       # stack limits: eligible (AFGK) stars only (19)
            s = SLm[SLm.cls == t].reset_index(drop=True)
            put(f"ReachAll{t}", sci(_cross(s), 1))
            fm = lambda L0: float(np.interp(np.log10(L0), np.log10(s.L_WHz), s.f95_mean))
            put(f"FMeanSix{t}", f"{fm(1e6):.2f}"); put(f"FBratAll{t}", f"{fm(4e5):.2f}")
            nm_ = lambda L0, col: float(np.interp(np.log10(L0), np.log10(s.L_WHz), s[col]))
            put(f"NeffSix{t}", com(round(nm_(1e6, "Neff_comb")))); put(f"WtenSix{t}", f"{100 * nm_(1e6, 'wfrac_within10pc'):.0f}")
            put(f"NeffBrat{t}", com(round(nm_(4e5, "Neff_comb")))); put(f"WtenBrat{t}", f"{100 * nm_(4e5, 'wfrac_within10pc'):.0f}")
            md = s.set_index("L_WHz").f95_median
            put(f"FMedSix{t}", f"{md.get(1e6, np.nan):.2f}"); put(f"FMedSeven{t}", f"{md.get(1e7, np.nan):.3f}"); put(f"FMedNine{t}", f"{md.get(1e9, np.nan):.3f}")
        reach = [_cross(SLm[SLm.cls == t].reset_index(drop=True)) for t in ["AFGK"]]
        for t in ["AFGK", "M"]:   # nearest shell alone, for comparison with Table 5 (not used for the limits)
            r0 = SA[(SA.cls == t) & (SA.dmin == 0)].iloc[0]; put(f"NearShell{t}", sci(max(r0.meanL, 0) + 1.645 * r0.meanL_err, 1))
        put("ReachEIRPAll", sci(float(f"{min(reach):.1e}") * 2e9))
        _COV = pd.read_csv(os.path.join(RES, "alleepoch_stack_coverage.csv"))
        sm_all = json.load(open(os.path.join(DER, "alleepoch_summary.json")))
        _ca = pd.read_csv(os.path.join(RES, "cutout_astrometry.csv")); _g = _ca[_ca.source_id == 349716170186042880]
        put("NCandSubCoadd", str(int((~pd.read_csv(os.path.join(RES, "alleepoch_candidates.csv")).cand_coadd).sum())))
        put("GoteSnrLo", f"{_g.snr_fit.min():.1f}"); put("GoteSnrHi", f"{_g.snr_fit.max():.1f}"); put("GoteNep", str(len(_g)))
        put("CovAFGKmin", f"{_COV[_COV.cls == 'AFGK'].coverage.min():.2f}"); put("NCovTrials", str(int(_COV.n_trials.iloc[0])))
        put("CovNangles", "8")
        _rl = {t: float(f"{L0:.1e}") for t, L0 in zip(["AFGK"], reach)}
        put("ReachRatioAFGK", f"{sm_all['f95_b0_reach1'] / _rl['AFGK']:.0f}")
        for t, L0 in zip(["AFGK"], reach):   # EIRP from the *displayed* (rounded) L so the abstract is self-consistent
            put(f"ReachEIRPAll{t}", sci(float(f"{L0:.1e}") * 2e9))
        ALL = dict(SA=SA, ST=ST[ST.scope == "all"])

    # ---------------- tables
    code = {c: k for c, _, k in CLS}; code["background (multi-epoch astrometry)"] = "BG"
    so_i, de_i, va_i = so.set_index("source_id"), de.set_index("source_id"), va.set_index("source_id")

    def mflags(e):
        e = e if isinstance(e, str) else ""
        return "".join(f for f, k in [("S", "SIMBAD"), ("N", "Gaia NSS"), ("R", "RUWE"), ("O", "over-luminous")] if k in e) or r"\nodata"

    def rflags(sid):
        f = ""
        if sid in so_i.index and so_i.loc[sid, "in_srsc"]: f += "P" if so_i.loc[sid, "n_stokesV"] > 0 else "K"
        if sid in de_i.index and de_i.loc[sid, "ours_secure"]: f += "D"
        if sid in va_i.index:
            f += "v" if va_i.loc[sid, "variable"] else ""
            f += "c" if va_i.loc[sid, "motion"] == "co-moving" else ""
        return f or r"\nodata"

    S = S.sort_values("dist_pc"); rows = []
    for _, r in S.iterrows():
        ND = r"\nodata"
        xflag = (("R" if np.isfinite(r.rxs_r) else "") + ("E" if np.isfinite(r.era_r) else "")) or ND
        gbs = f"{r.logGB:.2f}" if np.isfinite(r.logGB) else ND
        nm1 = clean_name(r.main_id, r.source_id); nm1 = r"\nodata" if nm1.startswith("{Gaia DR3") else nm1
        rows.append(f"{nm1} & {r.source_id} & {r.dist_pc:.1f} & {r.bp_rp:.2f} & {r.sep:.2f} & {r.rel:.3f} & {r.S_corr_mJy:.2f} & "
                    f"{np.log10(r.Lnu_R_WHz):.2f} & {xflag} & {gbs} & {mflags(r.multiplicity_evidence)} & {rflags(r.source_id)} & "
                    f"{code.get(r['class'], r['class'])} " + r"\\")
    open(os.path.join(HERE, "tables", "secure_matches.tex"), "w").write(r"""{\setlength{\tabcolsep}{2.5pt}  % group: keep the narrow spacing local to this table
\startlongtable
\begin{deluxetable*}{lrrrrrrrcrccl}
\tabletypesize{\scriptsize}
\tablecaption{VLASS Epoch 1 components securely associated (reliability $R>0.5$) with \NSecure\ main-sequence stars within 100\,pc.\label{tab:matches}}
\tablehead{\colhead{Name} & \colhead{Gaia DR3} & \colhead{$d$} & \colhead{$G_{BP}-G_{RP}$} & \colhead{Sep.} & \colhead{$R$} & \colhead{$S_{3\,\rm GHz}$} & \colhead{$\log L_\nu$} & \colhead{X-ray} & \colhead{$\log\frac{L_X}{L_\nu}$} & \colhead{Mult.} & \colhead{Radio} & \colhead{Class} \\
\colhead{} & \colhead{} & \colhead{(pc)} & \colhead{(mag)} & \colhead{(arcsec)} & \colhead{} & \colhead{(mJy)} & \colhead{(W\,Hz$^{-1}$)} & \colhead{} & \colhead{(Hz)} & \colhead{} & \colhead{} & \colhead{}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Sorted by distance. Separation at the epoch of the component's VLASS image. $S_{3\,\rm GHz}$ is the cataloged total flux density corrected for the Quick Look integrated-flux bias (divided by 0.90 in half-epoch 1.1; \citealt{Kimball2026}). X-ray: R = ROSAT 2RXS, E = eRASS1. Multiplicity evidence (Mult.): S = SIMBAD binary type, N = Gaia non-single-star solution, R = RUWE\,$>1.4$, O = over-luminous by $>0.5$\,mag. Radio-star evidence: P = in the Sydney Radio Star Catalogue \citep{Driessen2024} with a circularly polarized detection, K = in that catalog without one, D = in \citet{De2024}, v = significantly variable and c = co-moving with the star across VLASS epochs 1.1--4.1 (Section~\ref{sec:multiepoch}). Class (SIMBAD object types, except LQ~Hya and PW~And, young single stars; \citealt{Lehtinen2012,LopezSantiago2003}): Fl = flare star; Em = emission-line star; RS = RS~CVn binary; BY = BY~Dra variable; Y = young star; CV = cataclysmic variable; X = X-ray-active star; CP = magnetic chemically peculiar star; BG = not coincident with the star (Section~\ref{sec:lspm}). Machine-readable version in the reproducibility package.}
\end{deluxetable*}
}
""")
    rows = []
    for _, r in F.iterrows():
        rows.append(f"{str(r.shell_pc).replace('-', '--')} & {r.type} & {com(r.N)} & ${r.S_mean_uJy:.1f}\\pm{r.S_err_uJy:.1f}$ & ${r.S_SNR:.1f}$ & ${r.S_median_uJy:.1f}\\pm{r.S_median_err_uJy:.1f}$ & "
                    f"${r.Lmean_WHz/1e5:.1f}\\pm{r.L_err_WHz/1e5:.1f}$ & ${r.Lmean_UL95_WHz/1e5:.1f}$ & ${sci(r.EIRP_UL95_W_2GHz)}$ \\\\")
    open(os.path.join(HERE, "tables", "stack_results.tex"), "w").write(r"""\begin{deluxetable*}{llrrrrrrr}
\tablecaption{Inverse-variance-weighted mean stacks of VLASS Epoch 1 Quick Look images at the positions of individually undetected stars.\label{tab:stacks}}
\tablehead{\colhead{Shell} & \colhead{Type} & \colhead{$N$} & \colhead{$\langle S\rangle$} & \colhead{SNR} & \colhead{Median $S$} & \colhead{$\langle L_\nu\rangle$} & \colhead{$\langle L_\nu\rangle_{95}$} & \colhead{EIRP$_{95}$} \\
\colhead{(pc)} & \colhead{} & \colhead{} & \colhead{($\mu$Jy)} & \colhead{} & \colhead{($\mu$Jy)} & \colhead{($10^5$ W\,Hz$^{-1}$)} & \colhead{($10^5$ W\,Hz$^{-1}$)} & \colhead{(W)}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Means are measured against the empirical null (stacked image at 10\arcsec--18\arcsec from the centre); uncertainties are the larger of the formal and null-ring scatter. $\langle L_\nu\rangle_{95}$ is the one-sided 95\% upper limit on the mean spectral luminosity; EIRP$_{95}$ assumes a flat signal filling 2\,GHz. FGK/M split at $G_{BP}-G_{RP}=1.84$.}
\end{deluxetable*}
""")
    if ALL is not None:      # all-epoch steady limits for the same stars
        nearA = near.merge(PA[["source_id", "n", "S", "sig", "S_lim_steady", "L_lim_steady"]], on="source_id", how="left")
        _ELn = pd.read_parquet(os.path.join(DER, "activity_all.parquet")).set_index("source_id")
        def _elig(sid):   # E = eligible at any luminosity; L = eligible only above L_env (M dwarf / AFGK multiple); X = excluded
            return "E" if _ELn.eligible.get(sid, False) else ("L" if _ELn.env.get(sid, False) else "X")
        def _actf(sid): return "A" if _ELn.active.get(sid, False) else chr(92) + "nodata"
        FAMILIAR = {"G 272-61B": "UV Cet", "G 272-61A": "BL Cet", "HD 95735": "Lalande 21185", "CD-23 14742": "Ross 154", "HD 217987": "Lacaille 9352"}
        def _nm(r):
            n_ = clean_name(r.main_id, r.source_id).strip("{}"); return "{" + FAMILIAR.get(n_, n_) + "}"
        rows = [f"{_nm(r)} & {_actf(r.source_id)} & {_elig(r.source_id)} & {r.dist_pc:.2f} & {r.bp_rp:.2f} & {r.S_lim_mJy:.2f} & ${sci(r.Lnu_lim_WHz)}$ & "
                f"{int(r.n) if np.isfinite(r.n) else 0} & ${r.S:+.2f}\\pm{r.sig:.2f}$ & {r.S_lim_steady:.2f} & ${sci(r.L_lim_steady)}$ & ${sci(r.L_lim_steady * 2e9)}$ \\\\" for _, r in nearA.iterrows()]
        open(os.path.join(HERE, "tables", "nearest_limits.tex"), "w").write(r"""\begin{deluxetable*}{lccrrrrrrrrr}
\tabletypesize{\footnotesize}
\tablecaption{95\% limits on steady 2--4\,GHz emission at the positions of the twelve nearest stars in the sample (completeness thresholds, or upper confidence bounds for stars with emission; Section~\ref{sec:limits_method}).\label{tab:nearest}}
\tablehead{\colhead{Star} & \colhead{Act.} & \colhead{Elig.} & \colhead{$d$} & \colhead{$G_{BP}-G_{RP}$} & \colhead{$S_{\rm lim}^{\rm E1}$} & \colhead{$L_{\nu,\rm lim}^{\rm E1}$} & \colhead{$N_{\rm ep}$} & \colhead{$S_{\rm coadd}$} & \colhead{$S_{\rm lim}$} & \colhead{$L_{\nu,\rm lim}$} & \colhead{EIRP$_{\rm lim}$} \\
\colhead{} & \colhead{} & \colhead{} & \colhead{(pc)} & \colhead{(mag)} & \colhead{(mJy)} & \colhead{(W\,Hz$^{-1}$)} & \colhead{} & \colhead{(mJy)} & \colhead{(mJy)} & \colhead{(W\,Hz$^{-1}$)} & \colhead{(W)}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Act.: A = activity evidence (an X-ray counterpart or a SIMBAD activity type; Section~\ref{sec:candidates}), which at these distances is found for almost every star and is not used to select technosignature hosts. Elig.: technosignature eligibility (Section~\ref{sec:candidates}); E = eligible at any luminosity, L = eligible only above $L_{\rm env}=10^{10}\,\WHz$ (M dwarfs, and AFGK stars with Gaia multiplicity evidence), X = excluded (known radio-emitter class or $\log L_X/L_{\rm bol}\ge-4$). Every limit bounds the total emission at the star's position, natural or artificial, whatever its eligibility. E1: Epoch-1 catalog analysis. $N_{\rm ep}$, $S_{\rm coadd}$ (measured coadd flux density and its error), $S_{\rm lim}$, $L_{\nu,\rm lim}$: all-epoch coadded forced photometry. EIRP$_{\rm lim}=L_{\nu,\rm lim}\times2$\,GHz (steady, band-filling signal). The limits apply to emission within about one beam radius of the star; 1\arcsec\ corresponds to $d$\,au projected, and a transmitter on a planet at a wider separation would lie outside the measured position.}
\end{deluxetable*}
""")
    else:
        rows = [f"{clean_name(r.main_id, r.source_id)} & {r.dist_pc:.2f} & {r.bp_rp:.2f} & {r.S_lim_mJy:.2f} & ${sci(r.Lnu_lim_WHz)}$ & ${sci(r.EIRP_lim_W_2GHz)}$ \\\\" for _, r in near.iterrows()]
        open(os.path.join(HERE, "tables", "nearest_limits.tex"), "w").write(r"""\begin{deluxetable}{lrrrrr}
\tablecaption{95\% limits on steady 2--4\,GHz emission from the twelve nearest stars in the sample (completeness thresholds, or upper confidence bounds for stars with emission; Section~\ref{sec:limits_method}).\label{tab:nearest}}
\tablehead{\colhead{Star} & \colhead{$d$} & \colhead{$G_{BP}-G_{RP}$} & \colhead{$S_{\rm lim}$} & \colhead{$L_{\nu,\rm lim}$} & \colhead{EIRP$_{\rm lim}$} \\
\colhead{} & \colhead{(pc)} & \colhead{(mag)} & \colhead{(mJy)} & \colhead{(W\,Hz$^{-1}$)} & \colhead{(W)}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{EIRP$_{\rm lim}=L_{\nu,\rm lim}\times2$\,GHz (band-filling signal).}
\end{deluxetable}
""")
    rows = []
    for _, r in bl.iterrows():
        nm = "This work (VLASS QL)" if "VLASS" in r.survey else r.survey.replace("Price+2020 ", "BL ").replace(" (narrowband)", "")
        band = r.band.replace("-", "--"); dwell = re.sub(r"(\d)x(\d)", r"$\1\\times\2$", r.dwell).replace("~", r"$\sim$")
        pub = f" (${sci(r.EIRP_published_W)}$)" if np.isfinite(r.EIRP_published_W) else ""
        rows.append(f"{nm} & {band} & {dwell} & ${sci(r.S_lim_Jy_at50pc)}$ & ${sci(r.EIRP_lim_W_at50pc)}${pub} & {com(r.N_stars_constraining)} & ${100 * r.f95_common:.2g}$ " + r"\\")
    open(os.path.join(HERE, "tables", "bl_comparison.tex"), "w").write(r"""\begin{deluxetable*}{llllrrr}
\tablecaption{Comparison with the Breakthrough Listen narrowband survey of \citet{Price2020} for a star at 50\,pc.\label{tab:bl}}
\tablehead{\colhead{Survey} & \colhead{Band} & \colhead{Dwell} & \colhead{$S_{\rm lim}$ (Jy)} & \colhead{EIRP$_{\rm lim}$ (W)} & \colhead{$N_\star$} & \colhead{$f_{95}$ (\%)}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Breakthrough Listen (BL) limits apply to signals $\lesssim$3\,Hz wide with $|\dot\nu|\le4$\,Hz\,s$^{-1}$ and use the 1\,Hz transmitter-width convention of \citet{Price2020}, including the factor of 3.3 correction of \citet{Choza2024} (published values in parentheses), as in Figure~\ref{fig:tr}; the VLASS limit applies to the total in-band power of signals that survive the interference flagging (Section~\ref{sec:limits_method}). $N_\star$ is the number of stars constraining the occurrence limit (for BL GBT, the S-band targets, the band that overlaps VLASS; for VLASS, stars within 50\,pc without activity evidence). $f_{95}$ is the 95\% upper limit on the fraction of stars hosting a transmitter above EIRP$_{\rm lim}$ in a common construction: a 50\% detection probability per star, as adopted by \citet{Price2020}, and no background subtraction ($3/0.5N_\star$ for no detections; for VLASS $\mu_{95}=\MuFifty$ for the \KFifty\ steady-test detections among these stars). \citet{Price2020} quote 0.37\% (GBT S band) and 2.0\% (Parkes).}
\end{deluxetable*}
""")
    rows = [f"{r.image} & {r.epoch:.2f} & {r.peak_mJy:.2f} & {r.SNR:.1f} & ${r.tgt_E:+.2f}\\pm{r.err_E:.2f}$ & ${r.tgt_N:+.2f}\\pm{r.err_N:.2f}$ & ${r.star_E:+.2f}$ & ${r.star_N:+.2f}$ \\\\" for _, r in ast.iterrows()]
    open(os.path.join(HERE, "tables", "lspm_astrometry.tex"), "w").write(r"""\begin{deluxetable}{lrrrrrrr}
\tablecaption{Multi-epoch VLASS astrometry of the radio source near LSPM~J1943+5337.\label{tab:lspm}}
\tablehead{\colhead{Image} & \colhead{Epoch} & \colhead{$S_{\rm peak}$} & \colhead{SNR} & \colhead{$\Delta\alpha^*$} & \colhead{$\Delta\delta$} & \colhead{$\Delta\alpha^*_\star$} & \colhead{$\Delta\delta_\star$} \\
\colhead{} & \colhead{(yr)} & \colhead{(mJy\,b$^{-1}$)} & \colhead{} & \colhead{(arcsec)} & \colhead{(arcsec)} & \colhead{(arcsec)} & \colhead{(arcsec)}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Offsets are relative to the Epoch 1 catalog position, after tying each image to five compact reference components. Subscript $\star$: Gaia position of the star at the image epoch.}
\end{deluxetable}
""")
    # all-epoch tables
    if ALL is not None:
        rows = []
        _sp = {(r.shell, r.cls, r["pop"]): r for _, r in ALL["ST"].iterrows()}
        def _ms(sh, cl, pp):
            r = _sp.get((sh, cl, pp)); return r"\nodata" if r is None else f"${r.meanL/1e5:.1f}\\pm{r.meanL_err/1e5:.1f}$"
        for _, r in ALL["SA"].iterrows():
            rows.append(f"{r.shell.replace('-', '--')} & {r.cls} & {com(r.N_all)} & {com(r.N_clean)} & ${r.meanL/1e5:.1f}\\pm{r.meanL_err/1e5:.1f}$ & "
                        f"${r.meanL_snr:.1f}$ & {_ms(r.shell, r.cls, 'active')} & {_ms(r.shell, r.cls, 'inactive')} & {com(round(r.Neff_w))} & {100*r.top10_wfrac:.1f} & ${r.dmed_z:+.3f}\\pm{r.dmed_z_err:.3f}$ \\\\")
        open(os.path.join(HERE, "tables", "stack_results_all.tex"), "w").write(r"""\begin{deluxetable*}{llrrrrrrrrr}
\tabletypesize{\footnotesize}
\tablecaption{Stacks of the all-epoch coadded forced photometry.\label{tab:stacks_all}}
\tablehead{\colhead{Shell} & \colhead{Class} & \colhead{$N_{\rm all}$} & \colhead{$N_{\rm clean}$} & \colhead{$\langle L_\nu\rangle$} & \colhead{SNR} & \colhead{$\langle L_\nu\rangle_{\rm act}$} & \colhead{$\langle L_\nu\rangle_{\rm no\,act}$} & \colhead{$N_{\rm eff,w}$} & \colhead{Top 10} & \colhead{$\Delta\,{\rm med}(S/\sigma)$} \\
\colhead{(pc)} & \colhead{} & \colhead{} & \colhead{} & \colhead{($10^5$ W\,Hz$^{-1}$)} & \colhead{} & \colhead{($10^5$ W\,Hz$^{-1}$)} & \colhead{($10^5$ W\,Hz$^{-1}$)} & \colhead{} & \colhead{(\%)} & \colhead{}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{$\langle L_\nu\rangle$: inverse-variance-weighted mean spectral luminosity of clean stars (no secure association, no forced detection, no epoch with predicted catalog contamination $\ge0.5\sigma$), after subtracting the mean at the control positions; the error is the larger of the formal error and the scatter among the eight control positions. $N_{\rm eff,w}=(\sum w)^2/\sum w^2$; Top 10: weight fraction carried by the ten highest-weight stars. $\langle L_\nu\rangle_{\rm act}$ and $\langle L_\nu\rangle_{\rm no\,act}$: the same mean for the clean stars with and without activity evidence (Section~\ref{sec:candidates}); \nodata: fewer than ten stars. The control positions are subject to the same clean cuts as the stars. $\Delta\,{\rm med}(S/\sigma)$: median $S/\sigma$ of all unvetoed stars in the shell, including detected stars, minus the median at control positions; bootstrap error.}
\end{deluxetable*}
""")
        CA = pd.read_csv(os.path.join(RES, "alleepoch_candidates.csv"))
        # Table 3: forced detections at AFGK stars outside the excluded classes, i.e. the counted ones (eligible) and those
        # at AFGK stars with Gaia multiplicity evidence (eligible only above L_env, which none reaches)
        U = CA[CA.eligible | (CA.env & (CA.bp_rp < 1.84))].sort_values(["eligible", "dist_pc"], ascending=[False, True])
        # per-epoch catalog detections (21): "cat:sep"/flux/snr" entries, plus the QL median stack
        catd = {}
        fe, fm = os.path.join(RES, "epoch_catalog_candidates.csv"), os.path.join(RES, "medstack_star_matches.csv")
        if os.path.exists(fe):
            EC = pd.read_csv(fe).set_index("source_id")
            MS = pd.read_csv(fm).set_index("source_id") if os.path.exists(fm) else pd.DataFrame()
            for sid in U.source_id:
                ent = []
                cd = EC.cats_det.get(sid, np.nan) if sid in EC.index else np.nan
                if isinstance(cd, str):
                    for e in cd.split(";"):
                        c_, rest = e.split(":", 1); ent.append(f"{c_} ({fas(float(rest.split(chr(34))[0]))})")
                if sid in MS.index: ent.append(f"Med ({fas(MS.med_sep[sid])})")
                catd[sid] = ", ".join(ent) if ent else "---"
        rows = []
        for _, r in U.iterrows():
            nm = clean_name(r.main_id, r.source_id) if isinstance(r.main_id, str) and not r.main_id.startswith("Gaia DR3") else "---"
            mult = r.multiplicity_evidence if isinstance(r.multiplicity_evidence, str) and r.multiplicity_evidence else "---"
            mult = re.sub(r"over-luminous \((-?[0-9.]+) mag\)", lambda q: "$\\Delta M_G=" + q.group(1) + "$", mult)
            mult = re.sub(r"SIMBAD [^;]+", "SIMBAD", mult)
            eps = (r.epoch_snr if isinstance(r.epoch_snr, str) else "").replace(";", ", ")
            rows.append(f"{nm} & {r.source_id} & {'Y' if r.eligible else 'N'} & {r.dist_pc:.1f} & {r.bp_rp:.2f} & ${r.S_coadd_mJy:.2f}\\pm{r.sigma_coadd_mJy:.2f}$ & {r.snr_coadd:.1f} & {eps} & ${sci(r.L_max_epoch)}$ & {mult} & {catd.get(r.source_id, '---')} \\\\")
        open(os.path.join(HERE, "tables", "unexplained_candidates.tex"), "w").write(r"""{\setlength{\tabcolsep}{3pt}
\begin{deluxetable*}{lrcrrrr>{\raggedright\arraybackslash}p{2.5cm}r>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{2.0cm}}
\tabletypesize{\scriptsize}
\tablecaption{Forced detections (any epoch or the coadd at $\ge5\sigma$) at AFGK stars outside the known radio-emitter classes and with $\log L_X/L_{\rm bol}<-4$.\label{tab:unexplained}}
\tablehead{\colhead{Name} & \colhead{Gaia DR3} & \colhead{$k$} & \colhead{$d$} & \colhead{$G_{BP}-G_{RP}$} & \colhead{$S_{\rm coadd}$} & \colhead{SNR} & \colhead{Half-epoch:\,SNR} & \colhead{$L_{\nu,\max}$} & \colhead{Multiplicity} & \colhead{Catalogs} \\
\colhead{} & \colhead{} & \colhead{} & \colhead{(pc)} & \colhead{(mag)} & \colhead{(mJy)} & \colhead{} & \colhead{} & \colhead{(W\,Hz$^{-1}$)} & \colhead{} & \colhead{}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{$k$: Y = eligible star, counted in $k$; N = Gaia multiplicity evidence (RUWE\,$>1.4$, a non-single-star solution, or $\Delta M_G<-0.5$), so eligible only above $L_{\rm env}=10^{10}\,\WHz$, which none reaches (Section~\ref{sec:candidates}). $S_{\rm coadd}$, SNR: inverse-variance coadd over unvetoed epochs. In the steady test (coadd SNR\,$\ge5$) the counted forced detections give $k=\KSteady$ against $b=\BSteady$ expected at blank control positions under the same selection and veto; in the single-epoch test $k=\KSnap$ star-epochs against $b=\BSnap$. $L_{\nu,\max}$: spectral luminosity in the brightest epoch. Multiplicity: SIMBAD binary type, Gaia non-single-star solution (NSS), RUWE, and over-luminosity $\Delta M_G$ (mag) relative to the main-sequence ridge. Catalogs: independent per-epoch component catalogs with a $\ge5\sigma$ component within 4\arcsec\ of the star's position at that epoch, and the separation (E1: Epoch-1 QL; QL: NRAO QL lists; SE: Single Epoch lists; Med: QL median stack; Section~\ref{sec:catcheck}).}
\end{deluxetable*}
}
""")

        # Section 4.4 notes on the forced detections without activity evidence. The prose discusses the five steady-test
        # stars by name; if that set changes, the warning below says the text must be revisited.
        _SA = pd.read_parquet(os.path.join(DER, "ms_sample_vlass.parquet"), columns=["source_id", "ra", "dec"]).set_index("source_id")
        def _gal(ra, dec):   # ICRS -> Galactic (l, b) in degrees
            a, d = np.radians(ra), np.radians(dec)
            M = np.array([[-0.0548755604, -0.8734370902, -0.4838350155], [0.4941094279, -0.4448296300, 0.7469822445],
                          [-0.8676661490, -0.1980763734, 0.4559837762]])
            x = M @ np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
            return np.degrees(np.arctan2(x[1], x[0])) % 360, np.degrees(np.arcsin(x[2]))
        Uk = U[U.eligible]; Us, Ue = Uk[Uk.cand_coadd], Uk[~Uk.cand_coadd]       # counted (eligible) forced detections
        _IDS = {2071061615589030400}; _IDE = {1249218214732128640, 6180509689006780672}
        if set(Us.source_id) != _IDS or set(Ue.source_id) != _IDE:
            print("WARNING: counted forced detections changed:", sorted((set(Us.source_id) ^ _IDS) | (set(Ue.source_id) ^ _IDE)), "- revise ms.tex Section 4.4")
        _lb = [_gal(_SA.ra[i], _SA.dec[i]) for i in Us.source_id]
        put("NSteadyU", str(len(Us))); put("NSnapOnlyU", str(len(Ue)))
        put("NSteadyNoErass", str(sum(not (179.944 < l < 359.944) for l, b in _lb)))   # eRASS1 = German half, 179.944 < l < 359.944
        put("NSteadyCyg", str(sum((75 < l < 90) and abs(b) < 15 for l, b in _lb)))
        _mx = [max(float(e.split(":")[1]) for e in str(t).split(";")) for t in Ue.epoch_snr]
        put("SnapOnlyLo", f"{min(_mx):.1f}"); put("SnapOnlyHi", f"{max(_mx):.1f}")
        put("NSnapOnlyMult", str(int(sum(isinstance(m, str) and m != "" for m in Ue.multiplicity_evidence))))
        _g = CA.set_index("source_id")          # named M-dwarf forced detections discussed in Section 4.4
        _r = _g.loc[349716170186042880]; _L = 4 * np.pi * (_r.dist_pc * 3.0857e16) ** 2 * _r.S_coadd_mJy * 1e-29
        put("GOTEL", sci(_L)); put("GOTELcgs", sci(_L * 1e7)); put("GOTEd", f"{_r.dist_pc:.0f}"); put("GOTEdMG", f"{-_r.dMG:.1f}")
        _r = _g.loc[2080884927342950272]
        put("PMJSNR", f"{_r.snr_coadd:.1f}"); put("PMJMaxEp", f"{max(float(e.split(':')[1]) for e in _r.epoch_snr.split(';')):.1f}")
        _sid = 2563195186142791552; _ra, _de = _SA.ra[_sid], _SA.dec[_sid]
        _E = pd.read_parquet(os.path.join(RAW, "vlass_e1ql_components.parquet"), columns=["RAJ2000", "DEJ2000", "Ftot", "DCMaj"],
                             filters=[("DEJ2000", ">", _de - 0.01), ("DEJ2000", "<", _de + 0.01)])
        _sep = np.hypot((_E.RAJ2000 - _ra) * np.cos(np.radians(_de)), _E.DEJ2000 - _de) * 3600; _E = _E[_sep < 15].assign(sep=_sep[_sep < 15])
        put("GExtN", str(len(_E))); put("GExtSepLo", f"{_E.sep.min():.1f}"); put("GExtSepHi", f"{_E.sep.max():.1f}")
        put("GExtF", f"{_E.Ftot.sum():.0f}"); put("GExtMaj", f"{_E.DCMaj.min():.1f}"); put("GExtMajHi", f"{_E.DCMaj.max():.1f}")
        _EC = pd.read_csv(os.path.join(RES, "epoch_catalog_candidates.csv")).set_index("source_id")
        put("GExtVetoOwn", str(int(_EC.n_veto_own[_sid]))); put("GExtNep", str(int(_EC.n_epochs[_sid])))
        _cd = str(_EC.cats_det.get(349716170186042880, ""))
        _seps = [float(e.split(":", 1)[1].split('"')[0]) for e in _cd.split(";") if ":" in e]
        if _seps: put("GOTEsep", f"{min(_seps):.1f}")
        _ty = CA[CA.source_id == 2071061615589030400]
        if len(_ty): put("CandAstroScoadd", f"{_ty.S_coadd_mJy.iloc[0]:.2f}")
        import sys; sys.path.insert(0, os.path.join(ROOT, "code")); import common as _C
        _T = pd.read_csv(os.path.join(RES, "vlass_star_matches_classified.csv"))
        _Mn = _T[(_T["class"] != "UNEXPLAINED") & ~_T["class"].str.startswith("background") & (_T.bp_rp >= _C.BPRP_SPLIT)]
        put("NMSecNat", str(len(_Mn))); put("MSecLlo", sci(_Mn.Lnu_R_WHz.min())); put("MSecLhi", sci(_Mn.Lnu_R_WHz.max()))

    # per-epoch catalog cross-check (21)
    fx = os.path.join(DER, "epoch_catalog_crosscheck.json")
    if os.path.exists(fx):
        X = json.load(open(fx)); pc = X["per_catalogue"]
        ql = [c for c in ["QL3.1", "QL3.2", "QL4.1"] if c in pc]; se = [c for c in pc if c.startswith("SE")]
        cov = lambda cs: 100 * sum(pc[c]["n_covered"] for c in cs) / sum(pc[c]["n_star_epochs_in_half"] for c in cs)
        put("XcSEcovTwo", f"{cov([c for c in se if c.startswith('SE2')]):.0f}"); put("XcSEcovThree", f"{cov([c for c in se if c.startswith('SE3')]):.0f}")
        kq = [pc[c]["kappa_implied"] for c in ql]; put("XcKappaRange", f"{min(kq):.2f}--{max(kq):.2f}")
        w = np.array([pc[c]["ratio_true_err"] ** -2 for c in se]); v = np.array([pc[c]["ratio_true"] for c in se])
        put("XcSEratio", f"{np.sum(w * v) / w.sum():.2f}"); put("XcSEratioErr", f"{w.sum() ** -0.5:.2f}")
        put("XcSEratioN", str(sum(pc[c]["flux_n"] for c in se)))
        put("XcSEratioRange", f"{v.min():.2f}--{v.max():.2f}")
        put("XcBiasImplied", f"{0.92 * np.sum(w * v) / w.sum():.2f}")
        sq = X["SE3_vs_QL3"]; put("XcSEoverQL", f"{sq['peak_ratio_SE_over_QL']:.3f}"); put("XcSEoverQLErr", f"{sq['peak_ratio_err']:.3f}")
        put("XcSEoverQLN", str(sq["n"])); put("XcSEQLpos", f"{max(sq['pos_diff_rstd_E'], sq['pos_diff_rstd_N']):.2f}")
        asys = X["candidates"]["astrometric_sys_arcsec"]
        put("XcAstSE", fas(min(asys[c] for c in se), 2) + "--" + fas(max(asys[c] for c in se), 2))
        put("XcAstQL", fas(min(asys[c] for c in ql), 2) + "--" + fas(max(asys[c] for c in ql), 2))
        conc = lambda cs: (sum(pc[c]["n_forced_ge5_unvetoed_with_cat"] for c in cs), sum(pc[c]["n_forced_ge5_unvetoed"] for c in cs))
        a_, b_ = conc(ql); put("XcConcQL", f"{a_}"); put("XcConcQLN", f"{b_}"); put("XcConcQLpct", f"{100 * a_ / b_:.0f}")
        a_, b_ = conc(se); put("XcConcSE", f"{a_}"); put("XcConcSEN", f"{b_}"); put("XcConcSEpct", f"{100 * a_ / b_:.0f}")
        lt = ql + se
        put("XcCatLtFive", str(sum(pc[c]["n_cat_forced_lt5"] for c in lt))); put("XcCatLtFiveOff", str(sum(pc[c]["n_cat_forced_lt5_sep_gt1p5"] for c in lt)))
        put("XcCatChance", f"{sum(pc[c]['chance_expected'] for c in lt):.0f}")
        cd = X["candidates"]; put("XcUnexplCat", str(cd["n_unexplained_with_cat_det"])); put("XcCandCat", str(cd["n_with_cat_det"]))
        V = X["veto_variants"]
        for tag, k in (("Own", "own_epoch"), ("Union", "union")):
            put(f"Xc{tag}Ks", str(V[k]["k_steady"])); put(f"Xc{tag}Bs", f"{V[k]['b_steady']:.1f}"); put(f"Xc{tag}Mus", f"{V[k]['mu95_steady']:.1f}")
            if "f95_b0_floor" in V[k]:     # headline construction (no background subtraction) for this veto
                put(f"Xc{tag}FBz", sci(V[k]["f95_b0_floor"])); put(f"Xc{tag}MuBz", f"{V[k]['mu95_steady_b0']:.1f}")
                put(f"Xc{tag}FBzPct", f"{100 * (V[k]['f95_b0_floor'] / V['adopted_E1']['f95_b0_floor'] - 1):.0f}")
            put(f"Xc{tag}Ke", str(V[k]["k_snap"])); put(f"Xc{tag}Be", f"{V[k]['b_snap']:.1f}"); put(f"Xc{tag}Mue", f"{V[k]['mu95_snap']:.1f}")
        from scipy.stats import poisson as _pv
        _pp = {(k, t): _pv.sf(V[k][f"k_{t}"] - 1, V[k][f"b_{t}"]) for k in ("own_epoch", "union") for t in ("steady", "snap")}
        _kmin = min(_pp, key=_pp.get)
        if _kmin != ("union", "snap"): print("WARNING: largest veto-variant excess is", _kmin, "- update ms.tex Section 4.5 text")
        put("XcUnionPe", f"{_pp[('union', 'snap')]:.2f}")
        put("XcOwnAdded", str(len(V["own_epoch"]["added_vs_adopted"]))); put("XcOwnAddedUnexpl", str(len(V["own_epoch"]["added_unexplained"])))
        put("XcOwnNotEOne", str(X["star_epochs_veto"]["own_not_E1"]))
        put("XcMuSnapChange", f"{100 * (V['union']['mu95_snap'] / V['adopted_E1']['mu95_snap'] - 1):+.0f}")
        put("XcMuSteadyChange", f"{100 * (V['union']['mu95_steady'] / V['adopted_E1']['mu95_steady'] - 1):+.0f}")
        co = X["catalogue_only"]; put("XcCatOnlyN", str(co["n_stars_any"])); put("XcCatOnlyMulti", str(co["n_stars_ge2_epochs"]))
        M = X["medstack"]; put("XcMedN", com(M["n_stars_tested"])); put("XcMedDet", str(M["n_det"]))
        for s in ["secure", "explained", "unexplained", "catonly", "none"]:
            put(f"XcMed{s.capitalize()}", str(M["by_status"][s])); put(f"XcMed{s.capitalize()}Tested", str(M["tested_by_status"][s]))
        put("XcMedChance", f"{M['chance_expected']:.0f}"); put("XcMedRms", f"{1000 * M['Isl_rms_median_near_stars']:.0f}")
        put("XcNCat", str(len(ql) + len(se) + 1))
        put("XcSEcons", f"{100 * (np.sum(w * v) / w.sum() - 1):.0f}")
        # V376 Cep: known RS CVn radio star below R = 0.5 after the Epoch-1 astrometric correction
        _A = pd.read_parquet(os.path.join(DER, "matches_10arcsec.parquet")); _v = _A[_A.source_id == 2299477768899190400].sort_values("sep").iloc[0]
        _M = pd.read_parquet(os.path.join(DER, "epoch_catalog_matches.parquet")); _m = _M[(_M.source_id == 2299477768899190400) & _M.cat.str.startswith("QL3")]
        put("VCepSep", f"{_v.sep:.2f}"); put("VCepRel", f"{_v.rel:.2f}"); put("VCepLaterSep", f"{_m.m_sep.min():.2f}")
        EC = pd.read_csv(os.path.join(RES, "epoch_catalog_candidates.csv"))
        mv = EC[(EC.dof_2 >= 2) & (EC.motion_span_arcsec > 5)].sort_values("motion_span_arcsec", ascending=False)
        mv = mv.drop_duplicates(subset=["chi2_stationary"])      # a binary's two stars can share one radio source
        for tag, (_, r_) in zip(["A", "B"], mv.head(2).iterrows()):
            put(f"XcMove{tag}Name", clean_name(r_.main_id, r_.source_id).strip("{}"))
            put(f"XcMove{tag}Span", f"{r_.motion_span_arcsec:.0f}"); put(f"XcMove{tag}Co", f"{r_.chi2_comoving:.0f}")
            put(f"XcMove{tag}St", f"{r_.chi2_stationary:.0f}"); put(f"XcMove{tag}Dof", str(int(r_.dof_2)))
        MS = pd.read_csv(os.path.join(RES, "medstack_star_matches.csv")); nn = MS[MS.status == "none"]
        put("XcMedNoneSepMin", f"{nn.med_sep.min():.1f}"); put("XcMedNoneSepMax", f"{nn.med_sep.max():.1f}")
        uu = MS[MS.status == "unexplained"]; put("XcMedUnexplSepMax", f"{uu.med_sep.max():.1f}")

    # figures
    for f in ["vlass_separation_histogram", "vlass_detections_cmd", "vlass_perstar_limits", "vlass_population_limit",
              "vlass_technosig_summary", "vlass_ep1_stacking_summary", "transmitter_rate_vs_eirp", "lspm1943_archival_imaging", "lspm1943_vlass_multiepoch_astrometry", "vlass_detections_multiepoch",
              "alleepoch_perstar_limits", "alleepoch_population_limit", "alleepoch_stacks"]:
        for ext in ("pdf", "png"):
            src = os.path.join(RES, f"{f}.{ext}")
            if os.path.exists(src): shutil.copy(src, os.path.join(HERE, "figures", f"{f}.{ext}"))
    with open(os.path.join(HERE, "numbers.tex"), "w") as f:     # last, so that macros set by any block are included
        f.write("% AUTO-GENERATED by make_paper_inputs.py from ../results -- do not edit by hand\n")
        for k, val in macros.items(): f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
    print(f"wrote numbers.tex ({len(macros)} macros) and {len(os.listdir(os.path.join(HERE, 'tables')))} tables")
