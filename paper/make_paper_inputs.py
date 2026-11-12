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
    put("SigSys", f"{fit['sigma_sys_arcsec']:.2f}"); put("NTrue", f"{fit['N_true']:.0f}"); put("NSecure", str(fit["n_rel_gt_0p5"]))
    put("NRelNine", str(fit["n_rel_gt_0p9"])); put("NFalseSecure", f"{fit['expected_false_rel_gt_0p5']:.1f}")

    # --- classification
    T = pd.read_csv(os.path.join(RES, "vlass_star_matches_classified.csv")); xr = pd.read_csv(os.path.join(RAW, "xray_counterparts.csv"))
    put("NCand", str(len(xr))); put("NRxs", str(int(xr.rxs_r.notna().sum()))); put("NEra", str(int(xr.era_r.notna().sum())))
    S = T[T.secure].copy(); vc = S["class"].value_counts()
    grp = {"AB": ["binary/overluminous + X-ray", "RS CVn / active binary", "BY Dra / active binary", "binary/overluminous"],
           "FL": ["flare star", "X-ray active single star"], "OT": ["young star", "accreting WD binary (CV)", "magnetic CP star"]}
    for g_, cl in grp.items(): put(f"NClass{g_}", str(int(sum(vc.get(c, 0) for c in cl))))
    for c, k in [("binary/overluminous + X-ray", "BinX"), ("RS CVn / active binary", "RS"), ("BY Dra / active binary", "BY"), ("binary/overluminous", "Bin"),
                 ("flare star", "Flare"), ("X-ray active single star", "XA"), ("young star", "Young"), ("accreting WD binary (CV)", "CV"), ("magnetic CP star", "CP")]:
        put(f"NC{k}", str(int(vc.get(c, 0))))
    put("NSecureX", str(int(S.LX.notna().sum())))
    gb = S.logGB.dropna(); put("GBmed", f"{gb.median():.2f}"); put("GBmin", f"{gb.min():.2f}"); put("GBmax", f"{gb.max():.2f}")
    put("NUnexplained", str(int((S["class"] == "UNEXPLAINED").sum())))
    put("NNatural", str(int((~S["class"].str.startswith("background") & (S["class"] != "UNEXPLAINED")).sum())))
    put("NBackground", str(int(S["class"].str.startswith("background").sum())))

    # --- per-star limits and population
    L = pd.read_parquet(os.path.join(RES, "vlass_perstar_limits_100pc_MS.parquet"))
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
    lspm = T[T.source_id == 2137474969751117312].iloc[0]
    put("LspmS", f"{lspm.Ftot:.2f}"); put("LspmSep", f"{lspm.sep:.2f}"); put("LspmRel", f"{lspm.rel:.3f}"); put("LspmDist", f"{lspm.dist_pc:.0f}")
    put("LspmMaj", f"{lspm.DCMaj:.1f}"); put("LspmMin", f"{lspm.DCMin:.1f}")

    # --- stacking
    F = pd.read_csv(os.path.join(RES, "vlass_ep1_stack_results.csv")); fv = pd.read_csv(os.path.join(RES, "vlass_fraction_vs_power_limits.csv"))
    m24 = F[(F.shell_pc == "20-40") & (F.type == "M")].iloc[0]
    put("StackMS", f"{m24.S_mean_uJy:.1f}"); put("StackMSerr", f"{m24.S_err_uJy:.1f}"); put("StackMSNR", f"{m24.S_SNR:.1f}"); put("StackMN", com(m24.N))
    put("StackML", sci(m24.Lmean_WHz)); put("StackMMed", f"{m24.S_median_uJy:.1f}"); put("StackMMedSNR", f"{m24.S_median_uJy / m24.S_median_err_uJy:.1f}")
    put("StackFGKmaxSNR", f"{F[F.type == 'FGK'].L_SNR.max():.1f}"); put("NStackStars", com(F.N.sum()))
    Lg = fv.Lnu_T_WHz.values
    for t in ["FGK", "M"]:
        put(f"Reach{t}", sci(Lg[np.nanargmax(fv[f'f_UL95_stack_{t}'] < 1)], 2))
        put(f"DetReach{t}", sci(Lg[np.nanargmax(np.nan_to_num(fv[f'f_UL95_detection_{t}'], nan=9) < 1)], 1))
        i6 = np.argmin(np.abs(Lg - 1e6)); put(f"FStackSix{t}", f"{fv[f'f_UL95_stack_{t}'][i6]:.2f}")
    put("ReachEIRP", sci(min(Lg[np.nanargmax(fv[f'f_UL95_stack_{t}'] < 1)] for t in ['FGK', 'M']) * 2e9))
    z = np.load(os.path.join(RAW, "vlass_ep1_ql_cutouts_41px.npz"), mmap_mode="r"); put("NCutouts", com(len(z["source_id"])))

    # --- BL comparison
    bl = pd.read_csv(os.path.join(RES, "vlass_vs_BL_Price2020_comparison.csv")); v = bl.iloc[-1]
    put("VlassFiftyS", f"{v.S_lim_Jy_at50pc*1e3:.2f}"); put("VlassFiftyEIRP", sci(v.EIRP_lim_W_at50pc)); put("BLratio", sci(v.EIRP_lim_W_at50pc / 2.1e12))
    put("FluxRatio", com(round((2.1e12 / (4*np.pi*(50*3.0857e16)**2)) / (v.S_lim_Jy_at50pc * 1e-26), -2))); put("NFifty", com(v.N_stars_constraining))
    put("FFifty", sci(3.0 / v.N_stars_constraining)); put("FFiftyHalf", sci(6.0 / v.N_stars_constraining))

    with open(os.path.join(HERE, "numbers.tex"), "w") as f:
        f.write("% AUTO-GENERATED by make_paper_inputs.py from ../results -- do not edit by hand\n")
        for k, val in macros.items(): f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")

    # ---------------- tables
    code = {"binary/overluminous + X-ray": "B+X", "RS CVn / active binary": "RS", "BY Dra / active binary": "BY", "binary/overluminous": "B",
            "flare star": "Fl", "X-ray active single star": "X", "young star": "Y", "accreting WD binary (CV)": "CV", "magnetic CP star": "CP",
            "background (multi-epoch astrometry)": "BG"}
    S = S.sort_values("dist_pc"); rows = []
    for _, r in S.iterrows():
        ND = r"\nodata"
        xflag = (("R" if np.isfinite(r.rxs_r) else "") + ("E" if np.isfinite(r.era_r) else "")) or ND
        gbs = f"{r.logGB:.2f}" if np.isfinite(r.logGB) else ND
        nm1 = clean_name(r.main_id, r.source_id); nm1 = r"\nodata" if nm1.startswith("{Gaia DR3") else nm1
        rows.append(f"{nm1} & {r.source_id} & {r.dist_pc:.1f} & {r.bp_rp:.2f} & {r.dMG:+.2f} & {r.ruwe:.2f} & "
                    f"{r.sep:.2f} & {r.rel:.3f} & {r.Ftot/0.87:.2f} & {np.log10(r.Lnu_R_WHz):.2f} & {xflag} & {gbs} & {code.get(r['class'], r['class'])} " + r"\\")
    open(os.path.join(HERE, "tables", "secure_matches.tex"), "w").write(r"""\setlength{\tabcolsep}{2.5pt}
\begin{deluxetable*}{lrrrrrrrrrcrl}
\tabletypesize{\scriptsize}
\tablecaption{VLASS Epoch 1 components securely associated (reliability $R>0.5$) with \NSecure\ main-sequence stars within 100\,pc.\label{tab:matches}}
\tablehead{\colhead{Name} & \colhead{Gaia DR3} & \colhead{$d$} & \colhead{$G_{BP}-G_{RP}$} & \colhead{$\Delta M_G$} & \colhead{RUWE} & \colhead{Sep.} & \colhead{$R$} & \colhead{$S_{3\,\rm GHz}$} & \colhead{$\log L_\nu$} & \colhead{X-ray} & \colhead{$\log\frac{L_X}{L_\nu}$} & \colhead{Class} \\
\colhead{} & \colhead{} & \colhead{(pc)} & \colhead{(mag)} & \colhead{(mag)} & \colhead{} & \colhead{($''$)} & \colhead{} & \colhead{(mJy)} & \colhead{(W\,Hz$^{-1}$)} & \colhead{} & \colhead{(Hz)} & \colhead{}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Sorted by distance. Separation at the epoch of the component's VLASS image. $S_{3\,\rm GHz}$ is the catalogued total flux density divided by 0.87 (Quick Look flux-scale correction). X-ray: R = ROSAT 2RXS, E = eRASS1. Class: B+X binary/over-luminous with X-ray detection; RS = RS~CVn; BY = BY~Dra; B = binary/over-luminous; Fl = flare star; X = X-ray-active single star; Y = young star; CV = cataclysmic variable; CP = magnetic chemically peculiar star; BG = background source (Section~\ref{sec:lspm}). A machine-readable version with all columns is provided in the reproducibility package.}
\end{deluxetable*}
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
\tablecomments{Means are measured against the empirical null (stacked image at 10--18$''$ from the centre); uncertainties are the larger of the formal and null-ring scatter. $\langle L_\nu\rangle_{95}$ is the one-sided 95\% upper limit on the mean spectral luminosity; EIRP$_{95}$ assumes a flat signal filling 2\,GHz. FGK/M split at $G_{BP}-G_{RP}=1.84$.}
\end{deluxetable*}
""")
    rows = [f"{clean_name(r.main_id, r.source_id)} & {r.dist_pc:.2f} & {r.bp_rp:.2f} & {r.S_lim_mJy:.2f} & ${sci(r.Lnu_lim_WHz)}$ & ${sci(r.EIRP_lim_W_2GHz)}$ \\\\" for _, r in near.iterrows()]
    open(os.path.join(HERE, "tables", "nearest_limits.tex"), "w").write(r"""\begin{deluxetable}{lrrrrr}
\tablecaption{95\% upper limits on steady 2--4\,GHz emission from the twelve nearest stars in the sample.\label{tab:nearest}}
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
        rows.append(f"{nm} & {band} & {dwell} & ${sci(r.S_lim_Jy_at50pc)}$ & ${sci(r.EIRP_lim_W_at50pc)}$ & {com(r.N_stars_constraining)} " + r"\\")
    open(os.path.join(HERE, "tables", "bl_comparison.tex"), "w").write(r"""\begin{deluxetable*}{llllrr}
\tablecaption{Comparison with the Breakthrough Listen narrowband survey of \citet{Price2020} for a star at 50\,pc.\label{tab:bl}}
\tablehead{\colhead{Survey} & \colhead{Band} & \colhead{Dwell} & \colhead{$S_{\rm lim}$ (Jy)} & \colhead{EIRP$_{\rm lim}$ (W)} & \colhead{$N_\star$}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Breakthrough Listen (BL) limits apply to signals $\lesssim$3\,Hz wide with $|\dot\nu|\le4$\,Hz\,s$^{-1}$ and use the 1\,Hz transmitter-width convention of \citet{Price2020}; the VLASS limit applies to the total in-band power of any signal. $N_\star$ is the number of stars constraining the occurrence limit (for VLASS: stars within 50\,pc).}
\end{deluxetable*}
""")
    rows = [f"{r.image} & {r.epoch:.2f} & {r.peak_mJy:.2f} & {r.SNR:.1f} & ${r.tgt_E:+.2f}\\pm{r.err_E:.2f}$ & ${r.tgt_N:+.2f}\\pm{r.err_N:.2f}$ & ${r.star_E:+.2f}$ & ${r.star_N:+.2f}$ \\\\" for _, r in ast.iterrows()]
    open(os.path.join(HERE, "tables", "lspm_astrometry.tex"), "w").write(r"""\begin{deluxetable}{lrrrrrrr}
\tablecaption{Multi-epoch VLASS astrometry of the radio source near LSPM~J1943+5337.\label{tab:lspm}}
\tablehead{\colhead{Image} & \colhead{Epoch} & \colhead{$S_{\rm peak}$} & \colhead{SNR} & \colhead{$\Delta\alpha^*$} & \colhead{$\Delta\delta$} & \colhead{$\Delta\alpha^*_\star$} & \colhead{$\Delta\delta_\star$} \\
\colhead{} & \colhead{(yr)} & \colhead{(mJy\,b$^{-1}$)} & \colhead{} & \colhead{($''$)} & \colhead{($''$)} & \colhead{($''$)} & \colhead{($''$)}}
\startdata
""" + "\n".join(rows) + r"""
\enddata
\tablecomments{Offsets are relative to the Epoch 1 catalogue position, after tying each image to five compact reference components. Subscript $\star$: Gaia position of the star at the image epoch.}
\end{deluxetable}
""")
    # figures
    for f in ["vlass_technosig_summary", "vlass_ep1_stacking_summary", "lspm1943_archival_imaging", "lspm1943_vlass_multiepoch_astrometry"]:
        for ext in ("pdf", "png"):
            src = os.path.join(RES, f"{f}.{ext}")
            if os.path.exists(src): shutil.copy(src, os.path.join(HERE, "figures", f"{f}.{ext}"))
    print(f"wrote numbers.tex ({len(macros)} macros) and {len(os.listdir(os.path.join(HERE, 'tables')))} tables")
