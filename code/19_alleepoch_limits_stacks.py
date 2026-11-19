"""19 - All-epoch per-star limits, population limits from per-star detection probabilities, and stacks.

Per-star limits (95%), from the forced photometry of 17 and the vetoes of 18:
  steady    : inverse-variance coadd over unvetoed epochs. S_lim = (5 + 1.645) sigma_coadd for stars without emission;
              S_coadd + 1.645 sigma_coadd (if larger) for stars with natural or unexplained emission; for stars whose
              every epoch is confusion-vetoed, the largest single-epoch S + 1.645 sigma.
  any-epoch : a transmitter on in only one snapshot, at an unknown epoch, is excluded only if it would have been detected
              in whichever unvetoed epoch it was on: the largest (worst-epoch) single-epoch limit, (5 + 1.645) sigma_e,
              or S_e + 1.645 sigma_e where larger for stars with emission. The best-epoch value is kept as
              S_lim_bestepoch (the limit if the transmitter happened to be on in the lowest-noise epoch).
Population (steady transmitter of spectral luminosity L, flux S_L = L / 4 pi d^2):
  p_i(L) = Phi((S_L - t_i) / sigma_coadd)   (steady test: coadd only, one trial per star),
  with t_i = 5 sigma_coadd at every star without activity evidence, the rule that defines k (S_coadd + 5 sigma_coadd at the
  candidates, asking for an additional transmitter above the measured source, is kept as f95_b0_add; a transmitter
  must exceed it), and p_i = 0 for stars with activity evidence (04b), which would be classified as natural emitters.
  The expected number of detected transmitters is f * sum_i p_i = f * N_eff(L); with k unexplained candidates over an
  expected chance background b (18, steady test), f_95(L) = mu_95(k, b) / N_eff(L) (f95), and without background
  subtraction f_95 = mu_95(k, 0) / N_eff (f95_b0, the headline limit). Stars vetoed in coadd units (18) have p_i = 0.
  Duty cycle, counted per host star: a host on in a random fraction delta of snapshots is detected in at least one epoch
  with probability 1 - prod_e (1 - delta p_ie), so f_95(delta) = mu_95,host / sum_i [1 - prod_e (1 - delta p_ie)] (f95_duty*),
  with mu_95,host from the host-level snapshot test of 18; for small delta, (f delta)_95 = mu_95,host / sum_ie p_ie (fdelta95).
Stacks, in distance shells x (AFGK: BP-RP < 1.84; M), per epoch and for the all-epoch coadd:
  mean     : inverse-variance-weighted mean L_nu of clean stars (not secure, not candidates, no epoch with predicted
             catalogue contamination >= 0.5 sigma). Null from the same weighted mean at each of the 8 control positions;
             the null mean is subtracted and the error is the larger of the formal error and the null scatter.
             N_eff,w = (sum w)^2 / sum w^2 and the largest weight fraction are reported.
  median   : median S/sigma and median S of all stars in the shell except confusion-vetoed ones, *including* detected
             stars, compared with the median at control positions; error from 200 bootstrap resamples of stars.
  The stacks are reported for all clean stars and split by activity evidence (pop = all / active / inactive).
Stack limits (stars without activity evidence only):
  mean     : for each L_T, the clean stars at which such a transmitter would be at most 3 sigma (selection by noise alone,
             so hosts are rarely removed by the clean cut); weighted mean m with error s_m (formal error scaled by the
             control-position rms of S/sigma, or the scatter of the 8 control means with a Student-t quantile if larger),
             and host-sampling variance L_T^2 f (1 - f) / N_eff: f_95 solves f L_T - q sqrt(s_m^2 + L_T^2 f(1-f)/N_eff) = max(m, 0).
             Coverage is checked by injecting transmitters into the control-position measurements (alleepoch_stack_coverage.csv).
  median   : Monte Carlo, for the pre-specified pooled 0-100 pc sample of each class: give a random fraction f of the stars
             an extra S_L / sigma_i and find the f at which the median shift exceeds the observed shift + 1.645 sigma_med.

Inputs : data/derived/alleepoch_forced.parquet (+ alleepoch_forced_controls.parquet), alleepoch_candidates.json, forced_calibration.json,
         data/derived/ms_sample_vlass.parquet, results/vlass_perstar_limits_100pc_MS.parquet, results/alleepoch_candidates.csv
Outputs: results/alleepoch_perstar_limits.parquet, results/alleepoch_population_limits.csv,
         results/alleepoch_stacks.csv, results/alleepoch_stack_limits.csv, results/alleepoch_stack_coverage.csv,
         data/derived/alleepoch_summary.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json
import numpy as np, pandas as pd
from scipy.stats import norm

THR, Z = C.DET_SIGMA, C.Z95
N_CTRL = 8
LGRID = np.logspace(4, 11, 141)
rng = np.random.default_rng(20261113)


def lum(S_mJy, d_pc): return C.lum(S_mJy, d_pc)


def coadd(g):
    w = 1 / g.sigma_mJy ** 2
    return pd.Series(dict(S=np.sum(g.S_mJy * w) / w.sum(), sig=w.sum() ** -0.5, n=len(g), sig_best=g.sigma_mJy.min()))


def wmean_lum(S, sig, d):
    Lv, sL = lum(S, d), lum(sig, d); w = 1 / sL ** 2
    return np.sum(w * Lv) / w.sum(), w.sum() ** -0.5, w


def median_f_limit(z, s_over_sig, allowed, nmc=40):
    """Smallest f such that adding s_over_sig to a random fraction f of z shifts the median by more than `allowed`."""
    m0 = np.median(z)
    def shift(f):
        out = []
        for _ in range(nmc):
            k = rng.random(len(z)) < f; zz = z.copy(); zz[k] += s_over_sig[k]; out.append(np.median(zz) - m0)
        return np.mean(out)
    if shift(1.0) <= allowed: return np.nan
    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if shift(mid) > allowed: hi = mid
        else: lo = mid
    return hi


if __name__ == "__main__":
    LT = pd.read_parquet(os.path.join(C.DER, "alleepoch_forced.parquet"))
    LT = LT.merge(pd.read_parquet(os.path.join(C.DER, "alleepoch_forced_controls.parquet")), on=["source_id", "tag"], how="left", validate="one_to_one")
    cj = json.load(open(os.path.join(C.DER, "alleepoch_candidates.json")))
    CA = pd.read_csv(os.path.join(C.RES, "alleepoch_candidates.csv"))
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id", "dist_pc", "bp_rp"])
    tags = sorted(LT.tag.unique())
    LT = LT.merge(sv, on="source_id", how="left")
    natural = set(CA.source_id[CA.explained]) | set(LT.source_id[LT.detected])
    cand = set(CA.source_id)
    # stars with activity evidence (04b): a transmitter there would be classified as a natural emitter, so they cannot
    # contribute a detection to k and get p_i = 0 in N_eff (their per-star limits are unaffected)
    ACT = pd.read_parquet(os.path.join(C.DER, "activity_all.parquet"), columns=["source_id", "active"])
    active = set(ACT.source_id[ACT.active])
    # ---------------- per-star limits
    U = LT[~LT.veto]
    co = U.groupby("source_id")[["S_mJy", "sigma_mJy"]].apply(coadd)
    P = sv.set_index("source_id").join(co, how="left")
    P["n_ep_all"] = LT.groupby("source_id").size().reindex(P.index).fillna(0).astype(int)
    P["any_veto"] = LT.groupby("source_id").veto.any().reindex(P.index).fillna(False)
    P["natural"] = P.index.isin(natural); P["candidate"] = P.index.isin(cand)
    P["unexplained_candidate"] = P.candidate & ~P.natural
    emis = P.natural | P.candidate
    P["S_lim_steady"] = np.where(emis, np.maximum((THR + Z) * P.sig, P.S + Z * P.sig), (THR + Z) * P.sig)
    allv = P.n.isna() & (P.n_ep_all > 0)                     # every epoch vetoed: confusion-limited
    vmax = LT[LT.veto].assign(x=lambda d: d.S_mJy + Z * d.sigma_mJy).groupby("source_id").x.max()
    P.loc[allv, "S_lim_steady"] = vmax.reindex(P.index[allv]).values
    lim_e = (THR + Z) * U.sigma_mJy
    lim_e = pd.Series(np.where(U.source_id.isin(natural | cand), np.maximum(lim_e, U.S_mJy + Z * U.sigma_mJy), lim_e), index=U.index)
    P["S_lim_anyepoch"] = lim_e.groupby(U.source_id).max().reindex(P.index)          # worst unvetoed epoch
    P["S_lim_bestepoch"] = lim_e.groupby(U.source_id).min().reindex(P.index)
    P["L_lim_steady"] = lum(P.S_lim_steady, P.dist_pc); P["L_lim_anyepoch"] = lum(P.S_lim_anyepoch, P.dist_pc)
    P["class"] = np.where(P.bp_rp < C.BPRP_SPLIT, "AFGK", "M")
    P.reset_index().to_parquet(os.path.join(C.RES, "alleepoch_perstar_limits.parquet"), index=False)
    # ---------------- population: N_eff(L) = sum_i p_i(L)
    ok = P.n.notna().values
    d = P.dist_pc.values[ok]; sig = P.sig.values[ok]; Sco = P.S.values[ok]; nat = (P.natural | P.candidate).values[ok]
    act = P.index.isin(active)[ok]
    cvet = P.index.isin(cj.get("coadd_vetoed_ids", []))[ok]                   # steady-test (coadd-unit) confusion veto of 18
    # p_i uses the rule that defines k: a star counts when its coadd reaches 5 sigma, so t_i = 5 sigma_i at every star
    # without activity evidence, including the candidates themselves (each counted in k as a possible transmitter).
    # t_add (S_i + 5 sigma_i at the candidates) instead asks whether an additional transmitter would stand out above the
    # measured source; it is kept for comparison (f95_b0_add). p_i = 0 at stars with activity evidence.
    t_co = np.where(act | cvet, np.inf, THR * sig)
    t_add = np.where(act | cvet, np.inf, np.where(nat, Sco + THR * sig, THR * sig))
    Ue = U[~U.source_id.isin(natural | active)]   # single-epoch route: every star without activity evidence (as k_host counts)
    de, se = Ue.dist_pc.values, Ue.sigma_mJy.values; sid_e = Ue.source_id.values
    _, inv = np.unique(sid_e, return_inverse=True)
    mu95, kk, bb = cj["mu95_steady"], cj["k_steady"], cj["b_steady"]          # steady test: coadd, one trial per star
    mu95e = cj["mu95_snap"]                                                       # per-snapshot test: one trial per star-epoch
    mu95h = cj["mu95_host"]                     # host-level snapshot test: one trial per star, detected in any single epoch
    DELTAS = [0.1, 0.3, 1.0]
    rows = []
    for Lg in LGRID:
        SL = Lg / lum(1.0, d)                                 # mJy at each star
        p_co = norm.cdf((SL - t_co) / sig); neff_add = norm.cdf((SL - t_add) / sig).sum()
        SLe = Lg / lum(1.0, de); p_e = norm.cdf((SLe - THR * se) / se)
        p = p_co.copy()
        neff, nsnap = p.sum(), p_e.sum()
        r_ = dict(L_WHz=Lg, N_eff=neff, f95=min(mu95 / neff, np.inf) if neff > 0 else np.inf,
                  f95_b0=cj["mu95_steady_b0"] / neff if neff > 0 else np.inf,
                  f95_b0_add=cj["mu95_steady_b0"] / neff_add if neff_add > 0 else np.inf,
                  f95_k0b0=3.0 / neff if neff > 0 else np.inf, N_snap=nsnap,
                  # duty cycle, counted per host star: a host on in a random fraction delta of snapshots is detected at
                  # least once with probability 1 - prod_e (1 - delta p_ie); for small delta this tends to delta sum_e p_ie
                  fdelta95=mu95h / nsnap if nsnap > 0 else np.inf,
                  fdelta95_b0=cj["mu95_host_b0"] / nsnap if nsnap > 0 else np.inf,
                  fdelta95_epochcount=mu95e / nsnap if nsnap > 0 else np.inf)
        for dl in DELTAS:
            lq = np.zeros(inv.max() + 1); np.add.at(lq, inv, np.log1p(-dl * np.clip(p_e, 0, 1 - 1e-12)))
            nh = np.sum(1 - np.exp(lq)); r_[f"f95_duty{dl:g}"] = mu95h / nh if nh > 0 else np.inf
            r_[f"f95_duty{dl:g}_b0"] = cj["mu95_host_b0"] / nh if nh > 0 else np.inf
        rows.append(r_)
    POP = pd.DataFrame(rows); POP.to_csv(os.path.join(C.RES, "alleepoch_population_limits.csv"), index=False)
    # ---------------- stacks
    # pop: 'all' = every clean star (the astrophysical measurement, Table 5); 'active' / 'inactive' = split by activity
    # evidence (04b). Limits on transmitters use the 'inactive' stacks, the same population as the other limits.
    bad_star = set(LT.source_id[LT.contam_snr >= 0.5])
    st_rows, wts = [], {}
    for scope in tags + ["all"]:
        X = LT if scope == "all" else LT[LT.tag == scope]
        if scope == "all":    # per-star coadd of star and of each control position (unvetoed epochs)
            Y = X[~X.veto].copy(); Y["w"] = 1 / Y.sigma_mJy ** 2
            sw = np.sqrt(Y.w.groupby(Y.source_id).sum())
            agg = {"S": (Y.S_mJy * Y.w).groupby(Y.source_id).sum() / Y.w.groupby(Y.source_id).sum(), "sig": Y.w.groupby(Y.source_id).sum() ** -0.5}
            for j in range(N_CTRL):
                agg[f"c{j}"] = (Y[f"c{j}"] / Y.sigma_mJy).groupby(Y.source_id).sum() / sw
                # the clean cuts applied to the stars, applied to each control position: no >= 5 sigma value in the
                # coadd or any epoch, and predicted catalogue contamination below 0.5 sigma in every epoch
                agg[f"cbad{j}"] = ((Y[f"c{j}"] >= THR) | (Y[f"cz{j}"] >= 0.5)).groupby(Y.source_id).any()
            S = pd.DataFrame(agg); S["z"] = S.S / S.sig
            for j in range(N_CTRL): S[f"cbad{j}"] = S[f"cbad{j}"] | (S[f"c{j}"] >= THR)
            S = S.join(sv.set_index("source_id")); S["detected"] = S.index.isin(LT.source_id[LT.detected])
        else:
            S = X[~X.veto].set_index("source_id")[["S_mJy", "sigma_mJy", "snr", "detected", "dist_pc", "bp_rp"] + [f"c{j}" for j in range(N_CTRL)] + [f"cz{j}" for j in range(N_CTRL)]]
            S = S.rename(columns={"S_mJy": "S", "sigma_mJy": "sig", "snr": "z"})
            for j in range(N_CTRL): S[f"cbad{j}"] = (S[f"c{j}"] >= THR) | (S[f"cz{j}"] >= 0.5)
        S["cls"] = np.where(S.bp_rp < C.BPRP_SPLIT, "AFGK", "M")
        S["clean"] = ~S.detected & ~S.index.isin(cand) & ~S.index.isin(bad_star)
        S["active"] = S.index.isin(active)
        for (lo, hi) in C.SHELLS:
            for cl in ["AFGK", "M"]:
                sh0 = S[(S.dist_pc >= lo) & (S.dist_pc < hi) & (S.cls == cl)]
                for pop in ["all", "inactive", "active"]:
                    if scope != "all" and pop != "all": continue
                    sh = sh0 if pop == "all" else sh0[sh0.active == (pop == "active")]
                    cs = sh[sh.clean]
                    if len(cs) < 10: continue
                    mL, eL, w = wmean_lum(cs.S.values, cs.sig.values, cs.dist_pc.values)
                    nulls = []
                    for j in range(N_CTRL):
                        cj_ = cs[~cs[f"cbad{j}"].astype(bool) & np.isfinite(cs[f"c{j}"])]
                        nulls.append(wmean_lum(cj_[f"c{j}"].values * cj_.sig.values, cj_.sig.values, cj_.dist_pc.values)[0])
                    nulls = np.array(nulls)
                    err = max(eL, nulls.std(ddof=1)); mLc = mL - nulls.mean()
                    neffw = w.sum() ** 2 / np.sum(w ** 2); topw = np.sort(w)[::-1]
                    if scope == "all":
                        wts[(pop, cl, lo)] = dict(w=w, d=cs.dist_pc.values, sig=cs.sig.values,
                                                  ctrl=np.column_stack([np.where(cs[f"cbad{j}"].astype(bool), np.nan, cs[f"c{j}"].values) for j in range(N_CTRL)]))
                    # median stacks including detected stars
                    z = sh.z.values; zc = np.concatenate([sh[f"c{j}"].values for j in range(N_CTRL)]); zc = zc[np.isfinite(zc)]
                    bs = [np.median(rng.choice(z, len(z))) - np.median(rng.choice(zc, len(zc))) for _ in range(200)]
                    dmed = np.median(z) - np.median(zc)
                    mS = np.median(sh.S.values)
                    st_rows.append(dict(scope=scope, pop=pop, shell=f"{lo}-{hi}", cls=cl, N_all=len(sh), N_clean=len(cs),
                                        meanL=mLc, meanL_err=err, meanL_snr=mLc / err, null_mean=nulls.mean(), null_std=nulls.std(ddof=1),
                                        formal_err=eL, Neff_w=neffw, top1_wfrac=topw[0] / w.sum(), top10_wfrac=topw[:10].sum() / w.sum(),
                                        med_z=np.median(z), med_z_ctrl=np.median(zc), dmed_z=dmed, dmed_z_err=np.std(bs),
                                        med_S_mJy=mS, med_sig_mJy=np.median(sh.sig.values), dmin=lo, dmax=hi))
    ST = pd.DataFrame(st_rows); ST.to_csv(os.path.join(C.RES, "alleepoch_stacks.csv"), index=False)
    # ---------------- stack limits (all-epoch coadd, stars without activity evidence)
    # For each L_T the stack uses the clean stars without activity evidence at which such a transmitter would be at most
    # HOST_SNR sigma in the coadd (L_T <= HOST_SNR sigma_L,i), a selection fixed by the noise alone: a host then
    # rarely reaches the 5 sigma that would remove it from the clean sample, so the stack is not biased by the cuts.
    # The weighted mean is m = sum_i a_i L_i (a_i proportional to sigma_L,i^-2); if each star hosts a transmitter of
    # L_T with probability f, m has expectation f L_T and, besides the measurement error s_m (the larger of the formal
    # error, scaled by the empirical rms of S/sigma at the control positions, and the scatter of the 8 control means), host-sampling variance L_T^2 f (1 - f) / N_eff with
    # N_eff = 1 / sum a_i^2. f_95 solves f L_T - 1.645 sqrt(s_m^2 + L_T^2 f (1 - f) / N_eff) = max(m, 0).
    from scipy.optimize import brentq
    HOST_SNR = 3.0
    from scipy.stats import t as student_t
    def f95_samp(m, s_m, Lt, neff, q=Z):
        g_ = lambda f: f * Lt - q * np.sqrt(s_m ** 2 + Lt ** 2 * f * (1 - f) / neff) - max(m, 0)
        return brentq(g_, 0, 1) if g_(1.0) > 0 else np.nan
    IN = S[S.clean & ~S.active & np.isfinite(S.S)].copy()          # S is the all-epoch coadd table from the loop above
    IN["sigL"] = lum(IN.sig.values, IN.dist_pc.values); IN["L"] = lum(IN.S.values, IN.dist_pc.values)
    CZ = np.column_stack([np.where(IN[f"cbad{j}"].astype(bool), np.nan, IN[f"c{j}"].values) for j in range(N_CTRL)])
    def stack_lim(sel, L, sigL, ctrl, Lt):
        w = 1 / sigL[sel] ** 2; a = w / w.sum(); m = np.sum(a * L[sel]); eL = w.sum() ** -0.5
        # sigma is a robust (MAD) rms, blind to the non-Gaussian tail of faint sources and sidelobes: scale the formal
        # error by the actual rms of S/sigma at the (clean) control positions of the selected stars
        zall = ctrl[sel]; zall = zall[np.isfinite(zall)]; rms_z = float(np.sqrt(np.mean((zall - zall.mean()) ** 2))); eL *= max(rms_z, 1.0)
        nulls = []
        for j in range(ctrl.shape[1]):
            okj = np.isfinite(ctrl[sel, j]); wj = w[okj]; nulls.append(np.sum(wj * ctrl[sel, j][okj] * sigL[sel][okj]) / wj.sum())
        nulls = np.array(nulls); s_m = max(eL, nulls.std(ddof=1)); m = m - nulls.mean(); neff = 1 / np.sum(a ** 2)
        # when the scatter of the control means sets the error, it is estimated from only len(nulls) values: use the
        # Student-t quantile (1.89 for 8 control angles) instead of 1.645
        q = student_t.ppf(0.95, len(nulls) - 1) if nulls.std(ddof=1) > eL else Z
        return m, s_m, neff, f95_samp(m, s_m, Lt, neff, q), (max(m, 0) + q * s_m) / Lt
    lim_rows = []
    for cl in ["AFGK", "M"]:
        q = IN[IN.cls == cl]; L_, sL_, C_, d_ = q.L.values, q.sigL.values, CZ[(IN.cls == cl).values], q.dist_pc.values
        for Lt in LGRID:
            sel = Lt <= HOST_SNR * sL_
            if sel.sum() < 10: lim_rows.append(dict(cls=cl, L_WHz=Lt, n_stack=int(sel.sum()))); continue
            m, s_m, ne, f_, f0 = stack_lim(sel, L_, sL_, C_, Lt)
            w = 1 / sL_[sel] ** 2
            lim_rows.append(dict(cls=cl, L_WHz=Lt, n_stack=int(sel.sum()), f95_mean=f_, f95_mean_nosamp=f0, mean_comb=m, mean_comb_err=s_m,
                                 Neff_comb=ne, wfrac_within10pc=float(w[d_[sel] < 10].sum() / w.sum())))
    SL = pd.DataFrame(lim_rows)
    # coverage: inject transmitters (fraction f_true, luminosity L_T, hosts chosen at random) into the control-position
    # measurements of the same stars, apply the clean cuts of the real stars (pseudo-star removed if its coadd, or any
    # single unvetoed epoch, reaches 5 sigma after injection) and the same selection, and count how often f_95 >= f_true
    # (the null then comes from the other 7 angles)
    Yi = Y[Y.source_id.isin(IN.index)]; ie = IN.index.get_indexer(Yi.source_id); se_ = Yi.sigma_mJy.values
    lum1_e = lum(1.0, IN.dist_pc.values[ie])                       # W/Hz per mJy at each star-epoch
    def epoch_max(Lt, j):                                          # max single-epoch control S/N after adding the transmitter
        z = np.nan_to_num(Yi[f"c{j}"].values + Lt / lum1_e / se_, nan=-np.inf); out = np.full(len(IN), -np.inf)
        np.maximum.at(out, ie, z); return out
    cov_rows = []
    for cl in ["AFGK", "M"]:
        q = IN[IN.cls == cl]; sL_ = q.sigL.values; C_ = CZ[(IN.cls == cl).values]
        for Lt in [1.5e5, 2.2e5, 3e5, 4e5, 1e6, 1e7]:
            sel0 = Lt <= HOST_SNR * sL_
            if sel0.sum() < 10: continue
            EM = np.column_stack([epoch_max(Lt, j)[(IN.cls == cl).values] for j in range(N_CTRL)])
            for f_true in [0.1, 0.3, 0.6, 1.0]:
                ok_n = n_t = 0; ok_j = []; n_epcut = 0
                for j in range(N_CTRL):
                    ok0 = ok_n
                    zj = C_[:, j]
                    for rep in range(25):
                        host = rng.random(len(zj)) < f_true; zz = zj + host * Lt / sL_
                        keep = sel0 & np.isfinite(zz) & (zz < THR) & ~(host & (EM[:, j] >= THR)); n_epcut += int((sel0 & host & (EM[:, j] >= THR) & np.isfinite(zz) & (zz < THR)).sum())
                        others = np.delete(C_, j, axis=1)              # the other 7 control angles provide the null
                        Lp = zz * sL_
                        m_, s_m_, ne_, f_, _ = stack_lim(keep, Lp, sL_, others, Lt)
                        n_t += 1; ok_n += int((not np.isfinite(f_)) or (f_ >= f_true))
                    ok_j.append((ok_n - ok0) / 25)
                cov_rows.append(dict(cls=cl, L_WHz=Lt, f_true=f_true, coverage=ok_n / n_t, n_trials=n_t, n_noise=N_CTRL, n_host_draws=25,
                                     n_noise_below95=int(np.sum(np.array(ok_j) < 0.95)), n_sel=int(sel0.sum()), mean_epoch_cut=n_epcut / n_t,
                                     fail_angles=";".join(str(j) for j in range(N_CTRL) if ok_j[j] < 0.95),
                                     angle_mean_z=";".join(f"{np.nanmean(C_[sel0, j]):.4f}" for j in range(N_CTRL))))
    COV = pd.DataFrame(cov_rows); COV.to_csv(os.path.join(C.RES, "alleepoch_stack_coverage.csv"), index=False)
    # median-based limits at a few L_T (Monte Carlo), all-epoch coadd, for the pre-specified pooled 0-100 pc sample of
    # stars without activity evidence in each class (no choice of shell after seeing the data)
    Y = LT[~LT.veto].copy(); Y["w"] = 1 / Y.sigma_mJy ** 2
    sw = np.sqrt(Y.w.groupby(Y.source_id).sum())
    Sco_ = pd.DataFrame({"S": (Y.S_mJy * Y.w).groupby(Y.source_id).sum() / Y.w.groupby(Y.source_id).sum(),
                         "sig": Y.w.groupby(Y.source_id).sum() ** -0.5}).join(sv.set_index("source_id"))
    for j in range(N_CTRL): Sco_[f"c{j}"] = (Y[f"c{j}"] / Y.sigma_mJy).groupby(Y.source_id).sum() / sw
    Sco_["z"] = Sco_.S / Sco_.sig; Sco_["cls"] = np.where(Sco_.bp_rp < C.BPRP_SPLIT, "AFGK", "M")
    Sco_ = Sco_[~Sco_.index.isin(active)]
    med_rows = []
    for cl in ["AFGK", "M"]:
        sh = Sco_[Sco_.cls == cl]; z = sh.z.values
        zc = np.concatenate([sh[f"c{j}"].values for j in range(N_CTRL)]); zc = zc[np.isfinite(zc)]
        dmed = np.median(z) - np.median(zc); bs = [np.median(rng.choice(z, len(z))) - np.median(rng.choice(zc, len(zc))) for _ in range(200)]
        allowed = max(dmed, 0) + Z * np.std(bs)
        for Lt in [1e5, 1e6, 1e7, 1e8, 1e9]:
            fl = median_f_limit(z, (Lt / lum(1.0, sh.dist_pc.values)) / sh.sig.values, allowed)
            med_rows.append(dict(cls=cl, L_WHz=Lt, f95_median=fl, dmed_z_pooled=dmed, dmed_z_pooled_err=float(np.std(bs)), N_pooled=len(sh)))
    SL = SL.merge(pd.DataFrame(med_rows), on=["cls", "L_WHz"], how="left")
    SL.to_csv(os.path.join(C.RES, "alleepoch_stack_limits.csv"), index=False)
    # ---------------- summary
    g = lambda L0: float(np.interp(np.log10(L0), np.log10(POP.L_WHz), POP.f95))
    summ = dict(tags=tags, n_stars=int(len(P)), n_with_coadd=int(ok.sum()), median_epochs=float(P.n.median()),
                S_lim_steady_med=float(P.S_lim_steady.median()), L_lim_steady_med=float(P.L_lim_steady.median()),
                S_lim_any_med=float(P.S_lim_anyepoch.median()), S_lim_best_med=float(P.S_lim_bestepoch.median()),
                f95_1e7=g(1e7), f95_1e8=g(1e8), f95_1e9=g(1e9), f95_floor=float(POP.f95.min()), Neff_max=float(POP.N_eff.max()),
                mu95=mu95, k=kk, b=bb, mu95_snap=mu95e, k_snap=cj["k_snap"], b_snap=cj["b_snap"], fdelta95_floor=float(POP.fdelta95.min()),
                f95_b0_1e7=float(np.interp(7, np.log10(POP.L_WHz), POP.f95_b0)), f95_b0_1e8=float(np.interp(8, np.log10(POP.L_WHz), POP.f95_b0)),
                f95_b0_1e9=float(np.interp(9, np.log10(POP.L_WHz), POP.f95_b0)), f95_b0_floor=float(POP.f95_b0.min()),
                fdelta95_b0_floor=float(POP.fdelta95_b0.min()), fdelta95_epochcount_floor=float(POP.fdelta95_epochcount.min()),
                **{f"f95_duty{dl:g}_floor": float(POP[f"f95_duty{dl:g}"].min()) for dl in DELTAS},
                **{f"f95_duty{dl:g}_b0_floor": float(POP[f"f95_duty{dl:g}_b0"].min()) for dl in DELTAS},
                f95_b0_add_1e7=float(np.interp(7, np.log10(POP.L_WHz), POP.f95_b0_add)), f95_b0_add_floor=float(POP.f95_b0_add.min()),
                f95_b0_reach1=float(10 ** np.interp(0.0, -np.log10(POP.f95_b0.values[np.isfinite(POP.f95_b0.values)]),
                                                     np.log10(POP.L_WHz.values[np.isfinite(POP.f95_b0.values)]))),
                mu95_host=mu95h, k_host=cj["k_host"], b_host=cj["b_host"],
                n_unexplained_candidates=int(P.unexplained_candidate.sum()),
                n_active_with_photometry=int(act.sum()), n_inactive_with_photometry=int((~act).sum()))
    json.dump(summ, open(os.path.join(C.DER, "alleepoch_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(ST[ST.scope == "all"][["pop", "shell", "cls", "N_all", "N_clean", "meanL", "meanL_err", "meanL_snr", "Neff_w", "top10_wfrac", "dmed_z", "dmed_z_err"]].to_string(index=False, float_format="%.3g"))
    print(SL[SL.f95_median.notna()].to_string(index=False, float_format="%.3g"))
