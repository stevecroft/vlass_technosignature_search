"""19 - All-epoch per-star limits, population limits from per-star detection probabilities, and stacks.

Per-star limits (95%), from the forced photometry of 17 and the vetoes of 18:
  steady    : inverse-variance coadd over unvetoed epochs. S_lim = (5 + 1.645) sigma_coadd for stars without emission;
              S_coadd + 1.645 sigma_coadd (if larger) for stars with natural or unexplained emission; for stars whose
              every epoch is confusion-vetoed, the largest single-epoch S + 1.645 sigma.
  any-epoch : (5 + 1.645) sigma_e in the best (lowest-noise) unvetoed epoch, for a transmitter on in one snapshot.
Population (steady transmitter of spectral luminosity L, flux S_L = L / 4 pi d^2):
  p_i(L) = Phi((S_L - t_i) / sigma_coadd)   (steady test: coadd only, one trial per star),
  with t_i = 5 sigma_coadd, or S_coadd + 5 sigma_coadd for stars with natural emission (a transmitter must exceed it).
  The expected number of detected transmitters is f * sum_i p_i = f * N_eff(L); with k unexplained candidates over an
  expected chance background b (18, steady test), f_95(L) = mu_95(k, b) / N_eff(L).
  Duty cycle: a transmitter on in a random fraction delta of snapshots gives (f delta)_95 = mu_95,snap / sum_i sum_e p_ie(L),
  with mu_95,snap from the per-snapshot test (single-epoch S/N >= 5, one trial per star-epoch).
Stacks, in distance shells x (AFGK: BP-RP < 1.84; M), per epoch and for the all-epoch coadd:
  mean     : inverse-variance-weighted mean L_nu of clean stars (not secure, not candidates, no epoch with predicted
             catalogue contamination >= 0.5 sigma). Null from the same weighted mean at each of the 8 control positions;
             the null mean is subtracted and the error is the larger of the formal error and the null scatter.
             N_eff,w = (sum w)^2 / sum w^2 and the largest weight fraction are reported.
  median   : median S/sigma and median S of all stars in the shell except confusion-vetoed ones, *including* detected
             stars, compared with the median at control positions; error from 200 bootstrap resamples of stars.
Stack limits:
  mean     : f L_T <= <L>_95, using for each L_T only shells in which L_T is below the 5th percentile of the per-star
             steady limits (fainter transmitters would not have been individually detected); shells combined by
             inverse variance.
  median   : Monte Carlo: give a random fraction f of the shell's stars an extra S_L / sigma_i and find the f at which the
             median shift exceeds the observed shift + 1.645 sigma_med. Bounds f regardless of L_T once S_L >> sigma.

Inputs : data/derived/alleepoch_forced.parquet (+ alleepoch_forced_controls.parquet), alleepoch_candidates.json, forced_calibration.json,
         data/derived/ms_sample_vlass.parquet, results/vlass_perstar_limits_100pc_MS.parquet, results/alleepoch_candidates.csv
Outputs: results/alleepoch_perstar_limits.parquet, results/alleepoch_population_limits.csv,
         results/alleepoch_stacks.csv, results/alleepoch_stack_limits.csv, data/derived/alleepoch_summary.json
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
    P["S_lim_anyepoch"] = (THR + Z) * P.sig_best
    P["L_lim_steady"] = lum(P.S_lim_steady, P.dist_pc); P["L_lim_anyepoch"] = lum(P.S_lim_anyepoch, P.dist_pc)
    P["class"] = np.where(P.bp_rp < C.BPRP_SPLIT, "AFGK", "M")
    P.reset_index().to_parquet(os.path.join(C.RES, "alleepoch_perstar_limits.parquet"), index=False)
    # ---------------- population: N_eff(L) = sum_i p_i(L)
    ok = P.n.notna().values
    d = P.dist_pc.values[ok]; sig = P.sig.values[ok]; Sco = P.S.values[ok]; nat = (P.natural | P.candidate).values[ok]
    t_co = np.where(nat, Sco + THR * sig, THR * sig)
    Ue = U[~U.source_id.isin(natural | cand)]                # single-epoch route only for emission-free stars
    de, se = Ue.dist_pc.values, Ue.sigma_mJy.values; sid_e = Ue.source_id.values
    _, inv = np.unique(sid_e, return_inverse=True)
    mu95, kk, bb = cj["mu95_steady"], cj["k_steady"], cj["b_steady"]          # steady test: coadd, one trial per star
    mu95e = cj["mu95_snap"]                                                       # per-snapshot test: one trial per star-epoch
    rows = []
    for Lg in LGRID:
        SL = Lg / lum(1.0, d)                                 # mJy at each star
        p_co = norm.cdf((SL - t_co) / sig)
        SLe = Lg / lum(1.0, de); p_e = norm.cdf((SLe - THR * se) / se)
        p_ep_max = np.zeros(inv.max() + 1); np.maximum.at(p_ep_max, inv, p_e)
        p = p_co.copy()
        neff, nsnap = p.sum(), p_e.sum()
        rows.append(dict(L_WHz=Lg, N_eff=neff, f95=min(mu95 / neff, np.inf) if neff > 0 else np.inf,
                         f95_k0b0=3.0 / neff if neff > 0 else np.inf, N_snap=nsnap,
                         fdelta95=mu95e / nsnap if nsnap > 0 else np.inf))
    POP = pd.DataFrame(rows); POP.to_csv(os.path.join(C.RES, "alleepoch_population_limits.csv"), index=False)
    # ---------------- stacks
    clean_ep = ~LT.detected & ~LT.source_id.isin(cand) & ~LT.veto
    bad_star = set(LT.source_id[LT.contam_snr >= 0.5])
    st_rows = []
    for scope in tags + ["all"]:
        X = LT if scope == "all" else LT[LT.tag == scope]
        if scope == "all":    # per-star coadd of star and of each control position (unvetoed epochs)
            Y = X[~X.veto].copy(); Y["w"] = 1 / Y.sigma_mJy ** 2
            agg = {"S": (Y.S_mJy * Y.w).groupby(Y.source_id).sum() / Y.w.groupby(Y.source_id).sum(), "sig": Y.w.groupby(Y.source_id).sum() ** -0.5}
            for j in range(N_CTRL):
                agg[f"c{j}"] = (Y[f"c{j}"] / Y.sigma_mJy).groupby(Y.source_id).sum() / np.sqrt(Y.w.groupby(Y.source_id).sum())
            S = pd.DataFrame(agg); S["z"] = S.S / S.sig
            S = S.join(sv.set_index("source_id")); S["detected"] = S.index.isin(LT.source_id[LT.detected])
        else:
            S = X[~X.veto].set_index("source_id")[["S_mJy", "sigma_mJy", "snr", "detected", "dist_pc", "bp_rp"] + [f"c{j}" for j in range(N_CTRL)]]
            S = S.rename(columns={"S_mJy": "S", "sigma_mJy": "sig", "snr": "z"})
        S["cls"] = np.where(S.bp_rp < C.BPRP_SPLIT, "AFGK", "M")
        S["clean"] = ~S.detected & ~S.index.isin(cand) & ~S.index.isin(bad_star)
        for (lo, hi) in C.SHELLS:
            for cl in ["AFGK", "M"]:
                sh = S[(S.dist_pc >= lo) & (S.dist_pc < hi) & (S.cls == cl)]
                cs = sh[sh.clean]
                if len(cs) < 10: continue
                mL, eL, w = wmean_lum(cs.S.values, cs.sig.values, cs.dist_pc.values)
                nulls = np.array([wmean_lum(cs[f"c{j}"].values * cs.sig.values, cs.sig.values, cs.dist_pc.values)[0] for j in range(N_CTRL)])
                err = max(eL, nulls.std(ddof=1)); mLc = mL - nulls.mean()
                neffw = w.sum() ** 2 / np.sum(w ** 2); topw = np.sort(w)[::-1]
                # median stacks including detected stars
                z = sh.z.values; zc = np.concatenate([sh[f"c{j}"].values for j in range(N_CTRL)]); zc = zc[np.isfinite(zc)]
                bs = [np.median(rng.choice(z, len(z))) - np.median(rng.choice(zc, len(zc))) for _ in range(200)]
                dmed = np.median(z) - np.median(zc)
                mS = np.median(sh.S.values); mSc = np.median(zc * np.median(sh.sig.values))
                st_rows.append(dict(scope=scope, shell=f"{lo}-{hi}", cls=cl, N_all=len(sh), N_clean=len(cs),
                                    meanL=mLc, meanL_err=err, meanL_snr=mLc / err, null_mean=nulls.mean(), null_std=nulls.std(ddof=1),
                                    formal_err=eL, Neff_w=neffw, top1_wfrac=topw[0] / w.sum(), top10_wfrac=topw[:10].sum() / w.sum(),
                                    med_z=np.median(z), med_z_ctrl=np.median(zc), dmed_z=dmed, dmed_z_err=np.std(bs),
                                    med_S_mJy=mS, med_sig_mJy=np.median(sh.sig.values), dmin=lo, dmax=hi))
    ST = pd.DataFrame(st_rows); ST.to_csv(os.path.join(C.RES, "alleepoch_stacks.csv"), index=False)
    # ---------------- stack limits (all-epoch coadd)
    SA = ST[ST.scope == "all"]; lim_rows = []
    Pp = P[P.n.notna()]
    for cl in ["AFGK", "M"]:
        for Lt in LGRID:
            num = den = 0.0; fmed = []
            for _, r in SA[SA.cls == cl].iterrows():
                ps = Pp[(Pp.dist_pc >= r.dmin) & (Pp.dist_pc < r.dmax) & (Pp["class"] == cl)]
                if Lt >= np.percentile(ps.L_lim_steady, 5): continue
                w = 1 / r.meanL_err ** 2; num += w * r.meanL; den += w
            if den > 0:
                mean95 = max(num / den, 0) + Z * den ** -0.5; f_mean = mean95 / Lt
            else:
                f_mean = np.nan
            lim_rows.append(dict(cls=cl, L_WHz=Lt, f95_mean=f_mean))
    SL = pd.DataFrame(lim_rows)
    # median-based limits at a few L_T (Monte Carlo; expensive), all-epoch coadd, combining shells by taking the best
    Y = LT[~LT.veto].copy(); Y["w"] = 1 / Y.sigma_mJy ** 2
    Sco_ = pd.DataFrame({"S": (Y.S_mJy * Y.w).groupby(Y.source_id).sum() / Y.w.groupby(Y.source_id).sum(),
                         "sig": Y.w.groupby(Y.source_id).sum() ** -0.5}).join(sv.set_index("source_id"))
    Sco_["z"] = Sco_.S / Sco_.sig; Sco_["cls"] = np.where(Sco_.bp_rp < C.BPRP_SPLIT, "AFGK", "M")
    med_rows = []
    for cl in ["AFGK", "M"]:
        for Lt in [1e5, 1e6, 1e7, 1e8, 1e9]:
            best = np.nan
            for _, r in SA[SA.cls == cl].iterrows():
                sh = Sco_[(Sco_.dist_pc >= r.dmin) & (Sco_.dist_pc < r.dmax) & (Sco_.cls == cl)]
                allowed = max(r.dmed_z, 0) + Z * r.dmed_z_err
                fl = median_f_limit(sh.z.values, (Lt / lum(1.0, sh.dist_pc.values)) / sh.sig.values, allowed)
                if np.isfinite(fl) and not (best <= fl): best = fl
            med_rows.append(dict(cls=cl, L_WHz=Lt, f95_median=best))
    SL = SL.merge(pd.DataFrame(med_rows), on=["cls", "L_WHz"], how="left")
    SL.to_csv(os.path.join(C.RES, "alleepoch_stack_limits.csv"), index=False)
    # ---------------- summary
    g = lambda L0: float(np.interp(np.log10(L0), np.log10(POP.L_WHz), POP.f95))
    summ = dict(tags=tags, n_stars=int(len(P)), n_with_coadd=int(ok.sum()), median_epochs=float(P.n.median()),
                S_lim_steady_med=float(P.S_lim_steady.median()), L_lim_steady_med=float(P.L_lim_steady.median()),
                S_lim_any_med=float(P.S_lim_anyepoch.median()),
                f95_1e7=g(1e7), f95_1e8=g(1e8), f95_1e9=g(1e9), f95_floor=float(POP.f95.min()), Neff_max=float(POP.N_eff.max()),
                mu95=mu95, k=kk, b=bb, mu95_snap=mu95e, k_snap=cj["k_snap"], b_snap=cj["b_snap"], fdelta95_floor=float(POP.fdelta95.min()),
                n_unexplained_candidates=int(P.unexplained_candidate.sum()))
    json.dump(summ, open(os.path.join(C.DER, "alleepoch_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(ST[ST.scope == "all"][["shell", "cls", "N_all", "N_clean", "meanL", "meanL_err", "meanL_snr", "Neff_w", "top10_wfrac", "dmed_z", "dmed_z_err"]].to_string(index=False, float_format="%.3g"))
    print(SL[SL.f95_median.notna()].to_string(index=False, float_format="%.3g"))
