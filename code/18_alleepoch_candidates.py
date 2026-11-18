"""18 - All-epoch detection candidates from forced photometry, their classification, and the chance background.

A star-epoch is a candidate if its forced S/N (17) is >= 5 and the star is not an explained secure Epoch-1 association
(a natural emitter, or a background source not at the star); unexplained secure associations are tested like any star.
  * Confusion veto: the summed contribution at the measurement position of all Epoch-1 catalogue components within
    15" (other than the star's own secure component), each modelled as an elliptical Gaussian with its fitted
    (beam-convolved) Maj, Min, PA and peak, is at least half the detection threshold (kappa * pred >= 2.5 sigma_eff).
    A plain 3" radius is not enough: bright components 3-4" away put >5 sigma into the beam wing, especially at low
    declination where the QL beam is elongated. Vetoed stars are confusion-limited and reported separately.
    For the steady test the veto is also applied in coadd units: the predicted contamination, coadded over the unvetoed
    epochs like the flux, must stay below half the coadd threshold.
  * Classification of the remaining candidate stars with the same evidence as for the secure associations (07):
    SIMBAD object types (CfA TAP mirror) and ROSAT 2RXS / eRASS1 X-ray counterparts (CfA VizieR mirror). A candidate
    is "explained" if it has activity evidence (X-ray counterpart or SIMBAD X-ray type; flare/BY Dra/RS CVn/young/CV/
    magnetic CP/emission-line type). The *_strict values repeat the tests without the emission-line and SIMBAD X-ray types. Stars with
    only multiplicity evidence are listed separately and counted as unexplained (conservative).
  * Chance background: the same estimator and the same veto at 8 control positions 14" from each star. A component that
    would be a secure (R > 0.5) association of a control position, evaluated at the Epoch-1 epoch with the reliability
    model of 03 (assoc_radius), is exempt from that control's veto, exactly as a star's own secure component is, so that
    b counts chance coincidences such as LSPM J1943+5337 that enter k.
    b = (control rate) x (number of unvetoed, non-secure star-epochs), both counted only around stars without activity
    evidence (04b), because a detection at a star with activity evidence is classified as natural whatever its origin. Control positions carry no stellar emission,
    so b counts sidelobes and unrelated sources only, matching the unexplained candidates.
  * Upper limit on the number of transmitter detections hidden among the k unexplained candidates, given background b:
    the conditional (Zech 1989) 95% limit, P(n<=k | mu+b) / P(n<=k | b) = 0.05; and, without background subtraction,
    the classical Poisson limit for k (mu95_*_b0), adopted for the headline limits.
  * Diagnostics: k and b against threshold (results/alleepoch_k_b_threshold.csv), control hits per angle (8 independent
    realizations of the null), a bootstrap over stars for the error of b, a host-level snapshot test (one trial per
    star, detected in any single epoch; for the duty-cycle limit), and a catalog-level density test of the background
    (Epoch-1 components around stars and around control points 14"-180" away, by G magnitude).

Inputs : data/derived/forced_<tag>.parquet, forced_controls_<tag>.npz, data/derived/ms_sample_vlass.parquet,
         data/raw/vlass_e1ql_components.parquet, results/vlass_perstar_limits_100pc_MS.parquet,
         results/vlass_star_matches_classified.csv
Outputs: data/raw/simbad_forced_candidates.csv, data/raw/xray_forced_candidates.csv (fetched once, then cached),
         results/alleepoch_candidates.csv, results/alleepoch_k_b_threshold.csv, data/derived/alleepoch_candidates.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json, time, importlib, argparse
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import poisson
from scipy.optimize import brentq

ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="ep1,ep2,ep3,ep4"); ap.add_argument("--offline", action="store_true")
a = ap.parse_args()
THR, R_SEARCH, VETO_FRAC, R_CTRL, N_CTRL = 5.0, 15.0, 0.5, 14.0, 8
P04 = importlib.import_module("04_fetch_counterparts"); P07 = importlib.import_module("07_classify_and_limits")


def xyz(ra, de):
    ra, de = np.deg2rad(ra), np.deg2rad(de); return np.c_[np.cos(de) * np.cos(ra), np.cos(de) * np.sin(ra), np.sin(de)]


def predicted(ra, de, comp, tree, exclude_sid=None, own=None):
    """Summed catalogue-component contribution (mJy/beam, catalogue scale) at positions (ra, de) [deg]."""
    rr = 2 * np.sin(np.deg2rad(R_SEARCH / 3600) / 2); hits = tree.query_ball_point(xyz(ra, de), rr)
    ii = np.repeat(np.arange(len(ra)), [len(h) for h in hits]); jj = np.concatenate([np.asarray(h, int) for h in hits]) if len(ii) else np.array([], int)
    out = np.zeros(len(ra))
    if not len(ii): return out
    dE = (comp.RAJ2000.values[jj] - ra[ii]) * 3600 * np.cos(np.deg2rad(de[ii])); dN = (comp.DEJ2000.values[jj] - de[ii]) * 3600
    pa = np.deg2rad(comp.PA.values[jj]); u = dE * np.sin(pa) + dN * np.cos(pa); w = dE * np.cos(pa) - dN * np.sin(pa)
    val = comp.Fpeak.values[jj] * np.exp(-4 * np.log(2) * (u ** 2 / comp.Maj.values[jj] ** 2 + w ** 2 / comp.Min.values[jj] ** 2))
    if exclude_sid is not None:     # drop the component associated with each position (R > 0.5; see assoc_radius)
        val = np.where(own.reindex(list(zip(exclude_sid[ii], jj))).notna().values, 0.0, val)
    np.add.at(out, ii, val); return out


def assoc_radius(comp, dec_ref):
    """Radius (arcsec) within which a component at a position has reliability R > 0.5 under the reliability model of
    03 (Rayleigh true offsets + uniform background): R > 0.5 <=> r^2 < 2 s^2 ln[N_true Rmax^2 / (2 N_bg s^2)], with
    s^2 = (k e_pos)^2 + sigma_sys^2 + e_corr^2 the per-axis offset variance of that component."""
    fit = json.load(open(os.path.join(C.DER, "reliability_fit.json")))
    e_pos = np.hypot(comp.e_RAJ2000.values * 3600 * np.cos(np.deg2rad(dec_ref)), comp.e_DEJ2000.values * 3600) / np.sqrt(2)
    s2 = (fit["err_scale_k"] * e_pos) ** 2 + fit["sigma_sys_arcsec"] ** 2 + comp.e_corr.values ** 2
    arg = fit["N_true"] * C.MATCH_R_ARCSEC ** 2 / (2 * fit["N_bg_pairs"] * s2)
    return np.sqrt(np.where(arg > 1, 2 * s2 * np.log(np.maximum(arg, 1)), 0.0))


def zech_ul(k, b, cl=0.95):
    f = lambda mu: poisson.cdf(k, mu + b) / poisson.cdf(k, b) - (1 - cl)
    return brentq(f, 0, 100 + 10 * k)


def zech_ul_marg(k, hits, scale, cl=0.95, n=20000, seed=1):
    """Zech limit marginalized over the uncertainty in b, estimated from 'hits' control detections:
    b ~ Gamma(hits + 1/2) * scale (Jeffreys prior on the control rate), and mu_95 solves
    E_b[P(n<=k | mu+b)] / E_b[P(n<=k | b)] = 1 - cl."""
    bs = np.random.default_rng(seed).gamma(hits + 0.5, 1.0, n) * scale
    den = poisson.cdf(k, bs).mean()
    f = lambda mu: poisson.cdf(k, mu + bs).mean() / den - (1 - cl)
    return brentq(f, 0, 100 + 10 * k)


if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                         columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "dist_pc", "bp_rp", "MG", "dMG", "ruwe", "non_single_star", "obs_jyear"])
    L = pd.read_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet"), columns=["source_id", "explained_detection"])
    L = L.rename(columns={"explained_detection": "detected"})      # 'detected' below = excluded as an explained secure association
    v = C.e1_components(columns=["RAJ2000", "DEJ2000", "e_RAJ2000", "e_DEJ2000", "Ftot", "Fpeak", "Islrms", "Maj", "Min", "PA", "BMaj", "BMin", "DupFlag"])
    v = v[v.DupFlag < 2].reset_index(drop=True)
    v["Maj"] = np.maximum(v.Maj, v.BMaj); v["Min"] = np.maximum(v.Min, v.BMin)   # ~25% of fits report zero axes
    tree = cKDTree(xyz(v.RAJ2000.values, v.DEJ2000.values))
    Tm = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); Ts = Tm[Tm.secure]
    _, jown = tree.query(xyz(Ts.RAJ2000.values, Ts.DEJ2000.values))
    own = pd.Series(1, index=pd.MultiIndex.from_arrays([Ts.source_id.values, jown]))
    # The same association rule at the control positions: a component that would be a secure (R > 0.5) association of
    # a control position, evaluated like the stars at the Epoch-1 epoch, is exempt from that control's veto, so b counts
    # chance coincidences of the kind that enter k (e.g. LSPM J1943+5337) instead of vetoing them.
    v["r_assoc"] = assoc_radius(v, v.DEJ2000.values)
    ra1, de1 = C.propagate(sv.ra.values, sv.dec.values, sv.pmra.values, sv.pmdec.values, sv.obs_jyear.values, plx=sv.parallax.values)
    rmax_assoc = float(np.nanmax(v.r_assoc)); own_c = []
    for j in range(N_CTRL):
        t = j * 2 * np.pi / N_CTRL
        rc1 = ra1 - R_CTRL * np.cos(t) / 3600 / np.cos(np.deg2rad(de1)); dc1 = de1 + R_CTRL * np.sin(t) / 3600
        hits = tree.query_ball_point(xyz(rc1, dc1), 2 * np.sin(np.deg2rad(rmax_assoc / 3600) / 2))
        ii = np.repeat(np.arange(len(hits)), [len(h) for h in hits]); jj = np.concatenate([np.asarray(h, int) for h in hits])
        dE = (v.RAJ2000.values[jj] - rc1[ii]) * 3600 * np.cos(np.deg2rad(dc1[ii])); dN = (v.DEJ2000.values[jj] - dc1[ii]) * 3600
        m = np.hypot(dE, dN) < v.r_assoc.values[jj]
        own_c.append(pd.Series(1, index=pd.MultiIndex.from_arrays([sv.source_id.values[ii[m]], jj[m]])).groupby(level=[0, 1]).first())
    n_ctrl_assoc = int(sum(len(o) for o in own_c))
    cal = json.load(open(os.path.join(C.DER, "forced_calibration.json"))); kappa = cal["kappa"]
    longs, per_tag = [], {}
    for tag in a.tags.split(","):
        f = os.path.join(C.DER, f"forced_{tag}.parquet")
        if not os.path.exists(f): continue
        F = pd.read_parquet(f); cz = np.load(os.path.join(C.DER, f"forced_controls_{tag}.npz"))["ctrl_snr"]
        F["k"] = np.arange(len(F)); F = F[F.good].merge(sv, on="source_id", how="left")
        ra, de = C.propagate(F.ra.values, F.dec.values, F.pmra.values, F.pmdec.values, F.ep.values, plx=F.parallax.values)
        thr = VETO_FRAC * THR * cal[tag]["eta"] * F.sig_raw.values / kappa       # contamination (catalogue scale) that vetoes
        F["pred_contam"] = predicted(ra, de, v, tree, F.source_id.values, own); F["veto"] = F.pred_contam.values >= thr
        F["contam_snr"] = kappa * F.pred_contam.values / (cal[tag]["eta"] * F.sig_raw.values)
        for j in range(N_CTRL):
            t = j * 2 * np.pi / N_CTRL
            rc = ra - R_CTRL * np.cos(t) / 3600 / np.cos(np.deg2rad(de)); dc = de + R_CTRL * np.sin(t) / 3600   # +x = west
            pc = predicted(rc, dc, v, tree, F.source_id.values, own_c[j])
            F[f"c{j}"] = cz[F.k.values, j]; F[f"cv{j}"] = pc >= thr
            F[f"cz{j}"] = (kappa * pc / (cal[tag]["eta"] * F.sig_raw.values)).astype(np.float32)   # contamination S/N
        F["tag"] = tag
        longs.append(F[["source_id", "tag", "half_epoch", "ep", "S_mJy", "sigma_mJy", "snr", "pred_contam", "contam_snr", "veto"] +
                       [f"c{j}" for j in range(N_CTRL)] + [f"cv{j}" for j in range(N_CTRL)] + [f"cz{j}" for j in range(N_CTRL)]])
    LT = pd.concat(longs, ignore_index=True).merge(L, on="source_id", how="left"); LT["detected"] = LT.detected.fillna(False).astype(bool)
    # two files, each well under the 50 MB per-file limit of Overleaf's GitHub sync; join on (source_id, tag)
    cc = [f"c{j}" for j in range(N_CTRL)] + [f"cv{j}" for j in range(N_CTRL)] + [f"cz{j}" for j in range(N_CTRL)]   # control S/N, veto flag, contamination S/N
    LT.drop(columns=cc).to_parquet(os.path.join(C.DER, "alleepoch_forced.parquet"), index=False, compression="zstd")
    LT[["source_id", "tag"] + cc].to_parquet(os.path.join(C.DER, "alleepoch_forced_controls.parquet"), index=False, compression="zstd")
    # ---- per-star coadd (unvetoed epochs) for stars and for each control angle
    U = LT[~LT.detected & ~LT.veto].copy(); U["w"] = 1 / U.sigma_mJy ** 2
    g = U.groupby("source_id")
    CO = pd.DataFrame({"n_ep": g.size(), "S": g.apply(lambda d: np.sum(d.S_mJy * d.w) / d.w.sum(), include_groups=False),
                       "sig": g.w.sum() ** -0.5})
    CO["snr"] = CO.S / CO.sig
    # Steady-test veto in coadd units: the predicted catalogue contamination, coadded like the flux over the unvetoed
    # epochs, must stay below half the coadd threshold (per-epoch contributions just below the per-epoch veto add up).
    CO["contam_coadd"] = (U.contam_snr / U.sigma_mJy).groupby(U.source_id).sum() / np.sqrt((1 / U.sigma_mJy ** 2).groupby(U.source_id).sum())
    CO["veto_coadd"] = CO.contam_coadd >= VETO_FRAC * THR
    # control coadds: z_coadd = sum(z_e / sigma_e) / sqrt(sum sigma_e^-2), over epochs where that control is unvetoed
    # Two tests, each with its own trials and chance background:
    #   steady   : one trial per star, the coadd S/N >= 5           -> k_steady, b_steady (limits on f)
    #   snapshot : one trial per star-epoch, single-epoch S/N >= 5   -> k_snap, b_snap     (limits on f * delta)
    # Activity evidence for every star (04b). A forced detection at a star with activity evidence is classified as a
    # natural emitter and removed from k, whatever its origin, so chance detections around such stars must also be
    # removed from the background: b is the control-position rate around stars without activity evidence, times the
    # number of trials at those stars.
    ACT = pd.read_parquet(os.path.join(C.DER, "activity_all.parquet"), columns=["source_id", "active", "activity_evidence"])
    active_ids = set(ACT.source_id[ACT.active])
    _wk = {"emission-line", "X-ray (SIMBAD)"}                    # robustness variant without these two types (k_*_strict)
    active_strict = set(ACT.source_id[ACT.activity_evidence.apply(lambda e: any(x and x not in _wk for x in e.split("; ")))])
    def ctrl_tables(keep):
        """Per control angle: coadd S/N and coadd-veto flag per star, and single-epoch S/N, over the star-epochs in 'keep'."""
        out = []
        for j in range(N_CTRL):
            ok = ~U[f"cv{j}"].values & np.isfinite(U[f"c{j}"].values) & keep
            Uj = U[ok]; den = np.sqrt((1 / Uj.sigma_mJy ** 2).groupby(Uj.source_id).sum())
            zc = (Uj[f"c{j}"] / Uj.sigma_mJy).groupby(Uj.source_id).sum() / den
            cc = (Uj[f"cz{j}"].astype(float) / Uj.sigma_mJy).groupby(Uj.source_id).sum() / den
            out.append((zc[cc < VETO_FRAC * THR], Uj[f"c{j}"].values, Uj.source_id.values))
        return out
    def ctrl_counts(keep, thr=THR):
        """Control-position hits and trials (steady, single-epoch) over the star-epochs selected by 'keep'."""
        T_ = ctrl_tables(keep)
        return (sum(int((z >= thr).sum()) for z, _, _ in T_), sum(len(z) for z, _, _ in T_),
                sum(int((e >= thr).sum()) for _, e, _ in T_), sum(len(e) for _, e, _ in T_))
    inact = ~U.source_id.isin(active_ids).values
    CT_in = ctrl_tables(inact)
    co_hits, co_tot = sum(int((z >= THR).sum()) for z, _, _ in CT_in), sum(len(z) for z, _, _ in CT_in)
    ep_hits, ep_tot = sum(int((e >= THR).sum()) for _, e, _ in CT_in), sum(len(e) for _, e, _ in CT_in)
    # host-level snapshot test (duty cycle): one trial per star, "detected" if any single epoch reaches 5 sigma
    host_hits = sum(int(pd.Series(e >= THR).groupby(sid).any().sum()) for _, e, sid in CT_in)
    host_tot = sum(int(pd.Series(sid).nunique()) for _, e, sid in CT_in)
    n_inact_hosts = int(pd.Series(U.source_id[inact]).nunique())
    b_host = host_hits / host_tot * n_inact_hosts
    hits_per_angle_steady = [int((z >= THR).sum()) for z, _, _ in CT_in]
    hits_per_angle_snap = [int((e >= THR).sum()) for _, e, _ in CT_in]
    rate_co, rate_ep = co_hits / co_tot, ep_hits / ep_tot
    rate_any = rate_co                                    # (name kept for the summary)
    anyep_star = (U.snr >= THR).groupby(U.source_id).any()
    CO["cand_coadd"] = (CO.snr >= THR) & ~CO.veto_coadd; CO["cand_epoch"] = anyep_star.reindex(CO.index).fillna(False).astype(bool)
    CO["cand"] = CO.cand_coadd | CO.cand_epoch
    COs = CO[~CO.veto_coadd]                              # steady-test trials
    n_inact_star, n_inact_ep = int((~COs.index.isin(active_ids)).sum()), int(inact.sum())
    b_s = rate_co * n_inact_star; b_e = rate_ep * n_inact_ep; b = b_s
    # k and b against threshold (stars without activity evidence; explained secure associations excluded)
    inact_star = ~COs.index.isin(active_ids); thr_rows = []
    for t_ in [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0]:
        thr_rows.append(dict(threshold=t_, k_steady=int((COs.snr[inact_star] >= t_).sum()),
                             b_steady=sum(int((z >= t_).sum()) for z, _, _ in CT_in) / co_tot * n_inact_star,
                             k_steady_neg=int((COs.snr[inact_star] <= -t_).sum()),
                             b_steady_neg=sum(int((z <= -t_).sum()) for z, _, _ in CT_in) / co_tot * n_inact_star,
                             k_snap=int((U.snr[inact] >= t_).sum()),
                             b_snap=sum(int((e >= t_).sum()) for _, e, _ in CT_in) / ep_tot * n_inact_ep))
    pd.DataFrame(thr_rows).round(2).to_csv(os.path.join(C.RES, "alleepoch_k_b_threshold.csv"), index=False)
    # uncertainty of b including clustering of control hits (several angles or epochs of one star): bootstrap over stars
    rg = np.random.default_rng(3); sids = COs.index[inact_star].values
    hs = pd.Series(0.0, index=sids); ts = pd.Series(0.0, index=sids)
    for z, _, _ in CT_in:
        hs = hs.add((z >= THR).astype(float).reindex(sids).fillna(0)); ts = ts.add(pd.Series(1.0, index=z.index).reindex(sids).fillna(0))
    he = pd.Series(0.0, index=U.source_id[inact].unique()); te = he.copy()
    for _, e, sid in CT_in:
        he = he.add(pd.Series(e >= THR, index=sid).groupby(level=0).sum().reindex(he.index).fillna(0))
        te = te.add(pd.Series(1.0, index=sid).groupby(level=0).sum().reindex(he.index).fillna(0))
    bb_s, bb_e = [], []
    hv, tv, hev, tev = hs.values, ts.values, he.values, te.values
    for _ in range(2000):
        i_ = rg.integers(0, len(hv), len(hv)); bb_s.append(hv[i_].sum() / tv[i_].sum() * n_inact_star)
        i2 = rg.integers(0, len(hev), len(hev)); bb_e.append(hev[i2].sum() / tev[i2].sum() * n_inact_ep)
    n_hit_stars_steady = int((hv > 0).sum()); n_hit_stars_snap = int((hev > 0).sum())
    # for comparison: the unfiltered background (all stars), and the background excluding control positions around
    # stars with emission (sidelobes of the star's own emission could inflate b there)
    h0, t0_, e0, et0 = ctrl_counts(np.ones(len(U), bool))
    b_s_all, b_e_all = h0 / t0_ * len(COs), e0 / et0 * len(U)
    emit = CO.index[CO.cand].union(pd.Index(LT.source_id[LT.detected].unique()))
    h1, t1_, e1, et1 = ctrl_counts(inact & ~U.source_id.isin(emit).values)
    b_s_noemit, b_e_noemit = h1 / t1_ * n_inact_star, e1 / et1 * n_inact_ep
    inact0 = ~U.source_id.isin(active_strict).values
    h2, t2_, e2, et2 = ctrl_counts(inact0)
    b_s_strict, b_e_strict = h2 / t2_ * int((~COs.index.isin(active_strict)).sum()), e2 / et2 * int(inact0.sum())
    for tag in LT.tag.unique():
        x = LT[LT.tag == tag]; per_tag[tag] = dict(n_star_epochs=int(len(x)), n_veto=int(x.veto.sum()), n_ge5=int(((x.snr >= THR) & ~x.detected).sum()))
    cand_ids = CO.index[CO.cand].values
    # ---- counterparts (cached)
    fs, fx = os.path.join(C.RAW, "simbad_forced_candidates.csv"), os.path.join(C.RAW, "xray_forced_candidates.csv")
    sb = pd.read_csv(fs) if os.path.exists(fs) else pd.DataFrame(columns=["source_id"])
    xr = pd.read_csv(fx) if os.path.exists(fx) else pd.DataFrame(columns=["source_id"])
    todo = [i for i in cand_ids if i not in set(xr.source_id)]
    if todo and not a.offline:
        s0 = sv.set_index("source_id").loc[todo]; xrows = []
        for sid, r in s0.iterrows():
            x1 = P04.viz_cone("J/A+A/588/A103/cat2rxs", r.ra, r.dec, 40, "2RXS,CRate,e_CRate,ExiML")
            x2 = P04.viz_cone("J/A+A/682/A34/erass1-m", r.ra, r.dec, 20, "IAUName,MLFlux1,DETLIKE1")
            gv = lambda x, k: (x[k] if x is not None and k in x.index else np.nan)
            xrows.append(dict(source_id=sid, rxs_r=gv(x1, "_r"), rxs_crate=gv(x1, "CRate"), era_r=gv(x2, "_r"), era_F=gv(x2, "MLFlux1")))
            time.sleep(0.3)
        xr = pd.concat([xr, pd.DataFrame(xrows)], ignore_index=True); xr.to_csv(fx, index=False)
        sbn = P04.simbad_by_gaia(np.array(todo)); sb = pd.concat([sb, sbn], ignore_index=True).drop_duplicates("source_id"); sb.to_csv(fs, index=False)
    S = sv[sv.source_id.isin(cand_ids)].merge(xr, on="source_id", how="left").merge(
        sb[["source_id", "main_id", "otype", "otypes"]], on="source_id", how="left")
    S["LX"] = np.where(S.era_F.notna() | S.rxs_crate.notna(), 1.0, np.nan)
    ev = S.apply(P07.evidence, axis=1); S["multiplicity_evidence"] = [e[0] for e in ev]; S["activity_evidence"] = [e[1] for e in ev]
    S["explained"] = S.activity_evidence.str.len() > 0
    _weak = {"emission-line", "X-ray (SIMBAD)"}                  # robustness: criteria without these two types
    S["explained_strict"] = S.activity_evidence.apply(lambda e: len([x for x in e.split("; ") if x and x not in _weak]) > 0)
    S = S.merge(CO[["n_ep", "S", "sig", "snr", "cand_coadd", "cand_epoch"]].rename(columns={"S": "S_coadd_mJy", "sig": "sigma_coadd_mJy", "snr": "snr_coadd"}),
                left_on="source_id", right_index=True)
    n_ep_det = (U[U.source_id.isin(cand_ids)].snr >= THR).groupby(U.source_id[U.source_id.isin(cand_ids)]).sum().rename("n_ep_det")
    S = S.merge(n_ep_det, left_on="source_id", right_index=True, how="left")
    ep_info = U[U.source_id.isin(cand_ids)].groupby("source_id").apply(
        lambda d: ";".join(f"{h}:{z:.1f}" for h, z in zip(d.half_epoch, d.snr)), include_groups=False).rename("epoch_snr")
    S = S.merge(ep_info, left_on="source_id", right_index=True, how="left")
    S[["source_id", "main_id", "dist_pc", "bp_rp", "dMG", "ruwe", "otypes", "multiplicity_evidence", "activity_evidence", "explained",
       "n_ep", "S_coadd_mJy", "sigma_coadd_mJy", "snr_coadd", "epoch_snr", "cand_coadd", "cand_epoch", "n_ep_det"]].sort_values(
        ["explained", "snr_coadd"], ascending=[True, False]).to_csv(os.path.join(C.RES, "alleepoch_candidates.csv"), index=False)
    k = int((~S.explained).sum())
    k_host = int(((~S.explained) & S.cand_epoch).sum())
    k_s = int(((~S.explained) & S.cand_coadd).sum()); k_e = int(S.loc[~S.explained, "n_ep_det"].fillna(0).sum())
    k_s0 = int(((~S.explained_strict) & S.cand_coadd).sum()); k_e0 = int(S.loc[~S.explained_strict, "n_ep_det"].fillna(0).sum())
    res = dict(per_tag=per_tag, n_stars_coadd=int(len(CO)), n_star_epochs_trials=int(len(U)),
               ctrl_rate_coadd_per_star=rate_co, ctrl_rate_per_epoch=rate_ep,
               n_candidate_stars=int(len(S)), n_explained=int(S.explained.sum()),
               n_multiplicity_only=int(((~S.explained) & (S.multiplicity_evidence.str.len() > 0)).sum()),
               n_cand_coadd=int(S.cand_coadd.sum()), n_cand_epoch=int(S.cand_epoch.sum()),
               k_steady=k_s, b_steady=b_s, mu95_steady=zech_ul(k_s, b_s),
               k_snap=k_e, b_snap=b_e, mu95_snap=zech_ul(k_e, b_e),
               k_unexplained=k, b_expected=b_s, mu95_hidden=zech_ul(k_s, b_s),
               k_steady_strict=k_s0, mu95_steady_strict=zech_ul(k_s0, b_s_strict), k_snap_strict=k_e0, mu95_snap_strict=zech_ul(k_e0, b_e_strict),
               b_steady_strict=b_s_strict, b_snap_strict=b_e_strict,
               n_explained_strict=int(S.explained_strict.sum()),
               n_active_sample=int(len(active_ids)), f_active_sample=len(active_ids) / int(len(ACT)),
               n_inactive_stars_coadd=n_inact_star, n_inactive_star_epochs=n_inact_ep,
               ctrl_hits_steady=co_hits, ctrl_trials_steady=co_tot, ctrl_hits_snap=ep_hits, ctrl_trials_snap=ep_tot,
               b_steady_err=float(np.sqrt(co_hits) / co_tot * n_inact_star), b_snap_err=float(np.sqrt(ep_hits) / ep_tot * n_inact_ep),
               mu95_steady_marg=zech_ul_marg(k_s, co_hits, n_inact_star / co_tot), mu95_snap_marg=zech_ul_marg(k_e, ep_hits, n_inact_ep / ep_tot),
               b_steady_unfiltered=b_s_all, b_snap_unfiltered=b_e_all, mu95_steady_unfiltered=zech_ul(k_s, b_s_all), mu95_snap_unfiltered=zech_ul(k_e, b_e_all),
               b_steady_noemit=b_s_noemit, b_snap_noemit=b_e_noemit,
               n_vetoed_star_epochs=int(LT.veto.sum()), n_stars_any_veto=int(LT.groupby("source_id").veto.any().sum()),
               k_host=k_host, b_host=b_host, mu95_host=zech_ul(k_host, b_host), mu95_host_b0=float(C.poisson_ul95(k_host)),
               ctrl_hits_host=host_hits, ctrl_trials_host=host_tot, n_inactive_hosts=n_inact_hosts,
               coadd_vetoed_ids=[int(x) for x in CO.index[CO.veto_coadd]],
               n_coadd_vetoed=int(CO.veto_coadd.sum()), n_ctrl_assoc_exempt=n_ctrl_assoc,
               ctrl_hits_per_angle_steady=hits_per_angle_steady, ctrl_hits_per_angle_snap=hits_per_angle_snap,
               p_rank_steady=float((1 + sum(h <= k_s for h in hits_per_angle_steady)) / (N_CTRL + 1)),
               p_rank_snap=float((1 + sum(h >= k_e for h in hits_per_angle_snap)) / (N_CTRL + 1)),
               b_steady_err_boot=float(np.std(bb_s)), b_snap_err_boot=float(np.std(bb_e)),
               n_ctrl_hit_stars_steady=n_hit_stars_steady, n_ctrl_hit_stars_snap=n_hit_stars_snap,
               mu95_steady_b0=float(C.poisson_ul95(k_s)), mu95_snap_b0=float(C.poisson_ul95(k_e)),
               p_steady_low=float(poisson.cdf(k_s, b_s)), p_snap_high=float(poisson.sf(k_e - 1, b_e)))
    # ---- catalog-level check of the background (no forced photometry, no veto): Epoch-1 components (S/N >= 5) within
    # 0-1" and 1-2.5" of stars without activity evidence (at the Epoch-1 epoch) against the same counts around control
    # points 14", 30", 60" and 180" away (8 angles each, averaged), overall and by G magnitude (a selection against faint
    # stars blended with a radio source's optical host would appear as a deficit growing with G)
    gmag = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"), columns=["source_id", "phot_g_mean_mag"]).set_index("source_id").phot_g_mean_mag
    vs = v[v.Fpeak / v.Islrms >= 5]
    t5 = cKDTree(xyz(vs.RAJ2000.values, vs.DEJ2000.values)); inm = ~sv.source_id.isin(active_ids).values
    G_ = gmag.reindex(sv.source_id.values).values[inm]
    def ring_counts(r0, d0):
        d_, _ = t5.query(xyz(r0, d0), distance_upper_bound=2 * np.sin(np.deg2rad(2.5 / 3600) / 2))
        d_ = np.rad2deg(2 * np.arcsin(np.minimum(d_, 2) / 2)) * 3600; return d_ < 1.0, (d_ >= 1.0) & (d_ < 2.5)
    s0, s1 = ring_counts(ra1[inm], de1[inm]); c0 = np.zeros(inm.sum()); c1 = np.zeros(inm.sum()); per_ring = {}
    for rc in [14, 30, 60, 180]:
        rr = 0.0
        for j in range(8):
            t = j * 2 * np.pi / 8 + 0.1 * rc
            a0, a1 = ring_counts(ra1[inm] - rc * np.cos(t) / 3600 / np.cos(np.deg2rad(de1[inm])), de1[inm] + rc * np.sin(t) / 3600)
            c0 += a0 / 32; c1 += a1 / 32; rr += a1.sum() / 8
        per_ring[str(rc)] = float(rr)                    # mean count at 1-2.5" per set of control points at this radius
    gb = [(0, 13), (13, 15), (15, 17), (17, 30)]
    res["catalog_density"] = dict(n_stars=int(inm.sum()), star_0_1=int(s0.sum()), ctrl_0_1=float(c0.sum()), star_1_2p5=int(s1.sum()), ctrl_1_2p5=float(c1.sum()),
                                  p_low_1_2p5=float(poisson.cdf(int(s1.sum()), c1.sum())), ctrl_1_2p5_per_ring=per_ring,
                                  by_G={f"{lo}-{hi}": [int(s1[(G_ >= lo) & (G_ < hi)].sum()), float(c1[(G_ >= lo) & (G_ < hi)].sum())] for lo, hi in gb})
    json.dump(res, open(os.path.join(C.DER, "alleepoch_candidates.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
