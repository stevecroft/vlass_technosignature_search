"""18 - All-epoch detection candidates from forced photometry, their classification, and the chance background.

A star-epoch is a candidate if its forced S/N (17) is >= 5 and the star is not one of the secure Epoch-1 associations.
  * Confusion veto: the summed contribution at the measurement position of all Epoch-1 catalogue components within
    15" (other than the star's own secure component), each modelled as an elliptical Gaussian with its fitted
    (beam-convolved) Maj, Min, PA and peak, is at least half the detection threshold (kappa * pred >= 2.5 sigma_eff).
    A plain 3" radius is not enough: bright components 3-4" away put >5 sigma into the beam wing, especially at low
    declination where the QL beam is elongated. Vetoed stars are confusion-limited and reported separately.
  * Classification of the remaining candidate stars with the same evidence as for the secure associations (07):
    SIMBAD object types (CfA TAP mirror) and ROSAT 2RXS / eRASS1 X-ray counterparts (CfA VizieR mirror). A candidate
    is "explained" if it has activity evidence (X-ray, flare/BY Dra/RS CVn/young/CV/magnetic CP type). Stars with
    only multiplicity evidence are listed separately and counted as unexplained (conservative).
  * Chance background: the same estimator and the same 3" catalogue veto at 8 control positions 14" from each star.
    b = (control rate) x (number of unvetoed, non-secure star-epochs). Control positions carry no stellar emission,
    so b counts sidelobes and unrelated sources only, matching the unexplained candidates.
  * Upper limit on the number of transmitter detections hidden among the k unexplained candidates, given background b:
    the conditional (Zech 1989) 95% limit, P(n<=k | mu+b) / P(n<=k | b) = 0.05.

Inputs : data/derived/forced_<tag>.parquet, forced_controls_<tag>.npz, data/derived/ms_sample_vlass.parquet,
         data/raw/vlass_e1ql_components.parquet, results/vlass_perstar_limits_100pc_MS.parquet,
         results/vlass_star_matches_classified.csv
Outputs: data/raw/simbad_forced_candidates.csv, data/raw/xray_forced_candidates.csv (fetched once, then cached),
         results/alleepoch_candidates.csv, data/derived/alleepoch_candidates.json
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
    if exclude_sid is not None:     # drop each star's own secure component
        val = np.where(own.reindex(list(zip(exclude_sid[ii], jj))).notna().values, 0.0, val)
    np.add.at(out, ii, val); return out


def zech_ul(k, b, cl=0.95):
    f = lambda mu: poisson.cdf(k, mu + b) / poisson.cdf(k, b) - (1 - cl)
    return brentq(f, 0, 100 + 10 * k)


if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                         columns=["source_id", "ra", "dec", "pmra", "pmdec", "parallax", "dist_pc", "bp_rp", "MG", "dMG", "ruwe", "non_single_star"])
    L = pd.read_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet"), columns=["source_id", "detected"])
    v = pd.read_parquet(os.path.join(C.RAW, "vlass_e1ql_components.parquet"), columns=["RAJ2000", "DEJ2000", "Ftot", "Fpeak", "Maj", "Min", "PA", "BMaj", "BMin", "DupFlag"])
    v = v[v.DupFlag < 2].reset_index(drop=True)
    v["Maj"] = np.maximum(v.Maj, v.BMaj); v["Min"] = np.maximum(v.Min, v.BMin)   # ~25% of fits report zero axes
    tree = cKDTree(xyz(v.RAJ2000.values, v.DEJ2000.values))
    Tm = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); Ts = Tm[Tm.secure]
    _, jown = tree.query(xyz(Ts.RAJ2000.values, Ts.DEJ2000.values))
    own = pd.Series(1, index=pd.MultiIndex.from_arrays([Ts.source_id.values, jown]))
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
            F[f"c{j}"] = cz[F.k.values, j]; F[f"cv{j}"] = predicted(rc, dc, v, tree) >= thr
        F["tag"] = tag
        longs.append(F[["source_id", "tag", "half_epoch", "ep", "S_mJy", "sigma_mJy", "snr", "pred_contam", "contam_snr", "veto"] +
                       [f"c{j}" for j in range(N_CTRL)] + [f"cv{j}" for j in range(N_CTRL)]])
    LT = pd.concat(longs, ignore_index=True).merge(L, on="source_id", how="left"); LT["detected"] = LT.detected.fillna(False).astype(bool)
    LT.to_parquet(os.path.join(C.DER, "alleepoch_forced.parquet"), index=False)   # c<j>/cv<j>: control S/N and veto flag
    # ---- per-star coadd (unvetoed epochs) for stars and for each control angle
    U = LT[~LT.detected & ~LT.veto].copy(); U["w"] = 1 / U.sigma_mJy ** 2
    g = U.groupby("source_id")
    CO = pd.DataFrame({"n_ep": g.size(), "S": g.apply(lambda d: np.sum(d.S_mJy * d.w) / d.w.sum(), include_groups=False),
                       "sig": g.w.sum() ** -0.5})
    CO["snr"] = CO.S / CO.sig
    # control coadds: z_coadd = sum(z_e / sigma_e) / sqrt(sum sigma_e^-2), over epochs where that control is unvetoed
    ctrl_hits = ctrl_tot = 0; ep_hits = ep_tot = 0
    for j in range(N_CTRL):
        ok = ~U[f"cv{j}"].values & np.isfinite(U[f"c{j}"].values)
        Uj = U[ok]; ep_hits += int((Uj[f"c{j}"] >= THR).sum()); ep_tot += len(Uj)
        num = (Uj[f"c{j}"] / Uj.sigma_mJy).groupby(Uj.source_id).sum(); den = np.sqrt((1 / Uj.sigma_mJy ** 2).groupby(Uj.source_id).sum())
        zc = num / den; anyep = (Uj[f"c{j}"] >= THR).groupby(Uj.source_id).any()
        ctrl_hits += int(((zc >= THR) | anyep).sum()); ctrl_tot += len(zc)
    rate_any = ctrl_hits / ctrl_tot                      # per star: any epoch >= 5 or coadd >= 5, at a blank position
    anyep_star = (U.snr >= THR).groupby(U.source_id).any()
    CO["cand"] = (CO.snr >= THR) | anyep_star.reindex(CO.index).fillna(False)
    b = rate_any * len(CO)
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
    S = S.merge(CO[["n_ep", "S", "sig", "snr"]].rename(columns={"S": "S_coadd_mJy", "sig": "sigma_coadd_mJy", "snr": "snr_coadd"}),
                left_on="source_id", right_index=True)
    ep_info = U[U.source_id.isin(cand_ids)].groupby("source_id").apply(
        lambda d: ";".join(f"{h}:{z:.1f}" for h, z in zip(d.half_epoch, d.snr)), include_groups=False).rename("epoch_snr")
    S = S.merge(ep_info, left_on="source_id", right_index=True, how="left")
    S[["source_id", "main_id", "dist_pc", "bp_rp", "dMG", "ruwe", "otypes", "multiplicity_evidence", "activity_evidence", "explained",
       "n_ep", "S_coadd_mJy", "sigma_coadd_mJy", "snr_coadd", "epoch_snr"]].sort_values(["explained", "snr_coadd"], ascending=[True, False]).to_csv(
        os.path.join(C.RES, "alleepoch_candidates.csv"), index=False)
    k = int((~S.explained).sum())
    res = dict(per_tag=per_tag, n_stars_coadd=int(len(CO)), ctrl_rate_any_per_star=rate_any, ctrl_rate_per_epoch=ep_hits / ep_tot,
               n_candidate_stars=int(len(S)), n_explained=int(S.explained.sum()),
               n_multiplicity_only=int(((~S.explained) & (S.multiplicity_evidence.str.len() > 0)).sum()),
               k_unexplained=k, b_expected=b, mu95_hidden=zech_ul(k, b),
               n_vetoed_star_epochs=int(LT.veto.sum()), n_stars_any_veto=int(LT.groupby("source_id").veto.any().sum()))
    json.dump(res, open(os.path.join(C.DER, "alleepoch_candidates.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
