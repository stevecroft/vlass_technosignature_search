"""10 - Stack VLASS Epoch-1 QL cutouts in distance shells x spectral class, and derive fraction-vs-power limits.

Method: exclude detected, confused and near-component stars; per-cutout robust sigma from pixels r >= 8";
drop sigma > 5x median and non-finite centres; inverse-variance mean in L_nu (W/Hz) and in flux (uJy/beam);
QL flux bias /0.87; centring-loss correction mean(exp(-r^2/2 sigma_b^2)), sigma_b = 1.06 px; empirical null =
stacked image at r = 10-18" (subtract its mean; error = max(formal, null std)). Robustness: top-1% trimming
(bias measured at 12 null pixels), RA-half jackknife, median stack.
Fraction-vs-power: stack gives f <= UL95(mean L)/L_T using only shells whose 5th-percentile per-star limit > L_T;
individual non-detections give f < 3.0/N_eff(L_T).

Inputs : data/raw/vlass_ep1_ql_cutouts_41px.npz, results/vlass_perstar_limits_100pc_MS.parquet
Outputs: results/vlass_ep1_stack_results.csv, results/vlass_ep1_stack_robustness.csv,
         results/vlass_fraction_vs_power_limits.csv, data/derived/stack_images.npz
Expected (20-40 pc M): N = 9864, S = 11.6 +- 1.9 uJy, L = 1.20e6 +- 0.19e6 W/Hz
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import numpy as np, pandas as pd

HB = 20
yy, xx = np.mgrid[-HB:HB + 1, -HB:HB + 1]; RR = np.hypot(xx, yy)
ANN = RR >= 8; NULLRING = (RR >= 10) & (RR <= 18)
NULLPOS = [(int(round(14 * np.sin(t))), int(round(14 * np.cos(t)))) for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]


def robust_sigma(a):
    med = np.nanmedian(a, axis=1); return 1.4826 * np.nanmedian(np.abs(a - med[:, None]), axis=1)


def wmean(maps, w):
    fin = np.isfinite(maps); Wp = w[:, None, None] * fin
    return np.nansum(np.where(fin, maps, 0) * w[:, None, None], axis=0) / Wp.sum(0), Wp.sum(0)


def stack(cut, off, dist, beam_sigma_pix=1.06):
    S = cut / C.QL_FLUX_BIAS * 1e3                                        # mJy/beam
    sig = robust_sigma(S[:, ANN].reshape(len(S), -1))
    good = np.isfinite(sig) & (sig > 0) & (sig < 5 * np.nanmedian(sig)) & np.isfinite(S[:, HB, HB])
    S, sig, d, o = S[good], sig[good], dist[good], off[good]
    fac = 4 * np.pi * (d * C.PC_M) ** 2
    mL, WL = wmean(S * 1e-29 * fac[:, None, None], 1 / (sig * 1e-29 * fac) ** 2)
    mS, WS = wmean(S, 1 / sig ** 2); med = np.nanmedian(S, axis=0)
    cl = float(np.mean(np.exp(-(o ** 2).sum(1) / (2 * beam_sigma_pix ** 2))))
    return dict(N=int(good.sum()), centre_loss=cl, S=S, sig=sig, mL=mL, mS=mS, med=med,
                L_mean=mL[HB, HB] / cl, L_null_mean=np.mean(mL[NULLRING]) / cl, L_null_std=np.std(mL[NULLRING]) / cl, L_formal=1 / np.sqrt(WL[HB, HB]) / cl,
                S_mean=mS[HB, HB] / cl * 1e3, S_null_mean=np.mean(mS[NULLRING]) / cl * 1e3, S_null_std=np.std(mS[NULLRING]) / cl * 1e3,
                S_formal=1 / np.sqrt(WS[HB, HB]) / cl * 1e3,
                S_median=med[HB, HB] / cl * 1e3, S_med_null_mean=np.mean(med[NULLRING]) / cl * 1e3, S_med_null_std=np.std(med[NULLRING]) / cl * 1e3)


def trimmed(Spix, sig, pct=99):
    thr = np.nanpercentile(Spix, pct); m = np.isfinite(Spix) & (Spix < thr); w = 1 / sig[m] ** 2
    return np.sum(w * Spix[m]) / w.sum()


if __name__ == "__main__":
    z = np.load(os.path.join(C.RAW, "vlass_ep1_ql_cutouts_41px.npz"))
    L = pd.read_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet"))
    row = pd.Series(np.arange(len(L)), index=L.source_id.values)
    idx = row.reindex(z["source_id"]).values; okc = np.isfinite(idx)
    print("cutouts:", len(idx), " matched to sample:", int(okc.sum()))
    idx = idx[okc].astype(int); cut = z["cut"][okc]; off = z["off"][okc]
    excl = (L.detected | L.confused | L.has_nearby_component).values
    print("excluded stars (detected | confused | non-stellar component < 10\"):", int(excl.sum()))
    keep = ~excl[idx]; idx, cut, off = idx[keep], cut[keep], off[keep]
    dist = L.dist_pc.values[idx]; bprp = L.bp_rp.values[idx]; ra = L.ra.values[idx]
    fgk_all = L.bp_rp.values < C.BPRP_SPLIT

    rows, rob, images = [], [], {}
    for lo, hi in C.SHELLS:
        for typ, mt in [("FGK", bprp < C.BPRP_SPLIT), ("M", bprp >= C.BPRP_SPLIT)]:
            m = (dist > lo) & (dist <= hi) & mt
            b = stack(cut[m], off[m], dist[m]); cl = b["centre_loss"]
            Sm = b["S_mean"] - b["S_null_mean"]; Serr = max(b["S_formal"], b["S_null_std"])
            Lm = b["L_mean"] - b["L_null_mean"]; Lerr = max(b["L_formal"], b["L_null_std"])
            tc = trimmed(b["S"][:, HB, HB], b["sig"]); tn = np.mean([trimmed(b["S"][:, HB + dy, HB + dx], b["sig"]) for dy, dx in NULLPOS])
            msk = (L.dist_pc > lo) & (L.dist_pc <= hi) & (fgk_all if typ == "FGK" else ~fgk_all)
            rows.append(dict(shell_pc=f"{lo}-{hi}", type=typ, N=b["N"], centre_loss=round(cl, 3), S_mean_uJy=Sm, S_err_uJy=Serr, S_SNR=Sm / Serr,
                             S_trim_corr_uJy=(tc - tn) / cl * 1e3, S_median_uJy=b["S_median"] - b["S_med_null_mean"], S_median_err_uJy=b["S_med_null_std"],
                             Lmean_WHz=Lm, L_err_WHz=Lerr, L_SNR=Lm / Lerr, Lmean_UL95_WHz=max(Lm, 0) + C.Z95 * Lerr,
                             EIRP_UL95_W_2GHz=(max(Lm, 0) + C.Z95 * Lerr) * C.BAND_HZ,
                             Llim_5pct_WHz=float(np.percentile(L.Lnu_lim_WHz[msk], 5))))
            images[f"{lo}-{hi}_{typ}_S"] = b["mS"] * 1e3; images[f"{lo}-{hi}_{typ}_L"] = b["mL"]
            if (lo, hi) in [(10, 20), (20, 40), (40, 60), (60, 80)]:
                e = lambda q: max(q["S_formal"], q["S_null_std"])
                hN = stack(cut[m & (ra < 180)], off[m & (ra < 180)], dist[m & (ra < 180)])
                hS = stack(cut[m & (ra >= 180)], off[m & (ra >= 180)], dist[m & (ra >= 180)])
                rob.append(dict(stack=f"{lo}-{hi} {typ}", N=b["N"], S_minus_null_uJy=Sm, S_err_uJy=Serr, S_trim_corr_uJy=(tc - tn) / cl * 1e3,
                                RA_lt_180_uJy=hN["S_mean"], RA_lt_180_err=e(hN), RA_ge_180_uJy=hS["S_mean"], RA_ge_180_err=e(hS)))
    F = pd.DataFrame(rows); F.to_csv(os.path.join(C.RES, "vlass_ep1_stack_results.csv"), index=False)
    pd.DataFrame(rob).to_csv(os.path.join(C.RES, "vlass_ep1_stack_robustness.csv"), index=False)
    np.savez_compressed(os.path.join(C.DER, "stack_images.npz"), **images)
    pd.options.display.float_format = lambda x: f"{x:.3g}"
    print(F[["shell_pc", "type", "N", "S_mean_uJy", "S_err_uJy", "S_SNR", "Lmean_WHz", "L_err_WHz", "L_SNR"]].to_string(index=False))

    # fraction vs transmitter power
    Lgrid = np.logspace(5, 10.5, 300); out = {"Lnu_T_WHz": Lgrid, "EIRP_2GHz_W": Lgrid * C.BAND_HZ}
    for typ, mt in [("FGK", fgk_all), ("M", ~fgk_all)]:
        Ne = np.searchsorted(np.sort(L.Lnu_lim_WHz.values[mt]), Lgrid)
        out[f"f_UL95_detection_{typ}"] = np.where(Ne > 0, C.poisson_ul95(0) / np.maximum(Ne, 1), np.nan)
        s_ = F[F.type == typ]; fs = np.full(len(Lgrid), np.nan)
        for i, LT in enumerate(Lgrid):
            u_ = s_[s_.Llim_5pct_WHz > LT]
            if len(u_) == 0: continue
            w_ = 1 / u_.L_err_WHz ** 2; mu = np.sum(w_ * u_.Lmean_WHz) / w_.sum(); er = 1 / np.sqrt(w_.sum())
            fs[i] = (max(mu, 0) + C.Z95 * er) / LT
        out[f"f_UL95_stack_{typ}"] = fs
    fv = pd.DataFrame(out)[["Lnu_T_WHz", "EIRP_2GHz_W", "f_UL95_detection_FGK", "f_UL95_stack_FGK", "f_UL95_detection_M", "f_UL95_stack_M"]]
    fv.to_csv(os.path.join(C.RES, "vlass_fraction_vs_power_limits.csv"), index=False)
    for typ in ["FGK", "M"]:
        Ls = Lgrid[np.nanargmax(fv[f"f_UL95_stack_{typ}"] < 1)]; Ld = Lgrid[np.nanargmax(np.nan_to_num(fv[f"f_UL95_detection_{typ}"], nan=9) < 1)]
        print(f"{typ}: stack f<1 from L_T={Ls:.3g} W/Hz; detections f<1 from {Ld:.3g} W/Hz")
