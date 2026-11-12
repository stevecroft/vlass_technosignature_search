"""11 - Summary figures.
results/vlass_technosig_summary.png : (a) separation histogram vs offset control, (b) CMD with classified detections,
                                      (c) per-star limits vs distance, (d) population limit (clipped at f = 1)
results/vlass_ep1_stacking_summary.png : (a) stacked mean L_nu per shell, (b) stacked images, (c) fraction vs power (clipped at f = 1)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import numpy as np, pandas as pd

GROUP = {"RS CVn / active binary": "Active binary (RS CVn, BY Dra, other binary)", "BY Dra / active binary": "Active binary (RS CVn, BY Dra, other binary)",
         "binary/overluminous + X-ray": "Active binary (RS CVn, BY Dra, other binary)", "binary/overluminous": "Active binary (RS CVn, BY Dra, other binary)",
         "flare star": "Flare / X-ray-active single star", "X-ray active single star": "Flare / X-ray-active single star",
         "young star": "Other natural (young, CV, magnetic CP)", "accreting WD binary (CV)": "Other natural (young, CV, magnetic CP)",
         "magnetic CP star": "Other natural (young, CV, magnetic CP)",
         "background (multi-epoch astrometry)": "Background (LSPM J1943+5337, multi-epoch astrometry)",
         "UNEXPLAINED": "Unexplained"}
COLG = {"Active binary (RS CVn, BY Dra, other binary)": "#3b6ea8", "Flare / X-ray-active single star": "#e0a030",
        "Other natural (young, CV, magnetic CP)": "#6aa86a", "Background (LSPM J1943+5337, multi-epoch astrometry)": "#c0392b", "Unexplained": "k"}
COLT = {"FGK": "#e0a030", "M": "#3b6ea8"}


def clip1(f):
    f = np.asarray(f, float).copy(); f[~(f <= 1)] = np.nan; return f


def summary(plt):
    A = pd.read_parquet(os.path.join(C.DER, "matches_10arcsec.parquet")); offs = pd.read_parquet(os.path.join(C.DER, "offset_pairs.parquet"))
    L = pd.read_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet"))
    T = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); Ts = T[T.rel > 0.5].copy(); Ts["grp"] = Ts["class"].map(GROUP)
    pop = pd.read_csv(os.path.join(C.RES, "vlass_population_limit_curve.csv"))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4)); (a, b), (c, d) = axs
    bins = np.arange(0, 5.01, 0.25); hr, _ = np.histogram(A.sep, bins); ho, _ = np.histogram(offs.sep, bins)
    a.step(bins[:-1], hr, where="post", color="k", lw=1.2, label="Stars (epoch-propagated)")
    a.step(bins[:-1], ho / C.N_OFFSETS, where="post", color="0.55", lw=1.0, ls="--", label="3\u2032-offset positions (mean of 10)")
    a.set_xlabel("Star\u2013VLASS separation (arcsec)"); a.set_ylabel("Pairs per 0.25\u2033 bin")
    a.set_title("Genuine stellar matches lie within 1\u2033", loc="left"); a.legend(frameon=False, loc="upper right")
    b.hexbin(L.bp_rp, L.MG, gridsize=120, bins="log", cmap="Greys", mincnt=1, linewidths=0, extent=(-0.2, 5, -1, 17))
    for k, cc in COLG.items():
        m = Ts.grp == k
        if m.any(): b.scatter(Ts.bp_rp[m], Ts.MG[m], s=16, c=cc, edgecolor="k", lw=0.3, label=k, zorder=3)
    b.invert_yaxis(); b.set_xlabel("Gaia $G_{BP}-G_{RP}$ (mag)"); b.set_ylabel("$M_G$ (mag)")
    b.set_title("Detections are mostly active binaries", loc="left"); b.legend(frameon=False, fontsize=6, loc="upper right")
    far = L.dist_pc >= 10
    c.hexbin(L.dist_pc[far], L.Lnu_lim_WHz[far], xscale="log", yscale="log", gridsize=60, bins="log", cmap="Greys", mincnt=1, linewidths=0, extent=(1, 2, 5.5, 10.3))
    c.scatter(L.dist_pc[~far], L.Lnu_lim_WHz[~far], s=2, c="0.45", lw=0, zorder=2)
    for k, cc in COLG.items():
        m = Ts.grp == k
        if m.any(): c.scatter(Ts.dist_pc[m], Ts.Lnu_R_WHz[m], s=16, c=cc, edgecolor="k", lw=0.3, zorder=3)
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(1.5, 120); c.set_ylim(2e5, 1e10)
    c.set_xlabel("Distance (pc)"); c.set_ylabel("$L_\\nu$ at 3 GHz (W Hz$^{-1}$)"); c.set_title("Per-star 95% limits (grey) and detections", loc="left")
    c2 = c.secondary_yaxis("right", functions=(lambda y: y * C.BAND_HZ, lambda y: y / C.BAND_HZ)); c2.set_ylabel("EIRP, 2 GHz-wide (W)")
    r0 = L.nsmallest(1, "dist_pc").iloc[0]
    c.annotate("Barnard's star", (r0.dist_pc, r0.Lnu_lim_WHz), xytext=(4, 3e5), fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5), va="center")
    d.loglog(pop.Lnu_star_WHz, clip1(pop.f_UL95), color="k", lw=1.4)
    d.set_xlabel("Transmitter $L_\\nu$ threshold (W Hz$^{-1}$)"); d.set_ylabel("Fraction of stars hosting one (95% UL)")
    d2 = d.secondary_xaxis("top", functions=(lambda x: x * C.BAND_HZ, lambda x: x / C.BAND_HZ)); d2.set_xlabel("EIRP, 2 GHz-wide (W)")
    d.set_xlim(5e5, 3e10); d.set_ylim(1e-5, 1)
    i9 = np.argmin(np.abs(pop.Lnu_star_WHz - 1e9))
    d.text(0.97, 0.92, f"< 1 in {1/pop.f_UL95[i9]:,.0f} stars\nabove 10$^{{9}}$ W Hz$^{{-1}}$", transform=d.transAxes, ha="right", va="top", fontsize=7)
    for ax, l in zip([a, b, c, d], "abcd"): C.panel_letter(ax, l)
    fig.tight_layout(); fig.savefig(os.path.join(C.RES, "vlass_technosig_summary.png"), dpi=300); fig.savefig(os.path.join(C.RES, "vlass_technosig_summary.pdf"), dpi=300); plt.close(fig)


def stacking(plt):
    F = pd.read_csv(os.path.join(C.RES, "vlass_ep1_stack_results.csv")); fv = pd.read_csv(os.path.join(C.RES, "vlass_fraction_vs_power_limits.csv"))
    imgs = np.load(os.path.join(C.DER, "stack_images.npz"))
    fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    a = fig.add_subplot(gs[0, :2]); mids = {"0-10": 5, "10-20": 15, "20-40": 30, "40-60": 50, "60-80": 70, "80-100": 90}
    for typ, dx in [("FGK", -1.2), ("M", 1.2)]:
        s_ = F[F.type == typ]; x = np.array([mids[k] for k in s_.shell_pc]) + dx; sig = s_.L_SNR.values >= 3
        a.errorbar(x, s_.Lmean_WHz / 1e6, yerr=s_.L_err_WHz / 1e6, fmt="none", ecolor=COLT[typ], lw=1, capsize=2)
        a.scatter(x[sig], (s_.Lmean_WHz / 1e6)[sig], s=28, c=COLT[typ], edgecolor="k", lw=0.4, zorder=3, label=f"{typ} (\u22653\u03c3 filled)")
        a.scatter(x[~sig], (s_.Lmean_WHz / 1e6)[~sig], s=28, facecolor="white", edgecolor=COLT[typ], lw=1, zorder=3)
    a.axhline(0, color="0.6", lw=0.6); a.set_xticks(list(mids.values())); a.set_xticklabels([f"{k} pc" for k in mids])
    a.set_ylabel("Stacked mean $L_\\nu$ (10$^6$ W Hz$^{-1}$)"); a.set_title("Sub-threshold 3 GHz emission from M dwarfs at 10\u201360 pc", loc="left")
    a.legend(frameon=False, fontsize=6.5, loc="upper right"); a.margins(x=0.04)
    sub = gs[0, 2].subgridspec(2, 1, hspace=0.35)
    for j, typ in enumerate(["M", "FGK"]):
        ax = fig.add_subplot(sub[j]); im = imgs[f"20-40_{typ}_S"]; lim = np.nanpercentile(np.abs(im), 99.5)
        ax.imshow(im[20 - 12:20 + 13, 20 - 12:20 + 13], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=[12.5, -12.5, -12.5, 12.5])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{'M dwarfs' if typ == 'M' else 'FGK'}, 20\u201340 pc", fontsize=7, loc="left")
        r_ = F[(F.shell_pc == "20-40") & (F.type == typ)].iloc[0]
        ax.text(0.03, 0.04, f"{r_.S_mean_uJy:.1f}\u00b1{r_.S_err_uJy:.1f} \u00b5Jy", transform=ax.transAxes, fontsize=6, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        if j == 0: ax.plot([10, 5], [10, 10], color="k", lw=1); ax.text(7.5, 10.8, "5\u2033", ha="center", fontsize=6); C.panel_letter(ax, "b")
    c = fig.add_subplot(gs[1, :]); Lg = fv.Lnu_T_WHz.values
    for typ in ["FGK", "M"]:
        c.loglog(Lg, clip1(fv[f"f_UL95_stack_{typ}"]), color=COLT[typ], lw=1.6, label=f"{typ}: stacked mean (this work)")
        c.loglog(Lg, clip1(fv[f"f_UL95_detection_{typ}"]), color=COLT[typ], lw=1.2, ls="--", label=f"{typ}: individual non-detections")
    best = np.nanmin(fv[[k for k in fv.columns if k.startswith("f_UL95")]].values, axis=1); x_unc = Lg[np.argmax(best < 1)]
    c.axvspan(1e5, x_unc, color="0.88", lw=0, zorder=0)
    c.text(np.sqrt(1e5 * x_unc), 1e-3, "No constraint (f \u2265 1)", rotation=90, ha="center", va="center", fontsize=6.5, color="0.35")
    c.set_xlim(1e5, 3e10); c.set_ylim(1e-5, 1); c.set_xlabel("Transmitter $L_\\nu$ at 3 GHz (W Hz$^{-1}$)"); c.set_ylabel("Fraction of stars (95% UL)")
    c.set_title("Stacking reaches transmitters 4\u20136\u00d7 fainter than individual non-detections", loc="left")
    c2 = c.secondary_xaxis("top", functions=(lambda x: x * C.BAND_HZ, lambda x: x / C.BAND_HZ)); c2.set_xlabel("EIRP, 2 GHz-wide (W)")
    c.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.12, 0.0))
    C.panel_letter(a, "a"); C.panel_letter(c, "c"); fig.tight_layout()
    fig.savefig(os.path.join(C.RES, "vlass_ep1_stacking_summary.png"), dpi=300); fig.savefig(os.path.join(C.RES, "vlass_ep1_stacking_summary.pdf"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    plt = C.set_plot_style(); summary(plt); stacking(plt); print("figures written to", C.RES)
