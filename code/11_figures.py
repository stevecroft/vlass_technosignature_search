"""11 - Summary figures.
results/vlass_technosig_summary.png : (a) separation histogram vs offset control, (b) CMD with classified detections,
                                      (c) per-star limits vs distance, (d) population limit (clipped at f = 1);
  the same four panels are also written as stand-alone figures vlass_separation_histogram, vlass_detections_cmd,
  vlass_perstar_limits and vlass_population_limit (.png/.pdf) for the paper
results/vlass_ep1_stacking_summary.png : (a) stacked mean L_nu per shell, (b) stacked images, (c) fraction vs power (clipped at f = 1)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import numpy as np, pandas as pd

# groups come from the 'group' column written by 07_classify_and_limits.py (multiplicity is evidence, not a mechanism)
COLG = {"Active, with multiplicity evidence": "#3b6ea8", "Active, no multiplicity evidence": "#e0a030",
        "Young, CV or magnetic CP": "#6aa86a", "Background (not coincident with the star)": "#c0392b", "Unexplained": "k"}
COLT = {"FGK": "#e0a030", "M": "#3b6ea8"}
SHORT = {"Active, with multiplicity evidence": "Active, multiplicity evidence", "Active, no multiplicity evidence": "Active, no multiplicity evidence",
         "Young, CV or magnetic CP": "Young, CV or magnetic CP", "Background (not coincident with the star)": "Not coincident (LSPM J1943+5337)",
         "Unexplained": "Unexplained"}


def clip1(f):
    f = np.asarray(f, float).copy(); f[~(f <= 1)] = np.nan; return f


def load_summary():
    D = dict(A=pd.read_parquet(os.path.join(C.DER, "matches_10arcsec.parquet")), offs=pd.read_parquet(os.path.join(C.DER, "offset_pairs.parquet")),
             L=pd.read_parquet(os.path.join(C.RES, "vlass_perstar_limits_100pc_MS.parquet")),
             pop=pd.read_csv(os.path.join(C.RES, "vlass_population_limit_curve.csv")))
    T = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); Ts = T[T.rel > 0.5].copy(); Ts["grp"] = Ts["group"]
    D["Ts"] = Ts
    return D


def panel_separation(a, D, title=True):
    A, offs = D["A"], D["offs"]
    bins = np.arange(0, 5.01, 0.25); hr, _ = np.histogram(A.sep, bins); ho, _ = np.histogram(offs.sep, bins)
    a.step(bins[:-1], hr, where="post", color="k", lw=1.2, label="Stars (epoch-propagated)")
    a.step(bins[:-1], ho / C.N_OFFSETS, where="post", color="0.55", lw=1.0, ls="--", label="3\u2032-offset positions (mean of 10)")
    a.set_xlabel("Star\u2013VLASS separation (arcsec)"); a.set_ylabel("Pairs per 0.25\u2033 bin")
    if title: a.set_title("Genuine stellar matches lie within 1\u2033", loc="left")
    a.legend(frameon=False, loc="upper right")


def panel_cmd(b, D, title=True):
    L, Ts = D["L"], D["Ts"]
    b.hexbin(L.bp_rp, L.MG, gridsize=120, bins="log", cmap="Greys", mincnt=1, linewidths=0, extent=(-0.2, 5, -1, 17))
    for k, cc in COLG.items():
        m = Ts.grp == k
        if m.any(): b.scatter(Ts.bp_rp[m], Ts.MG[m], s=16, c=cc, edgecolor="k", lw=0.3, label=SHORT[k], zorder=3)
    b.invert_yaxis(); b.set_xlabel("Gaia $G_{BP}-G_{RP}$ (mag)"); b.set_ylabel("$M_G$ (mag)")
    if title: b.set_title("Detections are mostly active binaries", loc="left")
    b.legend(frameon=False, fontsize=6, loc="upper right", handletextpad=0.3)


def panel_limits(c, D, title=True):
    L, Ts = D["L"], D["Ts"]
    far = L.dist_pc >= 10
    c.hexbin(L.dist_pc[far], L.Lnu_lim_WHz[far], xscale="log", yscale="log", gridsize=60, bins="log", cmap="Greys", mincnt=1, linewidths=0, extent=(1, 2, 5.5, 10.3))
    c.scatter(L.dist_pc[~far], L.Lnu_lim_WHz[~far], s=2, c="0.45", lw=0, zorder=2)
    for k, cc in COLG.items():
        m = Ts.grp == k
        if m.any(): c.scatter(Ts.dist_pc[m], Ts.Lnu_R_WHz[m], s=16, c=cc, edgecolor="k", lw=0.3, zorder=3, label=None if title else k)
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(1.5, 120); c.set_ylim(2e5, 1e10)
    c.set_xlabel("Distance (pc)"); c.set_ylabel("$L_\\nu$ at 3 GHz (W Hz$^{-1}$)")
    if title: c.set_title("Per-star 95% limits (grey) and detections", loc="left")
    c2 = c.secondary_yaxis("right", functions=(lambda y: y * C.BAND_HZ, lambda y: y / C.BAND_HZ)); c2.set_ylabel("EIRP, 2 GHz-wide (W)")
    r0 = L.nsmallest(1, "dist_pc").iloc[0]
    c.annotate("Barnard's star", (r0.dist_pc, r0.Lnu_lim_WHz), xytext=(4, 3e5), fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5), va="center")


def panel_population(d, D, title=True):
    pop = D["pop"]
    d.loglog(pop.Lnu_star_WHz, clip1(pop.f_UL95), color="k", lw=1.4)
    d.set_xlabel("Transmitter $L_\\nu$ threshold (W Hz$^{-1}$)"); d.set_ylabel("Fraction of stars hosting one (95% UL)")
    d2 = d.secondary_xaxis("top", functions=(lambda x: x * C.BAND_HZ, lambda x: x / C.BAND_HZ)); d2.set_xlabel("EIRP, 2 GHz-wide (W)")
    d.set_xlim(5e5, 3e10); d.set_ylim(1e-5, 1)
    i9 = np.argmin(np.abs(pop.Lnu_star_WHz - 1e9))
    d.text(0.97, 0.92, f"< 1 in {1/pop.f_UL95[i9]:,.0f} stars\nabove 10$^{{9}}$ W Hz$^{{-1}}$", transform=d.transAxes, ha="right", va="top", fontsize=7)


PANELS = [("vlass_separation_histogram", panel_separation), ("vlass_detections_cmd", panel_cmd),
          ("vlass_perstar_limits", panel_limits), ("vlass_population_limit", panel_population)]


def summary(plt):
    """Combined 2x2 overview (results/) plus one stand-alone single-column figure per panel (used in the paper)."""
    D = load_summary()
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for ax, (_, fn), l in zip(axs.ravel(), PANELS, "abcd"): fn(ax, D); C.panel_letter(ax, l)
    fig.tight_layout(); fig.savefig(os.path.join(C.RES, "vlass_technosig_summary.png"), dpi=300); fig.savefig(os.path.join(C.RES, "vlass_technosig_summary.pdf"), dpi=300); plt.close(fig)
    for name, fn in PANELS:
        fig, ax = plt.subplots(figsize=(3.5, 3.0) if fn is not panel_limits else (3.6, 3.0))
        fn(ax, D, title=False); fig.tight_layout()
        fig.savefig(os.path.join(C.RES, name + ".png"), dpi=300); fig.savefig(os.path.join(C.RES, name + ".pdf"), dpi=300); plt.close(fig)


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


def multiepoch(plt):
    """Detected stars across VLASS epochs 1.1-4.1 (from 14_multiepoch_detections.py)."""
    fv = os.path.join(C.RES, "vlass_detections_variability.csv")
    if not os.path.exists(fv): print("multiepoch: run 14_multiepoch_detections.py first"); return
    S = pd.read_csv(fv); M = pd.read_csv(os.path.join(C.RES, "vlass_detections_multiepoch.csv"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
    col = {"co-moving": "#3b6ea8", "inconclusive": "0.55"}
    t = S[S.motion.isin(list(col))]
    for k, cc in col.items():
        m = t.motion == k
        a.scatter(t.lever_arm_arcsec[m], np.clip(t.dchi2_stationary_minus_comove[m], -20, None), s=16, c=cc, edgecolor="k", lw=0.3, label=k, zorder=3)
    bg = S[S["cls"].str.startswith("background")]
    a.scatter(bg.lever_arm_arcsec, bg.dchi2_stationary_minus_comove, s=40, marker="*", c="#c0392b", edgecolor="k", lw=0.3, zorder=4, label="LSPM J1943+5337 (not coincident)")
    a.axhline(9, color="k", lw=0.6, ls="--"); a.axhline(-9, color="k", lw=0.6, ls="--"); a.axhline(0, color="0.7", lw=0.5)
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=10)
    a.xaxis.set_minor_formatter(plt.NullFormatter())
    a.set_xlabel("Proper-motion lever arm (arcsec)"); a.set_ylabel(r"$\chi^2_{\rm stationary}-\chi^2_{\rm co}$ (co-moving)")
    a.legend(frameon=False, fontsize=6, loc="upper left")
    for sid, g in M.groupby("source_id"):
        f = g.peak_corr_mJy.where(g.detected, g.forced_peak_corr_mJy); mu = np.mean(f)
        if not np.isfinite(mu) or mu <= 0: continue
        isbg = S.loc[S.source_id == sid, "cls"].astype(str).str.startswith("background").any()
        b.plot(g.epoch, f / mu, "-", color="#c0392b" if isbg else "0.6", lw=1.4 if isbg else 0.5, zorder=3 if isbg else 1)
        b.scatter(g.epoch[g.detected], (f / mu)[g.detected], s=6, c="#c0392b" if isbg else "k", lw=0, zorder=4 if isbg else 2)
    b.axhline(1, color="k", lw=0.5, ls=":"); b.set_xlabel("Epoch (yr)"); b.set_ylabel(r"$S\,/\,\langle S\rangle$")
    C.panel_letter(a, "a"); C.panel_letter(b, "b"); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(C.RES, f"vlass_detections_multiepoch.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plt = C.set_plot_style(); summary(plt); stacking(plt); multiepoch(plt); print("figures written to", C.RES)
