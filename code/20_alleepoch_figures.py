"""20 - Figures for the all-epoch analysis (17-19).

Outputs (results/, PNG and PDF):
  alleepoch_population_limit : 95% upper limit on the fraction of stars hosting a transmitter vs L_nu (steady, all
                               epochs; Epoch-1 catalog analysis for comparison; per-snapshot duty-cycle limit f*delta;
                               mean- and median-stack limits by spectral class)
  alleepoch_perstar_limits   : steady per-star limits vs distance, with secure associations and candidates
  alleepoch_stacks           : (a) mean L_nu of clean stars per shell, (b) median S/sigma shift including detected
                               stars, (c) injection-recovery detection fraction vs injected S/N
Inputs : results/alleepoch_*.csv/parquet, results/vlass_population_limit_curve.csv, results/vlass_star_matches_classified.csv,
         data/derived/forced_calibration.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json
import numpy as np, pandas as pd
from scipy.stats import norm

COLT = {"AFGK": "#e0a030", "M": "#3b6ea8"}
MIDS = {"0-10": 5, "10-20": 15, "20-40": 30, "40-60": 50, "60-80": 70, "80-100": 90}


def clip1(f):
    f = np.asarray(f, float).copy(); f[~(f <= 1)] = np.nan; return f


def save(fig, name):
    for ext in ("png", "pdf"): fig.savefig(os.path.join(C.RES, f"{name}.{ext}"), dpi=300)


if __name__ == "__main__":
    plt = C.set_plot_style()
    POP = pd.read_csv(os.path.join(C.RES, "alleepoch_population_limits.csv"))
    SL = pd.read_csv(os.path.join(C.RES, "alleepoch_stack_limits.csv")); ST = pd.read_csv(os.path.join(C.RES, "alleepoch_stacks.csv"))
    old = pd.read_csv(os.path.join(C.RES, "vlass_population_limit_curve.csv"))
    P = pd.read_parquet(os.path.join(C.RES, "alleepoch_perstar_limits.parquet"))
    CA = pd.read_csv(os.path.join(C.RES, "alleepoch_candidates.csv"))
    T = pd.read_csv(os.path.join(C.RES, "vlass_star_matches_classified.csv")); Ts = T[T.secure]
    cal = json.load(open(os.path.join(C.DER, "forced_calibration.json")))
    tags = sorted(k for k in cal if k.startswith("ep"))

    # ---- population limit
    fig, d = plt.subplots(figsize=(3.6, 3.9))
    d.loglog(POP.L_WHz, clip1(POP.f95_b0), color="k", lw=1.6, label="All epochs, steady (no background subtracted)")
    d.loglog(POP.L_WHz, clip1(POP.f95), color="k", lw=0.8, ls="-.", label="All epochs, steady (control background)")
    d.loglog(old.Lnu_star_WHz, clip1(old.f_UL95), color="0.55", lw=1.0, ls="--", label="Epoch 1 catalog")
    d.loglog(POP.L_WHz, clip1(POP.fdelta95_b0), color="k", lw=1.0, ls=":", label="$f\\delta$, rare-on limit ($\\delta\\to0$)")
    for cl in ["AFGK"]:                      # stack limits use the eligible (AFGK) stars only (19)
        s = SL[SL.cls == cl]
        ind = np.interp(np.log10(s.L_WHz), np.log10(POP.L_WHz), POP.f95_b0)     # draw stacks only where they are tighter
        fm = clip1(s.f95_mean); fm[~(fm < ind)] = np.nan
        d.loglog(s.L_WHz, fm, color=COLT[cl], lw=1.3, label=f"{cl} mean stack (eligible stars)")
        m = s[s.f95_median.notna()]
        d.scatter(m.L_WHz, m.f95_median, s=14, marker="s", facecolor="white", edgecolor=COLT[cl], lw=1, zorder=3,
                  label=f"{cl} median stack (eligible stars)")
    d.axvline(C.L_ENV, color="0.6", lw=0.6, ls=(0, (2, 2)), zorder=0)
    d.text(C.L_ENV / 1.15, 3e-1, "$L_{\\rm env}$: M dwarfs and\nAFGK multiples\neligible above", fontsize=5, ha="right", va="top", color="0.35")
    d.set_xlim(1e5, 3e10); d.set_ylim(1e-5, 1)
    d.set_xlabel("Transmitter $L_\\nu$ at 3 GHz (W Hz$^{-1}$)"); d.set_ylabel("Fraction of stars hosting one (95% UL)")
    d2 = d.secondary_xaxis("top", functions=(lambda x: x * C.BAND_HZ, lambda x: x / C.BAND_HZ)); d2.set_xlabel("EIRP, 2 GHz-wide (W)")
    d.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(); save(fig, "alleepoch_population_limit"); plt.close(fig)

    # ---- per-star steady limits
    fig, c = plt.subplots(figsize=(3.6, 3.0))
    far = P.dist_pc >= 10
    c.hexbin(P.dist_pc[far], P.L_lim_steady[far], xscale="log", yscale="log", gridsize=60, bins="log", cmap="Greys", mincnt=1,
             linewidths=0, extent=(1, 2, 5.0, 10.5))
    c.scatter(P.dist_pc[~far], P.L_lim_steady[~far], s=2, c="0.45", lw=0, zorder=2)
    c.scatter(Ts.dist_pc, Ts.Lnu_R_WHz, s=12, c="#3b6ea8", edgecolor="k", lw=0.3, zorder=3, label="Secure associations (Epoch 1)")
    for ex, col, lab in [(True, "#e0a030", "Forced detections, not counted (ineligible host)"), (False, "#c0392b", "Forced detections at eligible stars (counted)")]:
        x = CA[CA.explained == ex]
        c.scatter(x.dist_pc, C.lum(x.S_coadd_mJy.clip(lower=1e-3), x.dist_pc), s=14, marker="D" if ex else "x", c=col, lw=0.8 if not ex else 0.3,
                  edgecolor="k" if ex else None, zorder=4, label=lab)
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(1.5, 120); c.set_ylim(1e5, 3e10)
    c.axhline(C.L_ENV, color="0.4", lw=0.7, ls="--", zorder=1); c.text(1.7, C.L_ENV * 1.25, "$L_{\\rm env}$", fontsize=6, color="0.3")
    c.set_xlabel("Distance (pc)"); c.set_ylabel("$L_\\nu$ at 3 GHz (W Hz$^{-1}$)")
    c2 = c.secondary_yaxis("right", functions=(lambda y: y * C.BAND_HZ, lambda y: y / C.BAND_HZ)); c2.set_ylabel("EIRP, 2 GHz-wide (W)")
    c.legend(frameon=False, fontsize=5.5, loc="lower right"); fig.tight_layout(); save(fig, "alleepoch_perstar_limits"); plt.close(fig)

    # ---- stacks
    fig = plt.figure(figsize=(7.2, 5.4)); gs = fig.add_gridspec(2, 2)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1]); e = fig.add_subplot(gs[1, :])
    for cl, dx in [("AFGK", -1.5), ("M", 1.5)]:
        for sc, alpha, ms in [(t, 0.35, 10) for t in tags] + [("all", 1.0, 26)]:
            s = ST[(ST.cls == cl) & (ST.scope == sc) & (ST["pop"] == "all")]
            if not len(s): continue
            x = np.array([MIDS[k] for k in s.shell]) + dx + (0 if sc == "all" else 0.4 * (tags.index(sc) - 1.5))
            a.errorbar(x, s.meanL / 1e6, yerr=s.meanL_err / 1e6, fmt="o", ms=np.sqrt(ms), color=COLT[cl], alpha=alpha, lw=0.8, capsize=1.5,
                       label=f"{cl}, all clean stars" if sc == "all" else None)
            b.errorbar(x, s.dmed_z, yerr=s.dmed_z_err, fmt="s", ms=np.sqrt(ms), color=COLT[cl], alpha=alpha, lw=0.8, capsize=1.5,
                       label=f"{cl}" if sc == "all" else None)
        for pp, mk, off in [("active", "^", 0.7), ("inactive", "v", -0.7)]:      # all-epoch coadd split by activity evidence
            s = ST[(ST.cls == cl) & (ST.scope == "all") & (ST["pop"] == pp)]
            x = np.array([MIDS[k] for k in s.shell]) + dx + off
            a.errorbar(x, s.meanL / 1e6, yerr=s.meanL_err / 1e6, fmt=mk, ms=3.5, mfc="white", color=COLT[cl], lw=0.7, capsize=1.2,
                       label=f"{cl}, {'with' if pp == 'active' else 'without'} activity evidence")
    for ax, yl in [(a, "Mean $L_\\nu$ of clean stars (10$^6$ W Hz$^{-1}$)"), (b, "Median $S/\\sigma$ shift vs controls")]:
        ax.axhline(0, color="0.6", lw=0.6); ax.set_xticks(list(MIDS.values())); ax.set_xticklabels([k for k in MIDS], fontsize=6)
        ax.set_xlabel("Distance shell (pc)"); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=6)
    a.set_title("Inverse-variance mean (faint stars only)", loc="left"); b.set_title("Median, including detected stars", loc="left")
    lv = np.linspace(1.5, 15.5, 200); e.plot(lv, norm.cdf(lv - 5), color="k", lw=1, label="Model $\\Phi(S/\\sigma - 5)$")
    for i, t in enumerate(tags):
        tb = pd.DataFrame(cal[t]["inject"]["table"]); e.plot(tb.level, tb.frac_det, "o", ms=4, label=f"Injected, {t}", color=plt.cm.viridis(i / max(len(tags) - 1, 1)))
    e.set_xlabel("Injected peak S/N (adopted calibration)"); e.set_ylabel("Fraction recovered at $\\geq5\\sigma$")
    e.set_title("Injection\u2013recovery through the forced-photometry pipeline", loc="left"); e.legend(frameon=False, fontsize=6, loc="lower right")
    for ax, l in zip([a, b, e], "abc"): C.panel_letter(ax, l)
    fig.tight_layout(); save(fig, "alleepoch_stacks"); plt.close(fig)
    print("figures written")
