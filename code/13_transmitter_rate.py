"""13 - Transmitter rate vs EIRP_min, VLASS limits compared with narrowband searches (cf. Painter et al. 2025, Fig. 15).

Transmitter rate TR = 1 / (N_star * nu_rel), with nu_rel = Delta_nu / nu_c = 2 GHz / 3 GHz for VLASS.

VLASS points
  * individual non-detections: a curve, TR(EIRP) = 1 / (N_eff(EIRP) * nu_rel), where N_eff is the number of stars whose
    individual limit lies below EIRP (results/vlass_population_limit_curve.csv); shown where N_eff >= 3 (f < 1).
  * stacks: f_95 is a fraction limit, not a count. We convert with N_equiv = 3 / f_95 (the Poisson 95% relation used for
    the individual limits), i.e. TR = f_95 / (3 nu_rel), and show it only where f_95 < 1 (distinct line style).
  VLASS EIRP is the total in-band power (L_nu x 2 GHz) and applies to a signal of any width; the narrowband EIRP_min
  values assume 1 Hz transmitters.

Literature points come from the user's seti_limits_py3 package (ET_power_law and the Choza et al. 2024 band modules,
used as in eirp.py from the forthcoming Jacobson-Bell et al. paper, with that paper's own points removed). Point
SETI_LIMITS_DIR at that folder; without it only the VLASS points are drawn.

Outputs: results/vlass_transmitter_rate_points.csv, results/transmitter_rate_vs_eirp.{pdf,png}
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import numpy as np, pandas as pd

NU_REL = 2.0 / 3.0
BRAT_EIRP = 4e5 * C.BAND_HZ        # Garrett (2026) Type I leakage benchmark, L_nu ~ 4e5 W/Hz, spread over 2 GHz


def vlass_points():
    """All-epoch limits (19) if present, else the Epoch-1 analysis (07, 10). With k unexplained candidates over a
    background b, TR = f_95 / (3 nu_rel), i.e. N_equiv = 3 / f_95; this reduces to 1 / (N_eff nu_rel) for k = b = 0."""
    rows = []
    fa = os.path.join(C.RES, "alleepoch_population_limits.csv")
    if os.path.exists(fa):
        pop = pd.read_csv(fa); sl = pd.read_csv(os.path.join(C.RES, "alleepoch_stack_limits.csv"))
        for _, r in pop[pop.f95_b0 < 1].iterrows():      # background-free limit (the headline value, Section 4.6)
            rows.append(("individual", r.L_WHz * C.BAND_HZ, 3.0 / r.f95_b0, r.f95_b0, r.f95_b0 / (3.0 * NU_REL)))
        for t in ["AFGK"]:                       # stack limits: eligible (AFGK) stars only (19)
            s = sl[(sl.cls == t) & (sl.f95_mean < 1)]
            for _, r in s.iterrows():
                rows.append((f"stack_{t}", r.L_WHz * C.BAND_HZ, 3.0 / r.f95_mean, r.f95_mean, r.f95_mean / (3.0 * NU_REL)))
    else:
        pop = pd.read_csv(os.path.join(C.RES, "vlass_population_limit_curve.csv"))
        fv = pd.read_csv(os.path.join(C.RES, "vlass_fraction_vs_power_limits.csv"))
        for _, r in pop[pop.N_eff >= 3].iterrows():
            rows.append(("individual", r.EIRP_2GHz_W, r.N_eff, np.nan, 1.0 / (r.N_eff * NU_REL)))
        for t, t2 in [("FGK", "AFGK"), ("M", "M")]:
            col = f"f_UL95_stack_{t}"
            for _, r in fv[fv[col] < 1].iterrows():
                rows.append((f"stack_{t2}", r.EIRP_2GHz_W, 3.0 / r[col], r[col], r[col] / (3.0 * NU_REL)))
    P = pd.DataFrame(rows, columns=["kind", "EIRP_W", "N_equiv", "f_UL95", "transmitter_rate"])
    P.to_csv(os.path.join(C.RES, "vlass_transmitter_rate_points.csv"), index=False)
    return P


def literature(plt):
    """Draw the literature points with the user's seti_limits_py3 package; returns legend handles and labels."""
    d = os.environ.get("SETI_LIMITS_DIR") or os.path.join(C.ROOT, "external", "seti_limits_py3")
    if not d or not os.path.isdir(d):
        print("SETI_LIMITS_DIR not set: plotting VLASS points only"); return [], []
    cwd = os.getcwd(); sys.path.insert(0, d); os.chdir(d)
    try:
        from color_coded_ET_power_law import ET_power_law
        from LBand_SETI_compare import lband_compare
        from SBand_SETI_compare import sband_compare
        from CBand_SETI_compare import cband_compare
        from XBand_SETI_compare import xband_compare
        dist = 2.92e7                                   # as in eirp.py
        bands = [f(dist, 33, save=False) for f in (lband_compare, sband_compare, cband_compare, xband_compare)]
        colors = ["dodgerblue", "firebrick", "seagreen", "peru"]; markers = [(6, 2, 0), (6, 2, 90), (6, 2, 180), (6, 2, 270)]
        for i, b in enumerate(bands):
            plt.plot(np.log10(b["EIRP"]), np.log10(1. / b["rarity"]), marker=markers[i], mew=5, ls="None", color=colors[i], ms=18, zorder=i)
        c24, = plt.plot(np.log10(np.max([b["EIRP"] for b in bands])), np.log10(1. / np.sum([b["rarity"] for b in bands])),
                        marker=markers[0], mew=5, ls="None", color="k", ms=18)
        h = ET_power_law(verbose=False)
    finally:
        os.chdir(cwd)
    keys = ["p1", "p2", "e", "gm", "h_a1", "s1", "pha", "hs", "m", "g", "uno_L", "uno_parkes_s"]
    labels = ["Price (2020 - Parkes)", "Price (2020 - GBT)", "Enriquez (2017)", "Gray & Mooley (2017)", "Harp (2016) All*",
              "Siemion (2013)", "Phoenix All*", "Horowitz & Sagan (1993)", "Tremblay (2020)", "Garrett & Siemion (2023)",
              "Uno (2023 - GBT)", "Uno (2023 - Parkes)"]
    return [c24] + [h[k] for k in keys], ["Choza (2024)"] + labels


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcdefaults()                                # ET_power_law is styled for a 15x10 inch figure (as in eirp.py)
    P = vlass_points()
    fig = plt.figure(figsize=(19, 10)); ax = plt.gca(); fs = 20
    hl, ll = literature(plt)
    ind = P[P.kind == "individual"].sort_values("EIRP_W")
    hv, = ax.plot(np.log10(ind.EIRP_W), np.log10(ind.transmitter_rate), color="k", lw=4, zorder=20)
    tr_ind = lambda e: np.interp(np.log10(e), np.log10(ind.EIRP_W), np.log10(ind.transmitter_rate), left=np.inf)
    hs = []
    for t, ls in [("AFGK", "--")]:
        s = P[P.kind == f"stack_{t}"].sort_values("EIRP_W")
        s = s[np.log10(s.transmitter_rate) < tr_ind(s.EIRP_W)]       # only where the stack is the tighter constraint
        h, = ax.plot(np.log10(s.EIRP_W), np.log10(s.transmitter_rate), color="k", lw=3, ls=ls, zorder=20); hs.append(h)
    ax.axvline(np.log10(BRAT_EIRP), color="0.35", lw=2, ls="-.", zorder=0)
    ax.text(np.log10(BRAT_EIRP), -8, "Type I BRaT leakage", va="center", ha="right", rotation=90, fontsize=18, color="0.35")
    ax.set_ylim(-15, 0); ax.set_xlabel(r"EIRP$_{\rm min}$ [W]", fontsize=fs)
    ax.set_ylabel(r"transmitter rate   $\left[\left(N_{\rm stars}\cdot\nu_{\rm rel}\right)^{-1}\right]$", fontsize=fs)
    ax.tick_params(labelsize=fs - 2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: rf"$10^{{{v:.0f}}}$"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: rf"$10^{{{v:.0f}}}$"))
    handles = [hv] + hs + hl
    labels = ["This work (VLASS)", "This work, stacks (eligible AFGK)"] + ll
    ax.legend(handles, labels, labelspacing=1.2, ncol=1, fancybox=True, framealpha=0.95, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=fs - 4, borderaxespad=0)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(os.path.join(C.RES, f"transmitter_rate_vs_eirp.{ext}"), dpi=150, bbox_inches="tight")
    print(P.groupby("kind").agg(n=("EIRP_W", "size"), EIRP_lo=("EIRP_W", "min"), EIRP_hi=("EIRP_W", "max"),
                                TR_lo=("transmitter_rate", "min"), TR_hi=("transmitter_rate", "max")).to_string())


if __name__ == "__main__":
    main()
