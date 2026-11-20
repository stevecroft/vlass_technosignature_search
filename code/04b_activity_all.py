"""04b - Activity evidence and technosignature eligibility for every sample star.

Two separate quantities are written for every star:

1. Activity evidence (descriptive; used for the astrophysical stack splits and the activity-by-distance table): a 2RXS
   counterpart within 40", an eRASS1 counterpart within 20" (each matched at the survey epoch, 1990.8 and 2020.3, with
   Gaia proper motions), or a SIMBAD X-ray/flare/BY Dra/RS CVn/emission-line/young/CV/magnetic-CP type (07.evidence).
   Its presence depends on survey depth and on how well a star has been studied, so it flags most of the nearest stars.

2. Technosignature eligibility (used for k, b and N_eff in 07, 12, 18, 19, 21). A detection at an eligible star counts as a
   possible transmitter; a non-eligible star gets p_i = 0.
   * AFGK stars (BP-RP < 1.84) are eligible unless they belong to a class of known radio emitters or contaminants
     (SIMBAD CV, RS CVn, young, magnetic CP; Gaia DR3 eclipsing binary, which includes Algols) or have a saturated or
     near-saturated corona, log L_X/L_bol >= LXLBOL_MAX = -4. The ROSAT and eRASS1 surveys reach this activity level
     for AFGK stars throughout 100 pc, so the cut does not depend on distance, unlike the presence of an X-ray detection.
     L_X uses the eRASS1 0.2-2.3 keV flux where available, else the 2RXS count rate x 6e-12 erg cm^-2 per count (07);
     L_bol uses M_G with a bolometric correction of zero (|BC_G| < 0.5 mag for these stars, ample for a threshold
     applied at the 1-dex level).
   * AFGK stars with Gaia evidence of multiplicity (RUWE > MULT_RUWE = 1.4, a DR3 non-single-star solution, or more
     than 0.5 mag above the main sequence, dMG < MULT_DMG = -0.5; column mult) are not eligible at all luminosities:
     an unresolved companion may be a flaring M dwarf. Like M dwarfs they are eligible above L_ENV (column env).
   * M dwarfs (BP-RP >= 1.84) are not eligible: their flares and coherent emission cannot be told from a transmitter by
     continuum flux. They are eligible only for transmitter luminosities above L_ENV = 1e10 W/Hz (common), above the most
     luminous incoherent radio flares from M dwarfs and above every natural stellar source in the survey.
   * env = (M dwarf, or AFGK with multiplicity evidence and log L_X/L_bol < -4) and not in the classes above. Stars in
     the classes, and AFGK stars with log L_X/L_bol >= -4, are excluded at all luminosities.

Inputs : data/derived/ms_sample_vlass.parquet, HEASARC TAP (cached), SIMBAD TAP (cached), Gaia archive (cached)
Outputs: data/raw/xray_{2rxs,erass1}_flux.parquet (full catalogs: position and flux), data/raw/simbad_otypes_all.parquet,
         data/raw/gaia_dr3_eclipsing_binaries_100pc.parquet,
         data/derived/activity_all.parquet (source_id, rxs_r, era_r, fX, log_lx_lbol, main_id, otypes, activity_evidence,
         active, excl_class, excl_reason, afgk, mult, eligible, env)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import time, importlib
import numpy as np, pandas as pd, pyvo
from scipy.spatial import cKDTree

P07 = importlib.import_module("07_classify_and_limits")
HEASARC_TAP = "https://heasarc.gsfc.nasa.gov/xamin/vo/tap"   # same catalogs as VizieR J/A+A/588/A103 and J/A+A/682/A34
# HEASARC table, flux column, match radius (arcsec), survey epoch (Julian year)
CATS = {"2rxs": ("rass2rxs", "count_rate", 40.0, 1990.8), "erass1": ("erass1main", "b1_flux", 20.0, 2020.3)}
ECF_2RXS = 6e-12                       # erg/cm^2/s per ct/s, coronal spectrum, 0.1-2.4 keV (as 07)
LSUN_CGS, MG_SUN = 3.828e33, 4.67
EXCL_OTYPES = {"CV*": "CV", "RS*": "RS CVn", "TT*": "young", "Y*O": "young", "a2*": "magnetic CP"}


def xyz(ra, de):
    a, d = np.radians(ra), np.radians(de)
    return np.c_[np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]


def _strip(svc, tb, col, lo, hi):
    """Rows with lo <= dec < hi; strips that reach the service's 100,000-row limit are split in two."""
    for k in range(4):
        try:
            t = svc.run_sync(f"SELECT ra, dec, {col} FROM {tb} WHERE dec >= {lo} AND dec < {hi}", maxrec=100_000).to_table(); break
        except Exception as e:
            print("retry", tb, lo, str(e)[:100], flush=True); time.sleep(30)
    else:
        raise RuntimeError(f"HEASARC query failed: {tb} {lo}")
    if len(t) >= 100_000:
        mid = 0.5 * (lo + hi); return _strip(svc, tb, col, lo, mid) + _strip(svc, tb, col, mid, hi)
    return [t.to_pandas().astype(float)]


def fetch_catalog(name):
    tb, col, _, _ = CATS[name]; f = os.path.join(C.RAW, f"xray_{name}_flux.parquet")
    if os.path.exists(f): return pd.read_parquet(f)
    svc = pyvo.dal.TAPService(HEASARC_TAP)
    n_exp = int(svc.run_sync(f"SELECT COUNT(*) AS n FROM {tb}").to_table()["n"][0])
    parts = []
    for lo in range(-90, 90, 10): parts += _strip(svc, tb, col, lo, lo + 10)
    out = pd.concat(parts, ignore_index=True); assert len(out) == n_exp, f"{name}: fetched {len(out)} of {n_exp}"
    out.to_parquet(f, index=False); return out


def match(sv, cat, r_arcsec, epoch):
    """Nearest catalog source within r_arcsec of each star propagated to the survey epoch: (separation, index or -1)."""
    ra, de = C.propagate(sv.ra.values, sv.dec.values, sv.pmra.values, sv.pmdec.values, epoch)
    d, j = cKDTree(xyz(cat.ra.values, cat.dec.values)).query(xyz(ra, de), k=1, distance_upper_bound=2 * np.sin(np.radians(r_arcsec / 3600) / 2))
    ok = np.isfinite(d)
    return np.where(ok, 2 * np.degrees(np.arcsin(np.clip(d / 2, 0, 1))) * 3600, np.nan), np.where(ok, j, -1)


ACT_OTYPES = ("X", "Er*", "Fl*", "BY*", "RS*", "TT*", "Y*O", "CV*", "a2*", "Em*")   # the activity types of 07.evidence


def simbad_otypes():
    """SIMBAD activity object types for every object with a Gaia DR3 identifier and parallax > 9 mas (cached)."""
    f = os.path.join(C.RAW, "simbad_otypes_all.parquet")
    if os.path.exists(f): return pd.read_parquet(f)
    lst = ",".join(f"'{t}'" for t in ACT_OTYPES)
    q = ("SELECT i.id AS gid, b.main_id, o.otype FROM otypes AS o JOIN basic AS b ON b.oid = o.oidref "
         f"JOIN ident AS i ON i.oidref = o.oidref WHERE o.otype IN ({lst}) AND b.plx_value > 9 AND i.id LIKE 'Gaia DR3 %'")
    for k in range(4):
        try:
            r = pyvo.dal.TAPService(C.SIMBAD_TAP).run_sync(q, maxrec=2_000_000).to_table().to_pandas(); break
        except Exception as e:
            print("retry", str(e)[:120], flush=True); time.sleep(30)
    else:
        raise RuntimeError("SIMBAD activity-type query failed")
    r["source_id"] = r.gid.str.replace("Gaia DR3 ", "").astype("int64")
    g = r.groupby("source_id").agg(main_id=("main_id", "first"), otypes=("otype", lambda x: "|".join(sorted(set(map(str, x))))))
    g = g.reset_index(); g.to_parquet(f, index=False)
    return g


def gaia_eclipsing():
    """Gaia DR3 eclipsing binaries (vari_eclipsing_binary) with parallax > 9.5 mas, in declination strips (cached)."""
    f = os.path.join(C.RAW, "gaia_dr3_eclipsing_binaries_100pc.parquet")
    if os.path.exists(f): return pd.read_parquet(f)
    svc = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")
    def run(lo, hi):
        q = ("SELECT v.source_id, v.global_ranking FROM gaiadr3.gaia_source AS g JOIN gaiadr3.vari_eclipsing_binary AS v "
             f"ON g.source_id = v.source_id WHERE g.parallax > 9.5 AND g.dec >= {lo} AND g.dec < {hi}")
        for k in range(3):
            try: return svc.run_sync(q, maxrec=100000).to_table().to_pandas()
            except Exception as e: print("retry", lo, hi, str(e)[:60], flush=True); time.sleep(5)
        if hi - lo > 1.5: m = 0.5 * (lo + hi); return pd.concat([run(lo, m), run(m, hi)])
        raise RuntimeError(f"Gaia query failed {lo} {hi}")
    r = pd.concat([run(lo, lo + 5) for lo in range(-45, 90, 5)]).drop_duplicates("source_id")
    r.to_parquet(f, index=False); return r


if __name__ == "__main__":
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"),
                         columns=["source_id", "ra", "dec", "pmra", "pmdec", "dist_pc", "bp_rp", "phot_g_mean_mag", "dMG", "ruwe", "non_single_star"])
    A = sv.copy(); val = {}
    for name, (tb, col, r, ep) in CATS.items():
        cat = fetch_catalog(name); sep, j = match(sv, cat, r, ep)
        A[f"{'rxs' if name == '2rxs' else 'era'}_r"] = sep; val[name] = np.where(j >= 0, cat[col].values[np.clip(j, 0, None)], np.nan)
        print(name, len(cat), "sources;", int(np.isfinite(sep).sum()), "sample stars matched", flush=True)
    A["fX"] = np.where(np.isfinite(val["erass1"]) & (val["erass1"] > 0), val["erass1"], val["2rxs"] * ECF_2RXS)
    Lbol = LSUN_CGS * 10 ** (-0.4 * (A.phot_g_mean_mag - 5 * np.log10(A.dist_pc / 10) - MG_SUN))
    A["log_lx_lbol"] = np.log10(A.fX * 4 * np.pi * (A.dist_pc * C.PC_M * 100) ** 2 / Lbol)
    sb = simbad_otypes()
    A = A.merge(sb, on="source_id", how="left")
    A["LX"] = np.where(A.rxs_r.notna() | A.era_r.notna(), 1.0, np.nan)
    ev = A.apply(P07.evidence, axis=1)
    A["activity_evidence"] = [e[1] for e in ev]; A["active"] = A.activity_evidence.str.len() > 0
    # ---- technosignature eligibility
    eb = set(gaia_eclipsing().source_id)
    ot = A.otypes.fillna("").str.split("|")
    reason = ot.apply(lambda o: "; ".join(sorted({EXCL_OTYPES[t] for t in o if t in EXCL_OTYPES})))
    reason = np.where(A.source_id.isin(eb), reason.str.cat(pd.Series("eclipsing binary", index=A.index), sep="; ").str.strip("; "), reason)
    A["excl_class"] = pd.Series(reason, index=A.index).str.len() > 0
    hot = (A.log_lx_lbol >= C.LXLBOL_MAX).fillna(False)
    A["afgk"] = A.bp_rp < C.BPRP_SPLIT
    A["excl_reason"] = [", ".join([x for x in (rs, f"log Lx/Lbol = {lx:.1f}" if h else "") if x]) for rs, lx, h in zip(reason, A.log_lx_lbol, hot)]
    A["mult"] = (A.ruwe > C.MULT_RUWE) | (A.non_single_star > 0) | (A.dMG < C.MULT_DMG)
    A["eligible"] = A.afgk & ~A.excl_class & ~hot & ~A.mult
    A["env"] = ~A.excl_class & (~A.afgk | (A.mult & ~hot))
    A[["source_id", "rxs_r", "era_r", "fX", "log_lx_lbol", "main_id", "otypes", "activity_evidence", "active", "excl_class", "excl_reason", "afgk", "mult", "eligible", "env"]].to_parquet(
        os.path.join(C.DER, "activity_all.parquet"), index=False, compression="zstd")
    print("stars:", len(A), "| with activity evidence:", int(A.active.sum()), f"({100 * A.active.mean():.2f}%)",
          "| X-ray:", int(A.LX.notna().sum()), "| SIMBAD type only:", int((A.active & A.LX.isna()).sum()))
    print("AFGK:", int(A.afgk.sum()), "| eligible:", int(A.eligible.sum()), "| excluded by class:", int((A.afgk & A.excl_class).sum()),
          "| by log Lx/Lbol >= %.1f:" % C.LXLBOL_MAX, int((A.afgk & ~A.excl_class & hot).sum()),
          "| with multiplicity evidence (else eligible):", int((A.afgk & ~A.excl_class & ~hot & A.mult).sum()),
          "| eligible above L_ENV:", int(A.env.sum()), "(M:", int((A.env & ~A.afgk).sum()), ")")
