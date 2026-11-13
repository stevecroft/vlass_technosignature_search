"""03 - Main-sequence sample, VLASS subtile assignment, epoch propagation, cross-match, offset controls,
and positional reliability (Rayleigh + uniform background, background density fixed from offsets).

Inputs : data/raw/gaia_dr3_plx10.parquet, data/raw/vlass_e1ql_components.parquet, data/raw/vlass_e1ql_subtiles.parquet
Outputs: data/derived/ms_sample_vlass.parquet   (MS stars in the VLASS footprint; expected 210,191)
         data/derived/matches_10arcsec.parquet  (all star-component pairs < 10", with reliability)
         data/derived/offset_pairs.parquet      (pairs from the 10 offset realisations)
         data/derived/ms_ridge.csv, data/derived/reliability_fit.json
Expected: sigma_sys = 0.231"/axis, N_true = 48.2, 50 pairs with rel > 0.5 (2.49 expected false)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import os, json
import numpy as np, pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u


def main_sequence(g):
    g = g[np.isfinite(g.bp_rp)].copy()
    g["MG"] = g.phot_g_mean_mag + 5 * np.log10(g.parallax / 100)
    g["dist_pc"] = 1000 / g.parallax
    bins = np.arange(-0.2, 5.01, 0.1); cen = 0.5 * (bins[1:] + bins[:-1])
    ridge = np.full(len(cen), np.nan)
    sel = (g.MG < 4.5 * g.bp_rp + 6.5)                    # coarse cut: excludes white dwarfs
    sel &= ~((g.MG < 4.0) & (g.bp_rp > 0.9))              # coarse cut: excludes giants/subgiants
    ok = np.isfinite(ridge)
    for it in range(3):                                   # iterative median ridge in 0.1-mag colour bins
        for i in range(len(cen)):
            m = sel & (g.bp_rp >= bins[i]) & (g.bp_rp < bins[i + 1])
            if m.sum() > 20:
                ridge[i] = np.median(g.MG[m] if it == 0 else
                                     g.MG[m & (np.abs(g.MG - np.interp(g.bp_rp, cen[ok], ridge[ok])) < 1.0)])
        ok = np.isfinite(ridge)
    r_s = ridge.copy(); r_s[ok] = median_filter(ridge[ok], 3, mode="nearest")
    g["dMG"] = g.MG - np.interp(g.bp_rp, cen[ok], r_s[ok])
    g["is_MS"] = (g.dMG > -1.5) & (g.dMG < 1.0) & (g.bp_rp > cen[ok].min()) & (g.bp_rp < cen[ok].max())
    pd.DataFrame({"bp_rp": cen[ok], "MG_ridge": r_s[ok]}).to_csv(os.path.join(C.DER, "ms_ridge.csv"), index=False)
    return g


def pairs(ra, dec, vc, R):
    c = SkyCoord(ra * u.deg, dec * u.deg)
    icomp, istar, sep, _ = c.search_around_sky(vc, R)     # NB: first index refers to vc (the search-around set)
    return pd.DataFrame({"istar": istar, "icomp": icomp, "sep": sep.arcsec})


if __name__ == "__main__":
    # sort: the Gaia archive does not guarantee row order, and the offset realisations below are seeded per row
    g = pd.read_parquet(os.path.join(C.RAW, "gaia_dr3_plx10.parquet")).sort_values("source_id").reset_index(drop=True)
    v = pd.read_parquet(os.path.join(C.RAW, "vlass_e1ql_components.parquet"))
    st = pd.read_parquet(os.path.join(C.RAW, "vlass_e1ql_subtiles.parquet"))
    print("Gaia rows:", len(g), " VLASS components:", len(v), " subtiles:", len(st))

    g = main_sequence(g)
    print("finite BP-RP:", len(g), " main sequence:", int(g.is_MS.sum()))

    # ---- subtile assignment (nearest image centre), observation epoch, footprint
    s = g[g.is_MS & (g.dec > -41)].copy().reset_index(drop=True)
    stc = SkyCoord(st.Crval1.values * u.deg, st.Crval2.values * u.deg)
    idx, d2d, _ = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg).match_to_catalog_sky(stc)
    s["subtile"] = st.Subtile.values[idx]; s["sub_sep_deg"] = d2d.deg
    s["sub_rms"] = st.MeanIslrms.values[idx]; s["vepoch"] = st.Epoch.values[idx]
    s["obs_jyear"] = Time(list(st.Obs.astype(str).str.strip().values[idx]), format="isot").jyear
    s["in_vlass"] = s.sub_sep_deg < 0.75
    s["ra_ep"], s["dec_ep"] = C.propagate(s.ra, s.dec, s.pmra, s.pmdec, s.obs_jyear, plx=s.parallax)
    sv = s[s.in_vlass].reset_index(drop=True)
    print("MS, dec>-41:", len(s), " in VLASS footprint:", len(sv))

    # ---- cross-match (real + 10 offset realisations, 3' in random PA)
    vg = v[(v.DupFlag < 2) & (v.SCode != "E")].reset_index(drop=True)
    vc = SkyCoord(vg.RAJ2000.values * u.deg, vg.DEJ2000.values * u.deg)
    R = C.MATCH_R_ARCSEC * u.arcsec
    real = pairs(sv.ra_ep.values, sv.dec_ep.values, vc, R)
    rng = np.random.default_rng(1); offs = []
    for k in range(C.N_OFFSETS):
        pa = rng.uniform(0, 2 * np.pi, len(sv)); r = C.OFFSET_ARCMIN / 60
        m = pairs(sv.ra_ep.values + r * np.sin(pa) / np.cos(np.deg2rad(sv.dec_ep.values)),
                  sv.dec_ep.values + r * np.cos(pa), vc, R)
        m["k"] = k; offs.append(m)
    offs = pd.concat(offs, ignore_index=True)

    # ---- re-propagate each pair to the epoch of the subtile the component came from
    stdate = dict(zip(st.Subtile, Time(list(st.Obs.astype(str).str.strip()), format="isot").jyear))
    A = real.copy()
    A["sep_star_epoch"] = A.sep          # separation at the epoch of the star's own (nearest) subtile
    A["comp_epoch"] = vg.Subtile.map(stdate).values[A.icomp]
    ra2, de2 = C.propagate(sv.ra.values[A.istar], sv.dec.values[A.istar], sv.pmra.values[A.istar],
                           sv.pmdec.values[A.istar], A.comp_epoch.values,
                           plx=sv.parallax.values[A.istar])
    A["sep"] = SkyCoord(ra2 * u.deg, de2 * u.deg).separation(
        SkyCoord(vg.RAJ2000.values[A.icomp] * u.deg, vg.DEJ2000.values[A.icomp] * u.deg)).arcsec
    A["e_pos"] = np.hypot(vg.e_RAJ2000.values[A.icomp] * 3600 * np.cos(np.deg2rad(sv.dec_ep.values[A.istar])),
                          vg.e_DEJ2000.values[A.icomp] * 3600) / np.sqrt(2)          # per-axis catalogue error

    # ---- reliability: extended ML fit of Rayleigh (true) + uniform (background) separations
    Rmax = C.MATCH_R_ARCSEC; N = len(sv)
    lam = (len(offs) / C.N_OFFSETS) / (N * np.pi * Rmax ** 2)                    # background pairs / arcsec^2 / star
    def nll(par):
        ss, logf = par; f = 1 / (1 + np.exp(-logf)); sig2 = A.e_pos ** 2 + ss ** 2
        pt = f * (A.sep / sig2) * np.exp(-A.sep ** 2 / (2 * sig2)); pb = lam * 2 * np.pi * A.sep
        return -(np.log(pt + pb).sum() - f * N - lam * np.pi * Rmax ** 2 * N)
    res = minimize(nll, [0.3, np.log(60 / N)], method="Nelder-Mead")
    ss, logf = res.x; f = 1 / (1 + np.exp(-logf)); sig2 = A.e_pos ** 2 + ss ** 2
    pt = f * (A.sep / sig2) * np.exp(-A.sep ** 2 / (2 * sig2)); pb = lam * 2 * np.pi * A.sep
    A["rel"] = pt / (pt + pb)
    fit = dict(sigma_sys_arcsec=float(ss), N_true=float(f * N), N_bg_pairs=float(lam * np.pi * Rmax ** 2 * N),
               n_rel_gt_0p5=int((A.rel > 0.5).sum()), n_rel_gt_0p9=int((A.rel > 0.9).sum()),
               expected_false_rel_gt_0p5=float((1 - A.rel[A.rel > 0.5]).sum()))
    print(json.dumps(fit, indent=1))

    # attach component columns to the pair table and save
    keepc = ["RAJ2000", "DEJ2000", "e_RAJ2000", "e_DEJ2000", "Ftot", "e_Ftot", "Fpeak", "e_Fpeak", "Islrms", "DCMaj",
             "DCMin", "DCPA", "SCode", "QualFlag", "DupFlag", "MainSample", "NNdist", "PeakToRing", "NVSSdist", "FIRSTdist", "Subtile"]
    A = pd.concat([A.reset_index(drop=True), vg.iloc[A.icomp][keepc].reset_index(drop=True),
                   sv.iloc[A.istar][["source_id"]].reset_index(drop=True)], axis=1)
    sv.to_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"))
    A.to_parquet(os.path.join(C.DER, "matches_10arcsec.parquet"))
    offs.to_parquet(os.path.join(C.DER, "offset_pairs.parquet"))
    json.dump(fit, open(os.path.join(C.DER, "reliability_fit.json"), "w"), indent=1)
    counts = dict(n_gaia=int(len(pd.read_parquet(os.path.join(C.RAW, "gaia_dr3_plx10.parquet"), columns=["source_id"]))),
                  n_finite_bprp=int(len(g)), n_ms=int(g.is_MS.sum()), n_ms_dec41=int(len(s)), n_in_vlass=int(len(sv)),
                  n_components=int(len(v)), n_mainsample=int((v.MainSample == 1).sum()), n_used_components=int(len(vg)),
                  n_subtiles=int(len(st)), obs_epoch_min=float(s.obs_jyear.min()), obs_epoch_max=float(s.obs_jyear.max()),
                  real_pairs_lt1=int((A.sep_star_epoch < 1).sum()), offset_pairs_lt1_per_realisation=float((offs.sep < 1).sum() / C.N_OFFSETS))
    json.dump(counts, open(os.path.join(C.DER, "sample_counts.json"), "w"), indent=1); print(json.dumps(counts))
