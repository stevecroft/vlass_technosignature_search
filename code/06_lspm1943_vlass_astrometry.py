"""06 - LSPM J1943+5337: multi-epoch VLASS astrometry (QL 1.2/2.2/3.2, SE 1.2/3.2) from CADC.
Each image is tied to a common frame with 5 bright, compact, isolated reference components; the target is then
compared with the Gaia track of the star (153 mas/yr north). CADC pixel cutouts (multi-SUB) replace the full
55/154 MB images used originally - the pixels are byte-identical.

Inputs : data/derived/matches_10arcsec.parquet, data/derived/ms_sample_vlass.parquet,
         data/raw/vlass_e1ql_components.parquet, data/raw/lspm1943/gaia_neighbours.csv (from 05)
Outputs: data/raw/lspm1943/vlass_<product>_{sci,rms}.fits (multi-extension cutouts: target + references)
         results/lspm1943_vlass_multiepoch_astrometry.csv, results/lspm1943_vlass_multiepoch_astrometry.png
         data/derived/lspm1943_astrometry.json
Expected: SE 3.2 radio-minus-star offset ~1.1" (N); chi2 = 85.7 for 3 dof for the 'at the star' model
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import io, json, requests
import numpy as np, pandas as pd, pyvo
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit

STAR, COMPANION = 2137474969751117312, 2137474969749746048
IMG = os.path.join(C.RAW, "lspm1943"); os.makedirs(IMG, exist_ok=True)
PRODUCTS = {("QL", "1.2"): "VLASS1.2.ql", ("QL", "2.2"): "VLASS2.2.ql", ("QL", "3.2"): "VLASS3.2.ql",
            ("SE", "1.2"): "VLASS1.2.se", ("SE", "3.2"): "VLASS3.2.se"}
INDEPENDENT = ["SE 1.2", "QL 2.2", "SE 3.2"]     # QL and SE of one epoch share visibilities


def g2d(xy, A, x0, y0, sx, sy, th, c):
    x, y = xy; ct, st = np.cos(th), np.sin(th)
    a = ct**2 / (2*sx**2) + st**2 / (2*sy**2); b = st*ct*(1/(2*sy**2) - 1/(2*sx**2)); cc = st**2/(2*sx**2) + ct**2/(2*sy**2)
    return A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + cc*(y-y0)**2)) + c


def fitsrc(d, w, coord, box_arcsec=10):
    ps = abs(w.proj_plane_pixel_scales()[1].to(u.arcsec).value)
    x0, y0 = [float(v) for v in w.world_to_pixel(coord)]; hb = int(box_arcsec / ps)
    xi, yi = int(round(x0)), int(round(y0))
    if yi - hb < 0 or xi - hb < 0: return None
    st = d[yi-hb:yi+hb+1, xi-hb:xi+hb+1].astype(float) * 1e3
    if st.shape != (2*hb+1, 2*hb+1) or not np.isfinite(st).all(): return None
    Y, X = np.mgrid[yi-hb:yi+hb+1, xi-hb:xi+hb+1]
    jy, jx = np.unravel_index(np.argmax(st * (np.hypot(X-x0, Y-y0) < 3/ps)), st.shape)
    try:
        p, cov = curve_fit(g2d, (X.ravel(), Y.ravel()), st.ravel(), p0=[st[jy, jx], X[jy, jx], Y[jy, jx], 1.2/ps, 1.0/ps, 0, 0], maxfev=20000)
    except Exception:
        return None
    e = np.sqrt(np.diag(cov)); c_ = w.pixel_to_world(p[1], p[2])
    return dict(ra=c_.ra.deg, dec=c_.dec.deg, peak=p[0], ex=e[1]*ps, ey=e[2]*ps, fwhm_a=2.355*abs(p[3])*ps, fwhm_b=2.355*abs(p[4])*ps)


def cadc_files(ra0, de0):
    q = f"""SELECT p.productID, a.uri, c.position_axis_function_dimension_naxis1 AS nx, c.position_axis_function_dimension_naxis2 AS ny,
    c.position_axis_function_refCoord_coord1_pix AS crpix1, c.position_axis_function_refCoord_coord2_pix AS crpix2,
    c.position_axis_function_refCoord_coord1_val AS crval1, c.position_axis_function_refCoord_coord2_val AS crval2,
    c.position_axis_function_cd11 AS cd11, c.position_axis_function_cd12 AS cd12, c.position_axis_function_cd21 AS cd21, c.position_axis_function_cd22 AS cd22
    FROM caom2.Observation AS o JOIN caom2.Plane AS p ON o.obsID=p.obsID JOIN caom2.Artifact AS a ON a.planeID=p.planeID
    JOIN caom2.Part AS pt ON pt.artifactID=a.artifactID JOIN caom2.Chunk AS c ON c.partID=pt.partID
    WHERE o.collection='VLASS' AND INTERSECTS(CIRCLE('ICRS',{ra0},{de0},0.001), p.position_bounds)=1"""
    t = pyvo.dal.TAPService(C.CADC_TAP).search(q).to_table().to_pandas()
    return t[t.uri.str.contains(r"tt0(\.rms)?\.subim\.fits$")]


def fetch_cutouts(files, coords, half_arcsec=14):
    """One multi-SUB request per file: box 0 = target, boxes 1..n = references."""
    out = {}
    for (kind, ep), pre in PRODUCTS.items():
        for tag, pat in [("sci", "tt0.subim.fits"), ("rms", "tt0.rms.subim.fits")]:
            f = os.path.join(IMG, f"vlass_{kind}{ep}_{tag}.fits"); out[(kind, ep, tag)] = f
            if os.path.exists(f): continue
            rw = files[files.uri.str.contains(pre.replace(".", r"\.")) & files.uri.str.endswith(pat)].iloc[0]
            w = WCS(naxis=2); w.wcs.ctype = ["RA---SIN", "DEC--SIN"]; w.wcs.crpix = [rw.crpix1, rw.crpix2]
            w.wcs.crval = [rw.crval1, rw.crval2]; w.wcs.cd = [[rw.cd11, rw.cd12], [rw.cd21, rw.cd22]]
            hb = int(np.ceil(half_arcsec / abs(rw.cd22 * 3600))); xs, ys = w.world_to_pixel(coords); subs = []
            for x, y in zip(xs, ys):
                X, Y = int(round(float(x))) + 1, int(round(float(y))) + 1
                subs.append(f"[{max(X-hb,1)}:{min(X+hb,int(rw.nx))},{max(Y-hb,1)}:{min(Y+hb,int(rw.ny))},*,*]")
            r = requests.get(C.CADC_DATA + rw.uri.split("/")[-1], params=[("SUB", s) for s in subs], timeout=300)
            assert r.status_code == 200 and r.content[:6] == b"SIMPLE", r.status_code
            open(f, "wb").write(r.content)
    return out


if __name__ == "__main__":
    A = pd.read_parquet(os.path.join(C.DER, "matches_10arcsec.parquet"))
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"))
    v = pd.read_parquet(os.path.join(C.RAW, "vlass_e1ql_components.parquet"))
    star = sv[sv.source_id == STAR].iloc[0]
    comp = pd.read_csv(os.path.join(IMG, "gaia_neighbours.csv")).query("source_id == @COMPANION").iloc[0]
    m = A[A.source_id == STAR].sort_values("sep").iloc[0]; RA0, DE0 = m.RAJ2000, m.DEJ2000
    vl = SkyCoord(RA0 * u.deg, DE0 * u.deg); cosd = np.cos(np.deg2rad(DE0))
    # reference components: bright, compact, isolated, within 0.45 deg
    vg = v[(v.DupFlag < 2) & (v.SCode != "E")]
    cr = vg[np.abs(vg.DEJ2000 - DE0) < 0.5].copy()
    cr["dist"] = SkyCoord(cr.RAJ2000.values * u.deg, cr.DEJ2000.values * u.deg).separation(vl).deg
    refs = cr[(cr.dist < 0.45) & (cr.Fpeak > 4) & (cr.DCMaj < 1.5) & (cr.NNdist > 30) & (cr.SCode == "S") & (cr.dist > 0.005)]
    print("reference components:", len(refs))
    coords = SkyCoord(np.r_[RA0, refs.RAJ2000.values] * u.deg, np.r_[DE0, refs.DEJ2000.values] * u.deg)
    paths = fetch_cutouts(cadc_files(RA0, DE0), coords)

    rows = []
    for (kind, ep), _ in PRODUCTS.items():
        sci = fits.open(paths[(kind, ep, "sci")]); rms = fits.open(paths[(kind, ep, "rms")])
        hs = [h for h in sci if h.data is not None]; hr = [h for h in rms if h.data is not None]
        h0 = hs[0].header; w0 = WCS(h0).celestial
        tgt = fitsrc(np.squeeze(hs[0].data), w0, vl)
        xr_, yr_ = [int(round(float(q))) for q in WCS(hr[0].header).celestial.world_to_pixel(vl)]
        rms_t = float(np.squeeze(hr[0].data)[yr_, xr_]) * 1e3
        rf = []
        for k, rr in enumerate(refs.itertuples(), start=1):
            f = fitsrc(np.squeeze(hs[k].data), WCS(hs[k].header).celestial, SkyCoord(rr.RAJ2000 * u.deg, rr.DEJ2000 * u.deg), box_arcsec=8)
            if f and f["peak"] > 3: f["ra0"], f["de0"] = rr.RAJ2000, rr.DEJ2000; rf.append(f)
        rf = pd.DataFrame(rf)
        dra = (rf.ra - rf.ra0) * 3600 * cosd; dde = (rf.dec - rf.de0) * 3600
        ora, ode = np.median(dra), np.median(dde)
        sra = 1.4826 * np.median(np.abs(dra - ora)) / np.sqrt(len(rf)); sde = 1.4826 * np.median(np.abs(dde - ode)) / np.sqrt(len(rf))
        epoch = Time(h0["DATE-OBS"][:19]).jyear
        bmaj, bmin = h0["BMAJ"] * 3600, h0["BMIN"] * 3600
        snr = tgt["peak"] / rms_t; eb = 0.5 * np.sqrt(bmaj * bmin) / snr
        tE = (tgt["ra"] - RA0) * 3600 * cosd - ora; tN = (tgt["dec"] - DE0) * 3600 - ode
        sa, sd = C.propagate(star.ra, star.dec, star.pmra, star.pmdec, epoch, plx=star.parallax)
        rows.append(dict(image=f"{kind} {ep}", epoch=round(epoch, 3), frame_off_E=round(ora, 2), frame_off_N=round(ode, 2), n_ref=len(rf),
                         ref_scatter_E=round(sra * np.sqrt(len(rf)), 2), ref_scatter_N=round(sde * np.sqrt(len(rf)), 2),
                         peak_mJy=round(tgt["peak"], 2), rms_mJy=round(rms_t, 3), SNR=round(snr, 1),
                         fwhm_a=round(tgt["fwhm_a"], 2), fwhm_b=round(tgt["fwhm_b"], 2), beam=f"{bmaj:.2f}x{bmin:.2f}",
                         tgt_E=round(tE, 2), tgt_N=round(tN, 2), err_E=round(np.sqrt(tgt["ex"]**2 + eb**2 + sra**2), 2),
                         err_N=round(np.sqrt(tgt["ey"]**2 + eb**2 + sde**2), 2),
                         star_E=round((sa - RA0) * 3600 * cosd, 2), star_N=round((sd - DE0) * 3600, 2)))
    ast = pd.DataFrame(rows); ast.to_csv(os.path.join(C.RES, "lspm1943_vlass_multiepoch_astrometry.csv"), index=False)
    print(ast[["image", "epoch", "peak_mJy", "SNR", "tgt_E", "tgt_N", "err_N", "star_N"]].to_string(index=False))

    # frame tie of the E1 catalogue to Gaia from the secure stellar matches
    sec = A[A.rel > 0.5]; s_ = sv.iloc[sec.istar]
    ra_s, de_s = C.propagate(s_.ra.values, s_.dec.values, s_.pmra.values, s_.pmdec.values, sec.comp_epoch.values, plx=s_.parallax.values)
    dE = (sec.RAJ2000.values - ra_s) * 3600 * np.cos(np.deg2rad(de_s)); dN = (sec.DEJ2000.values - de_s) * 3600
    out = dict(frame_tie_dE=float(np.median(dE)), frame_tie_dN=float(np.median(dN)))
    ind = ast[ast.image.isin(INDEPENDENT)]; t0 = ind.epoch.values - ind.epoch.values.min()
    for comp_, errc, starc, mu in [("tgt_N", "err_N", "star_N", star.pmdec / 1000), ("tgt_E", "err_E", "star_E", star.pmra / 1000)]:
        y = ind[comp_].values; Wt = 1 / ind[errc].values ** 2
        Am = np.vstack([np.ones_like(t0), t0]).T; cov = np.linalg.inv(Am.T @ (Am * Wt[:, None])); beta = cov @ (Am.T @ (Wt * y))
        out[f"{comp_}_mu_mas_yr"] = float(beta[1] * 1000); out[f"{comp_}_mu_err"] = float(np.sqrt(cov[1, 1]) * 1000)
        out[f"{comp_}_chi2_stationary"] = float(np.sum(Wt * (y - np.average(y, weights=Wt)) ** 2))
        off = np.average(y - mu * t0, weights=Wt); out[f"{comp_}_chi2_comoving"] = float(np.sum(Wt * (y - off - mu * t0) ** 2))
        out[f"{comp_}_chi2_at_star"] = float(np.sum(Wt * (y - ind[starc].values) ** 2))
    out["chi2_at_star_2d"] = out["tgt_N_chi2_at_star"] + out["tgt_E_chi2_at_star"]; out["dof_at_star_2d"] = 2 * len(ind)
    from scipy.stats import chi2 as _c2
    out["p_at_star_2d"] = float(_c2.sf(out["chi2_at_star_2d"], out["dof_at_star_2d"]))
    # the test is whether the emission is located at the star; stationary vs co-moving-with-offset is not decided here
    out["verdict"] = "not coincident" if out["p_at_star_2d"] < 1e-4 else "inconclusive"
    json.dump(out, open(os.path.join(C.DER, "lspm1943_astrometry.json"), "w"), indent=1); print(json.dumps(out, indent=1))

    # figure
    plt = C.set_plot_style(); from matplotlib.patches import Ellipse
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    tt = np.linspace(2018.8, 2025, 50)
    trk = C.propagate(star.ra, star.dec, star.pmra, star.pmdec, tt, plx=star.parallax)
    for ax, comp_, errc, ytrk, lab in [(a0, "tgt_E", "err_E", (trk[0] - RA0) * 3600 * cosd, "\u0394RA cos\u03b4"),
                                       (a1, "tgt_N", "err_N", (trk[1] - DE0) * 3600, "\u0394Dec")]:
        ax.plot(tt, ytrk, color="#3b6ea8", lw=1.4, label="Star (Gaia DR3, with parallax)")
        for _, rw in ast.iterrows():
            used = rw.image in INDEPENDENT
            ax.errorbar(rw.epoch + (0 if used else 0.08), rw[comp_], yerr=rw[errc], fmt="o" if used else "s", mfc="#c0392b" if used else "white",
                        mec="#c0392b", ecolor="#c0392b", ms=4.5, capsize=2, lw=0.8)
        ax.set_xlabel("Epoch (yr)"); ax.set_ylabel(lab + " from 2019 radio position (arcsec)"); ax.margins(0.05)
    a1.errorbar([], [], fmt="o", mfc="#c0392b", mec="#c0392b", color="#c0392b", label="Radio (SE 1.2, QL 2.2, SE 3.2)")
    a1.errorbar([], [], fmt="s", mfc="white", mec="#c0392b", color="#c0392b", label="QL 1.2 / 3.2 (same data as SE)")
    a1.set_ylim(-1.35, 1.25); a1.legend(frameon=False, fontsize=5.5, loc="lower right")
    a0.set_title("Right ascension", loc="left"); a1.set_title("Declination", loc="left")
    se3 = fits.open(paths[("SE", "3.2", "sci")])[0]; w3 = WCS(se3.header).celestial; d3 = np.squeeze(se3.data)
    ps = abs(w3.proj_plane_pixel_scales()[1].to(u.arcsec).value); xv, yv = [float(q) for q in w3.world_to_pixel(vl)]; hb = int(8 / ps)
    stamp = d3[int(yv)-hb:int(yv)+hb+1, int(xv)-hb:int(xv)+hb+1] * 1e3
    ext = [(int(xv)-hb-xv-0.5)*ps*-1, (int(xv)+hb-xv+0.5)*ps*-1, (int(yv)-hb-yv-0.5)*ps, (int(yv)+hb-yv+0.5)*ps]
    imh = a2.imshow(stamp, origin="lower", cmap="magma", extent=ext, vmin=-0.3, vmax=1.1)
    a2.contour(stamp, levels=[0.375, 0.625, 0.875], colors="w", linewidths=0.5, extent=ext, origin="lower")
    r3 = ast[ast.image == "SE 3.2"].iloc[0]
    for o, cc, mk, lab in [(star, "#5dade2", "+", "Star, 2024.4"), (comp, "#58d68d", "x", "Companion, 2024.4")]:
        sa, sd = C.propagate(o.ra, o.dec, o.pmra, o.pmdec, r3.epoch, plx=o.parallax)
        a2.plot((sa - RA0) * 3600 * cosd + r3.frame_off_E, (sd - DE0) * 3600 + r3.frame_off_N, mk, color=cc, ms=9, mew=1.4, label=lab)
    a2.set_xlim(8, -8); a2.set_ylim(-8, 8); a2.set_xlabel("\u0394RA (arcsec, E left)"); a2.set_ylabel("\u0394Dec (arcsec)")
    a2.set_title("VLASS 3.2 SE image, 2024.4", loc="left"); a2.legend(frameon=False, fontsize=6, loc="lower left", labelcolor="w")
    cb = fig.colorbar(imh, ax=a2, fraction=0.046, pad=0.02); cb.set_label("mJy beam$^{-1}$")
    C.panel_letter(a0, "a"); C.panel_letter(a1, "b"); C.panel_letter(a2, "c"); fig.tight_layout()
    fig.savefig(os.path.join(C.RES, "lspm1943_vlass_multiepoch_astrometry.png"), dpi=300); fig.savefig(os.path.join(C.RES, "lspm1943_vlass_multiepoch_astrometry.pdf"), dpi=300)
