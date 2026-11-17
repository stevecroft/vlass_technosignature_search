"""05 - LSPM J1943+5337: archival optical imaging at the VLASS position while the high-proper-motion star
was displaced (POSS-I 1953: 9.8"; POSS-II 1989-92: ~4"; PS1 ~2012: 0.8"). Ring-aperture photometry cancels the
stellar halo: the aperture at the radio position is compared with apertures at the same radius from the star's
epoch position at other position angles (excluding +-40 deg around the target and companion directions).
Limits are calibrated against PS1 stars in a 4' field.

Inputs : data/derived/matches_10arcsec.parquet (for the VLASS component position)
Outputs: data/raw/lspm1943/{dss_*.fits, ps1_*.fits, ps1_calibration.csv, gaia_neighbours.csv, gaia_astrometry_errors.csv},
         data/derived/lspm1943_gaia_position_errors.csv
         results/lspm1943_imaging_limits.csv, results/lspm1943_archival_imaging.png
Expected: no source at the radio position; 3-sigma limits g~20.4 / r~19.7 (POSS-I), ~20.9 (POSS-II B_J, R_F)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import io, requests
import numpy as np, pandas as pd, pyvo
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

STAR = 2137474969751117312          # LSPM J1943+5337
COMPANION = 2137474969749746048     # co-moving Gaia DR3 neighbour
IMG = os.path.join(C.RAW, "lspm1943"); os.makedirs(IMG, exist_ok=True)
DSS = ["poss1_blue", "poss1_red", "poss2ukstu_blue", "poss2ukstu_red", "poss2ukstu_ir"]
LABEL = {"dss_poss1_blue": "POSS-I O", "dss_poss1_red": "POSS-I E", "dss_poss2ukstu_blue": "POSS-II B_J",
         "dss_poss2ukstu_red": "POSS-II R_F", "dss_poss2ukstu_ir": "POSS-II I_N",
         "ps1_g": "PS1 g", "ps1_r": "PS1 r", "ps1_i": "PS1 i", "ps1_z": "PS1 z", "ps1_y": "PS1 y"}
BAND = {"dss_poss1_blue": "g", "dss_poss1_red": "r", "dss_poss2ukstu_blue": "g", "dss_poss2ukstu_red": "r",
        "dss_poss2ukstu_ir": "i", "ps1_g": "g", "ps1_r": "r", "ps1_i": "i", "ps1_z": "z", "ps1_y": "y"}
PS1_EPOCH = 2012.3                   # PS1 DR2 mean epoch of the star (MJD 56020)


def fetch(ra0, de0):
    for sv in DSS:
        f = os.path.join(IMG, f"dss_{sv}.fits")
        if not os.path.exists(f):
            b = requests.get(f"https://archive.stsci.edu/cgi-bin/dss_search?v={sv}&r={ra0:.6f}&d={de0:.6f}&e=J2000&h=4.0&w=4.0&f=fits", timeout=180).content
            assert b[:6] == b"SIMPLE"; open(f, "wb").write(b)
    fl = requests.get(f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra0}&dec={de0}&filters=grizy&type=stack", timeout=60).text
    for _, row in pd.read_csv(io.StringIO(fl), sep=" ").iterrows():
        f = os.path.join(IMG, f"ps1_{row['filter']}.fits")
        if not os.path.exists(f):   # NB: fitscut needs ra=/dec=; x=/y= are skycell pixels and return the wrong region
            b = requests.get("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi", params={"red": row.filename, "format": "fits", "ra": ra0, "dec": de0, "size": 960}, timeout=180).content
            assert b[:6] == b"SIMPLE"; open(f, "wb").write(b)
    f = os.path.join(IMG, "ps1_calibration.csv")
    if not os.path.exists(f):
        t = requests.get("https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv", params={"ra": ra0, "dec": de0, "radius": 150 / 3600, "nDetections.gte": 5,
              "columns": "[raMean,decMean,gMeanPSFMag,rMeanPSFMag,iMeanPSFMag,zMeanPSFMag,yMeanPSFMag,iMeanKronMag]"}, timeout=120).text
        open(f, "w").write(t)
    f = os.path.join(IMG, "gaia_neighbours.csv")
    if not os.path.exists(f):
        q = f"""SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec,
        DISTANCE(POINT({ra0},{de0}), POINT(ra,dec))*3600 AS sep_vlass FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT(ra,dec), CIRCLE({ra0},{de0},{20/3600}))"""
        pyvo.dal.TAPService(C.GAIA_TAP).search(q).to_table().to_pandas().to_csv(f, index=False)


def load(name):
    hd = fits.open(os.path.join(IMG, name + ".fits"))[0]; im = hd.data.astype(float); h = hd.header
    if "BSOFTEN" in h:   # PS1 asinh-compressed stacks
        a = 2.5 / np.log(10); im = h["BOFFSET"] + h["BSOFTEN"] * 2 * np.sinh(im / a)
    ep = Time(h["DATE-OBS"].strip()[:10]).jyear if name.startswith("dss") else PS1_EPOCH   # DSS times are malformed
    return im, WCS(h), ep


def apsum(im, x, y, r):
    Y, X = np.ogrid[:im.shape[0], :im.shape[1]]; m = (X - x) ** 2 + (Y - y) ** 2 <= r ** 2
    return np.nansum(im[m])


def measure(name, vl, star, comp, pcal, rng):
    im, w, ep = load(name); ps = abs(w.proj_plane_pixel_scales()[1].to(u.arcsec).value)
    rap = max((2.0 if name.startswith("dss") else 1.0) / ps, 1.2)
    xv, yv = [float(v) for v in w.world_to_pixel(vl)]
    xs, ys = [float(v) for v in w.world_to_pixel(SkyCoord(*C.propagate(star.ra, star.dec, star.pmra, star.pmdec, ep, plx=star.parallax), unit="deg"))]
    xc, yc = [float(v) for v in w.world_to_pixel(SkyCoord(*C.propagate(comp.ra, comp.dec, comp.pmra, comp.pmdec, ep, plx=comp.parallax), unit="deg"))]
    imb = im - np.nanmedian(im); R = np.hypot(xv - xs, yv - ys)
    thv, thc = np.arctan2(yv - ys, xv - xs), np.arctan2(yc - ys, xc - xs)
    angd = lambda a, b: np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    ring = np.array([apsum(imb, xs + R * np.cos(t), ys + R * np.sin(t), rap) for t in np.linspace(0, 2 * np.pi, 90, endpoint=False)
                     if angd(t, thv) > np.deg2rad(40) and angd(t, thc) > np.deg2rad(40)])
    tgt = apsum(imb, xv, yv, rap); sig_ring = 1.4826 * np.median(np.abs(ring - np.median(ring)))
    cs = SkyCoord(pcal.raMean.values * u.deg, pcal.decMean.values * u.deg); xp, yp = w.world_to_pixel(cs); sky = []
    for _ in range(6000):        # blank-sky apertures: >8" from any PS1 source, >20" from the star
        xa, ya = rng.uniform(10, im.shape[1] - 10), rng.uniform(10, im.shape[0] - 10)
        if np.min(np.hypot(xp - xa, yp - ya)) * ps < 8 or np.hypot(xa - xs, ya - ys) * ps < 20: continue
        v_ = apsum(imb, xa, ya, rap)
        if np.isfinite(v_): sky.append(v_)
        if len(sky) >= 300: break
    sky = np.array(sky); sig_sky = 1.4826 * np.median(np.abs(sky - np.median(sky)))
    mags = pcal[f"{BAND[name]}MeanPSFMag"].values
    ok = (xp > rap + 3) & (yp > rap + 3) & (xp < im.shape[1] - rap - 3) & (yp < im.shape[0] - rap - 3) & (cs.separation(vl).arcsec > 15)
    fl = np.array([apsum(imb, x, y, rap) - np.median(sky) for x, y in zip(xp, yp)])
    sel = ok & (mags > 17.5) & (mags < 21.0) & (fl > 0) & np.isfinite(fl)
    zpv = mags[sel] + 2.5 * np.log10(fl[sel]); zp = np.median(zpv)
    return dict(image=LABEL[name], epoch=round(ep, 2), PS1_band=BAND[name], star_offset_arcsec=round(R * ps, 2),
                ap_radius_arcsec=round(rap * ps, 2), n_ring=len(ring), SNR_vs_ring=round((tgt - np.median(ring)) / sig_ring, 2),
                SNR_vs_sky=round((tgt - np.median(sky)) / sig_sky, 2), ring_over_sky_noise=round(sig_ring / sig_sky, 2),
                n_cal=int(sel.sum()), zp_scatter_mag=round(1.4826 * np.median(np.abs(zpv - zp)), 2),
                mag_limit_3sigma=round(zp - 2.5 * np.log10(3 * max(sig_ring, sig_sky)), 2),
                sky_only_limit_3sigma=round(zp - 2.5 * np.log10(3 * sig_sky), 2))


def figure(vl, star, comp, res):
    plt = C.set_plot_style(); from matplotlib.patches import Ellipse; from matplotlib.lines import Line2D
    show = ["dss_poss1_blue", "dss_poss1_red", "dss_poss2ukstu_blue", "dss_poss2ukstu_red", "dss_poss2ukstu_ir", "ps1_g", "ps1_r", "ps1_i"]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.0))
    for ax, name in zip(axs.flat, show):
        im, w, ep = load(name); ps = abs(w.proj_plane_pixel_scales()[1].to(u.arcsec).value)
        xv, yv = [float(v) for v in w.world_to_pixel(vl)]; half = 12 / ps
        cut = im[int(yv - half) - 2:int(yv + half) + 3, int(xv - half) - 2:int(xv + half) + 3]
        lo, hi = np.nanpercentile(cut, [10, 99.5])
        ax.imshow(np.arcsinh((im - lo) / (0.05 * (hi - lo))), origin="lower", cmap="gray_r", vmin=0, vmax=np.arcsinh(20))
        ax.add_patch(Ellipse((xv, yv), 2 * 3.34 / ps, 2 * 1.32 / ps, angle=90 + 108.95, fill=False, ec="#c0392b", lw=0.9))
        for o, cc, mk, ms in [(star, "#3b6ea8", "+", 7), (comp, "#2e8b57", "s", 8)]:   # open square: leaves the white dwarf visible
            xs, ys = [float(v) for v in w.world_to_pixel(SkyCoord(*C.propagate(o.ra, o.dec, o.pmra, o.pmdec, ep, plx=o.parallax), unit="deg"))]
            ax.plot(xs, ys, mk, ms=ms, mew=1.0, color=cc, mfc="none")
        ax.set_xlim(xv - half, xv + half); ax.set_ylim(yv - half, yv + half); ax.set_xticks([]); ax.set_yticks([])
        rr = res[res.image == LABEL[name]].iloc[0]
        ax.set_title(f"{LABEL[name]}, {ep:.1f}", loc="left", fontsize=7)
        ax.text(0.03, 0.04, f"star {rr.star_offset_arcsec:.1f}\u2033 away\n3\u03c3 limit {rr.mag_limit_3sigma:.1f}", transform=ax.transAxes,
                fontsize=6, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    h = [Line2D([], [], color="#c0392b", lw=0.9, label="VLASS component (deconvolved size, 2019.2)"),
         Line2D([], [], marker="+", ls="", color="#3b6ea8", mew=1, label="LSPM J1943+5337 at image epoch"),
         Line2D([], [], marker="s", ls="", color="#2e8b57", mfc="none", mew=1, label=f"Co-moving companion (Gaia DR3 {COMPANION}) at image epoch")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.subplots_adjust(hspace=0.28, top=0.94, bottom=0.12)
    fig.savefig(os.path.join(C.RES, "lspm1943_archival_imaging.png"), dpi=300); fig.savefig(os.path.join(C.RES, "lspm1943_archival_imaging.pdf"), dpi=300)


if __name__ == "__main__":
    A = pd.read_parquet(os.path.join(C.DER, "matches_10arcsec.parquet"))
    sv = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet"))
    m = A[A.source_id == STAR].sort_values("sep").iloc[0]
    ra0, de0 = m.RAJ2000, m.DEJ2000; vl = SkyCoord(ra0 * u.deg, de0 * u.deg)
    fetch(ra0, de0)
    nb = pd.read_csv(os.path.join(IMG, "gaia_neighbours.csv"))
    star = sv[sv.source_id == STAR].iloc[0]; comp = nb[nb.source_id == COMPANION].iloc[0]
    # Gaia DR3 astrometric uncertainties of the star and companion, propagated to each image epoch (linear, including the
    # position-proper-motion correlation, with the parallax error added in quadrature as an upper bound on its contribution)
    fe = os.path.join(IMG, "gaia_astrometry_errors.csv")
    if not os.path.exists(fe):
        q = f"""SELECT source_id, ref_epoch, ra_error, dec_error, parallax_error, pmra_error, pmdec_error, ra_pmra_corr, dec_pmdec_corr
        FROM gaiadr3.gaia_source WHERE source_id IN ({STAR}, {COMPANION})"""
        pyvo.dal.TAPService(C.GAIA_TAP).search(q).to_table().to_pandas().to_csv(fe, index=False)
    ge = pd.read_csv(fe); rows = []
    for _, g in ge.iterrows():
        for nm in LABEL:
            dt = load(nm)[2] - g.ref_epoch
            sa = np.sqrt(g.ra_error**2 + dt**2 * g.pmra_error**2 + 2 * dt * g.ra_pmra_corr * g.ra_error * g.pmra_error + g.parallax_error**2)
            sd = np.sqrt(g.dec_error**2 + dt**2 * g.pmdec_error**2 + 2 * dt * g.dec_pmdec_corr * g.dec_error * g.pmdec_error + g.parallax_error**2)
            rows.append(dict(source_id=int(g.source_id), object="star" if g.source_id == STAR else "companion", image=LABEL[nm],
                             epoch=round(dt + g.ref_epoch, 2), sig_ra_mas=sa, sig_dec_mas=sd))
    pd.DataFrame(rows).to_csv(os.path.join(C.DER, "lspm1943_gaia_position_errors.csv"), index=False)
    pcal = pd.read_csv(os.path.join(IMG, "ps1_calibration.csv"))
    pcal = pcal[(pcal.iMeanPSFMag > 0) & (np.abs(pcal.iMeanPSFMag - pcal.iMeanKronMag) < 0.1)]
    rng = np.random.default_rng(3)
    res = pd.DataFrame([measure(n, vl, star, comp, pcal, rng) for n in LABEL])
    res.to_csv(os.path.join(C.RES, "lspm1943_imaging_limits.csv"), index=False)
    print(res[["image", "epoch", "star_offset_arcsec", "SNR_vs_ring", "mag_limit_3sigma"]].to_string(index=False))
    figure(vl, star, comp, res)
