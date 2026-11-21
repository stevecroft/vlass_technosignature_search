"""24 - Positional context of the radio emission offset from TYC 3573-1391-1 (Gaia DR3 2071061615589030400).

SIMBAD identifications elsewhere in the pipeline are matched by Gaia DR3 identifier, so an object distinct from the
star at the offset radio position would not be found. This script (1) forms the radio position as the mean of the
per-epoch catalog positions (star position propagated to each image epoch plus the catalog offset recorded by 21),
(2) cone-searches SIMBAD (basic table, 60") and NVSS (VIII/65, 60") and Gaia DR3 (I/355, 10") around it, and
(3) records the projected separation from the star. Query results are cached in data/raw/tyc3573_field_*.csv so that
re-runs are offline.
Inputs : data/derived/epoch_catalog_matches.parquet, data/derived/alleepoch_forced.parquet, data/derived/ms_sample_vlass.parquet
Outputs: data/derived/tyc3573_field.json
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import json
import numpy as np, pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

SID = 2071061615589030400
R_SIMBAD, R_NVSS, R_GAIA = 60.0, 60.0, 10.0      # arcsec


def cached(name, fetch):
    f = os.path.join(C.RAW, f"tyc3573_field_{name}.csv")
    if os.path.exists(f): return pd.read_csv(f)
    df = fetch(); df.to_csv(f, index=False); return df


def viz(cat, ra, de, r, cols):
    def go():
        txt = C.viz_get({"-source": cat, "-c": f"{ra} {de}", "-c.rs": r, "-out.max": 50, "-out.add": "_r",
                         "-sort": "_r", "-out": cols}, timeout=120)
        df = C.vizier_tsv_to_df(txt)
        return pd.DataFrame(columns=["_r"] + cols.split(",")) if df is None else df
    return go


if __name__ == "__main__":
    st = pd.read_parquet(os.path.join(C.DER, "ms_sample_vlass.parquet")).set_index("source_id").loc[SID]
    m = pd.read_parquet(os.path.join(C.DER, "epoch_catalog_matches.parquet"))
    m = m[(m.source_id == SID) & m.m_dE.notna()].copy()
    fe = pd.read_parquet(os.path.join(C.DER, "alleepoch_forced.parquet"), columns=["source_id", "tag", "ep"])
    ep_tag = fe[fe.source_id == SID].set_index("tag").ep
    m["epoch"] = np.where(m.m_MJD.notna(), Time(m.m_MJD.fillna(51544.0).values, format="mjd").decimalyear,
                          m.tag.map(ep_tag).values)            # the CIRADA Epoch-1 catalog has no MJD: forced-photometry epoch
    pos = []
    for r in m.itertuples():
        ra, de = C.propagate(np.array([st.ra]), np.array([st.dec]), np.array([st.pmra]), np.array([st.pmdec]), r.epoch)
        pos.append(SkyCoord(ra[0] * u.deg, de[0] * u.deg).spherical_offsets_by(r.m_dE * u.arcsec, r.m_dN * u.arcsec))
    P = SkyCoord([p.ra for p in pos], [p.dec for p in pos])
    c = SkyCoord(P.ra.deg.mean() * u.deg, P.dec.deg.mean() * u.deg)
    ep_mean = float(m.epoch.mean())
    ra0, de0 = C.propagate(np.array([st.ra]), np.array([st.dec]), np.array([st.pmra]), np.array([st.pmdec]), ep_mean)
    sep_star = float(SkyCoord(ra0[0] * u.deg, de0[0] * u.deg).separation(c).arcsec)

    def simbad():
        import pyvo
        q = (f"SELECT b.main_id, b.otype, b.plx_value, DISTANCE(POINT('ICRS', b.ra, b.dec), POINT('ICRS', {c.ra.deg}, {c.dec.deg}))*3600 AS sep "
             f"FROM basic AS b WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {c.ra.deg}, {c.dec.deg}, {R_SIMBAD / 3600}))=1 ORDER BY sep")
        return pyvo.dal.TAPService(C.SIMBAD_TAP).search(q).to_table().to_pandas()
    sb = cached("simbad", simbad)
    nv = cached("nvss", viz("VIII/65/nvss", c.ra.deg, c.dec.deg, R_NVSS, "NVSS,RAJ2000,DEJ2000,e_RAJ2000,e_DEJ2000,S1.4,e_S1.4"))
    ga = cached("gaia", viz("I/355/gaiadr3", c.ra.deg, c.dec.deg, R_GAIA, "Source,Gmag,Plx,e_Plx"))

    others = sb[sb.main_id.str.strip() != "TYC 3573-1391-1"]
    out = {"radio_ra": c.ra.deg, "radio_dec": c.dec.deg, "radio_l": c.galactic.l.deg, "radio_b": c.galactic.b.deg,
           "n_catalog_positions": int(len(m)), "position_scatter_arcsec": [round(float(x), 3) for x in P.separation(c).arcsec],
           "mean_epoch": ep_mean, "sep_star_arcsec": sep_star, "proj_sep_au": sep_star * float(st.dist_pc),
           "simbad_radius_arcsec": R_SIMBAD, "simbad_n_other": int(len(others)),
           "simbad_nearest_other_arcsec": float(others.sep.min()) if len(others) else None,
           "simbad_nearest_other": others.sort_values("sep").main_id.iloc[0].strip() if len(others) else None,
           "gaia_nearest_other_arcsec": float(ga[ga.Source != SID]._r.min()) if (ga.Source != SID).any() else None,
           "gaia_star_arcsec": float(ga[ga.Source == SID]._r.iloc[0]), "star_G": float(st.phot_g_mean_mag)}
    if len(nv):
        n = nv.sort_values("_r").iloc[0]
        nc = SkyCoord(str(n.RAJ2000), str(n.DEJ2000), unit=(u.hourangle, u.deg))
        dra, dde = c.spherical_offsets_to(nc)
        era = float(n.e_RAJ2000) * 15 * np.cos(np.radians(c.dec.deg))       # e_RAJ2000 is in seconds of time
        out.update({"nvss_name": "NVSS J" + str(n.NVSS).strip(), "nvss_sep_arcsec": float(n._r), "nvss_S_mJy": float(n["S1.4"]),
                    "nvss_eS_mJy": float(n["e_S1.4"]), "nvss_offset_sigma": float(np.hypot(dra.arcsec / era, dde.arcsec / float(n.e_DEJ2000)))})
    json.dump(out, open(os.path.join(C.DER, "tyc3573_field.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
