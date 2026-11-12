"""01 - Gaia DR3 sources within 100 pc (parallax > 10 mas, parallax/error > 10, dec > -45 deg).
Output: data/raw/gaia_dr3_plx10.parquet  (expected: 321,417 rows)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH-safe
import common as C
import os
import pyvo

OUT = os.path.join(C.RAW, "gaia_dr3_plx10.parquet")
QUERY = """SELECT g.source_id, g.ra, g.dec, g.ra_error, g.dec_error, g.parallax, g.parallax_error, g.pmra, g.pmdec,
g.pmra_error, g.pmdec_error, g.radial_velocity, g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
g.ruwe, g.phot_bp_rp_excess_factor, g.non_single_star, g.phot_variable_flag, g.astrometric_params_solved,
g.ipd_frac_multi_peak, g.duplicated_source, g.l, g.b
FROM gaiadr3.gaia_source AS g
WHERE g.parallax > 10 AND g.parallax_over_error > 10 AND g.dec > -45"""

if __name__ == "__main__":
    svc = pyvo.dal.TAPService(C.GAIA_TAP)
    job = svc.submit_job(QUERY, maxrec=2000000)
    job.run(); job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=3600)
    if job.phase != "COMPLETED":
        raise SystemExit(f"Gaia job ended in phase {job.phase}")
    df = job.fetch_result().to_table().to_pandas()
    df["phot_variable_flag"] = df["phot_variable_flag"].astype(str)   # mixed-width strings break FITS writers
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} rows")
